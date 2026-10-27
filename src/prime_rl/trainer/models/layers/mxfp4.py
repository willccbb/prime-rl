"""Frozen MXFP4 expert weights.

Checkpoints such as ``openai/gpt-oss-20b`` ship routed-expert weights as MXFP4: two e2m1 values
per byte plus one e8m0 scale per 32 values along the input dimension. Every MXFP4 value is exactly
representable in bfloat16, so dequantizing reproduces the checkpoint's weights bit for bit.

``Mxfp4GroupedExperts`` keeps those bytes on the GPU in place of bfloat16 (or fp32) expert
parameters. Each grouped matmul dequantizes one projection just before use and dequantizes it
again in backward for the input gradient, so no bfloat16 copy of the experts outlives the call.
The weights are frozen; adapters (LoRA) train on top.
"""

import torch
import triton
import triton.language as tl
from torch import nn

from prime_rl.trainer.models.layers.activations import ActivationDispatch
from prime_rl.trainer.models.layers.expert_compute import BF16ExpertCompute, broadcast_expert_bias, to_local
from prime_rl.trainer.models.layers.moe import GroupedExperts

MXFP4_BLOCK_SIZE = 32
_PACKED_BLOCK_BYTES = MXFP4_BLOCK_SIZE // 2


@triton.jit
def _e2m1_to_float(nibble):
    # e2m1: 1 sign, 2 exponent and 1 mantissa bit. Exponent 0 is subnormal (0 or 0.5).
    exponent = (nibble >> 1) & 0x3
    mantissa = (nibble & 0x1).to(tl.float32)
    magnitude = tl.where(
        exponent == 0,
        mantissa * 0.5,
        (1.0 + mantissa * 0.5) * tl.exp2((exponent - 1).to(tl.float32)),
    )
    return tl.where((nibble & 0x8) != 0, -magnitude, magnitude)


@triton.jit
def _mxfp4_dequantize_kernel(blocks_ptr, scales_ptr, out_ptr, num_bytes, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_bounds = offsets < num_bytes
    packed = tl.load(blocks_ptr + offsets, mask=in_bounds, other=0).to(tl.int32)
    scale_exponent = tl.load(scales_ptr + offsets // 16, mask=in_bounds, other=127).to(tl.float32)
    # exp2 flushes the subnormal 2^-127 (exponent 0) to zero, so that one scale is 2^-126 * 0.5.
    scale = tl.exp2(tl.maximum(scale_exponent, 1.0) - 127.0)
    half = tl.where(scale_exponent == 0, 0.5, 1.0)
    # The low nibble holds the even element, the high nibble the odd one.
    low = (_e2m1_to_float(packed & 0xF) * scale * half).to(tl.bfloat16)
    high = (_e2m1_to_float(packed >> 4) * scale * half).to(tl.bfloat16)
    tl.store(out_ptr + 2 * offsets, low, mask=in_bounds)
    tl.store(out_ptr + 2 * offsets + 1, high, mask=in_bounds)


@torch.library.custom_op("prime_rl::mxfp4_dequantize", mutates_args=())
def mxfp4_dequantize(blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """``blocks`` ``[..., groups, 16]`` uint8 and ``scales`` ``[..., groups]`` uint8 -> ``[..., groups * 32]`` bf16."""
    blocks, scales = blocks.contiguous(), scales.contiguous()
    out = torch.empty(
        (*scales.shape[:-1], scales.shape[-1] * MXFP4_BLOCK_SIZE), dtype=torch.bfloat16, device=blocks.device
    )
    num_bytes = blocks.numel()
    block = 1024
    _mxfp4_dequantize_kernel[(triton.cdiv(num_bytes, block),)](blocks, scales, out, num_bytes, BLOCK=block)
    return out


@mxfp4_dequantize.register_fake
def _(blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return blocks.new_empty((*scales.shape[:-1], scales.shape[-1] * MXFP4_BLOCK_SIZE), dtype=torch.bfloat16)


class _FrozenMxfp4GroupedMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, blocks: torch.Tensor, scales: torch.Tensor, offsets: torch.Tensor):
        ctx.save_for_backward(blocks, scales, offsets)
        weight = mxfp4_dequantize(blocks, scales)
        return torch._grouped_mm(x, weight.transpose(-2, -1), offs=offsets)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        blocks, scales, offsets = ctx.saved_tensors
        weight = mxfp4_dequantize(blocks, scales)
        return torch._grouped_mm(grad_output.contiguous(), weight, offs=offsets), None, None, None


def frozen_mxfp4_grouped_mm(
    x: torch.Tensor, blocks: torch.Tensor, scales: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    """``x [tokens, in] @ W[e]^T`` per expert group, with ``W [experts, out, in]`` stored as MXFP4."""
    return _FrozenMxfp4GroupedMM.apply(x.bfloat16(), blocks, scales, offsets)


class Mxfp4GroupedExperts(GroupedExperts):
    """Gated routed experts whose projection weights stay in the checkpoint's MXFP4 format.

    Each projection ``{gate,up,down}_proj`` is a pair of frozen uint8 buffers, ``*_blocks``
    ``[experts, out, in / 32, 16]`` and ``*_scales`` ``[experts, out, in / 32]``. Biases stay bf16.
    """

    projections = ("gate_proj", "up_proj", "down_proj")

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        *,
        activation: str,
        bias: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        self.num_experts = num_experts
        self.dim = dim
        self.hidden_dim = hidden_dim
        for proj, (out_features, in_features) in zip(
            self.projections, ((hidden_dim, dim), (hidden_dim, dim), (dim, hidden_dim))
        ):
            if in_features % MXFP4_BLOCK_SIZE:
                raise ValueError(f"MXFP4 {proj} input dim {in_features} is not a multiple of {MXFP4_BLOCK_SIZE}")
            groups = in_features // MXFP4_BLOCK_SIZE
            self.register_buffer(
                f"{proj}_blocks",
                torch.empty(num_experts, out_features, groups, _PACKED_BLOCK_BYTES, dtype=torch.uint8),
            )
            self.register_buffer(f"{proj}_scales", torch.empty(num_experts, out_features, groups, dtype=torch.uint8))
        self.gate_proj = self.up_proj = self.down_proj = None
        self.register_parameter("gate_up_proj", None)
        self.gate_proj_bias = nn.Parameter(torch.empty(num_experts, hidden_dim)) if bias else None
        self.up_proj_bias = nn.Parameter(torch.empty(num_experts, hidden_dim)) if bias else None
        self.down_proj_bias = nn.Parameter(torch.empty(num_experts, dim)) if bias else None
        self.activation = ActivationDispatch[activation]
        self.supported_fusions = {}
        self.compute = BF16ExpertCompute()

    def packed_weight_numel(self) -> int:
        """Weight elements held in the packed buffers (two per byte)."""
        return sum(2 * getattr(self, f"{proj}_blocks").numel() for proj in self.projections)

    def project(self, proj: str, x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return frozen_mxfp4_grouped_mm(x, getattr(self, f"{proj}_blocks"), getattr(self, f"{proj}_scales"), offsets)

    def forward(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

        def biased(proj: str, inputs: torch.Tensor) -> torch.Tensor:
            out = self.project(proj, inputs, offsets)
            bias = getattr(self, f"{proj}_bias")
            if bias is None:
                return out
            return out + broadcast_expert_bias(to_local(bias), num_tokens_per_expert, out.shape[0]).bfloat16()

        hidden = self.activation.apply(biased("gate_proj", x), biased("up_proj", x))
        return biased("down_proj", hidden).type_as(x)

    def init_weights(self, init_std: float) -> None:
        for bias in (self.gate_proj_bias, self.up_proj_bias, self.down_proj_bias):
            if bias is not None:
                nn.init.zeros_(bias)

    def set_compute(self, compute) -> None:
        if not isinstance(compute, BF16ExpertCompute):
            raise ValueError("MXFP4 experts dequantize to bf16 grouped GEMMs; model.moe.compute must stay bf16")
        self.compute = compute
