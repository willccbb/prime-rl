import pytest
import torch

from prime_rl.trainer.models.deepseek_v4.dequantize import FP4_E2M1_LUT
from prime_rl.trainer.models.gpt_oss.converting_gpt_oss import GptOssExperts
from prime_rl.trainer.models.layers.lora.experts import LoRAGptOssGroupedExperts
from prime_rl.trainer.models.layers.moe import GroupedExperts
from prime_rl.trainer.models.layers.mxfp4 import Mxfp4GroupedExperts, frozen_mxfp4_grouped_mm, mxfp4_dequantize

pytestmark = [pytest.mark.gpu]


def _random_mxfp4(*shape: int, generator: torch.Generator, scale_range=(118, 130)) -> tuple[torch.Tensor, torch.Tensor]:
    """Random packed blocks ``[*shape, 16]`` and e8m0 scales ``[*shape]``."""
    blocks = torch.randint(0, 256, (*shape, 16), dtype=torch.uint8, generator=generator)
    scales = torch.randint(*scale_range, shape, dtype=torch.uint8, generator=generator)
    return blocks.cuda(), scales.cuda()


def _reference_dequantize(blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    lut = FP4_E2M1_LUT.to(blocks.device)
    values = torch.stack((lut[(blocks & 0xF).long()], lut[(blocks >> 4).long()]), dim=-1).flatten(-2)
    return (values * torch.exp2(scales.float() - 127).unsqueeze(-1)).flatten(-2).bfloat16()


def test_mxfp4_dequantize_matches_reference_bitwise():
    generator = torch.Generator().manual_seed(0)
    # Every finite e8m0 exponent (255 is NaN).
    blocks, scales = _random_mxfp4(4, 96, 3, generator=generator, scale_range=(0, 255))
    assert torch.equal(mxfp4_dequantize(blocks, scales), _reference_dequantize(blocks, scales))


def test_frozen_mxfp4_grouped_mm_matches_bf16_weights():
    generator = torch.Generator().manual_seed(0)
    blocks, scales = _random_mxfp4(4, 64, 4, generator=generator)
    weight = _reference_dequantize(blocks, scales)
    counts = torch.tensor([24, 0, 8, 32], device="cuda")
    offsets = torch.cumsum(counts, dim=0, dtype=torch.int32)
    x = torch.randn(int(counts.sum()), 128, device="cuda", dtype=torch.bfloat16)
    grad_output = torch.randn(x.shape[0], 64, device="cuda", dtype=torch.bfloat16)

    x_mxfp4 = x.clone().requires_grad_()
    out = frozen_mxfp4_grouped_mm(x_mxfp4, blocks, scales, offsets)
    out.backward(grad_output)

    x_bf16 = x.clone().requires_grad_()
    reference = torch._grouped_mm(x_bf16, weight.transpose(-2, -1), offs=offsets)
    reference.backward(grad_output)

    assert torch.equal(out, reference)
    assert torch.equal(x_mxfp4.grad, x_bf16.grad)


def test_lora_over_mxfp4_experts_matches_bf16_experts():
    num_experts, dim, hidden_dim = 4, 64, 64
    generator = torch.Generator().manual_seed(0)
    prefix = "experts"
    packed = {}
    for name in ("gate_up_proj", "down_proj"):
        rows = 2 * hidden_dim if name == "gate_up_proj" else dim
        packed[f"{prefix}.{name}_blocks"], packed[f"{prefix}.{name}_scales"] = _random_mxfp4(
            num_experts, rows, 2, generator=generator
        )
    packed[f"{prefix}.gate_up_proj_bias"] = torch.randn(num_experts, 2 * hidden_dim, device="cuda")
    packed[f"{prefix}.down_proj_bias"] = torch.randn(num_experts, dim, device="cuda")
    GptOssExperts(prefix).hf_to_prime(packed)
    packed = {name.removeprefix(f"{prefix}."): tensor for name, tensor in packed.items()}

    kwargs = dict(dim=dim, hidden_dim=hidden_dim, num_experts=num_experts, activation="clamped_swiglu", bias=True)
    with torch.device("cuda"):
        mxfp4_experts = Mxfp4GroupedExperts(**kwargs)
        bf16_experts = GroupedExperts(expert_type="gated", **kwargs)
    mxfp4_experts.load_state_dict(packed)
    bf16_experts.load_state_dict(
        {
            **{name: tensor for name, tensor in packed.items() if name.endswith("_bias")},
            **{
                proj: _reference_dequantize(packed[f"{proj}_blocks"], packed[f"{proj}_scales"])
                for proj in ("gate_proj", "up_proj", "down_proj")
            },
        }
    )

    torch.manual_seed(0)
    mxfp4_lora = LoRAGptOssGroupedExperts(mxfp4_experts, rank=8)
    bf16_lora = LoRAGptOssGroupedExperts(bf16_experts, rank=8)
    bf16_params = dict(bf16_lora.named_parameters())
    for name, mxfp4_param in mxfp4_lora.named_parameters():
        if mxfp4_param.requires_grad:
            mxfp4_param.data.normal_(0, 0.05)
            bf16_params[name].data.copy_(mxfp4_param.data)

    counts = torch.tensor([16, 0, 40, 8], device="cuda")
    x = torch.randn(int(counts.sum()), dim, device="cuda", dtype=torch.bfloat16)
    outputs, grads = [], []
    for lora in (mxfp4_lora, bf16_lora):
        out = lora(x, counts)
        trainable = [param for param in lora.parameters() if param.requires_grad]
        outputs.append(out)
        grads.append(torch.autograd.grad(out.float().square().sum(), trainable))

    assert torch.equal(outputs[0], outputs[1])
    for mxfp4_grad, bf16_grad in zip(*grads):
        assert torch.equal(mxfp4_grad, bf16_grad)
