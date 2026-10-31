import torch
from torch.distributed.tensor import DTensor

from prime_rl.trainer.models.layers.expert_compute import broadcast_expert_bias
from prime_rl.trainer.models.layers.lora.base import LoRAModule, lora_parameter, pad_rank
from prime_rl.trainer.models.layers.moe import GroupedExperts


def _to_local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _full(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach()
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def _grouped_mm(x: torch.Tensor, weight: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """x [tokens, in] @ weight [experts, out, in] per expert group."""
    return torch._grouped_mm(x.bfloat16(), _to_local(weight).bfloat16().transpose(-2, -1), offs=offsets)


def _dense(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """x [tokens, in] @ weight [out, in] for an adapter matrix shared by every expert."""
    return x.bfloat16() @ _to_local(weight).bfloat16().T


def _flat_a(a: torch.Tensor) -> torch.Tensor:
    """(num_experts, rank, in) -> (num_experts*rank, in)"""
    return a.reshape(-1, a.shape[-1]).clone()


def _flat_b(b: torch.Tensor) -> torch.Tensor:
    """(num_experts, out, rank) -> (out, rank*num_experts) with experts fast-varying"""
    return b.permute(1, 2, 0).contiguous().reshape(b.shape[1], -1)


def _check_grouped_mm_dims(*dims: int) -> None:
    if any(dim % 8 != 0 for dim in dims):
        raise ValueError("grouped_mm requires rank and expert dimensions divisible by 8")


class LoRAGroupedExperts(LoRAModule):
    """GroupedExperts + per-expert LoRA on each projection (gate_proj if gated, up_proj, down_proj).

    Exports the vLLM per-expert MoE LoRA format. Tokens arrive already permuted by the token dispatcher.
    """

    def __init__(self, base_layer: GroupedExperts, rank: int, alpha: float = 32.0, dropout: float = 0.0):
        super().__init__(base_layer, rank, alpha, dropout)
        self.num_experts = base_layer.num_experts
        gated = base_layer.gate_proj is not None
        self.projections = ("gate_proj", "down_proj", "up_proj") if gated else ("up_proj", "down_proj")
        _check_grouped_mm_dims(rank, *base_layer.down_proj.shape[1:])
        for proj in self.projections:
            weight = getattr(base_layer, proj)
            num_experts, out_features, in_features = weight.shape
            setattr(self, f"{proj}_lora_A", lora_parameter(num_experts, rank, in_features, like=weight))
            setattr(self, f"{proj}_lora_B", lora_parameter(num_experts, out_features, rank, like=weight))
        self.reset_parameters()

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        """Per-expert slices, e.g. ``{expert_id}.gate_proj.lora_A.weight`` (vLLM 2D MoE LoRA format)."""
        weights = {
            f"{proj}.lora_{ab}": _full(getattr(self, f"{proj}_lora_{ab}")) for proj in self.projections for ab in "AB"
        }
        # Clone so each tensor owns its storage instead of viewing the full stacked weight.
        return {
            f"{expert_id}.{name}.weight": weight[expert_id].clone()
            for expert_id in range(self.num_experts)
            for name, weight in weights.items()
        }

    def _project(self, proj: str, x: torch.Tensor, lora_x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        base = self.base_layer.project(proj, x, offsets)
        lora = _grouped_mm(
            _grouped_mm(lora_x, getattr(self, f"{proj}_lora_A"), offsets), getattr(self, f"{proj}_lora_B"), offsets
        )
        return base + self.scaling * lora

    def forward(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
        lora_x = self.lora_dropout(x)
        gate = self._project("gate_proj", x, lora_x, offsets) if "gate_proj" in self.projections else None
        up = self._project("up_proj", x, lora_x, offsets)
        h = self.base_layer.activation.apply(gate, up)
        return self._project("down_proj", h, self.lora_dropout(h), offsets).type_as(x)


class LoRAGptOssGroupedExperts(LoRAModule):
    """GPT-OSS GroupedExperts + LoRA.

    Keeps GPT-OSS's combined, interleaved gate/up adapter (vLLM 3D MoE LoRA format) on top of the split
    gate_proj/up_proj runtime weights.
    """

    def __init__(self, base_layer: GroupedExperts, rank: int, alpha: float = 32.0, dropout: float = 0.0):
        super().__init__(base_layer, rank, alpha, dropout)
        n, hidden_size, intermediate_size = base_layer.num_experts, base_layer.dim, base_layer.hidden_dim
        self.num_experts = n
        _check_grouped_mm_dims(rank, hidden_size, intermediate_size)
        # The biases share the expert weights' device and dtype, also when the weights are MXFP4 bytes.
        like = base_layer.up_proj_bias
        self.gate_up_lora_A = lora_parameter(n, rank, hidden_size, like=like)
        self.gate_up_lora_B = lora_parameter(n, 2 * intermediate_size, rank, like=like)
        self.down_lora_A = lora_parameter(n, rank, intermediate_size, like=like)
        self.down_lora_B = lora_parameter(n, hidden_size, rank, like=like)
        self.reset_parameters()

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        """vLLM 3D MoE adapter format (gpt-oss has ``is_3d_moe_weight = True`` in vLLM).

        ``base_layer.lora_{A,B}.weight`` is the gate_up projection and ``lora_{A,B}.weight`` the down
        projection, with experts stacked into the rank dim. vLLM's ``_stack_moe_lora_weights`` reshapes
            lora_A: (num_experts*rank, in)  -> (num_experts, rank, in)
            lora_B: (out, rank*num_experts) -> (out, rank, num_experts) -> (num_experts, out, rank)
        """
        return {
            "base_layer.lora_A.weight": _flat_a(_full(self.gate_up_lora_A)),
            "base_layer.lora_B.weight": _flat_b(_full(self.gate_up_lora_B)),
            "lora_A.weight": _flat_a(_full(self.down_lora_A)),
            "lora_B.weight": _flat_b(_full(self.down_lora_B)),
        }

    def forward(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        base = self.base_layer
        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

        def bias(tensor: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
            return broadcast_expert_bias(_to_local(tensor), num_tokens_per_expert, out.shape[0]).bfloat16()

        def lora(x: torch.Tensor, lora_A: torch.Tensor, lora_B: torch.Tensor) -> torch.Tensor:
            return _grouped_mm(_grouped_mm(x, lora_A, offsets), lora_B, offsets)

        lora_x = self.lora_dropout(x)
        gate = base.project("gate_proj", x, offsets)
        gate = gate + bias(base.gate_proj_bias, gate)
        up = base.project("up_proj", x, offsets)
        up = up + bias(base.up_proj_bias, up)

        gate_up_lora = lora(lora_x, self.gate_up_lora_A, self.gate_up_lora_B)
        gate = gate + self.scaling * gate_up_lora[..., ::2]
        up = up + self.scaling * gate_up_lora[..., 1::2]

        h = base.activation.apply(gate, up)
        out = base.project("down_proj", h, offsets)
        out = out + bias(base.down_proj_bias, out)
        return (out + self.scaling * lora(self.lora_dropout(h), self.down_lora_A, self.down_lora_B)).type_as(x)


class LoRATinkerGptOssExperts(LoRAModule):
    """GPT-OSS GroupedExperts + LoRA in Tinker's layout.

    Gate and up each have one A shared across experts and a B per expert; down has an A per expert and
    one B shared across experts. Exports the same vLLM 3D MoE format as ``LoRAGptOssGroupedExperts`` at
    twice the rank: the fused gate_up adapter stacks the gate and up A matrices, gate rows of B in the
    first ``rank`` columns and up rows in the rest, and down is zero-padded to match.
    """

    def __init__(self, base_layer: GroupedExperts, rank: int, alpha: float = 32.0, dropout: float = 0.0):
        super().__init__(base_layer, rank, alpha, dropout, export_rank=2 * rank)
        n, hidden_size, intermediate_size = base_layer.num_experts, base_layer.dim, base_layer.hidden_dim
        self.num_experts = n
        _check_grouped_mm_dims(rank, hidden_size, intermediate_size)
        like = base_layer.up_proj_bias
        self.gate_lora_A = lora_parameter(rank, hidden_size, like=like)
        self.gate_lora_B = lora_parameter(n, intermediate_size, rank, like=like)
        self.up_lora_A = lora_parameter(rank, hidden_size, like=like)
        self.up_lora_B = lora_parameter(n, intermediate_size, rank, like=like)
        self.down_lora_A = lora_parameter(n, rank, intermediate_size, like=like)
        self.down_lora_B = lora_parameter(hidden_size, rank, like=like)
        self.reset_parameters()

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        n, r = self.num_experts, self.rank
        gate_B, up_B = _full(self.gate_lora_B), _full(self.up_lora_B)
        gate_up_A = torch.cat([_full(self.gate_lora_A), _full(self.up_lora_A)]).expand(n, -1, -1)
        gate_up_B = gate_B.new_zeros(n, 2 * gate_B.shape[1], 2 * r)
        gate_up_B[:, 0::2, :r] = gate_B
        gate_up_B[:, 1::2, r:] = up_B
        down_A, down_B = pad_rank(_full(self.down_lora_A), _full(self.down_lora_B).expand(n, -1, -1), 2 * r)
        return {
            "base_layer.lora_A.weight": _flat_a(gate_up_A),
            "base_layer.lora_B.weight": _flat_b(gate_up_B),
            "lora_A.weight": _flat_a(down_A),
            "lora_B.weight": _flat_b(down_B),
        }

    def forward(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        base = self.base_layer
        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

        def bias(tensor: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
            return broadcast_expert_bias(_to_local(tensor), num_tokens_per_expert, out.shape[0]).bfloat16()

        lora_x = self.lora_dropout(x)
        gate = base.project("gate_proj", x, offsets)
        gate = gate + bias(base.gate_proj_bias, gate)
        gate = gate + self.scaling * _grouped_mm(_dense(lora_x, self.gate_lora_A), self.gate_lora_B, offsets)
        up = base.project("up_proj", x, offsets)
        up = up + bias(base.up_proj_bias, up)
        up = up + self.scaling * _grouped_mm(_dense(lora_x, self.up_lora_A), self.up_lora_B, offsets)

        h = base.activation.apply(gate, up)
        out = base.project("down_proj", h, offsets)
        out = out + bias(base.down_proj_bias, out)
        down = _dense(_grouped_mm(self.lora_dropout(h), self.down_lora_A, offsets), self.down_lora_B)
        return (out + self.scaling * down).type_as(x)
