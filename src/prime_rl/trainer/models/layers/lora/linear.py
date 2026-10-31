import torch
from torch import nn

from prime_rl.trainer.models.layers.lora.base import LoRAModule, lora_parameter, pad_rank


class LoRALinear(LoRAModule):
    """Linear layer with a low-rank adapter: base(x) + alpha / rank * B(A(dropout(x)))."""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float = 32.0,
        dropout: float = 0.0,
        export_rank: int | None = None,
    ):
        super().__init__(base_layer, rank, alpha, dropout, export_rank)
        self.lora_A = lora_parameter(rank, base_layer.in_features, like=base_layer.weight)
        self.lora_B = lora_parameter(base_layer.out_features, rank, like=base_layer.weight)
        self.reset_parameters()

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        lora_A, lora_B = pad_rank(self.lora_A.detach(), self.lora_B.detach(), self.export_rank)
        return {"lora_A.weight": lora_A, "lora_B.weight": lora_B}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2d = x.view(-1, x.shape[-1])
        base_out = self.base_layer(x2d)
        lora_out = self.lora_dropout(x2d) @ self.lora_A.T @ self.lora_B.T
        return (base_out + self.scaling * lora_out).view(*x.shape[:-1], -1)


class LoRAOutputLinear(LoRALinear):
    """PrimeRL LM head (``FusedOutputLinear`` or ``VanillaOutputLinear``) with a low-rank adapter.

    The adapter rides in the head's own pass, ``[h, s·hAᵀ] @ [W | B]ᵀ = hWᵀ + s·hAᵀBᵀ``, so the chunked
    head still never materializes the full logits.
    """

    def forward(self, hidden_states: torch.Tensor, labels: torch.Tensor | None = None, **kwargs: torch.Tensor | None):
        lora_hidden = self.scaling * (self.lora_dropout(hidden_states) @ self.lora_A.T)
        return self.base_layer(
            torch.cat([hidden_states, lora_hidden], dim=-1),
            labels,
            weight=torch.cat([self.base_layer.weight, self.lora_B], dim=1),
            **kwargs,
        )
