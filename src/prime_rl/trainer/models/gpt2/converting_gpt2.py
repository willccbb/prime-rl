"""HF <-> PrimeRL weight conversion for GPT-2.

HF GPT-2 stores its projections as ``Conv1D`` weights of shape ``(in, out)`` and fuses
query, key and value into one ``c_attn``. PrimeRL uses ``nn.Linear`` layout ``(out, in)``
with separate q/k/v projections, so each projection is renamed and transposed, and
``c_attn`` is also split in three.
"""

from torch import Tensor

from prime_rl.trainer.models.conversion_ops import (
    Conditional,
    ConvOp,
    Drop,
    MapValue,
    PrefixRename,
    Rename,
    SplitConcat,
    key_present,
)


def _transpose(t: Tensor) -> Tensor:
    return t.t().contiguous()


def _linear(hf: str, prime: str) -> list[ConvOp]:
    return [
        Rename(f"{hf}.weight", f"{prime}.weight"),
        MapValue(f"{prime}.weight", forward=_transpose, backward=_transpose),
        Rename(f"{hf}.bias", f"{prime}.bias"),
    ]


def _norm(hf: str, prime: str) -> list[ConvOp]:
    return [Rename(f"{hf}.weight", f"{prime}.weight"), Rename(f"{hf}.bias", f"{prime}.bias")]


def is_hf_state_dict(state_dict: dict[str, Tensor]) -> bool:
    return any(name.endswith("attn.c_attn.weight") for name in state_dict)


def is_prime_state_dict(state_dict: dict[str, Tensor]) -> bool:
    return any(name.endswith("self_attn.q_proj.weight") for name in state_dict)


def conversion_chain(config) -> list[ConvOp]:
    hidden = config.hidden_size
    operations: list[ConvOp] = [
        # Hub checkpoints name the backbone without a prefix; `save_pretrained` writes it
        # under `transformer.`. Accept both and emit the unprefixed names.
        Conditional(key_present("transformer.wte.weight"), then=[PrefixRename("transformer.", "")]),
        Rename("wte.weight", "model.embed_tokens.weight"),
        Rename("wpe.weight", "model.embed_positions.weight"),
        *_norm("ln_f", "model.norm"),
    ]
    for layer_idx in range(config.num_hidden_layers):
        hf = f"h.{layer_idx}"
        prime = f"model.layers.{layer_idx}"
        fused = f"{prime}.self_attn.c_attn"
        operations.extend(
            [
                # Causal-mask buffers that older checkpoints serialize.
                Drop(f"{hf}.attn.bias"),
                Drop(f"{hf}.attn.masked_bias"),
                *_norm(f"{hf}.ln_1", f"{prime}.input_layernorm"),
                *_linear(f"{hf}.attn.c_attn", fused),
                SplitConcat(
                    f"{fused}.weight",
                    [(f"{prime}.self_attn.{p}_proj.weight", hidden) for p in "qkv"],
                    dim=0,
                ),
                SplitConcat(
                    f"{fused}.bias",
                    [(f"{prime}.self_attn.{p}_proj.bias", hidden) for p in "qkv"],
                    dim=0,
                ),
                *_linear(f"{hf}.attn.c_proj", f"{prime}.self_attn.o_proj"),
                *_norm(f"{hf}.ln_2", f"{prime}.post_attention_layernorm"),
                *_linear(f"{hf}.mlp.c_fc", f"{prime}.mlp.up_proj"),
                *_linear(f"{hf}.mlp.c_proj", f"{prime}.mlp.down_proj"),
            ]
        )
    return operations
