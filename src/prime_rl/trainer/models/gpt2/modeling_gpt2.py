# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional, Union

import torch
from torch import Tensor, nn
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.gpt2.converting_gpt2 import conversion_chain, is_hf_state_dict, is_prime_state_dict
from prime_rl.trainer.models.layers.attn import ATTN_IMPL2CLASS, AttentionConfig
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput, TokenSupport
from prime_rl.trainer.models.layers.mlp import FeedForward
from prime_rl.trainer.models.layers.norms import LayerNorm
from prime_rl.utils.sequence import get_cu_seqlens_from_seq_lens


def _check_supported(config: GPT2Config) -> None:
    if config.activation_function != "gelu_new":
        raise ValueError(f"GPT-2 activation {config.activation_function!r} is not supported, only 'gelu_new'")
    if not config.scale_attn_weights or config.scale_attn_by_inverse_layer_idx or config.reorder_and_upcast_attn:
        raise ValueError("Only standard GPT-2 attention scaling is supported")


class GPT2DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: GPT2Config, layer_idx: int):
        super().__init__()
        attn_config = AttentionConfig(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_attention_heads,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_attention_heads,
            is_causal=True,
            attention_bias=True,
            use_qk_norm=False,
            rms_norm_eps=config.layer_norm_epsilon,
            output_bias=True,
        )
        self.self_attn = ATTN_IMPL2CLASS[config._attn_implementation](attn_config)
        self.mlp = FeedForward(
            dim=config.hidden_size,
            hidden_dim=config.n_inner or 4 * config.hidden_size,
            expert_type="non_gated",
            activation="gelu_tanh",
            bias=True,
        )
        self.input_layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.post_attention_layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: Optional[torch.LongTensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states, _ = self.self_attn(
            hidden_states=self.input_layernorm(hidden_states),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        hidden_states = residual + hidden_states
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


@auto_docstring
class GPT2PreTrainedModel(PreTrainedModelPrimeRL):
    config: GPT2Config
    base_model_prefix = "model"

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return is_hf_state_dict(state_dict)

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return is_prime_state_dict(state_dict)

    @classmethod
    def conversion_chain(cls, config: GPT2Config):
        return conversion_chain(config)


@auto_docstring
class GPT2Model(GPT2PreTrainedModel):
    def __init__(self, config: GPT2Config):
        super().__init__(config)
        _check_supported(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_positions = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layers = nn.ModuleList([GPT2DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.gradient_checkpointing = False
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        *,
        seq_lens: torch.LongTensor,
        seq_lens_are_pre_shard: bool = False,
    ) -> BaseModelOutputWithPast:
        r"""
        seq_lens (`torch.LongTensor` of shape `(num_documents,)`):
            Per-document lengths of the packed row (PrimeRL packed-batch contract).
        seq_lens_are_pre_shard (`bool`, *optional*, defaults to `False`):
            Whether `seq_lens` holds pre-CP-shard (global) document boundaries.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        cu_seqlens, max_seqlen = get_cu_seqlens_from_seq_lens(
            seq_lens.to(device=inputs_embeds.device),
            total_tokens=None if seq_lens_are_pre_shard else inputs_embeds.shape[1],
        )
        torch._dynamo.mark_dynamic(cu_seqlens, 0)

        # Learned absolute positions: position_ids restart at 0 for every packed document.
        hidden_states = inputs_embeds + self.embed_positions(position_ids)
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(hidden_states, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

        return BaseModelOutputWithPast(last_hidden_state=self.norm(hidden_states))


@auto_docstring
class GPT2ForCausalLM(GPT2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = GPT2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        temperature: Optional[torch.Tensor] = None,
        support: Optional[TokenSupport] = None,
        *,
        seq_lens: torch.LongTensor,
        seq_lens_are_pre_shard: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> PrimeLmOutput:
        r"""
        seq_lens (`torch.LongTensor` of shape `(num_documents,)`):
            Per-document lengths of the packed row (PrimeRL packed-batch contract).
        seq_lens_are_pre_shard (`bool`, *optional*, defaults to `False`):
            Whether `seq_lens` holds pre-CP-shard (global) document boundaries.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Accepted only for HuggingFace API compatibility; training does no autoregressive decoding.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels used by PrimeRL's wrapped LM head to optionally compute per-token logprobs/entropy.
        temperature (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Per-token temperatures for logprobs/entropy computation when `labels` are provided.
        support (`TokenSupport`, *optional*):
            Per-token sampler support (CSR rows), forwarded to the wrapped LM head.
        """
        assert use_cache is None, "use_cache is not supported for custom gpt2"
        assert past_key_values is None, "past_key_values is not supported for custom gpt2"

        if position_ids is None:
            reference = inputs_embeds if inputs_embeds is not None else input_ids
            position_ids = torch.arange(reference.shape[1], device=reference.device).unsqueeze(0)

        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            seq_lens=seq_lens,
            seq_lens_are_pre_shard=seq_lens_are_pre_shard,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        return self.lm_head(
            hidden_states[:, slice_indices, :],
            labels[:, slice_indices] if labels is not None else None,
            temperature=temperature,
            support=support,
        )

    def init_buffers_post_meta(self):
        pass
