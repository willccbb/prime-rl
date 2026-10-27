import os

import pytest
import torch
import torch.distributed as dist

from prime_rl.trainer.models.gpt_oss import GptOssConfig
from prime_rl.trainer.models.gpt_oss import GptOssForCausalLM as PrimeRLGptOssForCausalLM
from prime_rl.trainer.models.gpt_oss.attention import GptOssAttention, substitute_gpt_oss_ulysses_attention
from prime_rl.trainer.models.gpt_oss.converting_gpt_oss import GptOssExperts


def _config() -> GptOssConfig:
    return GptOssConfig(
        num_hidden_layers=1,
        num_local_experts=4,
        vocab_size=128,
        hidden_size=64,
        intermediate_size=32,
        head_dim=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        num_experts_per_tok=2,
        sliding_window=4,
        rope_parameters={"rope_type": "default", "rope_theta": 150000.0},
        attn_implementation="flash_attention_4",
        use_cache=False,
    )


def test_gpt_oss_checkpoint_conversion_roundtrip():
    with torch.device("meta"):
        model = PrimeRLGptOssForCausalLM(_config())

    prime_state_dict = {name: torch.randn(tensor.shape) for name, tensor in model.state_dict().items()}
    expected_prime = {name: tensor.clone() for name, tensor in prime_state_dict.items()}
    hf_state_dict = model.convert_to_hf(prime_state_dict)

    assert model.is_hf_state_dict(hf_state_dict)
    assert not model.is_prime_state_dict(hf_state_dict)

    roundtrip = model.convert_to_prime(hf_state_dict)
    assert roundtrip.keys() == expected_prime.keys()
    for name, tensor in roundtrip.items():
        torch.testing.assert_close(tensor, expected_prime[name])


def test_mxfp4_checkpoint_conversion_roundtrip():
    prefix = "model.layers.0.mlp.experts"
    hf = {
        f"{prefix}.gate_up_proj_blocks": torch.randint(0, 256, (2, 8, 3, 16), dtype=torch.uint8),
        f"{prefix}.gate_up_proj_scales": torch.randint(0, 256, (2, 8, 3), dtype=torch.uint8),
        f"{prefix}.gate_up_proj_bias": torch.randn(2, 8),
        f"{prefix}.down_proj_blocks": torch.randint(0, 256, (2, 96, 1, 16), dtype=torch.uint8),
        f"{prefix}.down_proj_scales": torch.randint(0, 256, (2, 96, 1), dtype=torch.uint8),
        f"{prefix}.down_proj_bias": torch.randn(2, 96),
    }
    state_dict = dict(hf)
    converter = GptOssExperts(prefix)
    converter.hf_to_prime(state_dict)
    assert torch.equal(state_dict[f"{prefix}.up_proj_blocks"], hf[f"{prefix}.gate_up_proj_blocks"][:, 1::2])
    converter.prime_to_hf(state_dict)
    assert state_dict.keys() == hf.keys()
    for name, tensor in hf.items():
        assert torch.equal(state_dict[name], tensor)


def test_gpt_oss_mxfp4_experts_count_as_active_params():
    from prime_rl.configs.trainer import LoRAConfig
    from prime_rl.trainer.lora import apply_lora_to_model

    counts = []
    for quantization_config in (None, {"quant_method": "mxfp4"}):
        config = _config()
        config.hidden_size = config.intermediate_size = 64
        config.quantization_config = quantization_config
        with torch.device("meta"):
            model = PrimeRLGptOssForCausalLM(config)
        apply_lora_to_model(model, LoRAConfig(rank=8))
        counts.append(model.num_active_params())
    assert counts[0] == counts[1]


@pytest.fixture(scope="module")
def cp_process_group():
    if int(os.environ.get("WORLD_SIZE", 1)) != 2:
        pytest.skip("run with torchrun --nproc-per-node=2")
    if torch.cuda.get_device_capability()[0] not in (9, 10, 11):
        pytest.skip("GPT-OSS learned sinks require SM90 or SM100/SM110")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    yield dist.group.WORLD
    dist.destroy_process_group()


@pytest.mark.gpu
def test_gpt_oss_context_parallel_attention(cp_process_group):
    local_rank = int(os.environ["LOCAL_RANK"])
    process_group = cp_process_group
    original_compute_attention = GptOssAttention.compute_attention

    try:
        torch.manual_seed(0)
        config = _config()
        config.head_dim = 64
        attention = GptOssAttention(config, layer_idx=0).cuda().to(torch.bfloat16)
        query = torch.randn(8, 4, 64, device="cuda", dtype=torch.bfloat16)
        key = torch.randn(8, 2, 64, device="cuda", dtype=torch.bfloat16)
        value = torch.randn(8, 2, 64, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            for tensor in (attention.sinks, query, key, value):
                dist.broadcast(tensor, src=0)
        cu_seqlens = torch.tensor([0, 5, 8], device="cuda", dtype=torch.int32)

        reference_query = query.detach().clone().requires_grad_()
        reference_key = key.detach().clone().requires_grad_()
        reference_value = value.detach().clone().requires_grad_()
        reference = original_compute_attention(
            attention,
            reference_query,
            reference_key,
            reference_value,
            cu_seqlens,
            5,
        )
        reference_output = reference.detach().clone()
        reference.float().square().sum().backward()
        reference_sink_grad = attention.sinks.grad.detach().clone()
        attention.sinks.grad = None

        substitute_gpt_oss_ulysses_attention(process_group)

        local_slice = slice(local_rank * 4, (local_rank + 1) * 4)
        local_query = query[local_slice].detach().clone().requires_grad_()
        local_key = key[local_slice].detach().clone().requires_grad_()
        local_value = value[local_slice].detach().clone().requires_grad_()
        actual = attention.compute_attention(local_query, local_key, local_value, cu_seqlens, 5)
        actual.float().square().sum().backward()

        gathered = [torch.empty_like(actual) for _ in range(2)]
        dist.all_gather(gathered, actual)
        torch.testing.assert_close(torch.cat(gathered), reference_output, rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(local_query.grad, reference_query.grad[local_slice], rtol=5e-2, atol=5e-2)
        torch.testing.assert_close(local_key.grad, reference_key.grad[local_slice], rtol=5e-2, atol=5e-2)
        torch.testing.assert_close(local_value.grad, reference_value.grad[local_slice], rtol=5e-2, atol=5e-2)

        sink_grad = attention.sinks.grad
        dist.all_reduce(sink_grad)
        torch.testing.assert_close(sink_grad, reference_sink_grad, rtol=5e-2, atol=5e-2)
    finally:
        GptOssAttention.compute_attention = original_compute_attention
