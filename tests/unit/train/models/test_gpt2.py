import pytest
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import GPT2Config
from transformers import GPT2LMHeadModel as HFGPT2LMHeadModel

from prime_rl.trainer.models.gpt2 import GPT2ForCausalLM as PrimeRLGPT2ForCausalLM
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.utils.utils import default_dtype

pytestmark = [pytest.mark.gpu]


def tiny_config() -> GPT2Config:
    config = GPT2Config(n_embd=256, n_layer=3, n_head=4, n_positions=512, vocab_size=1000)
    config._attn_implementation = "flash_attention_2"
    return config


def prime_from_hf_state_dict(config: GPT2Config, state_dict: dict) -> PrimeRLGPT2ForCausalLM:
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        prime_model = PrimeRLGPT2ForCausalLM._from_config(config)
    prime_keys = set(prime_model.state_dict().keys())
    state_dict = {k: v.to("cuda", torch.bfloat16) for k, v in state_dict.items()}
    prime_model.convert_to_prime(state_dict)
    state_dict.setdefault("lm_head.weight", state_dict["model.embed_tokens.weight"])
    assert set(state_dict) == prime_keys
    prime_model.load_state_dict(state_dict)
    inject_prime_lm_head(prime_model, chunk_size=None)
    return prime_model


def hf_model(config: GPT2Config) -> HFGPT2LMHeadModel:
    config = GPT2Config.from_dict(config.to_dict())
    config._attn_implementation = "sdpa"
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        return HFGPT2LMHeadModel._from_config(config).eval()


def test_gpt2_matches_hf_and_roundtrips():
    config = tiny_config()
    hf = hf_model(config)
    prime = prime_from_hf_state_dict(config, hf.state_dict())

    input_ids = torch.randint(0, config.vocab_size, (1, 100), device="cuda")
    position_ids = torch.arange(100, device="cuda").unsqueeze(0)
    hf_logits = hf(input_ids, position_ids=position_ids).logits
    prime_logits = prime(input_ids, position_ids, seq_lens=torch.tensor([100], device="cuda"))["logits"]
    assert (prime_logits - hf_logits).abs().max() < 0.1

    # Weights go back out under the hub's unprefixed names, which is what vLLM loads.
    expected = {name.removeprefix("transformer."): tensor for name, tensor in hf.named_parameters()}
    converted = prime.convert_to_hf(prime.state_dict())
    assert set(converted) == set(expected) | {"lm_head.weight"}
    for name, tensor in expected.items():
        assert torch.equal(converted[name], tensor), name


def test_gpt2_packed_documents_match_separate_runs():
    config = tiny_config()
    hf = hf_model(config)
    prime = prime_from_hf_state_dict(config, hf.state_dict())

    lengths = [37, 63]
    docs = [torch.randint(0, config.vocab_size, (1, n), device="cuda") for n in lengths]
    packed_ids = torch.cat(docs, dim=1)
    packed_positions = torch.cat([torch.arange(n, device="cuda") for n in lengths]).unsqueeze(0)
    packed_logits = prime(packed_ids, packed_positions, seq_lens=torch.tensor(lengths, device="cuda"))["logits"]
    separate_logits = torch.cat([hf(doc).logits for doc in docs], dim=1)
    assert (packed_logits - separate_logits).abs().max() < 0.1


def test_gpt2_hub_checkpoint_matches_hf():
    """The hub checkpoint names its backbone without the ``transformer.`` prefix. In bf16 the
    port should sit as close to the fp32 reference as HF's own bf16 model does."""
    config = GPT2Config.from_pretrained("openai-community/gpt2")
    config._attn_implementation = "flash_attention_2"
    state_dict = load_file(hf_hub_download("openai-community/gpt2", "model.safetensors"))
    assert "wte.weight" in state_dict
    prime = prime_from_hf_state_dict(config, state_dict)
    reference = HFGPT2LMHeadModel.from_pretrained("openai-community/gpt2", dtype=torch.float32).cuda().eval()
    hf_bf16 = HFGPT2LMHeadModel.from_pretrained("openai-community/gpt2", dtype=torch.bfloat16).cuda().eval()

    input_ids = torch.randint(0, config.vocab_size, (1, 200), device="cuda")
    with torch.no_grad():
        reference_logprobs = reference(input_ids).logits.float().log_softmax(-1)
        hf_logprobs = hf_bf16(input_ids).logits.float().log_softmax(-1)
        prime_logits = prime(input_ids, seq_lens=torch.tensor([200], device="cuda"))["logits"]
    prime_logprobs = prime_logits.float().log_softmax(-1)

    def kl(logprobs):
        return (reference_logprobs.exp() * (reference_logprobs - logprobs)).sum(-1).mean()

    assert kl(prime_logprobs) < 1.2 * kl(hf_logprobs), (kl(prime_logprobs), kl(hf_logprobs))
