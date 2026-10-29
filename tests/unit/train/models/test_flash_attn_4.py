import pytest
import torch

from prime_rl.trainer.models.layers.attn import _flash_attn_fwd, flash_attn_4_varlen_op

pytestmark = [pytest.mark.gpu]


def _eager_causal_attention(query, key, value, cu_seqlens, scale):
    groups = query.shape[1] // key.shape[1]
    outputs = []
    for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
        q = query[start:end].float().transpose(0, 1)
        k = key[start:end].float().repeat_interleave(groups, dim=1).transpose(0, 1)
        v = value[start:end].float().repeat_interleave(groups, dim=1).transpose(0, 1)
        scores = q @ k.transpose(-1, -2) * scale
        causal = torch.ones(end - start, end - start, dtype=torch.bool, device=query.device).tril()
        outputs.append((scores.masked_fill(~causal, float("-inf")).softmax(-1) @ v).transpose(0, 1))
    return torch.cat(outputs)


def test_flash_attn_4_varlen_op_matches_eager_reference():
    if _flash_attn_fwd is None:
        pytest.skip("flash-attn-4 is not installed")
    torch.manual_seed(0)
    # Packed GQA sequences whose boundaries are not tile-aligned.
    cu_seqlens = torch.tensor([0, 333, 1000, 1001, 2048], device="cuda", dtype=torch.int32)
    total, heads, kv_heads, head_dim = int(cu_seqlens[-1]), 8, 2, 64
    query = torch.randn(total, heads, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    key = torch.randn(total, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    value = torch.randn(total, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max())

    output, _ = flash_attn_4_varlen_op(
        query,
        key,
        value,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=True,
    )
    grad_output = torch.randn_like(output, dtype=torch.float32)
    grads = torch.autograd.grad((output.float() * grad_output).sum(), (query, key, value))

    reference = _eager_causal_attention(query, key, value, cu_seqlens, head_dim**-0.5)
    reference_grads = torch.autograd.grad((reference * grad_output).sum(), (query, key, value))

    torch.testing.assert_close(output.float(), reference, rtol=2e-2, atol=2e-2)
    for grad, reference_grad in zip(grads, reference_grads):
        relative_error = (grad.float() - reference_grad.float()).norm() / reference_grad.float().norm()
        assert relative_error < 1e-2
