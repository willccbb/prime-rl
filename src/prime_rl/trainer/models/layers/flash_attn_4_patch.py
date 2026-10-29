"""Fix for FlashAttention 4's SM80-family backward with GQA and packed sequences.

FA4's SM80 backward (``FlashAttentionBackwardSm80``, which ``FlashAttentionBackwardSm120`` extends)
accumulates GQA dK/dV in fp32 at a per-sequence offset of ``offset_k + batch_idx * n_block_size``,
while the postprocess that converts the accumulator reads at the tile-rounded
``seqlen.padded_offset_k``. Whenever a packed sequence starts at an offset that is not a multiple of
the tile size, dK and dV come out wrong. ``epilogue`` below is FA4 98c1cee's own, changed only to use
``seqlen.padded_offset_k``. The patch refuses to install unless FA4's epilogue is still exactly 98c1cee's,
compared by syntax tree so formatting and comments don't count.

CuTe DSL compiles kernels from their Python source on first use, so the replacement takes effect when
it is installed before the first backward call. FA4's optional persistent kernel cache
(``FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1``) fingerprints only FA4's own files; clear it after
changing this module.
"""

import ast
import hashlib
import inspect
import textwrap

import cutlass
import cutlass.cute as cute
from flash_attn.cute import utils
from flash_attn.cute.flash_bwd import FlashAttentionBackwardSm80
from flash_attn.cute.seqlen_info import SeqlenInfoQK

# sha256 of ast.dump of FlashAttentionBackwardSm80.epilogue at FA4 98c1cee (flash_bwd.py:1071-1197).
_UPSTREAM_EPILOGUE_SHA256 = "3578d7dcc2ef44417034f1e5a9429ed110cc82952cc18be15cd88843ee36b229"


@cute.jit
def _epilogue(
    self,
    acc_dK: cute.Tensor,
    acc_dV: cute.Tensor,
    mdK: cute.Tensor,
    mdV: cute.Tensor,
    sdK: cute.Tensor,
    sdV: cute.Tensor,
    gmem_tiled_copy_dK: cute.TiledCopy,
    gmem_tiled_copy_dV: cute.TiledCopy,
    tiled_mma: cute.TiledMma,
    tidx: cutlass.Int32,
    n_block: cutlass.Int32,
    num_head: cutlass.Int32,
    batch_size: cutlass.Int32,
    seqlen: SeqlenInfoQK,
    d_head: cutlass.Int32,
    d_head_v: cutlass.Int32,
):
    rdV = cute.make_fragment_like(acc_dV, self.dtype)
    rdV.store(acc_dV.load().to(self.dtype))
    rdK = cute.make_fragment_like(acc_dK, self.dtype)
    rdK.store(acc_dK.load().to(self.dtype))
    gmem_thr_copy_dK = gmem_tiled_copy_dK.get_slice(tidx)
    gmem_thr_copy_dV = gmem_tiled_copy_dV.get_slice(tidx)

    batch_idx = batch_size
    head_idx_kv = num_head // self.qhead_per_kvhead if cutlass.const_expr(not self.pack_gqa) else num_head

    if cutlass.const_expr(self.qhead_per_kvhead == 1):
        # Make sure all threads have finished reading K and V, otherwise we get racy dQ
        # because smem_q could be changed.
        cute.arch.barrier()
        # smem copy atom for dKV
        smem_copy_atom_dKV = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self.dtype, num_bits_per_copy=2 * self.dtype.width
        )
        smem_thr_copy_dKV = cute.make_tiled_copy_C(smem_copy_atom_dKV, tiled_mma).get_slice(tidx)
        taccdVrdV = smem_thr_copy_dKV.retile(rdV)
        taccdKrdK = smem_thr_copy_dKV.retile(rdK)
        taccdVsdV = smem_thr_copy_dKV.partition_D(sdV)
        taccdKsdK = smem_thr_copy_dKV.partition_D(sdK)
        # copy acc O from rmem to smem with the smem copy atom
        cute.copy(smem_copy_atom_dKV, taccdVrdV, taccdVsdV)
        cute.copy(smem_copy_atom_dKV, taccdKrdK, taccdKsdK)

        if cutlass.const_expr(not seqlen.has_cu_seqlens_k):
            mdK_cur, mdV_cur = [t[batch_idx, None, head_idx_kv, None] for t in (mdK, mdV)]
        else:
            mdK_cur, mdV_cur = [
                cute.domain_offset((seqlen.offset_k, 0), t[None, head_idx_kv, None]) for t in (mdK, mdV)
            ]

        blkdK_shape = (self.n_block_size, self.head_dim_padded)
        blkdV_shape = (self.n_block_size, self.head_dim_v_padded)
        gdK = cute.local_tile(mdK_cur, blkdK_shape, (n_block, 0))
        gdV = cute.local_tile(mdV_cur, blkdV_shape, (n_block, 0))
        tdKsdK = gmem_thr_copy_dK.partition_S(sdK)
        tdKgdK = gmem_thr_copy_dK.partition_D(gdK)
        tdVsdV = gmem_thr_copy_dV.partition_S(sdV)
        tdVgdV = gmem_thr_copy_dV.partition_D(gdV)
        tdKrdK = cute.make_fragment_like(tdKgdK, self.dtype)
        tdVrdV = cute.make_fragment_like(tdVgdV, self.dtype)
        # sync before all smem stores are done.
        cute.arch.barrier()
        # load acc dK and dV from smem to rmem for wider vectorization
        # Need to check OOB when reading from smem if kBlockN isn't evenly tiled
        # TODO
        cute.autovec_copy(tdKsdK, tdKrdK)
        cute.autovec_copy(tdVsdV, tdVrdV)

        cdK = cute.make_identity_tensor((self.n_block_size, self.head_dim_padded))
        tdKcdK = gmem_thr_copy_dK.partition_S(cdK)
        t0dKcdK = gmem_tiled_copy_dK.get_slice(0).partition_S(cdK)
        if cutlass.const_expr(self.head_dim_padded == self.head_dim_v_padded):
            tdVcdV = tdKcdK
            t0dVcdV = t0dKcdK
        else:
            cdV = cute.make_identity_tensor((self.n_block_size, self.head_dim_v_padded))
            tdVcdV = gmem_thr_copy_dV.partition_S(cdV)
            t0dVcdV = gmem_tiled_copy_dV.get_slice(0).partition_S(cdV)
        tdKpdK = utils.predicate_k(tdKcdK, limit=d_head)
        if cutlass.const_expr(self.same_hdim_kv):
            tdVpdV = tdKpdK
        else:
            tdVpdV = utils.predicate_k(tdVcdV, limit=d_head_v)
        # copy acc dK and acc_dV from rmem to gmem
        for rest_m in cutlass.range_constexpr(cute.size(tdKrdK.shape[1])):
            if t0dKcdK[0, rest_m, 0][0] < seqlen.seqlen_k - n_block * self.n_block_size - tdKcdK[0][0]:
                cute.copy(
                    gmem_tiled_copy_dK,
                    tdKrdK[None, rest_m, None],
                    tdKgdK[None, rest_m, None],
                    pred=tdKpdK[None, rest_m, None] if cutlass.const_expr(self.check_hdim_oob) else None,
                )
        for rest_m in cutlass.range_constexpr(cute.size(tdVrdV.shape[1])):
            if t0dVcdV[0, rest_m, 0][0] < seqlen.seqlen_k - n_block * self.n_block_size - tdVcdV[0][0]:
                cute.copy(
                    gmem_tiled_copy_dV,
                    tdVrdV[None, rest_m, None],
                    tdVgdV[None, rest_m, None],
                    pred=tdVpdV[None, rest_m, None] if cutlass.const_expr(self.check_hdim_v_oob) else None,
                )

    else:  # qhead_per_kvhead > 1, do atomic add
        # For Sm90, we need to sync to avoid racy writes to smem_q
        # For Sm80, we don't need to sync since we're not touching smem
        head_idx_kv = num_head // self.qhead_per_kvhead if cutlass.const_expr(not self.pack_gqa) else num_head

        if cutlass.const_expr(not seqlen.has_cu_seqlens_k):
            mdK_cur, mdV_cur = [t[batch_idx, head_idx_kv, None] for t in (mdK, mdV)]
        else:
            # The postprocess reads the accumulator at the tile-rounded per-sequence offset.
            padded_offset_k = seqlen.padded_offset_k
            mdK_cur = cute.domain_offset((padded_offset_k * self.head_dim_padded,), mdK[head_idx_kv, None])
            mdV_cur = cute.domain_offset((padded_offset_k * self.head_dim_v_padded,), mdV[head_idx_kv, None])

        gdV = cute.local_tile(mdV_cur, (self.n_block_size * self.head_dim_v_padded,), (n_block,))
        gdK = cute.local_tile(mdK_cur, (self.n_block_size * self.head_dim_padded,), (n_block,))
        tdVgdVaccum = gmem_thr_copy_dV.partition_S(gdV)
        tdKgdKaccum = gmem_thr_copy_dK.partition_S(gdK)
        acc_dV_atomic = gmem_thr_copy_dV.retile(acc_dV)
        acc_dK_atomic = gmem_thr_copy_dK.retile(acc_dK)
        assert cute.size(acc_dV_atomic) == cute.size(tdVgdVaccum)
        assert cute.size(acc_dK_atomic) == cute.size(tdKgdKaccum)
        for i in cutlass.range(cute.size(acc_dV_atomic), unroll_full=True):
            utils.atomic_add_fp32(acc_dV_atomic[i], utils.elem_pointer(tdVgdVaccum, i))
        for i in cutlass.range(cute.size(acc_dK_atomic), unroll_full=True):
            utils.atomic_add_fp32(acc_dK_atomic[i], utils.elem_pointer(tdKgdKaccum, i))


def patch_flash_attn_4_sm80_backward() -> None:
    """Install the fixed ``epilogue`` on FA4's SM80-family backward kernel."""
    if FlashAttentionBackwardSm80.epilogue is _epilogue:
        return
    upstream = ast.parse(textwrap.dedent(inspect.getsource(FlashAttentionBackwardSm80.epilogue))).body[0]
    if hashlib.sha256(ast.dump(upstream).encode()).hexdigest() != _UPSTREAM_EPILOGUE_SHA256:
        raise RuntimeError(
            "FlashAttention 4's SM80 backward epilogue is no longer 98c1cee's, which this patch replaces; "
            "check whether the padded_offset_k fix is upstream, then update or remove "
            "prime_rl/trainer/models/layers/flash_attn_4_patch.py"
        )
    FlashAttentionBackwardSm80.epilogue = _epilogue
