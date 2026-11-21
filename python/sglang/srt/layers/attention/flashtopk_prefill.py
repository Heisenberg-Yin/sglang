"""Optional FlashTopK block selection for SM100 eager prefill."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.utils import is_sm100_supported

# Bound workspace for selected-block remapping and expansion, independent of
# the native backend's dense logits memory budget.
PREFILL_ROWS_PER_CHUNK = 8192


def can_use_flashtopk_prefill(q: torch.Tensor, heads: int, dtype: torch.dtype) -> bool:
    return (
        envs.SGLANG_OPT_USE_FLASHTOPK_PREFILL.get()
        and q.is_cuda
        and q.dtype == dtype
        and q.shape[1:] == (heads, 128)
        and is_sm100_supported()
        and not torch.compiler.is_compiling()
        and not torch.cuda.is_current_stream_capturing()
    )


def _sequential_blocks(lengths: torch.Tensor, topk: int) -> torch.Tensor:
    cols = torch.arange(topk, device=lengths.device, dtype=torch.int32)
    return torch.where(cols < lengths[:, None], cols, -1)


@triton.jit
def _finalize_indices_kernel(
    OUT,
    STARTS,
    LENGTHS,
    stride_row,
    topk: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SUBTRACT_STARTS: tl.constexpr,
):
    """In-place post-processing per row: fill short rows (length <= topk) sequentially,
    and optionally convert packed QSA indices to request-relative offsets. Long rows with
    start == 0 return immediately without touching global memory."""
    row = tl.program_id(0).to(tl.int64)
    length = tl.load(LENGTHS + row)
    if length <= topk:
        col = tl.arange(0, BLOCK_K)
        mask = col < topk
        vals = tl.where(col < length, col, -1).to(tl.int32)
        tl.store(OUT + row * stride_row + col, vals, mask=mask)
        return
    if SUBTRACT_STARTS:
        start = tl.load(STARTS + row)
        if start != 0:
            col = tl.arange(0, BLOCK_K)
            mask = col < topk
            ptr = OUT + row * stride_row + col
            idx = tl.load(ptr, mask=mask, other=-1)
            tl.store(ptr, tl.where(idx >= 0, idx - start, -1), mask=mask)


def _finalize_indices(
    out: torch.Tensor,
    starts: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    *,
    subtract_starts: bool,
) -> torch.Tensor:
    if out.numel() == 0:
        return out
    if out.is_cuda:
        _finalize_indices_kernel[(out.shape[0],)](
            out,
            starts,
            lengths,
            out.stride(0),
            topk=topk,
            BLOCK_K=triton.next_power_of_2(topk),
            SUBTRACT_STARTS=subtract_starts,
            num_warps=4,
        )
        return out
    short = lengths <= topk
    sequential = _sequential_blocks(lengths, topk)
    if subtract_starts:
        out = torch.where(out >= 0, out - starts[:, None], -1)
    return torch.where(short[:, None], sequential, out)


def select_qsa_blocks(
    q: torch.Tensor,
    compressed_keys: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    topk: int,
    max_row_keys: int | None = None,
) -> torch.Tensor:
    """Select request-relative block IDs from packed, full-block windows."""
    starts = row_starts.to(torch.int32).contiguous()
    ends = row_ends.to(torch.int32).contiguous()
    lengths = ends - starts
    # The CUDA fast top-k enumerates a short row in block order.
    if compressed_keys.shape[0] <= topk or (
        max_row_keys is not None and max_row_keys <= topk
    ):
        return _sequential_blocks(lengths, topk)

    from flashtopk import select

    short = lengths <= topk
    out = torch.empty((q.shape[0], topk), dtype=torch.int32, device=q.device)
    indices = select(
        q.contiguous(),
        compressed_keys.squeeze(1).contiguous(),
        None,
        starts,
        torch.where(short, starts, ends),
        topk=topk,
        pruning="off",
        chunk_size=PREFILL_ROWS_PER_CHUNK,
        out=out,
        score_divisor=math.sqrt(q.shape[-1]),
    )
    if indices is not out:
        out = indices
    return _finalize_indices(out, starts, lengths, topk, subtract_starts=True)


def select_c4_indices(
    q: torch.Tensor,
    keys: torch.Tensor,
    weights: torch.Tensor,
    key_scales: torch.Tensor,
    starts: torch.Tensor,
    lengths: torch.Tensor,
    page_table: torch.Tensor,
    page_size: int,
    out_pages: torch.Tensor,
    out_raw: torch.Tensor | None = None,
) -> None:
    """Write logical and physical C4 slots for one gathered request."""
    from sglang.kernels.ops.attention.topk_page_table import topk_page_table_transform_

    topk = out_pages.shape[1]
    lengths = lengths.reshape(-1).to(torch.int32).contiguous()
    if keys.shape[0] <= topk:
        out_pages.copy_(_sequential_blocks(lengths, topk))
    else:
        from flashtopk import select

        short = lengths <= topk
        indices = select(
            q.contiguous(),
            keys.contiguous(),
            weights.float().contiguous(),
            starts,
            torch.where(short, starts, lengths),
            kv_scales=key_scales.contiguous(),
            topk=topk,
            pruning="adaptive",
            chunk_size=PREFILL_ROWS_PER_CHUNK,
            out=out_pages,
        )
        if indices is not out_pages:
            out_pages.copy_(indices)
        finalized = _finalize_indices(
            out_pages, starts, lengths, topk, subtract_starts=False
        )
        if finalized is not out_pages:
            out_pages.copy_(finalized)
    if out_raw is not None:
        out_raw.copy_(out_pages)
    # There is one row in the compact page table. The start of every C4 window
    # is zero, so starts is also the row-to-request map.
    topk_page_table_transform_(out_pages, page_table, starts, page_size)
