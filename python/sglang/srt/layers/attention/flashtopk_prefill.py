"""Optional FlashTopK block selection for SM100 eager prefill."""

from __future__ import annotations

import math

import torch

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
    sequential = _sequential_blocks(lengths, topk)
    if compressed_keys.shape[0] <= topk or (
        max_row_keys is not None and max_row_keys <= topk
    ):
        return sequential

    from flashtopk import select

    short = lengths <= topk
    weights = torch.ones(q.shape[:2], dtype=torch.float32, device=q.device)
    indices = select(
        q.contiguous(),
        compressed_keys.squeeze(1).contiguous(),
        weights,
        starts,
        torch.where(short, starts, ends),
        topk=topk,
        pruning="off",
        score_divisor=math.sqrt(q.shape[-1]),
    )
    relative = torch.where(indices >= 0, indices - starts[:, None], -1)
    return torch.where(short[:, None], sequential, relative)


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
    # All keys in short rows survive. Preserve the native backend's sequential
    # order there, including the zero-complete-block case.
    result = _sequential_blocks(lengths, topk)
    if keys.shape[0] > topk:
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
            pruning="off",
        )
        result = torch.where(short[:, None], result, indices)
    out_pages.copy_(result)
    if out_raw is not None:
        out_raw.copy_(result)
    # There is one row in the compact page table. The start of every C4 window
    # is zero, so starts is also the row-to-request map.
    topk_page_table_transform_(out_pages, page_table, starts, page_size)
