"""SM100 integration checks of the block selector, slot conversion and attention output."""

import math

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="requires an SM100 GPU and the FlashTopK block selector",
)


def index_operands(rows, heads, length, dtype):
    # Integer dots are exactly representable in all tested input dtypes. Shuffling
    # scores within each request exposes accidental batch-global/local ID mixups.
    rank = (torch.arange(length, device="cuda") * 137) % 1021
    bits = torch.arange(10, device="cuda")
    keys = torch.zeros(length, 128, device="cuda")
    keys[:, :10] = (rank[:, None] >> bits) & 1
    q = torch.zeros(rows, heads, 128, device="cuda")
    q[:, 0, :10] = 2.0**bits
    # FP8 E4M3 tops out at 448. Halving each query component keeps max=256.
    return (q * 0.5).to(dtype), keys.to(dtype), rank


def dense_attention(q, k, v, slots):
    keys = k[slots].repeat_interleave(q.size(0) // k.size(1), dim=1)
    values = v[slots].repeat_interleave(q.size(0) // v.size(1), dim=1)
    return F.scaled_dot_product_attention(
        q[None, :, None].double(),
        keys.transpose(0, 1)[None].double(),
        values.transpose(0, 1)[None].double(),
    )[0, :, 0].float()


def test_qsa_packed_token_attention():
    from sglang.srt.layers.attention.flashtopk_prefill import select_qsa_blocks
    from sglang.srt.layers.attention.qsa.kernel import (
        expand_qsa_block_indices,
        qsa_sparse_attention,
    )

    lengths = [2079, 2087]
    block_counts = [n // 4 for n in lengths]
    # The last chunk of each request starts partway through the prefix.
    positions = [2, 3, 2049, 2078, 3, 2048, 2086]
    requests = [0, 0, 0, 0, 1, 1, 1]
    rows = len(positions)
    qidx, kidx, ranks = index_operands(rows, 4, sum(block_counts), torch.bfloat16)
    pos = torch.tensor(positions, device="cuda", dtype=torch.int32)
    req = torch.tensor(requests, device="cuda")
    starts = torch.tensor([0, block_counts[0]], device="cuda", dtype=torch.int32)[req]
    ends = starts + (pos + 1) // 4
    blocks = select_qsa_blocks(qidx, kidx[:, None], starts, ends, 512)
    seq_lens = torch.tensor(lengths, device="cuda", dtype=torch.int32)[req]
    tokens = expand_qsa_block_indices(blocks, pos, seq_lens, 4, 2048)
    bases = torch.tensor([0, lengths[0]], device="cuda")[req]
    physical = torch.where(tokens >= 0, tokens + bases[:, None], -1)
    rng = torch.Generator(device="cuda").manual_seed(47)
    q = torch.randn(rows, 4, 16, generator=rng, device="cuda")
    k, v = torch.randn(2, sum(lengths), 2, 16, generator=rng, device="cuda")
    actual = qsa_sparse_attention(q, k, v, physical)
    for r, (p, batch) in enumerate(zip(positions, requests)):
        first = sum(block_counts[:batch])
        count = (p + 1) // 4
        best = ranks[first : first + count].argsort(descending=True, stable=True)[:512]
        full = (best[:, None] * 4 + torch.arange(4, device="cuda")).flatten()
        logical = torch.cat([full, torch.arange(4 * count, p + 1, device="cuda")])
        got = tokens[r][tokens[r] >= 0].long()
        assert torch.equal(got.sort().values, logical.sort().values)
        want = dense_attention(q[r], k, v, logical + sum(lengths[:batch]))
        torch.testing.assert_close(actual[r], want, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("budget", [512, 1024])
def test_csa_physical_slot_attention(budget):
    from sglang.srt.layers.attention.flashtopk_prefill import select_c4_indices
    from sglang.srt.layers.attention.qsa.kernel import qsa_sparse_attention

    # The C4 indexer maps logical blocks through its compressed-cache page table.
    size = budget + 17
    qidx, kidx, ranks = index_operands(3, 64, size, torch.float8_e4m3fn)
    lengths = torch.tensor([1, budget - 1, size], device="cuda", dtype=torch.int32)
    page_size = 64
    n_pages = math.ceil(size / page_size)
    pages = torch.arange(n_pages, 0, -1, device="cuda", dtype=torch.int32)[None]
    pages_out = torch.empty(3, budget, device="cuda", dtype=torch.int32)
    raw = torch.empty_like(pages_out)
    select_c4_indices(
        qidx,
        kidx,
        torch.ones(3, 64, device="cuda"),
        torch.ones(size, device="cuda"),
        torch.zeros_like(lengths),
        lengths,
        pages,
        page_size,
        pages_out,
        raw,
    )
    rng = torch.Generator(device="cuda").manual_seed(48)
    q = torch.randn(3, 4, 16, generator=rng, device="cuda")
    compressed = torch.randn(size, 1, 16, generator=rng, device="cuda")
    cache = torch.zeros((n_pages + 1) * page_size, 1, 16, device="cuda")
    ids = torch.arange(size, device="cuda")
    cache[pages[0, ids // page_size].long() * page_size + ids % page_size] = compressed
    # Compressed Shared-KV MQA uses the same latent as key and value.
    actual = qsa_sparse_attention(q, cache, cache, pages_out)
    for r, count in enumerate([1, budget - 1, size]):
        logical = ranks[:count].argsort(descending=True, stable=True)[:budget]
        got = raw[r][raw[r] >= 0].long()
        assert torch.equal(got.sort().values, logical.sort().values)
        want = dense_attention(q[r], compressed, compressed, logical)
        torch.testing.assert_close(actual[r], want, rtol=2e-5, atol=2e-5)
