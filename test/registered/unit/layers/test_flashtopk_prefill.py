import math
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention.flashtopk_prefill import (
    select_c4_indices,
    select_qsa_blocks,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestFlashTopKPrefill(unittest.TestCase):
    def test_qsa_returns_request_relative_blocks_and_enumerates_short_rows(self):
        q = torch.zeros(4, 4, 128, dtype=torch.bfloat16)
        keys = torch.zeros(15, 1, 128, dtype=q.dtype)
        starts = torch.tensor([0, 6, 6, 6], dtype=torch.int32)
        ends = torch.tensor([0, 8, 9, 14], dtype=torch.int32)
        result = torch.tensor(
            [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1], [12, 8, 11]],
            dtype=torch.int32,
        )
        op = Mock(return_value=result)
        with patch.dict(sys.modules, {"flashtopk": SimpleNamespace(select=op)}):
            blocks = select_qsa_blocks(q, keys, starts, ends, 3)

        q_arg, keys_arg, weights, ks, ke = op.call_args.args
        self.assertEqual(keys_arg.shape, (15, 128))
        self.assertEqual(q_arg.shape, (4, 4, 128))
        torch.testing.assert_close(weights, torch.ones(4, 4))
        torch.testing.assert_close(ks, starts)
        self.assertEqual(ke.tolist(), [0, 6, 6, 14])
        self.assertEqual(op.call_args.kwargs["score_divisor"], math.sqrt(128))
        self.assertEqual(
            blocks.tolist(), [[-1, -1, -1], [0, 1, -1], [0, 1, 2], [6, 2, 5]]
        )

    def test_qsa_prefill_entry_preserves_partial_tail(self):
        from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer

        module = "sglang.srt.layers.attention.qsa.qsa_indexer"
        shape = SimpleNamespace(token_topk=2048, compress_ratio=4, block_topk=512)
        q = torch.zeros(2, 4, 128, dtype=torch.bfloat16)
        keys = torch.zeros(4, 1, 128, dtype=q.dtype)
        starts = torch.tensor([0, 2], dtype=torch.int32)
        ends = starts + torch.tensor([2, 1], dtype=torch.int32)
        pos = torch.tensor([7, 5], dtype=torch.int32)
        lengths = torch.tensor([8, 8], dtype=torch.int32)
        op = Mock(return_value=torch.full((2, 512), -1, dtype=torch.int32))
        with (
            patch(f"{module}.can_use_flashtopk_prefill", return_value=True),
            patch(f"{module}.qsa_mqa_prefill") as dense,
            patch.dict(sys.modules, {"flashtopk": SimpleNamespace(select=op)}),
        ):
            tokens = QSAIndexer.select_prefill_tokens(
                shape, q, keys, starts, ends, pos, lengths
            )
        dense.assert_not_called()
        op.assert_not_called()
        self.assertEqual(tokens.shape, (2, 2051))
        self.assertEqual(tokens[0, :8].tolist(), list(range(8)))
        self.assertEqual(tokens[1, :6].tolist(), list(range(6)))
        self.assertTrue((tokens[0, 8:] == -1).all())
        self.assertTrue((tokens[1, 6:] == -1).all())

    def test_c4_prefill_entry_uses_lengths_before_trivial_row_elision(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.attention.dsv4.indexer import C4IndexerBackendMixin
        from sglang.srt.layers.attention.dsv4.metadata import PagedIndexerMetadata
        from sglang.srt.model_executor.forward_mode import ForwardMode

        module = "sglang.srt.layers.attention.dsv4.indexer"
        lengths = torch.tensor([0, 511, 512, 513], dtype=torch.int32)
        starts = torch.zeros_like(lengths)
        table = torch.arange(9, dtype=torch.int32)[None, :]
        metadata = object.__new__(PagedIndexerMetadata)
        metadata.page_table = table.expand(4, -1)
        metadata.compressed_page_size = 64
        metadata.compressed_seq_lens = lengths
        core = SimpleNamespace(
            positions=torch.arange(4),
            c4_sparse_page_indices=torch.empty(4, 512, dtype=torch.int32),
            c4_sparse_raw_indices=torch.empty(4, 512, dtype=torch.int32),
        )
        backend = C4IndexerBackendMixin()
        backend.token_to_kv_pool = object()
        backend.hisparse_coordinator = None
        backend.forward_metadata = SimpleNamespace(
            indexer_metadata=metadata, core_metadata=core
        )
        backend._forward_prepare_normal = Mock(
            return_value=(
                torch.zeros(4, 64, 128, dtype=torch.float8_e4m3fn),
                torch.ones(4, 64, 1),
            )
        )
        backend._get_nonpaged_indexer_plan = Mock(
            return_value=SimpleNamespace(
                ks=starts,
                ke=torch.tensor([0, 0, 0, 513]),
                page_table=table,
                query_rows=4,
            )
        )
        backend._gather_nonpaged_index_k = Mock(return_value=(object(), object()))
        backend._nonpaged_mqa_logits = Mock()
        with (
            patch(f"{module}.can_use_flashtopk_prefill", return_value=True),
            patch(f"{module}.get_global_indexer_capturer", return_value=None),
            patch(f"{module}.select_c4_indices") as fused,
            envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.override(True),
        ):
            backend.forward_c4_indexer(
                x=torch.empty(4, 1),
                q_lora=torch.empty(4, 1),
                c4_indexer=SimpleNamespace(use_fp4_indexer=False, layer_id=0),
                forward_batch=SimpleNamespace(forward_mode=ForwardMode.EXTEND),
            )
        backend._nonpaged_mqa_logits.assert_not_called()
        self.assertTrue(
            backend._get_nonpaged_indexer_plan.call_args.kwargs["use_flashtopk"]
        )
        fused.assert_called_once()
        args = fused.call_args.args
        torch.testing.assert_close(args[5], lengths)
        self.assertIs(args[6], table)
        self.assertEqual(args[8].data_ptr(), core.c4_sparse_page_indices.data_ptr())
        self.assertEqual(args[9].data_ptr(), core.c4_sparse_raw_indices.data_ptr())

    def test_c4_raw_and_physical_slots_use_true_lengths(self):
        q = torch.zeros(4, 64, 128, dtype=torch.float8_e4m3fn)
        keys = torch.zeros(9, 128, dtype=q.dtype)
        scales = torch.ones(9)
        weights = torch.ones(4, 64)
        starts = torch.zeros(4, dtype=torch.int32)
        lengths = torch.tensor([0, 2, 3, 9], dtype=torch.int32)
        pages = torch.tensor([[5, 1, 11]], dtype=torch.int32)
        raw = torch.empty(4, 3, dtype=torch.int32)
        physical = torch.empty_like(raw)

        def map_slots(ids, table, row_to_batch, page_size):
            self.assertEqual(row_to_batch.tolist(), [0, 0, 0, 0])
            addr = table[0, ids.clamp_min(0).long() // page_size] * page_size
            ids.copy_(torch.where(ids < 0, -1, addr + ids % page_size))

        op = Mock(
            return_value=torch.tensor(
                [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1], [8, 0, 6]], dtype=torch.int32
            )
        )
        with patch.dict(
            sys.modules,
            {
                "flashtopk": SimpleNamespace(select=op),
                "sglang.kernels.ops.attention.topk_page_table": SimpleNamespace(
                    topk_page_table_transform_=map_slots
                ),
            },
        ):
            select_c4_indices(
                q, keys, weights, scales, starts, lengths, pages, 4, physical, raw
            )

        self.assertEqual(op.call_args.args[4].tolist(), [0, 0, 0, 9])
        self.assertIs(op.call_args.kwargs["kv_scales"], scales)
        self.assertEqual(raw.tolist(), [[-1, -1, -1], [0, 1, -1], [0, 1, 2], [8, 0, 6]])
        self.assertEqual(
            physical.tolist(), [[-1, -1, -1], [20, 21, -1], [20, 21, 22], [44, 20, 6]]
        )


if __name__ == "__main__":
    unittest.main()
