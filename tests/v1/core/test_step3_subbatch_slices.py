# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""IDE_006 Step 3.0 — ``SubBatch.*_token_slice`` helpers.

These slices are the foundation for cdec dispatch (TSK_015 4.5 + TSK_018
3.1). The helpers themselves carry no dispatch logic — they only
report the (start, end) row ranges each request subset occupies in
the kernel layout. Verifying the math here guards the layers that
will consume them later.

See ``shadow_assists/features/IDE_006/TSK_015.md`` and
``TSK_018.md`` Phase 3.
"""

from __future__ import annotations

from dataclasses import dataclass

from vllm.v1.core.sched.sub_batch import SubBatch


@dataclass
class _Req:
    request_id: int
    prompt_len: int
    num_tokens: int


def _make_sb() -> SubBatch:
    # ZeroPerfPredictor (default) — slices don't depend on perfdata.
    return SubBatch()


# ----------------------------------------------------------------------
# Empty sub-batch — every slice is (0, 0) and total_tokens == 0.
# ----------------------------------------------------------------------
def test_empty_subbatch_all_slices_are_zero():
    sb = _make_sb()
    assert sb.cprf_token_slice == (0, 0)
    assert sb.gprf_token_slice == (0, 0)
    assert sb.gdec_token_slice == (0, 0)
    assert sb.cdec_token_slice == (0, 0)
    assert sb.total_tokens == 0


# ----------------------------------------------------------------------
# Single-subset cases — slice spans the entire contribution.
# ----------------------------------------------------------------------
def test_only_cprfs():
    sb = _make_sb()
    sb.add_pref(_Req(0, 16, 16), is_gpu=False)
    sb.add_pref(_Req(1, 32, 32), is_gpu=False)
    assert sb.cprf_token_slice == (0, 48)
    # all later subsets start at 48 with zero length
    assert sb.gprf_token_slice == (48, 48)
    assert sb.gdec_token_slice == (48, 48)
    assert sb.cdec_token_slice == (48, 48)
    assert sb.total_tokens == 48


def test_only_gprfs():
    sb = _make_sb()
    sb.add_pref(_Req(0, 64, 64), is_gpu=True)
    assert sb.cprf_token_slice == (0, 0)
    assert sb.gprf_token_slice == (0, 64)
    assert sb.gdec_token_slice == (64, 64)
    assert sb.cdec_token_slice == (64, 64)
    assert sb.total_tokens == 64


def test_only_gdecs_each_one_token():
    sb = _make_sb()
    for i in range(5):
        sb.add_gdec(_Req(i, 4, 5))
    assert sb.gdec_token_slice == (0, 5)        # 5 tokens, one per req
    assert sb.cdec_token_slice == (5, 5)
    assert sb.total_tokens == 5


def test_only_cdecs_each_one_token():
    sb = _make_sb()
    for i in range(3):
        sb.add_cdec(_Req(i, 4, 5))
    assert sb.cdec_token_slice == (0, 3)
    assert sb.total_tokens == 3


# ----------------------------------------------------------------------
# Mixed sub-batch — verify all four slices stack correctly under
# the documented order [cprf, gprf, gdec, cdec].
# ----------------------------------------------------------------------
def test_mixed_subbatch_slices_compose_correctly():
    sb = _make_sb()
    sb.add_pref(_Req(10, 8, 8), is_gpu=False)         # cprf: 8 tokens
    sb.add_pref(_Req(11, 16, 16), is_gpu=True)        # gprf: 16 tokens
    sb.add_pref(_Req(12, 8, 8), is_gpu=True)          # gprf: another 8
    for i in range(4):
        sb.add_gdec(_Req(20 + i, 4, 5))               # gdec: 4 tokens
    for i in range(2):
        sb.add_cdec(_Req(30 + i, 4, 5))               # cdec: 2 tokens

    assert sb.cprf_token_slice == (0, 8)
    assert sb.gprf_token_slice == (8, 32)            # 8 + 24
    assert sb.gdec_token_slice == (32, 36)
    assert sb.cdec_token_slice == (36, 38)
    assert sb.total_tokens == 38


def test_slices_match_all_reqs_kernel_layout():
    """The slice contract must match the actual ``all_reqs`` order
    (cprf → gprf → gdec → cdec). This catches any future re-ordering
    of ``all_reqs`` that would break dispatch row mapping."""
    sb = _make_sb()
    sb.add_pref(_Req(1, 4, 4), is_gpu=False)
    sb.add_pref(_Req(2, 8, 8), is_gpu=True)
    sb.add_gdec(_Req(3, 4, 5))
    sb.add_cdec(_Req(4, 4, 5))

    all_ids = [r.request_id for r in sb.all_reqs]
    assert all_ids == [1, 2, 3, 4]                   # cprf → gprf → gdec → cdec


def test_total_tokens_consistent_with_slices():
    """``total_tokens`` must equal the sum of slice widths."""
    sb = _make_sb()
    sb.add_pref(_Req(1, 12, 12), is_gpu=False)
    sb.add_pref(_Req(2, 20, 20), is_gpu=True)
    sb.add_gdec(_Req(3, 4, 5))
    sb.add_cdec(_Req(4, 4, 5))
    sb.add_cdec(_Req(5, 4, 5))

    s1, e1 = sb.cprf_token_slice
    s2, e2 = sb.gprf_token_slice
    s3, e3 = sb.gdec_token_slice
    s4, e4 = sb.cdec_token_slice
    width = (e1 - s1) + (e2 - s2) + (e3 - s3) + (e4 - s4)
    assert width == sb.total_tokens


# ----------------------------------------------------------------------
# Behavior under add → pop sequences (slice positions track mutations).
# ----------------------------------------------------------------------
def test_slices_update_after_pop():
    sb = _make_sb()
    sb.add_gdec(_Req(1, 4, 5))
    sb.add_gdec(_Req(2, 4, 5))
    sb.add_cdec(_Req(3, 4, 5))

    assert sb.gdec_token_slice == (0, 2)
    assert sb.cdec_token_slice == (2, 3)

    sb.pop_gdec()
    assert sb.gdec_token_slice == (0, 1)
    assert sb.cdec_token_slice == (1, 2)              # cdec start shifts down


def test_variable_prompt_lengths():
    """prompt_len varies per cprf/gprf request — slice must sum exactly."""
    sb = _make_sb()
    sb.add_pref(_Req(1, 100, 100), is_gpu=True)
    sb.add_pref(_Req(2, 7, 7), is_gpu=True)
    sb.add_pref(_Req(3, 23, 23), is_gpu=True)
    assert sb.gprf_token_slice == (0, 130)


# ----------------------------------------------------------------------
# Step 3.2.C-1 — seq-row slices (one row per request — for block_table_tensor).
# Key contract: cprf / gprf contribute multi-token but ONE seq row.
# ----------------------------------------------------------------------
def test_seq_slices_count_one_row_per_request():
    sb = _make_sb()
    sb.add_pref(_Req(1, 100, 100), is_gpu=False)        # cprf: 1 seq, 100 tokens
    sb.add_pref(_Req(2, 50, 50), is_gpu=True)           # gprf: 1 seq, 50 tokens
    sb.add_pref(_Req(3, 30, 30), is_gpu=True)           # gprf: 1 seq, 30 tokens
    sb.add_gdec(_Req(4, 4, 5))                          # gdec: 1 seq
    sb.add_gdec(_Req(5, 4, 5))                          # gdec: 1 seq
    sb.add_cdec(_Req(6, 4, 5))                          # cdec: 1 seq

    # token slice (existing): cprf=100, gprf=80, gdec=2, cdec=1
    assert sb.cprf_token_slice == (0, 100)
    assert sb.gprf_token_slice == (100, 180)
    assert sb.gdec_token_slice == (180, 182)
    assert sb.cdec_token_slice == (182, 183)

    # seq slice (new): each request contributes 1 row
    assert sb.cprf_seq_slice == (0, 1)
    assert sb.gprf_seq_slice == (1, 3)
    assert sb.gdec_seq_slice == (3, 5)
    assert sb.cdec_seq_slice == (5, 6)
    assert sb.total_seqs == 6


def test_empty_seq_slices_all_zero():
    sb = _make_sb()
    assert sb.cprf_seq_slice == (0, 0)
    assert sb.gprf_seq_slice == (0, 0)
    assert sb.gdec_seq_slice == (0, 0)
    assert sb.cdec_seq_slice == (0, 0)
    assert sb.total_seqs == 0


def test_seq_slice_decouples_from_token_slice_when_prefill_present():
    """The whole point of cdec_seq_slice — when prefill is present,
    token positions and seq positions diverge."""
    sb = _make_sb()
    sb.add_pref(_Req(1, 64, 64), is_gpu=True)    # 1 seq, 64 tokens
    sb.add_cdec(_Req(2, 4, 5))                   # 1 seq, 1 token

    # token: cdec starts at 64
    assert sb.cdec_token_slice == (64, 65)
    # seq: cdec starts at 1 (since 1 prefill seq before)
    assert sb.cdec_seq_slice == (1, 2)


def test_seq_slice_matches_token_slice_when_only_decodes():
    """When only gdec/cdec are present, 1 token = 1 seq, so the
    two slices coincide."""
    sb = _make_sb()
    sb.add_gdec(_Req(1, 4, 5))
    sb.add_gdec(_Req(2, 4, 5))
    sb.add_cdec(_Req(3, 4, 5))
    sb.add_cdec(_Req(4, 4, 5))

    assert sb.cdec_token_slice == sb.cdec_seq_slice == (2, 4)
    assert sb.gdec_token_slice == sb.gdec_seq_slice == (0, 2)
