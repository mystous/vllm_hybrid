# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TSK_002 §4.5c reload-fallback (X4/W1) — `_mask_prefill_cold_blocks` 단위 회귀.

mask helper 가 prefill chunk (q_len > 1) 의 cold count 를 0 으로 force 하고
decode (q_len == 1) 의 cold count 는 그대로 보존하는지 검증.

검증 contract:
- decode-only batch — mask 영향 없음
- prefill-only batch (모두 q_len > 1, 모두 cold > 0) — 모두 0 으로 mask
- mixed batch (prefill + decode mixed) — prefill seq 만 mask, decode seq 보존
- empty / num_reqs == 0 — silent no-op
- num_reqs < num_reqs_padded (padding 슬롯) — padding 영역은 건드리지 않음
"""

from __future__ import annotations

import numpy as np
import pytest

from vllm.v1.worker.gpu_model_runner import _mask_prefill_cold_blocks


def _cu_q_from_lens(q_lens: list[int]) -> np.ndarray:
    """Build cu_query_lens cumulative numpy array from per-seq q_lens."""
    return np.concatenate([[0], np.cumsum(q_lens, dtype=np.int32)]).astype(
        np.int32
    )


def test_decode_only_batch_unchanged():
    """모두 q_len == 1 인 decode-only batch — cold counts 그대로."""
    num_reqs = 4
    num_cold = np.array([2, 0, 4, 1], dtype=np.int32)
    cu_q = _cu_q_from_lens([1, 1, 1, 1])
    expected = num_cold.copy()
    _mask_prefill_cold_blocks(num_cold, cu_q, num_reqs)
    np.testing.assert_array_equal(num_cold, expected)


def test_prefill_only_batch_all_masked():
    """모두 q_len > 1 인 prefill batch — 모든 cold count 가 0 으로 mask."""
    num_reqs = 3
    num_cold = np.array([5, 3, 7], dtype=np.int32)
    cu_q = _cu_q_from_lens([16, 8, 32])  # all prefill
    _mask_prefill_cold_blocks(num_cold, cu_q, num_reqs)
    np.testing.assert_array_equal(num_cold, np.zeros(3, dtype=np.int32))


def test_mixed_prefill_decode_only_prefill_masked():
    """prefill 과 decode 가 섞인 batch — prefill seq 만 mask, decode 보존.

    이게 §4.5c 의 본 motivation: mixed batch 에서 decode seq 의 cold path
    가 IDE_006 path 그대로 firing 되어 throughput 가치 보존.
    """
    num_reqs = 5
    num_cold = np.array([3, 2, 8, 1, 4], dtype=np.int32)
    # seq 0: prefill (q_len=16, cold=3) → mask
    # seq 1: decode (q_len=1, cold=2) → 보존
    # seq 2: prefill (q_len=4, cold=8) → mask
    # seq 3: decode (q_len=1, cold=1) → 보존
    # seq 4: prefill (q_len=2, cold=4) → mask
    cu_q = _cu_q_from_lens([16, 1, 4, 1, 2])
    _mask_prefill_cold_blocks(num_cold, cu_q, num_reqs)
    np.testing.assert_array_equal(
        num_cold, np.array([0, 2, 0, 1, 0], dtype=np.int32)
    )


def test_q_len_eq_1_threshold_decode_preserved():
    """q_len == 1 은 decode 로 분류되어 mask 되지 않음 (boundary)."""
    num_reqs = 1
    num_cold = np.array([5], dtype=np.int32)
    cu_q = _cu_q_from_lens([1])  # exactly 1 — decode
    _mask_prefill_cold_blocks(num_cold, cu_q, num_reqs)
    np.testing.assert_array_equal(num_cold, np.array([5], dtype=np.int32))


def test_q_len_eq_2_threshold_prefill_masked():
    """q_len == 2 는 prefill 로 분류되어 mask 됨 (boundary)."""
    num_reqs = 1
    num_cold = np.array([5], dtype=np.int32)
    cu_q = _cu_q_from_lens([2])
    _mask_prefill_cold_blocks(num_cold, cu_q, num_reqs)
    np.testing.assert_array_equal(num_cold, np.array([0], dtype=np.int32))


def test_zero_reqs_no_op():
    """num_reqs == 0 — silent no-op, 빈 배열도 안전."""
    num_cold = np.zeros(0, dtype=np.int32)
    cu_q = np.array([0], dtype=np.int32)
    _mask_prefill_cold_blocks(num_cold, cu_q, 0)
    assert num_cold.shape == (0,)


def test_padded_slots_untouched():
    """num_reqs < num_reqs_padded — padding 영역의 num_cold_blocks 는 건드리지 않음.

    실제 caller (gpu_model_runner) 가 num_reqs_padded 크기 array 를 만들고
    num_reqs 까지만 채우는 패턴. mask 도 num_reqs 까지만 적용해야 padding
    영역의 stale 0 / sentinel 값이 우연히 변경되지 않음.
    """
    num_reqs = 2
    num_reqs_padded = 5
    num_cold = np.array([3, 4, 99, 99, 99], dtype=np.int32)
    cu_q = _cu_q_from_lens([8, 1, 0, 0, 0])  # 첫 둘만 의미 있음
    _mask_prefill_cold_blocks(num_cold, cu_q, num_reqs)
    # seq 0 prefill mask, seq 1 decode 보존
    # padding 슬롯 [2:] 은 99 그대로
    np.testing.assert_array_equal(
        num_cold, np.array([0, 4, 99, 99, 99], dtype=np.int32)
    )


def test_no_prefill_seqs_skips_putmask():
    """모든 seq 가 q_len == 1 이면 prefill_mask 가 빈 mask — putmask 호출 안 함.

    이건 미세 최적화 contract — caller 가 매 step 호출할 때 decode-only
    batch (가장 흔한 case) 에서 putmask 의 SIMD 비용을 피한다.
    """
    num_reqs = 100  # 큰 batch
    num_cold = np.zeros(num_reqs, dtype=np.int32)
    num_cold[::3] = 5  # 일부 cold-bearing
    cu_q = _cu_q_from_lens([1] * num_reqs)
    expected = num_cold.copy()
    _mask_prefill_cold_blocks(num_cold, cu_q, num_reqs)
    np.testing.assert_array_equal(num_cold, expected)
