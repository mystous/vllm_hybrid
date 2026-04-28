# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TSK_002 §4.5c — `_pack_cold_cpu_block_ids` packing helper 단위 회귀.

사용자 코드 review 에서 발견된 shape mismatch 버그의 가드:
prefill seq 의 *원래 (mask 전) cold ids 길이* 가 decode seq 의 mask 후
max 보다 클 때, 이전 코드처럼 ``cold_ids_np[i, :len(ids)] = ids`` 로
원래 길이 그대로 채우면 row width (= max_cold_per_req) 초과로
``ValueError``. 본 helper 는 *masked count* 만큼만 채우고 prefill seq
(n=0) 는 skip 하므로 안전.

전형 시나리오 — prefill cold 630, decode cold 10:
  - max_cold_per_req = 10 (mask 후)
  - cold_ids_np shape = (..., 10)
  - prefill row: n=0 → skip (모두 0)
  - decode row: n=10 → ids[:10] 로 채움
"""

from __future__ import annotations

import numpy as np
import pytest

from vllm.v1.worker.gpu_model_runner import _pack_cold_cpu_block_ids


def test_max_zero_returns_empty_tensor():
    """mask 후 모두 0 — packing 은 (..., 0) 빈 텐서."""
    num_cold = np.zeros(4, dtype=np.int32)
    cold_ids_np, max_cold = _pack_cold_cpu_block_ids(
        num_cold,
        cold_ids_per_req={"a": [1, 2], "b": [3]},
        req_ids=["a", "b", "c", "d"],
        num_reqs=4,
        num_reqs_padded=4,
    )
    assert max_cold == 0
    assert cold_ids_np.shape == (4, 0)


def test_decode_only_packing():
    """모두 decode (mask 영향 없음) — 각 row 가 mask 후 길이만큼 채워짐."""
    num_cold = np.array([3, 5, 0, 2], dtype=np.int32)
    cold_ids_per_req = {
        "a": [10, 11, 12],
        "b": [20, 21, 22, 23, 24],
        "c": [],  # cold 없음
        "d": [40, 41],
    }
    cold_ids_np, max_cold = _pack_cold_cpu_block_ids(
        num_cold,
        cold_ids_per_req,
        req_ids=["a", "b", "c", "d"],
        num_reqs=4,
        num_reqs_padded=4,
    )
    assert max_cold == 5
    assert cold_ids_np.shape == (4, 5)
    np.testing.assert_array_equal(cold_ids_np[0], [10, 11, 12, 0, 0])
    np.testing.assert_array_equal(cold_ids_np[1], [20, 21, 22, 23, 24])
    np.testing.assert_array_equal(cold_ids_np[2], [0, 0, 0, 0, 0])
    np.testing.assert_array_equal(cold_ids_np[3], [40, 41, 0, 0, 0])


def test_prefill_orig_cold_larger_than_decode_max_no_mismatch():
    """**핵심 회귀** — prefill seq 의 *원래* cold ids 가 매우 길고 (630),
    decode seq 가 짧은 (10) 시나리오. mask 후 prefill row 는 0 으로 되어
    있어 packing 에서 skip 되어야 한다. 이전 코드는 prefill row 에
    ids[630개] 를 row width=10 자리에 넣으려다 ``ValueError`` 발생."""
    num_cold = np.array([0, 10, 0], dtype=np.int32)  # mask 후
    prefill_orig_ids = list(range(630))  # mask 전 원래 길이 630
    cold_ids_per_req = {
        "prefill_a": prefill_orig_ids,
        "decode_b": list(range(100, 110)),  # decode 의 cold 10 개
        "prefill_c": list(range(200, 230)),  # mask 전 30 개
    }
    cold_ids_np, max_cold = _pack_cold_cpu_block_ids(
        num_cold,
        cold_ids_per_req,
        req_ids=["prefill_a", "decode_b", "prefill_c"],
        num_reqs=3,
        num_reqs_padded=3,
    )
    assert max_cold == 10, "mask 후 max — decode_b 의 10"
    assert cold_ids_np.shape == (3, 10)
    # prefill row 는 모두 0 (mask 후 n=0 → skip)
    np.testing.assert_array_equal(cold_ids_np[0], np.zeros(10, dtype=np.int32))
    np.testing.assert_array_equal(cold_ids_np[2], np.zeros(10, dtype=np.int32))
    # decode row 는 ids 그대로
    np.testing.assert_array_equal(
        cold_ids_np[1], np.arange(100, 110, dtype=np.int32)
    )


def test_padded_slots_left_zero():
    """num_reqs < num_reqs_padded — padding row 는 모두 0."""
    num_cold = np.array([2, 3, 0, 0], dtype=np.int32)
    cold_ids_per_req = {"a": [1, 2], "b": [3, 4, 5]}
    cold_ids_np, max_cold = _pack_cold_cpu_block_ids(
        num_cold,
        cold_ids_per_req,
        req_ids=["a", "b"],  # num_reqs=2
        num_reqs=2,
        num_reqs_padded=4,
    )
    assert max_cold == 3
    assert cold_ids_np.shape == (4, 3)
    np.testing.assert_array_equal(cold_ids_np[0], [1, 2, 0])
    np.testing.assert_array_equal(cold_ids_np[1], [3, 4, 5])
    np.testing.assert_array_equal(cold_ids_np[2], [0, 0, 0])
    np.testing.assert_array_equal(cold_ids_np[3], [0, 0, 0])


def test_missing_or_empty_ids_skipped():
    """cold_ids_per_req 에 entry 없거나 빈 list — silent skip (raise 안 함)."""
    num_cold = np.array([0, 5, 0], dtype=np.int32)
    cold_ids_per_req = {
        # "a" 없음
        "b": [1, 2, 3, 4, 5],
        "c": [],  # 빈 list
    }
    cold_ids_np, max_cold = _pack_cold_cpu_block_ids(
        num_cold,
        cold_ids_per_req,
        req_ids=["a", "b", "c"],
        num_reqs=3,
        num_reqs_padded=3,
    )
    assert max_cold == 5
    np.testing.assert_array_equal(cold_ids_np[0], [0, 0, 0, 0, 0])
    np.testing.assert_array_equal(cold_ids_np[1], [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(cold_ids_np[2], [0, 0, 0, 0, 0])


def test_assert_inconsistent_metadata():
    """``num_cold_blocks > len(ids)`` — connector metadata 불일치는 silent
    truncation 대신 AssertionError 로 raise."""
    num_cold = np.array([5], dtype=np.int32)  # 5 개 요구
    cold_ids_per_req = {"a": [1, 2]}  # 그러나 ids 는 2 개만
    with pytest.raises(AssertionError, match=r"connector metadata inconsistency"):
        _pack_cold_cpu_block_ids(
            num_cold,
            cold_ids_per_req,
            req_ids=["a"],
            num_reqs=1,
            num_reqs_padded=1,
        )


def test_takes_first_n_ids_when_ids_longer():
    """``len(ids) > n`` 은 정상 — ``ids[:n]`` 만 사용. mask 후 n 이 줄어든
    decode 시나리오를 흉내. (실제로 OffloadingConnector 는 lookup hit
    수만큼 ids 를 보내므로 보통 ``len(ids) == n`` 이지만, 향후 connector
    가 더 많이 보낼 수도 있으니 안전하게)."""
    num_cold = np.array([3], dtype=np.int32)
    cold_ids_per_req = {"a": [10, 11, 12, 99, 99, 99]}  # 앞 3 만 의미
    cold_ids_np, max_cold = _pack_cold_cpu_block_ids(
        num_cold,
        cold_ids_per_req,
        req_ids=["a"],
        num_reqs=1,
        num_reqs_padded=1,
    )
    assert max_cold == 3
    np.testing.assert_array_equal(cold_ids_np[0], [10, 11, 12])
