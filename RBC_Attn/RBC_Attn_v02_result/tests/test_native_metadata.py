"""native page 매핑과 절대 causal 경계 검증 (CUDA 불필요한 부분).

지시서 §2: `run_host_tests.sh` 에 포함된다.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rbc import data, native  # noqa: E402


def test_metadata_maps_csr_directly():
    """indptr/indices 는 원래 CSR 을 그대로 쓴다 (K/V 복제 없음)."""
    ptr = np.array([0, 2, 5], dtype=np.int64)
    ids = np.array([0, 3, 1, 2, 7], dtype=np.int32)
    qpos = np.array([1000, 2000], dtype=np.int64)
    indptr, indices, lpl = native.build_metadata(ptr, ids, qpos, bk=128, causal=False)
    assert indptr.tolist() == ptr.tolist()
    assert indices.tolist() == ids.tolist()
    assert lpl.tolist() == [128, 128]


def test_last_page_len_partial_block():
    """q_pos 가 마지막 선택 block 안에 있으면 부분 길이가 나와야 한다."""
    bk = 128
    # query 0: 마지막 선택 block 5 -> 위치 640..767. q_pos=700 -> 700-640+1 = 61
    ptr = np.array([0, 3], dtype=np.int64)
    ids = np.array([0, 2, 5], dtype=np.int32)
    qpos = np.array([700], dtype=np.int64)
    _, _, lpl = native.build_metadata(ptr, ids, qpos, bk=bk, causal=True)
    assert lpl.tolist() == [61]


def test_last_page_len_full_when_block_before_qpos():
    bk = 128
    ptr = np.array([0, 2], dtype=np.int64)
    ids = np.array([0, 1], dtype=np.int32)      # 마지막 block 1 -> 128..255
    qpos = np.array([5000], dtype=np.int64)     # 훨씬 뒤
    _, _, lpl = native.build_metadata(ptr, ids, qpos, bk=bk, causal=True)
    assert lpl.tolist() == [128]


def test_last_page_len_exact_block_boundary():
    """q_pos 가 block 의 마지막 원소일 때 last_page_len == BK."""
    bk = 128
    ptr = np.array([0, 1], dtype=np.int64)
    ids = np.array([3], dtype=np.int32)         # 384..511
    qpos = np.array([511], dtype=np.int64)
    _, _, lpl = native.build_metadata(ptr, ids, qpos, bk=bk, causal=True)
    assert lpl.tolist() == [128]

    qpos = np.array([384], dtype=np.int64)      # block 의 첫 원소
    _, _, lpl = native.build_metadata(ptr, ids, qpos, bk=bk, causal=True)
    assert lpl.tolist() == [1]


def test_empty_row_is_unsupported():
    """설계 §6: 빈 선택 행은 native 매핑이 없다 → 미지원으로 올린다 (우회 금지)."""
    ptr = np.array([0, 0, 2], dtype=np.int64)
    ids = np.array([0, 1], dtype=np.int32)
    qpos = np.array([100, 500], dtype=np.int64)
    with pytest.raises(native.NativeUnsupported):
        native.build_metadata(ptr, ids, qpos, bk=128, causal=True)


def test_future_block_is_unsupported():
    """선택 block 이 q_pos 보다 완전히 뒤면 native 로 표현할 수 없다."""
    ptr = np.array([0, 1], dtype=np.int64)
    ids = np.array([10], dtype=np.int32)        # 1280..1407
    qpos = np.array([100], dtype=np.int64)
    with pytest.raises(native.NativeUnsupported):
        native.build_metadata(ptr, ids, qpos, bk=128, causal=True)


def test_unsorted_row_is_rejected():
    ptr = np.array([0, 3], dtype=np.int64)
    ids = np.array([5, 2, 9], dtype=np.int32)
    qpos = np.array([5000], dtype=np.int64)
    with pytest.raises(native.NativeUnsupported):
        native.build_metadata(ptr, ids, qpos, bk=128, causal=True)


def test_synthetic_sample_metadata_roundtrip():
    """합성 입력(causal 적용 후)이 native 메타데이터로 변환되는지."""
    s = data.make_synthetic("random", hkv=1, nq=16, nk=32768, g=8, d=64, bk=128,
                            sel=12, seed=61, causal=True)
    ptr, ids = s["ptr"][0], s["block_ids"][0]
    indptr, indices, lpl = native.build_metadata(ptr, ids, s["q_positions"],
                                                 bk=s["bk"], causal=True)
    assert indptr[-1] == len(indices)
    assert np.all(lpl >= 1) and np.all(lpl <= s["bk"])
    # causal_filter 를 거쳤으므로 모든 마지막 block 은 q_pos 이하에서 시작한다
    for i in range(len(ptr) - 1):
        last_b = int(ids[ptr[i + 1] - 1])
        assert last_b * s["bk"] <= s["q_positions"][i]


def test_versions_reports_availability():
    v = native.versions()
    assert "flashinfer" in v and "available" in v


def test_lse_base_convention_documented():
    """flashinfer LSE 가 log2 밑임을 모듈이 환산 상수로 반영하고 있는지."""
    assert abs(native.LN2 - 0.6931471805599453) < 1e-15
