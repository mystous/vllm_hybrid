"""v0.2/P2 — direct / partial / empty 출력 소유권 (지시서 §10.1).

이 파일은 host 측 계약을 검사한다. 커널이 실제로 그 계약대로 쓰는지는
tests/test_gpu.py 의 direct-output 검사가 담당한다.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rbc import data, planner  # noqa: E402

POLICIES = ["q_outer", "signature_only", "rbc", "kv_outer"]


def _plan(pattern, policy, *, nq=64, nk=4096, g=8, sel=8, max_m=32, seed=7, min_tasks=0):
    s = data.make_synthetic(pattern, hkv=1, nq=nq, nk=nk, g=g, d=64, bk=128, sel=sel,
                            seed=seed)
    ptr, ids = s["ptr"][0], s["block_ids"][0]
    p = planner.plan_head(ptr, ids, bk=128, d=64, g=g, max_m=max_m, policy=policy,
                          min_tasks=min_tasks)
    return s, ptr, ids, p


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("pattern", ["common_private", "random", "clustered"])
def test_degree_partition_is_exhaustive(policy, pattern):
    """모든 query 는 direct(1) / multi-owner(>=2) / empty(0) 중 정확히 하나다."""
    s, ptr, ids, p = _plan(pattern, policy)
    nq = len(ptr) - 1
    deg = np.zeros(nq, dtype=np.int64)
    for slot, qq in enumerate(p.task_q):
        deg[int(qq)] += 1
    assert int((deg == 1).sum()) == p.direct_rows
    assert int((deg >= 2).sum()) == len(p.multi_rows)
    assert int((deg == 0).sum()) == p.empty_rows
    assert p.direct_rows + len(p.multi_rows) + p.empty_rows == nq


@pytest.mark.parametrize("policy", POLICIES)
def test_slot_mode_matches_degree(policy):
    """slot_mode==1 인 슬롯은 degree==1 행에만 붙는다."""
    s, ptr, ids, p = _plan("random", policy)
    nq = len(ptr) - 1
    deg = np.zeros(nq, dtype=np.int64)
    for qq in p.task_q:
        deg[int(qq)] += 1
    for slot, qq in enumerate(p.task_q):
        want = 1 if deg[int(qq)] == 1 else 0
        assert int(p.slot_mode[slot]) == want, f"슬롯 {slot} (query {qq}) 모드 불일치"


@pytest.mark.parametrize("policy", POLICIES)
def test_reduction_csr_covers_only_multi_owner_rows(policy):
    """reduction CSR 은 multi-owner 행만 담고, direct 슬롯은 들어가지 않는다."""
    s, ptr, ids, p = _plan("random", policy)
    assert len(p.red_ptr) == len(p.multi_rows) + 1
    assert int(p.red_ptr[-1]) == len(p.red_slot)
    direct = {i for i, m in enumerate(p.slot_mode) if int(m) == 1}
    assert not (set(int(x) for x in p.red_slot) & direct)
    for i, row in enumerate(p.multi_rows):
        lo, hi = int(p.red_ptr[i]), int(p.red_ptr[i + 1])
        assert hi - lo >= 2, f"multi 행 {row} 의 degree 가 2 미만"
        for sl in p.red_slot[lo:hi]:
            assert int(p.task_q[int(sl)]) == int(row)


def test_empty_rows_have_no_slot():
    """빈 행은 슬롯을 받지 않는다 (out=0, LSE=-inf 는 별도 커널이 채운다)."""
    nq, sel = 64, 8
    s = data.make_synthetic("random", hkv=1, nq=nq, nk=4096, g=8, d=64, bk=128, sel=sel,
                            seed=11)
    ptr = np.array(s["ptr"][0], dtype=np.int64)
    ids = np.array(s["block_ids"][0], dtype=np.int32)
    # query 0 과 마지막 query 의 선택을 비운다
    keep = np.ones(len(ids), dtype=bool)
    keep[ptr[0]:ptr[1]] = False
    keep[ptr[nq - 1]:ptr[nq]] = False
    new_ids = ids[keep]
    cnt = np.diff(ptr).copy()
    cnt[0] = 0
    cnt[nq - 1] = 0
    new_ptr = np.zeros(nq + 1, dtype=np.int64)
    new_ptr[1:] = np.cumsum(cnt)
    for policy in POLICIES:
        p = planner.plan_head(new_ptr, new_ids, bk=128, d=64, g=8, max_m=32,
                              policy=policy, min_tasks=0)
        assert p.empty_rows == 2, f"{policy}: empty_rows={p.empty_rows}"
        assert 0 not in set(int(x) for x in p.task_q)
        assert (nq - 1) not in set(int(x) for x in p.task_q)
        assert planner.verify_exact_partition(p, new_ptr, new_ids)[0] is True


@pytest.mark.parametrize("policy", POLICIES)
def test_q_outer_has_no_partial_slots(policy):
    """Q-outer 는 그룹당 task 하나이므로 partial 이 0 이어야 한다 (P2 의 구조적 결과)."""
    s, ptr, ids, p = _plan("common_private", policy)
    st = p.stats(n_edges=len(ids))
    if policy == "q_outer":
        assert st["partial_slots"] == 0
        assert st["multi_owner_rows"] == 0
    # 다른 정책은 partial 이 있을 수 있고, 있으면 CSR 과 일치해야 한다
    assert st["partial_slots"] == int((p.slot_mode == 0).sum())
