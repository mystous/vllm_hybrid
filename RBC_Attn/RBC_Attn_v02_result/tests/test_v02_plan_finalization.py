"""v0.2/P1 — 선택 후 계획 불변, 분할 보존 (지시서 §4.3, §4.5, §10.1)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rbc import data, planner, select  # noqa: E402

POLICIES = ["q_outer", "signature_only", "rbc", "kv_outer"]
SM = 132


def _inp(pattern="random", *, nq=128, nk=8192, g=8, sel=8, seed=3):
    s = data.make_synthetic(pattern, hkv=1, nq=nq, nk=nk, g=g, d=64, bk=128, sel=sel,
                            seed=seed)
    return s, s["ptr"][0], s["block_ids"][0]


def test_default_has_no_forced_split():
    """P1: 기본값은 강제 분할 없음. min_tasks 를 주지 않으면 2*SM 로 늘리지 않는다."""
    s, ptr, ids = _inp()
    for policy in POLICIES:
        p = planner.plan_head(ptr, ids, bk=128, d=64, g=8, max_m=32, policy=policy)
        q = planner.plan_head(ptr, ids, bk=128, d=64, g=8, max_m=32, policy=policy,
                              min_tasks=0)
        assert p.n_tasks == q.n_tasks, f"{policy}: 기본 min_tasks 가 0 이 아니다"
        if policy in ("q_outer", "signature_only", "rbc"):
            assert p.n_tasks < 2 * SM or policy == "kv_outer", (
                f"{policy}: 자연 task 수 {p.n_tasks} 가 2*SM 이상이면 이 검사는 무의미")


@pytest.mark.parametrize("policy", POLICIES)
def test_split_is_a_candidate_not_a_postprocess(policy):
    """분할은 후보로만 생성된다. 같은 min_tasks 는 항상 같은 계획을 준다 (결정성)."""
    s, ptr, ids = _inp()
    for mt in (0, SM, 2 * SM):
        a = planner.plan_head(ptr, ids, bk=128, d=64, g=8, max_m=32, policy=policy,
                              min_tasks=mt)
        b = planner.plan_head(ptr, ids, bk=128, d=64, g=8, max_m=32, policy=policy,
                              min_tasks=mt)
        assert a.n_tasks == b.n_tasks
        assert np.array_equal(a.task_qptr, b.task_qptr)
        assert np.array_equal(a.task_k, b.task_k)
        assert np.array_equal(a.slot_mode, b.slot_mode)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("min_tasks", [0, SM, 2 * SM])
def test_split_preserves_exact_partition(policy, min_tasks):
    """분할 후에도 모든 selected edge 가 정확히 한 번 배정된다."""
    s, ptr, ids = _inp()
    nq = len(ptr) - 1
    p = planner.plan_head(ptr, ids, bk=128, d=64, g=8, max_m=32, policy=policy,
                          min_tasks=min_tasks)
    assert planner.verify_exact_partition(p, ptr, ids)[0] is True


@pytest.mark.parametrize("policy", POLICIES)
def test_split_only_increases_task_count(policy):
    """min_tasks 를 키우면 task 수가 줄지 않는다 (KV 방향 분할이므로)."""
    s, ptr, ids = _inp()
    counts = [planner.plan_head(ptr, ids, bk=128, d=64, g=8, max_m=32, policy=policy,
                                min_tasks=mt).n_tasks for mt in (0, SM, 2 * SM)]
    assert counts[0] <= counts[1] <= counts[2], counts


@pytest.mark.parametrize("policy", POLICIES)
def test_split_preserves_kv_visit_total(policy):
    """분할은 KV block 방문 총량을 바꾸지 않는다 (같은 block 을 나눠 가질 뿐)."""
    s, ptr, ids = _inp()
    base = None
    for mt in (0, SM, 2 * SM):
        p = planner.plan_head(ptr, ids, bk=128, d=64, g=8, max_m=32, policy=policy,
                              min_tasks=mt)
        qs = np.diff(p.task_qptr)
        ks = np.diff(p.task_kptr)
        total = int(ks[qs > 0].sum())
        if base is None:
            base = total
        else:
            assert total == base, f"{policy} min_tasks={mt}: 방문 총량 {total} != {base}"


def test_probe_matches_real_plan_features():
    """probe 로 뽑은 Q-outer/signature 특징이 실제 계획 통계와 같아야 한다 (P4 전제)."""
    for pattern in ("common_private", "random", "clustered"):
        s, ptr, ids = _inp(pattern)
        for mm in (16, 32, 64, 128):
            pr = planner.probe_head(ptr, ids, bk=128, d=64, g=8, max_m=mm)
            for which, policy in (("q", "q_outer"), ("s", "signature_only")):
                f = select.features_from_probe(pr, which, d=64, bk=128)
                p = planner.plan_head(ptr, ids, bk=128, d=64, g=8, max_m=mm,
                                      policy=policy, min_tasks=0)
                g = select.features_from_plan(p, d=64, bk=128)
                assert f.n_tasks == g.n_tasks, (pattern, mm, which, "n_tasks")
                assert f.sum_k == g.sum_k, (pattern, mm, which, "sum_k")
                assert f.max_k == g.max_k, (pattern, mm, which, "max_k")
                assert f.max_q == g.max_q, (pattern, mm, which, "max_q")
                assert f.bm == g.bm, (pattern, mm, which, "bm")
                assert f.padded == g.padded, (pattern, mm, which, "padded")
                assert f.multi_rows == g.multi_rows, (pattern, mm, which, "multi_rows")
                assert f.empty_rows == g.empty_rows, (pattern, mm, which, "empty_rows")


def test_selector_returns_finalized_plan_unchanged():
    """선택기가 돌려준 계획은 그대로 실행 대상이다 — 이후 재분할하지 않는다."""
    table_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "configs", "plan_cost_table.json")
    if not os.path.exists(table_path):
        pytest.skip("교정 표가 없다 (tools/calibrate_plan_cost.py 를 먼저 실행)")
    table = select.CostTable.load(table_path)
    s, ptr, ids = _inp()
    nq = len(ptr) - 1
    p, dec = select.choose_plan(ptr, ids, bk=128, d=64, g=8, max_m=32, table=table)
    assert dec.selected in ("q_outer", "signature", "bounded_merge")
    assert planner.verify_exact_partition(p, ptr, ids)[0] is True
    # 선택된 계획을 다시 만들면 동일해야 한다 (선택 이후 변형 금지)
    pol = {"q_outer": "q_outer", "signature": "signature_only",
           "bounded_merge": "rbc"}[dec.selected]
    q = planner.plan_head(ptr, ids, bk=128, d=64, g=8, max_m=32, policy=pol, min_tasks=0)
    assert np.array_equal(p.task_qptr, q.task_qptr)
    assert np.array_equal(p.task_k, q.task_k)
    assert np.array_equal(p.slot_mode, q.slot_mode)
