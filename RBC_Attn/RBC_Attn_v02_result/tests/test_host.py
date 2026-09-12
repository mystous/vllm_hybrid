"""호스트 테스트 — edge partition 정확성과 FP64 수식 검산 (CUDA 불필요).

지시서 §2: `run_host_tests.sh` 로 실행. CUDA 없음으로 skip 된 것은 통과가 아니다.
"""

import itertools
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rbc import data, planner  # noqa: E402

POLICIES = ["rbc", "signature_only", "q_outer", "kv_outer"]
PATTERNS = ["common_private", "random", "clustered", "disjoint"]


# --------------------------------------------------------------------------
# CSR 계약
# --------------------------------------------------------------------------
def test_csr_validation_accepts_valid():
    ptr = np.array([0, 2, 3], dtype=np.int64)
    ids = np.array([1, 5, 2], dtype=np.int32)
    data.validate_csr(ptr, ids, nq=2, n_blocks=8)


@pytest.mark.parametrize("ptr,ids,nq,why", [
    (np.array([0, 2], dtype=np.int64), np.array([1, 5], np.int32), 2, "ptr 길이가 Nq+1 이 아니다"),
    (np.array([1, 2, 3], dtype=np.int64), np.array([1, 5], np.int32), 2, "ptr[0] != 0"),
    (np.array([0, 2, 2], dtype=np.int64), np.array([5, 1], np.int32), 2, "행이 정렬돼 있지 않다"),
    (np.array([0, 2, 2], dtype=np.int64), np.array([1, 1], np.int32), 2, "행에 중복이 있다"),
    (np.array([0, 2, 3], dtype=np.int64), np.array([1, 5], np.int32), 2, "ptr[-1] 과 길이 불일치"),
    (np.array([0, 1, 2], dtype=np.int64), np.array([1, 99], np.int32), 2, "block id 범위 밖"),
])
def test_csr_validation_rejects(ptr, ids, nq, why):
    with pytest.raises(ValueError):
        data.validate_csr(ptr, ids, nq=nq, n_blocks=8)


def test_causal_filter_drops_future_blocks():
    # q_pos=100, bk=128 -> block 0 만 남는다 (block 1 은 128 부터 시작)
    ptr = np.array([0, 3], dtype=np.int64)
    ids = np.array([0, 1, 2], dtype=np.int32)
    p2, i2 = data.causal_filter(ptr, ids, np.array([100]), bk=128)
    assert i2.tolist() == [0]
    assert p2.tolist() == [0, 1]


# --------------------------------------------------------------------------
# exact-edge partition
# --------------------------------------------------------------------------
@pytest.mark.parametrize("pattern,policy", list(itertools.product(PATTERNS, POLICIES)))
def test_exact_partition(pattern, policy):
    nk = 65536
    sel = 8 if pattern == "disjoint" else 16
    nq = 8 if pattern in ("common_private", "disjoint") else 16
    s = data.make_synthetic(pattern, hkv=1, nq=nq, nk=nk, g=8, d=64, bk=128,
                            sel=sel, seed=3, causal=False)
    ptr, ids = s["ptr"][0], s["block_ids"][0]
    p = planner.plan_head(ptr, ids, bk=128, d=64, g=8, max_m=128, policy=policy)
    ok, msg = planner.verify_exact_partition(p, ptr, ids)
    assert ok, f"{pattern}/{policy}: {msg}"


@pytest.mark.parametrize("policy", POLICIES)
def test_empty_rows_are_handled(policy):
    """빈 선택 행이 있어도 계획이 성립하고 그 query 는 어떤 task 에도 들어가지 않는다.

    v0.2/P2: reduction CSR 은 multi-owner 행만 담으므로 빈 행은 애초에 목록에 없다.
    """
    ptr = np.array([0, 2, 2, 4], dtype=np.int64)      # query 1 은 빈 행
    ids = np.array([0, 1, 0, 3], dtype=np.int32)
    p = planner.plan_head(ptr, ids, bk=128, d=64, g=8, max_m=128, policy=policy)
    ok, msg = planner.verify_exact_partition(p, ptr, ids)
    assert ok, msg
    assert 1 not in set(int(x) for x in p.task_q), "빈 행이 task 에 들어갔다"
    assert 1 not in set(int(x) for x in p.multi_rows), "빈 행이 multi_rows 에 들어갔다"
    assert p.empty_rows == 1


@pytest.mark.parametrize("policy", POLICIES)
def test_all_rows_empty(policy):
    ptr = np.array([0, 0, 0], dtype=np.int64)
    ids = np.zeros(0, dtype=np.int32)
    p = planner.plan_head(ptr, ids, bk=128, d=64, g=8, max_m=128, policy=policy)
    assert p.n_tasks == 0
    assert p.red_ptr.tolist() == [0]        # multi-owner 행 0개
    assert p.empty_rows == 2 and p.direct_rows == 0


@pytest.mark.parametrize("g,max_m", [(1, 16), (4, 64), (8, 128), (16, 128), (16, 64)])
def test_group_size_respects_max_m(g, max_m):
    """한 계획 그룹의 query 수는 min(64, max_m/G) 이하 (설계 §2)."""
    s = data.make_synthetic("random", hkv=1, nq=200, nk=65536, g=g, d=64, bk=128,
                            sel=12, seed=5, causal=False)
    ptr, ids = s["ptr"][0], s["block_ids"][0]
    p = planner.plan_head(ptr, ids, bk=128, d=64, g=g, max_m=max_m, policy="rbc")
    ok, msg = planner.verify_exact_partition(p, ptr, ids)
    assert ok, msg
    limit = min(64, max(1, max_m // g))
    qs = np.diff(p.task_qptr)
    assert qs.max() <= limit, f"task 의 query 수 {qs.max()} > 한도 {limit}"


# --------------------------------------------------------------------------
# 구조 지표 — 설계 §3 표
# --------------------------------------------------------------------------
def test_design_section3_table():
    """설계 §3: 8 query, G=16, 공통 8 + 개인 8 에서의 task/partial/방문/padded."""
    s = data.make_synthetic("common_private", hkv=1, nq=8, nk=65536, g=16, d=128,
                            bk=128, sel=16, seed=0, causal=False)
    ptr, ids = s["ptr"][0], s["block_ids"][0]
    n_edges = len(ids)
    assert n_edges == 8 * 16

    got = {}
    for pol in POLICIES:
        p = planner.plan_head(ptr, ids, bk=128, d=128, g=16, max_m=128, policy=pol)
        got[pol] = p.stats(n_edges=n_edges)

    # 설계 §3 표의 "query별 partial 슬롯 총합" 은 v0.2 에서 query_task_incidences 다.
    assert got["q_outer"]["n_tasks"] == 1
    assert got["q_outer"]["query_task_incidences"] == 8
    assert abs(got["q_outer"]["padded_ratio"] - 4.5) < 1e-6

    assert got["kv_outer"]["n_tasks"] == 72
    assert got["kv_outer"]["query_task_incidences"] == 128
    assert abs(got["kv_outer"]["padded_ratio"] - 1.0) < 1e-6

    assert got["rbc"]["n_tasks"] == 9
    assert got["rbc"]["query_task_incidences"] == 16
    assert abs(got["rbc"]["padded_ratio"] - 1.0) < 1e-6

    # v0.2/P2 의 직접 효과: Q-outer 는 task 가 하나라 모든 행이 단일 소유가 되어
    # partial workspace 를 전혀 쓰지 않는다. RBC 는 각 query 가 공통+개인 두 task 에
    # 걸치므로 여전히 merge 가 필요하다. 이 구조적 차이를 기록해 둔다.
    assert got["q_outer"]["partial_slots"] == 0
    assert got["q_outer"]["direct_rows"] == 8
    assert got["q_outer"]["multi_owner_rows"] == 0
    assert got["rbc"]["partial_slots"] == 16
    assert got["rbc"]["multi_owner_rows"] == 8
    assert got["kv_outer"]["partial_slots"] == 128

    # 모든 정책의 KV block 방문 수가 같아야 한다 (edge 를 보존하므로)
    visits = {k: v["kv_block_visits"] for k, v in got.items()}
    assert len(set(visits.values())) == 1, visits


def test_rbc_never_worse_than_signature_only_in_cost():
    """RBC 는 signature-only 에서 출발해 순비용이 양인 병합만 하므로 task 수가 같거나 적다."""
    for pattern in PATTERNS:
        sel = 8 if pattern == "disjoint" else 16
        s = data.make_synthetic(pattern, hkv=1, nq=16, nk=65536, g=8, d=64, bk=128,
                                sel=sel, seed=7, causal=False)
        ptr, ids = s["ptr"][0], s["block_ids"][0]
        a = planner.plan_head(ptr, ids, bk=128, d=64, g=8, max_m=128, policy="signature_only")
        b = planner.plan_head(ptr, ids, bk=128, d=64, g=8, max_m=128, policy="rbc")
        assert b.n_tasks <= a.n_tasks, f"{pattern}: rbc {b.n_tasks} > sig {a.n_tasks}"


def test_min_tasks_increases_task_count():
    s = data.make_synthetic("common_private", hkv=1, nq=8, nk=65536, g=16, d=128,
                            bk=128, sel=16, seed=0, causal=False)
    ptr, ids = s["ptr"][0], s["block_ids"][0]
    base = planner.plan_head(ptr, ids, bk=128, d=128, g=16, max_m=128, policy="rbc")
    more = planner.plan_head(ptr, ids, bk=128, d=128, g=16, max_m=128, policy="rbc",
                             min_tasks=base.n_tasks + 5)
    assert more.n_tasks >= base.n_tasks + 5
    ok, msg = planner.verify_exact_partition(more, ptr, ids)
    assert ok, msg


def test_cost_weights_change_decisions():
    """비용 계수를 바꾸면 병합 결정이 달라져야 한다 (교정되지 않은 휴리스틱임을 드러낸다)."""
    s = data.make_synthetic("random", hkv=1, nq=16, nk=8192, g=8, d=64, bk=128,
                            sel=10, seed=11, causal=False)
    ptr, ids = s["ptr"][0], s["block_ids"][0]
    cheap_flops = planner.plan_head(ptr, ids, bk=128, d=64, g=8, max_m=128, policy="rbc",
                                    cost=dict(flop_weight=0.0))
    dear_flops = planner.plan_head(ptr, ids, bk=128, d=64, g=8, max_m=128, policy="rbc",
                                   cost=dict(flop_weight=10.0))
    # FLOPs 가 비싸면 병합을 덜 해 task 가 많아진다
    assert cheap_flops.n_tasks <= dear_flops.n_tasks
    for p in (cheap_flops, dear_flops):
        ok, msg = planner.verify_exact_partition(p, ptr, ids)
        assert ok, msg


# --------------------------------------------------------------------------
# FP64 수식 검산 (partial 결합)
# --------------------------------------------------------------------------
def _merge_reference(states):
    """설계 §2.2 수식을 FP64 로 계산한다. states = [(u, m, l), ...]"""
    m = max(s[1] for s in states)
    l = sum(np.exp(s[1] - m) * s[2] for s in states)
    u = sum(np.exp(s[1] - m) * s[0] for s in states)
    return u / l, m + np.log(l)


def test_partial_merge_formula_matches_direct_softmax():
    rng = np.random.default_rng(0)
    d = 16
    for trial in range(20):
        n_parts = int(rng.integers(1, 5))
        scores, vals = [], []
        states = []
        for _ in range(n_parts):
            n = int(rng.integers(1, 7))
            s = rng.standard_normal(n) * 3.0
            v = rng.standard_normal((n, d))
            scores.append(s); vals.append(v)
            m = s.max()
            p = np.exp(s - m)
            states.append((p @ v, m, p.sum()))
        out_merged, lse_merged = _merge_reference(states)

        s_all = np.concatenate(scores)
        v_all = np.concatenate(vals, axis=0)
        m = s_all.max()
        p = np.exp(s_all - m)
        out_direct = (p @ v_all) / p.sum()
        lse_direct = m + np.log(p.sum())

        assert np.allclose(out_merged, out_direct, atol=1e-12), trial
        assert abs(lse_merged - lse_direct) < 1e-12, trial


def test_reference_attention_matches_dense_on_full_mask():
    """모든 block 을 선택하면 참조 구현이 dense attention 과 같아야 한다."""
    nq, nk, g, d, bk = 4, 512, 2, 32, 128
    rng = np.random.default_rng(1)
    n_blocks = nk // bk
    ptr = np.arange(nq + 1, dtype=np.int64) * n_blocks
    ids = np.tile(np.arange(n_blocks, dtype=np.int32), nq)
    sample = dict(
        q=rng.standard_normal((1, nq, g, d), dtype=np.float32),
        k=rng.standard_normal((1, nk, d), dtype=np.float32),
        v=rng.standard_normal((1, nk, d), dtype=np.float32),
        ptr=ptr[None, :], block_ids=[ids], bk=bk,
        q_positions=np.full(nq, nk - 1, dtype=np.int64), causal=False,
        scale=np.float32(1.0 / np.sqrt(d)))
    out, lse = data.reference_attention(sample)

    q = sample["q"][0].astype(np.float64); k = sample["k"][0].astype(np.float64)
    v = sample["v"][0].astype(np.float64)
    for i in range(nq):
        s = (q[i] @ k.T) / np.sqrt(d)
        m = s.max(axis=1, keepdims=True)
        p = np.exp(s - m)
        want = (p @ v) / p.sum(axis=1, keepdims=True)
        assert np.allclose(out[0, i], want, atol=1e-10)


def test_empty_row_reference_is_zero_and_neg_inf():
    nq, nk, g, d, bk = 2, 256, 2, 16, 128
    sample = dict(
        q=np.zeros((1, nq, g, d), np.float32), k=np.zeros((1, nk, d), np.float32),
        v=np.zeros((1, nk, d), np.float32),
        ptr=np.array([[0, 0, 1]], dtype=np.int64), block_ids=[np.array([0], np.int32)],
        bk=bk, q_positions=np.array([10, 200]), causal=True,
        scale=np.float32(0.25))
    out, lse = data.reference_attention(sample)
    assert np.all(out[0, 0] == 0.0)
    assert np.all(np.isneginf(lse[0, 0]))


# --------------------------------------------------------------------------
# 스레드
# --------------------------------------------------------------------------
@pytest.mark.parametrize("threads", [1, 2, 8])
def test_threads_produce_identical_plans(threads):
    s = data.make_synthetic("random", hkv=1, nq=128, nk=65536, g=8, d=64, bk=128,
                            sel=12, seed=13, causal=False)
    ptr, ids = s["ptr"][0], s["block_ids"][0]
    a = planner.plan_head(ptr, ids, bk=128, d=64, g=8, max_m=128, policy="rbc", threads=1)
    b = planner.plan_head(ptr, ids, bk=128, d=64, g=8, max_m=128, policy="rbc",
                          threads=threads)
    assert np.array_equal(a.task_qptr, b.task_qptr)
    assert np.array_equal(a.task_q, b.task_q)
    assert np.array_equal(a.task_kptr, b.task_kptr)
    assert np.array_equal(a.task_k, b.task_k)
    assert np.array_equal(a.task_kmask, b.task_kmask)
