"""C++ 플래너의 ctypes 바인딩과 계획 자료구조.

계획(Plan)은 GPU 실행기에 넘길 descriptor 묶음이다.

  task_qptr[T+1], task_q[]   : task 별 query 전역 인덱스
  task_kptr[T+1], task_k[]   : task 별 KV block 인덱스
  red_ptr[Nq+1], red_task[]  : query 별 기여 task 목록 (partial 결합용)

정확성 계약: 원래 (query, KV block) edge 를 중복 없이 정확히 한 번 배정한다.
`verify_exact_partition` 이 이를 검사한다.
"""

from __future__ import annotations

import ctypes
import os
import time
from dataclasses import dataclass, field

import numpy as np

_LIB = None

POLICIES = {"rbc": 0, "signature_only": 1, "q_outer": 2, "kv_outer": 3}


def _load():
    global _LIB
    if _LIB is not None:
        return _LIB
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "build", "librbc_planner.so")
    if not os.path.exists(path):
        raise RuntimeError(f"플래너가 빌드돼 있지 않다: {path} (bash build.sh)")
    lib = ctypes.CDLL(path)

    class Sizes(ctypes.Structure):
        _fields_ = [("n_tasks", ctypes.c_int32), ("n_task_q", ctypes.c_int32),
                    ("n_task_k", ctypes.c_int32), ("n_red", ctypes.c_int32),
                    ("n_multi_rows", ctypes.c_int32), ("direct_rows", ctypes.c_int32),
                    ("empty_rows", ctypes.c_int32)]

    lib.rbc_plan.restype = ctypes.c_int
    lib.rbc_plan.argtypes = [
        ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_int, ctypes.POINTER(Sizes)]
    lib.rbc_fetch.restype = ctypes.c_int
    lib.rbc_fetch.argtypes = ([ctypes.POINTER(ctypes.c_int32)] * 6
                              + [ctypes.POINTER(ctypes.c_uint64)]
                              + [ctypes.POINTER(ctypes.c_int32)] * 2)
    lib.rbc_max_threads.restype = ctypes.c_int
    # v0.2/P4: 계획을 만들지 않고 후보 특징만 뽑는 probe
    lib.rbc_probe.restype = ctypes.c_int
    lib.rbc_probe.argtypes = [
        ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double)]
    lib.rbc_probe_len.restype = ctypes.c_int
    lib._Sizes = Sizes
    _LIB = lib
    return lib


@dataclass
class Plan:
    task_qptr: np.ndarray
    task_q: np.ndarray
    task_kptr: np.ndarray
    task_k: np.ndarray
    red_ptr: np.ndarray           # 길이 n_multi_rows+1 (v0.2/P2: multi-owner 행만)
    red_slot: np.ndarray          # partial 슬롯 번호 (task_q 안의 위치)
    task_kmask: np.ndarray        # task_k 와 같은 길이. block 별 query 슬롯 비트마스크
    slot_mode: np.ndarray         # 슬롯별 0=partial, 1=direct (P2)
    multi_rows: np.ndarray        # degree>=2 인 query 행 번호
    direct_rows: int = 0
    empty_rows: int = 0
    policy: str = "rbc"
    plan_ms: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def n_tasks(self) -> int:
        return len(self.task_qptr) - 1

    def task_shape(self, t: int):
        return (int(self.task_qptr[t + 1] - self.task_qptr[t]),
                int(self.task_kptr[t + 1] - self.task_kptr[t]))

    def stats(self, n_edges: int | None = None) -> dict:
        """설계 §3 표에 대응하는 구조 지표. **논리값이며 실측 바이트가 아니다.**

        padded_units 는 설계 §2.1 의 padded dot 정의를 따른다:
            sum_t round_pow2(max(16, q_t * G)) * k_t
        선택 edge 의 이론 최소치는 (edge 수 * G) 이므로 그 비율을 padded_ratio 로 낸다.
        """
        g = int(self.meta.get("g", 1))
        qs = np.diff(self.task_qptr).astype(np.int64)
        ks = np.diff(self.task_kptr).astype(np.int64)

        def rp2(x):
            v = 16
            while v < x:
                v <<= 1
            return v

        padded = int(sum(rp2(max(16, int(q) * g)) * int(k) for q, k in zip(qs, ks)))
        # v0.2/P2: query_task_incidences 는 전체 슬롯 수, partial_slots 는 그중 실제
        # partial workspace 에 쓰는 슬롯 수다 (direct 행 제외). 이름을 섞지 않는다.
        n_partial = int((self.slot_mode == 0).sum()) if self.slot_mode.size else 0
        out = dict(
            n_tasks=int(self.n_tasks),
            query_task_incidences=int(qs.sum()),
            partial_slots=n_partial,
            direct_rows=int(self.direct_rows),
            multi_owner_rows=int(len(self.multi_rows)),
            empty_rows=int(self.empty_rows),
            kv_block_visits=int((qs > 0).astype(np.int64).dot(ks)),
            padded_units=padded,
            max_q=int(qs.max()) if qs.size else 0,
            max_k=int(ks.max()) if ks.size else 0,
        )
        if n_edges:
            out["padded_ratio"] = round(padded / (n_edges * g), 4)
        return out


PROBE_FIELDS = (
    "n_groups", "group_q", "n_edges", "empty_rows", "nonempty_rows",
    "q_ntasks", "q_max_q", "q_max_k", "q_sum_k",
    "s_ntasks", "s_max_q", "s_max_k", "s_sum_k",
    "q_vis_bm1", "q_vis_bm2", "q_vis_bm4", "q_vis_bm8",
    "q_vis_bm16", "q_vis_bm32", "q_vis_bm64", "q_vis_bm128",
    "s_vis_bm1", "s_vis_bm2", "s_vis_bm4", "s_vis_bm8",
    "s_vis_bm16", "s_vis_bm32", "s_vis_bm64", "s_vis_bm128",
    "q_padded", "s_padded", "q_slots", "s_slots",
    "q_direct_rows", "s_direct_rows", "q_multi_rows", "s_multi_rows",
    "max_atoms_in_group", "deg_p50", "deg_p90", "deg_max",
    "unique_blocks", "issued_blocks",
    "q_k_p50", "q_k_p90", "s_k_p50", "s_k_p90",
    "q_full_ratio", "s_full_ratio",
)


def probe_head(ptr, block_ids, *, bk=128, d=128, g=8, max_m=128, threads=None) -> dict:
    """계획을 만들지 않고 Q-outer / signature 후보의 예측 특징만 계산한다 (P4 §7.7).

    반환값에 `probe_us` (특징 계산에 쓴 CPU 시간) 를 포함한다. 선택기의 CPU 비용은
    이 값까지 포함해서 보고해야 한다.
    """
    lib = _load()
    ptr = np.ascontiguousarray(ptr, dtype=np.int64)
    block_ids = np.ascontiguousarray(block_ids, dtype=np.int32)
    nq = len(ptr) - 1
    n = lib.rbc_probe_len()
    buf = np.zeros(n, dtype=np.float64)
    th = int(os.environ.get("RBC_THREADS", 1)) if threads is None else int(threads)
    t0 = time.perf_counter()
    used = lib.rbc_probe(ptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                         block_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                         nq, bk, d, g, max_m, th,
                         buf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    us = (time.perf_counter() - t0) * 1e6
    out = {name: float(buf[i]) for i, name in enumerate(PROBE_FIELDS[:used])}
    out["probe_us"] = us
    out["nq"] = nq
    out["g"] = g
    out["max_m"] = max_m
    return out


def plan_head(ptr: np.ndarray, block_ids: np.ndarray, *, bk=128, d=128, g=8,
              max_m=128, policy="rbc", min_tasks=0, threads=None,
              cost=None) -> Plan:
    """한 KV head 의 CSR 로부터 계획을 만든다."""
    lib = _load()
    if policy not in POLICIES:
        raise ValueError(f"알 수 없는 정책: {policy} ({list(POLICIES)})")
    cost = cost or {}
    threads = int(threads if threads is not None else os.environ.get("RBC_THREADS", 1))
    ptr = np.ascontiguousarray(ptr, dtype=np.int64)
    block_ids = np.ascontiguousarray(block_ids, dtype=np.int32)
    nq = len(ptr) - 1
    sizes = lib._Sizes()

    t0 = time.perf_counter()
    rc = lib.rbc_plan(ptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                      block_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                      nq, bk, d, g, max_m, POLICIES[policy], min_tasks,
                      float(cost.get("byte_weight", 1.0)),
                      float(cost.get("flop_weight", 0.015)),
                      float(cost.get("task_weight", 0.0)),
                      float(cost.get("partial_weight", 1.0)),
                      threads, ctypes.byref(sizes))
    if rc != 0:
        raise RuntimeError(f"rbc_plan 실패 rc={rc}")

    task_qptr = np.empty(sizes.n_tasks + 1, dtype=np.int32)
    task_q = np.empty(max(sizes.n_task_q, 1), dtype=np.int32)
    task_kptr = np.empty(sizes.n_tasks + 1, dtype=np.int32)
    task_k = np.empty(max(sizes.n_task_k, 1), dtype=np.int32)
    red_ptr = np.empty(nq + 1, dtype=np.int32)   # 상한 할당, 아래에서 절단
    red_slot = np.empty(max(sizes.n_red, 1), dtype=np.int32)
    task_kmask = np.empty(max(sizes.n_task_k, 1), dtype=np.uint64)
    slot_mode = np.empty(max(sizes.n_task_q, 1), dtype=np.int32)
    multi_rows = np.empty(max(sizes.n_multi_rows, 1), dtype=np.int32)
    P = ctypes.POINTER(ctypes.c_int32)
    P64 = ctypes.POINTER(ctypes.c_uint64)
    lib.rbc_fetch(task_qptr.ctypes.data_as(P), task_q.ctypes.data_as(P),
                  task_kptr.ctypes.data_as(P), task_k.ctypes.data_as(P),
                  red_ptr.ctypes.data_as(P), red_slot.ctypes.data_as(P),
                  task_kmask.ctypes.data_as(P64), slot_mode.ctypes.data_as(P),
                  multi_rows.ctypes.data_as(P))
    plan_ms = (time.perf_counter() - t0) * 1e3

    return Plan(task_qptr[:sizes.n_tasks + 1], task_q[:sizes.n_task_q],
                task_kptr[:sizes.n_tasks + 1], task_k[:sizes.n_task_k],
                red_ptr[:sizes.n_multi_rows + 1], red_slot[:sizes.n_red],
                task_kmask[:sizes.n_task_k], slot_mode[:sizes.n_task_q],
                multi_rows[:sizes.n_multi_rows],
                direct_rows=int(sizes.direct_rows), empty_rows=int(sizes.empty_rows),
                policy=policy, plan_ms=plan_ms,
                meta=dict(bk=bk, d=d, g=g, max_m=max_m, min_tasks=min_tasks,
                          threads=threads))



# --------------------------------------------------------------------------
# v0.2/P5: descriptor 를 호출자가 준 버퍼 (pinned slab) 에 C++ 이 직접 쓰게 한다.
# Python list -> NumPy -> pinned Tensor -> device Tensor 연쇄를 없앤다 (§8.2).
# --------------------------------------------------------------------------

class PlanSizes:
    """rbc_plan 호출 결과 크기. 이 시점에 descriptor 는 C++ 안에만 있다."""

    __slots__ = ("n_tasks", "n_task_q", "n_task_k", "n_red", "n_multi_rows",
                 "direct_rows", "empty_rows", "nq", "meta", "plan_call_ms")

    def __init__(self, sz, nq, meta, plan_call_ms):
        self.n_tasks = int(sz.n_tasks)
        self.n_task_q = int(sz.n_task_q)
        self.n_task_k = int(sz.n_task_k)
        self.n_red = int(sz.n_red)
        self.n_multi_rows = int(sz.n_multi_rows)
        self.direct_rows = int(sz.direct_rows)
        self.empty_rows = int(sz.empty_rows)
        self.nq = int(nq)
        self.meta = meta
        self.plan_call_ms = float(plan_call_ms)

    def lengths(self) -> dict:
        """slab 에 예약해야 할 배열별 원소 수 (int32 배열 / int64 배열 구분)."""
        return dict(
            task_qptr=self.n_tasks + 1, task_q=self.n_task_q,
            task_kptr=self.n_tasks + 1, task_k=self.n_task_k,
            red_ptr=self.n_multi_rows + 1, red_slot=self.n_red,
            slot_mode=self.n_task_q, multi_rows=self.n_multi_rows,
            task_kmask=self.n_task_k)


def plan_sizes(ptr, block_ids, *, bk=128, d=128, g=8, max_m=128, policy="rbc",
               min_tasks=0, threads=None, cost=None) -> PlanSizes:
    """계획을 계산하고 크기만 돌려준다. descriptor 는 `fetch_into` 로 꺼낸다.

    같은 스레드에서 `fetch_into` 를 바로 호출해야 한다 (C++ 보관소가 thread_local).
    """
    lib = _load()
    if policy not in POLICIES:
        raise ValueError(f"알 수 없는 정책: {policy} ({list(POLICIES)})")
    cost = cost or {}
    threads = int(threads if threads is not None else os.environ.get("RBC_THREADS", 1))
    ptr = np.ascontiguousarray(ptr, dtype=np.int64)
    block_ids = np.ascontiguousarray(block_ids, dtype=np.int32)
    nq = len(ptr) - 1
    sz = lib._Sizes()
    t0 = time.perf_counter()
    rc = lib.rbc_plan(ptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                      block_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                      nq, bk, d, g, max_m, POLICIES[policy], min_tasks,
                      float(cost.get("byte_weight", 1.0)),
                      float(cost.get("flop_weight", 0.015)),
                      float(cost.get("task_weight", 0.0)),
                      float(cost.get("partial_weight", 1.0)),
                      threads, ctypes.byref(sz))
    if rc != 0:
        raise RuntimeError(f"rbc_plan 실패 rc={rc}")
    ms = (time.perf_counter() - t0) * 1e3
    return PlanSizes(sz, nq, dict(bk=bk, d=d, g=g, max_m=max_m, min_tasks=min_tasks,
                                  threads=threads, policy=policy), ms)


def fetch_into(addrs: dict) -> None:
    """`plan_sizes` 로 만든 descriptor 를 주어진 주소에 C++ 이 직접 쓴다.

    addrs: {"task_qptr": int 주소, ..., "task_kmask": int 주소} — 9개 전부 필요하다.
    호출자가 각 구간이 `PlanSizes.lengths()` 만큼 담을 수 있음을 보장해야 한다.
    """
    lib = _load()
    P = ctypes.POINTER(ctypes.c_int32)
    P64 = ctypes.POINTER(ctypes.c_uint64)
    need = ("task_qptr", "task_q", "task_kptr", "task_k", "red_ptr", "red_slot",
            "task_kmask", "slot_mode", "multi_rows")
    missing = [n for n in need if n not in addrs]
    if missing:
        raise ValueError(f"주소가 빠졌다: {missing}")
    c = {n: ctypes.cast(ctypes.c_void_p(int(addrs[n])), P64 if n == "task_kmask" else P)
         for n in need}
    rc = lib.rbc_fetch(c["task_qptr"], c["task_q"], c["task_kptr"], c["task_k"],
                       c["red_ptr"], c["red_slot"], c["task_kmask"], c["slot_mode"],
                       c["multi_rows"])
    if rc != 0:
        raise RuntimeError(f"rbc_fetch 실패 rc={rc}")


def max_threads() -> int:
    return int(_load().rbc_max_threads())


# --------------------------------------------------------------------------
# 정확성 검사
# --------------------------------------------------------------------------
def verify_exact_partition(plan: Plan, ptr: np.ndarray, block_ids: np.ndarray):
    """계획이 원래 edge 를 중복 없이 정확히 덮는지 검사한다.

    반환 (ok, 설명). 병합 직사각형 때문에 task 는 원래 없던 (q,b) 짝을 포함할 수 있다.
    그 잉여는 커널에서 mask 로 제외되므로 여기서는 '덮개(cover)' 와 '중복' 을 나눠 본다.
    """
    nq = len(ptr) - 1
    true_edges = set()
    for i in range(nq):
        for b in block_ids[ptr[i]:ptr[i + 1]]:
            true_edges.add((i, int(b)))

    covered, dup = set(), []
    for t in range(plan.n_tasks):
        qs = plan.task_q[plan.task_qptr[t]:plan.task_qptr[t + 1]]
        ks = plan.task_k[plan.task_kptr[t]:plan.task_kptr[t + 1]]
        for q in qs:
            for b in ks:
                e = (int(q), int(b))
                if e in true_edges:
                    if e in covered:
                        dup.append(e)
                    covered.add(e)

    missing = true_edges - covered
    if missing:
        return False, f"누락된 edge {len(missing)}개 (예: {sorted(missing)[:3]})"
    if dup:
        return False, f"중복 배정 edge {len(dup)}개 (예: {dup[:3]})"

    # v0.2/P2: 출력 소유권. degree 를 다시 세어 slot_mode·multi_rows·red CSR 을 검사한다.
    deg = {}
    for si, qq in enumerate(plan.task_q):
        deg.setdefault(int(qq), []).append(si)
    exp_direct = sum(1 for v in deg.values() if len(v) == 1)
    exp_multi = sorted(q for q, v in deg.items() if len(v) >= 2)
    exp_empty = nq - len(deg)
    if plan.direct_rows != exp_direct:
        return False, f"direct_rows {plan.direct_rows} != 기대 {exp_direct}"
    if plan.empty_rows != exp_empty:
        return False, f"empty_rows {plan.empty_rows} != 기대 {exp_empty}"
    if sorted(int(x) for x in plan.multi_rows) != exp_multi:
        return False, "multi_rows 불일치"
    for q, slots in deg.items():
        want_mode = 1 if len(slots) == 1 else 0
        for si in slots:
            if int(plan.slot_mode[si]) != want_mode:
                return False, (f"슬롯 {si}(query {q}, degree {len(slots)}) 의 "
                               f"slot_mode={plan.slot_mode[si]} != {want_mode}")
    for r, q in enumerate(plan.multi_rows):
        got = set(int(x) for x in plan.red_slot[plan.red_ptr[r]:plan.red_ptr[r + 1]])
        if got != set(deg[int(q)]):
            return False, f"multi row {q} 의 reduction 슬롯 불일치"
    # direct 슬롯은 reduction CSR 에 들어가면 안 된다
    direct_slots = {v[0] for v in deg.values() if len(v) == 1}
    if direct_slots & set(int(x) for x in plan.red_slot):
        return False, "direct 슬롯이 reduction CSR 에 포함됐다"

    # kmask: 비트가 선 슬롯만 실제 edge 여야 한다
    for t in range(plan.n_tasks):
        qs = plan.task_q[plan.task_qptr[t]:plan.task_qptr[t + 1]]
        k0, k1 = plan.task_kptr[t], plan.task_kptr[t + 1]
        for j in range(k0, k1):
            b = int(plan.task_k[j])
            m = int(plan.task_kmask[j])
            for si, qq in enumerate(qs):
                bit = (m >> si) & 1
                real = (int(qq), b) in true_edges
                if bool(bit) != real:
                    return False, (f"task {t} block {b} 슬롯 {si}(query {qq}) 의 "
                                   f"kmask={bit} 인데 실제 edge={real}")
    return True, "ok"
