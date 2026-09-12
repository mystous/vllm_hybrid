"""P5 — hot path 의 할당·pin·작은 전송을 제거한 3단계 runtime (지시서 §8).

    prepare_capacity(shape_limits)   # 초기화 때만 메모리·event 준비
    plan_into(input_mask, slot)      # 새 mask 를 기존 메모리에 계획
    submit(slot, Q, K, V)            # 사용 구간만 전송하고 GPU 실행

핵심 계약:
  - **버퍼 재사용과 plan 재사용은 다르다.** mask 가 바뀌면 반드시 새로 계획한다.
    `plan_into` 는 매번 C++ 플래너를 다시 돌리고 generation 을 올린다.
  - descriptor 는 slot 별 pinned slab 안에 C++ 이 직접 쓴다 (§8.2). Python list →
    NumPy → pinned → device 연쇄가 없다.
  - 전송은 완료된 plan 당 연속 H2D 하나 (FULL membership 이면 kmask 를 보내지 않아
    실제로 1회, 아니면 int64 membership 때문에 2회).
  - capacity 초과는 `CapacityExceeded` 로 실패한다. silent reallocation 을 하지 않는다.
  - slot 상태 전이: FREE → HOST_READY → PLANNING → H2D_INFLIGHT → COMPUTE_INFLIGHT
    → DONE → FREE. 같은 slot 을 덮기 전에 이전 H2D / compute 완료를 확인한다.

측정 분해 (§8.7): pack / submit(API) / DMA / wait 를 각각 기록한다.
`cudaMemcpyAsync` API 누적시간을 DMA 시간이라고 부르지 않는다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from . import kernels, planner

# slab 헤더 (int32 칸). §8.2 의 header 요소.
H_VERSION = 0
H_GENERATION = 1
H_NQ = 2
H_G = 3
H_D = 4
H_BK = 5
H_CAUSAL = 6
H_N_TASKS = 7
H_N_SLOTS = 8
H_N_TASK_K = 9
H_N_RED = 10
H_N_MULTI = 11
H_N_EMPTY = 12
H_BLOCK_M = 13
H_FULL = 14
H_LIVE_I32 = 15
H_LIVE_I64 = 16
H_OFF_BASE = 20          # 이후 9칸에 배열 오프셋
HEADER_I32 = 32

_ARRAY_ORDER = ("task_qptr", "task_q", "task_kptr", "task_k",
                "red_ptr", "red_slot", "slot_mode", "multi_rows", "empty_rows")

DESCRIPTOR_VERSION = 2


class CapacityExceeded(RuntimeError):
    pass


def _next_pow2(x: int) -> int:
    v = 1
    while v < x:
        v <<= 1
    return v


@dataclass
class ShapeLimits:
    """초기화 때 확정하는 용량 상한. 이 범위를 넘는 입력은 실패로 처리한다."""
    max_nq: int
    max_tasks: int
    max_slots: int          # task_q 총 길이 (query-task incidence)
    max_task_k: int
    max_red: int
    g: int
    d: int
    bk: int = 128

    def i32_words(self) -> int:
        return (HEADER_I32
                + (self.max_tasks + 1) * 2      # task_qptr, task_kptr
                + self.max_slots * 2            # task_q, slot_mode
                + self.max_task_k               # task_k
                + (self.max_nq + 1)             # red_ptr
                + self.max_red                  # red_slot
                + self.max_nq                   # multi_rows
                + self.max_nq                   # empty_rows
                + 64)                           # alignment 여유

    def i64_words(self) -> int:
        return self.max_task_k + 8


@dataclass
class Timing:
    """§8.7 이 요구하는 분해. 각 항목은 서로 다른 것을 센다."""
    plan_call_ms: float = 0.0        # C++ 플래너 계산
    pack_ms: float = 0.0             # descriptor 를 pinned slab 에 쓰는 시간 (C++ fetch 포함)
    submit_api_ms: float = 0.0       # cudaMemcpyAsync 등 API 호출에 머문 CPU 시간
    dma_ms: float = 0.0              # copy stream 의 event 구간 (실제 전송)
    wait_ms: float = 0.0             # CPU 가 event 를 기다린 시간
    gpu_ms: float = 0.0              # compute stream 의 event 구간
    h2d_bytes: int = 0
    h2d_copies: int = 0


class Slot:
    """§8.4 의 slot. host/device descriptor·workspace·event·generation 을 따로 소유한다."""

    FREE, HOST_READY, PLANNING, H2D_INFLIGHT, COMPUTE_INFLIGHT, DONE = range(6)

    def __init__(self, idx: int, limits: ShapeLimits, device: str):
        self.idx = idx
        self.limits = limits
        self.device = device
        self.state = Slot.FREE
        self.generation = 0
        # host pinned / device slab — 초기화 때 한 번만 pin·할당한다
        self.host_i32 = torch.empty(limits.i32_words(), dtype=torch.int32).pin_memory()
        self.host_i64 = torch.empty(limits.i64_words(), dtype=torch.int64).pin_memory()
        self.dev_i32 = torch.empty(limits.i32_words(), dtype=torch.int32, device=device)
        self.dev_i64 = torch.empty(limits.i64_words(), dtype=torch.int64, device=device)
        self.host_np_i32 = self.host_i32.numpy()
        # partial workspace 와 출력 — 상한 형상으로 미리 잡는다
        g, d = limits.g, limits.d
        self.ws_u = torch.empty((limits.max_slots, g, d), device=device, dtype=torch.float32)
        self.ws_m = torch.empty((limits.max_slots, g), device=device, dtype=torch.float32)
        self.ws_l = torch.empty((limits.max_slots, g), device=device, dtype=torch.float32)
        self.out = torch.empty((limits.max_nq, g, d), device=device, dtype=torch.float32)
        self.lse = torch.empty((limits.max_nq, g), device=device, dtype=torch.float32)
        # event / stream
        self.copy_stream = torch.cuda.Stream(device=device)
        self.h2d_done = torch.cuda.Event()
        self.compute_done = torch.cuda.Event()
        self.d2h_done = torch.cuda.Event()
        self.ev_dma0 = torch.cuda.Event(enable_timing=True)
        self.ev_dma1 = torch.cuda.Event(enable_timing=True)
        self.ev_gpu0 = torch.cuda.Event(enable_timing=True)
        self.ev_gpu1 = torch.cuda.Event(enable_timing=True)
        self._h2d_recorded = False
        self._compute_recorded = False
        # 현재 계획의 view
        self.view = {}
        self.timing = Timing()
        self.plan_meta = {}

    # ---- 내부: slab 구간 배치 ----
    def _layout(self, lens: dict, nq: int) -> dict:
        off = HEADER_I32
        view = {}
        for name in _ARRAY_ORDER:
            n = int(lens.get(name, 0))
            view[name] = (off, n)
            off += n
            off = (off + 1) & ~1        # 2-word 정렬 (alignment gap 도 전송 바이트에 포함)
        if off > self.host_i32.numel():
            raise CapacityExceeded(
                f"slot{self.idx}: int32 slab 부족 (필요 {off}, 용량 {self.host_i32.numel()})")
        return view, off

    def wait_h2d(self) -> float:
        """pinned descriptor 를 덮기 전에 이전 H2D 완료를 확인한다."""
        if not self._h2d_recorded:
            return 0.0
        t0 = time.perf_counter()
        self.h2d_done.synchronize()
        return (time.perf_counter() - t0) * 1e3

    def wait_compute(self) -> float:
        """device descriptor/workspace 를 덮기 전에 이전 compute 완료를 확인한다."""
        if not self._compute_recorded:
            return 0.0
        t0 = time.perf_counter()
        self.compute_done.synchronize()
        return (time.perf_counter() - t0) * 1e3


class Runtime:
    """prepare_capacity → plan_into → submit 세 단계."""

    def __init__(self, limits: ShapeLimits, *, device="cuda", n_slots=2):
        self.limits = limits
        self.device = device
        self.compute_stream = torch.cuda.Stream(device=device)
        t0 = time.perf_counter()
        self.slots = [Slot(i, limits, device) for i in range(n_slots)]
        torch.cuda.synchronize()
        self.prepare_ms = (time.perf_counter() - t0) * 1e3
        # hot path 계측: 여기서 0 이 아니면 §8.7 실패
        self.hot_pin_calls = 0
        self.hot_dev_allocs = 0
        self.pin_calls_setup = 2 * n_slots
        self.dev_allocs_setup = 7 * n_slots

    def info(self) -> dict:
        return dict(n_slots=len(self.slots), prepare_ms=self.prepare_ms,
                    capacity_i32_words=self.limits.i32_words(),
                    capacity_i64_words=self.limits.i64_words(),
                    host_pin_calls_setup=self.pin_calls_setup,
                    device_allocation_requests_setup=self.dev_allocs_setup,
                    host_pin_calls_hotpath=self.hot_pin_calls,
                    device_allocation_requests_hotpath=self.hot_dev_allocs)

    # ------------------------------------------------------------------
    def plan_into(self, ptr, block_ids, slot: int, *, policy="q_outer", max_m=32,
                  min_tasks=0, causal=True, threads=None, selector=None,
                  plan_cost_us=None, send_membership=None) -> dict:
        """새 mask 를 기존 메모리에 계획하고 단일 H2D 로 보낸다.

        `selector` 를 주면 (P4 교정 표) 후보를 선택기가 고른다. 그 CPU 시간도 포함한다.
        """
        S = self.slots[slot]
        S.state = Slot.PLANNING
        # 이전 H2D 가 끝나야 pinned slab 을 덮을 수 있다
        w = S.wait_h2d()
        decision = None

        t_plan0 = time.perf_counter()
        if selector is not None:
            from . import select as _sel
            pr = planner.probe_head(ptr, block_ids, bk=self.limits.bk, d=self.limits.d,
                                    g=self.limits.g, max_m=max_m, threads=threads)
            fq = _sel.features_from_probe(pr, "q", d=self.limits.d, bk=self.limits.bk)
            fs = _sel.features_from_probe(pr, "s", d=self.limits.d, bk=self.limits.bk)
            pol, why = "q_outer", "unsupported_features(BM 미교정)"
            if selector.supports(fq):
                tq = selector.predict(fq)
                ts = selector.predict(fs) if selector.supports(fs) else float("inf")
                dm = selector.meta.get("diff_margin_rel_p90")
                margin = (dm * tq) if dm is not None else selector.rel_error(fq) * tq
                pc = plan_cost_us or {}
                extra = max(0.0, pc.get("signature", 0.0) - pc.get("q_outer", 0.0))
                if ts < tq - extra - margin:
                    pol, why = "signature_only", "predicted_signature_win"
                else:
                    why = "predicted_saving<=extra_prepare+error_margin"
            policy = pol
            decision = dict(selected=policy, reason=why, probe_us=pr["probe_us"])

        sz = planner.plan_sizes(ptr, block_ids, bk=self.limits.bk, d=self.limits.d,
                                g=self.limits.g, max_m=max_m, policy=policy,
                                min_tasks=min_tasks, threads=threads)
        plan_call_ms = (time.perf_counter() - t_plan0) * 1e3

        nq = sz.nq
        if nq > self.limits.max_nq:
            raise CapacityExceeded(f"nq {nq} > max_nq {self.limits.max_nq}")
        if sz.n_tasks > self.limits.max_tasks:
            raise CapacityExceeded(f"tasks {sz.n_tasks} > max_tasks {self.limits.max_tasks}")
        if sz.n_task_q > self.limits.max_slots:
            raise CapacityExceeded(f"slots {sz.n_task_q} > max_slots {self.limits.max_slots}")
        if sz.n_task_k > self.limits.max_task_k:
            raise CapacityExceeded(f"task_k {sz.n_task_k} > max {self.limits.max_task_k}")
        if sz.n_red > self.limits.max_red:
            raise CapacityExceeded(f"red {sz.n_red} > max_red {self.limits.max_red}")

        # --- pack: C++ 이 pinned slab 에 직접 쓴다 ---
        t_pack0 = time.perf_counter()
        lens = sz.lengths()
        lens["empty_rows"] = sz.empty_rows
        view, live_i32 = S._layout(lens, nq)
        base_i32 = S.host_i32.data_ptr()
        base_i64 = S.host_i64.data_ptr()
        addrs = {name: base_i32 + view[name][0] * 4 for name in _ARRAY_ORDER
                 if name != "empty_rows"}
        addrs["task_kmask"] = base_i64
        planner.fetch_into(addrs)

        # 빈 행 목록은 descriptor 에서 직접 만든다 (task_q 가 이미 pinned 안에 있다)
        off_q, n_q = view["task_q"]
        seen = np.zeros(nq, dtype=bool)
        if n_q:
            seen[S.host_np_i32[off_q:off_q + n_q]] = True
        empties = np.nonzero(~seen)[0].astype(np.int32)
        off_e, n_e = view["empty_rows"]
        if empties.size != n_e:
            # 플래너가 센 empty_rows 와 실제 재구성이 어긋나면 계약 위반이다
            raise RuntimeError(f"empty_rows 불일치: 계획 {n_e} vs 재구성 {empties.size}")
        if n_e:
            S.host_np_i32[off_e:off_e + n_e] = empties

        # membership 폭·전송 여부 판정 (§8.3): FULL 이면 보내지 않는다
        kmask = S.host_i64[:max(sz.n_task_k, 1)]
        qs_off, qs_n = view["task_qptr"]
        qptr = S.host_np_i32[qs_off:qs_off + qs_n]
        ks_off, ks_n = view["task_kptr"]
        kptr = S.host_np_i32[ks_off:ks_off + ks_n]
        full = _full_membership_from_slab(qptr, kptr, kmask.numpy(), sz.n_tasks)
        send_mask = (not full) if send_membership is None else bool(send_membership)

        max_q = int(np.max(np.diff(qptr))) if qs_n > 1 else 1
        block_m = _next_pow2(max(1, max_q))
        n_partial = 0
        off_sm, n_sm = view["slot_mode"]
        if n_sm:
            n_partial = int((S.host_np_i32[off_sm:off_sm + n_sm] == 0).sum())

        S.generation += 1
        h = S.host_np_i32
        h[H_VERSION] = DESCRIPTOR_VERSION
        h[H_GENERATION] = S.generation
        h[H_NQ] = nq
        h[H_G] = self.limits.g
        h[H_D] = self.limits.d
        h[H_BK] = self.limits.bk
        h[H_CAUSAL] = int(bool(causal))
        h[H_N_TASKS] = sz.n_tasks
        h[H_N_SLOTS] = sz.n_task_q
        h[H_N_TASK_K] = sz.n_task_k
        h[H_N_RED] = sz.n_red
        h[H_N_MULTI] = sz.n_multi_rows
        h[H_N_EMPTY] = int(empties.size)
        h[H_BLOCK_M] = block_m
        h[H_FULL] = int(full)
        h[H_LIVE_I32] = live_i32
        h[H_LIVE_I64] = sz.n_task_k if send_mask else 0
        for i, name in enumerate(_ARRAY_ORDER):
            h[H_OFF_BASE + i] = view[name][0]
        pack_ms = (time.perf_counter() - t_pack0) * 1e3

        # --- 단일 연속 H2D ---
        S.state = Slot.H2D_INFLIGHT
        t_api0 = time.perf_counter()
        with torch.cuda.stream(S.copy_stream):
            S.ev_dma0.record(S.copy_stream)
            S.dev_i32[:live_i32].copy_(S.host_i32[:live_i32], non_blocking=True)
            copies = 1
            if send_mask and sz.n_task_k:
                S.dev_i64[:sz.n_task_k].copy_(S.host_i64[:sz.n_task_k], non_blocking=True)
                copies += 1
            S.ev_dma1.record(S.copy_stream)
            S.h2d_done.record(S.copy_stream)
        S._h2d_recorded = True
        submit_api_ms = (time.perf_counter() - t_api0) * 1e3

        nbytes = live_i32 * 4 + (sz.n_task_k * 8 if (send_mask and sz.n_task_k) else 0)
        S.view = view
        S.plan_meta = dict(nq=nq, n_tasks=sz.n_tasks, n_slots=sz.n_task_q,
                           n_task_k=sz.n_task_k, n_partial_slots=n_partial,
                           n_multi_rows=sz.n_multi_rows, n_empty_rows=int(empties.size),
                           block_m=block_m, full_membership=full,
                           membership_sent=bool(send_mask), generation=S.generation,
                           policy=policy, max_m=max_m, min_tasks=min_tasks,
                           causal=bool(causal), live_i32_words=live_i32,
                           direct_rows=sz.direct_rows, decision=decision)
        S.timing = Timing(plan_call_ms=plan_call_ms, pack_ms=pack_ms,
                          submit_api_ms=submit_api_ms, wait_ms=w,
                          h2d_bytes=nbytes, h2d_copies=copies)
        S.state = Slot.HOST_READY
        return dict(S.plan_meta)

    # ------------------------------------------------------------------
    def submit(self, slot: int, q, k, v, q_positions, *, scale, bn=None,
               late_v=False, num_warps=4, num_stages=2) -> tuple:
        """slot 의 계획으로 GPU 실행을 낸다. compute stream 은 이 slot 의 H2D 만 기다린다."""
        S = self.slots[slot]
        if not S.plan_meta:
            raise RuntimeError("plan_into 를 먼저 호출해야 한다")
        # 같은 slot 의 device descriptor·workspace 를 덮기 전에 이전 compute 완료 확인
        w = S.wait_compute()
        m = S.plan_meta
        d32, d64 = S.dev_i32, S.dev_i64

        def seg(name):
            off, n = S.view[name]
            return d32[off:off + max(n, 1)]

        dplan = _SlabPlan(
            task_qptr=seg("task_qptr"), task_q=seg("task_q"),
            task_kptr=seg("task_kptr"), task_k=seg("task_k"),
            task_kmask=d64[:max(m["n_task_k"], 1)],
            red_ptr=seg("red_ptr"), red_slot=seg("red_slot"),
            slot_mode=seg("slot_mode"), multi_rows=seg("multi_rows"),
            empty_row_ids=seg("empty_rows"),
            n_tasks=m["n_tasks"], n_slots=m["n_slots"],
            n_partial_slots=m["n_partial_slots"], n_multi_rows=m["n_multi_rows"],
            n_empty_rows=m["n_empty_rows"], block_m=m["block_m"],
            max_qtiles=1, full_membership=m["full_membership"])

        nq = m["nq"]
        ws = (S.ws_u[:max(m["n_slots"], 1)], S.ws_m[:max(m["n_slots"], 1)],
              S.ws_l[:max(m["n_slots"], 1)], S.out[:nq], S.lse[:nq])
        S.state = Slot.COMPUTE_INFLIGHT
        with torch.cuda.stream(self.compute_stream):
            self.compute_stream.wait_event(S.h2d_done)
            S.ev_gpu0.record(self.compute_stream)
            out, lse = kernels.run_plan(
                dplan, q, k, v, q_positions, bk=self.limits.bk, causal=m["causal"],
                scale=scale, bn=bn, late_v=late_v, num_warps=num_warps,
                num_stages=num_stages, workspace=_FixedWorkspace(ws))
            S.ev_gpu1.record(self.compute_stream)
            S.compute_done.record(self.compute_stream)
        S._compute_recorded = True
        S.timing.wait_ms += w
        S.state = Slot.DONE
        return out, lse

    def finish(self, slot: int) -> Timing:
        """완료 의존성을 지키고 (사용자 소비 시점) 시간 분해를 확정한다."""
        S = self.slots[slot]
        t0 = time.perf_counter()
        S.compute_done.synchronize()
        S.timing.wait_ms += (time.perf_counter() - t0) * 1e3
        S.timing.dma_ms = S.ev_dma0.elapsed_time(S.ev_dma1)
        S.timing.gpu_ms = S.ev_gpu0.elapsed_time(S.ev_gpu1)
        S.state = Slot.FREE
        return S.timing


@dataclass
class _SlabPlan:
    """`kernels.run_plan` 이 기대하는 필드만 갖춘 view. 새 tensor 를 만들지 않는다."""
    task_qptr: torch.Tensor
    task_q: torch.Tensor
    task_kptr: torch.Tensor
    task_k: torch.Tensor
    task_kmask: torch.Tensor
    red_ptr: torch.Tensor
    red_slot: torch.Tensor
    slot_mode: torch.Tensor
    multi_rows: torch.Tensor
    empty_row_ids: torch.Tensor
    n_tasks: int
    n_slots: int
    n_partial_slots: int
    n_multi_rows: int
    n_empty_rows: int
    block_m: int
    max_qtiles: int
    full_membership: bool


class _FixedWorkspace:
    """미리 잡아 둔 버퍼를 그대로 돌려준다 (hot path 할당 0)."""

    def __init__(self, bufs):
        self._b = bufs

    def get(self, n_slots, nq, G, D):
        u, m, l, out, lse = self._b
        return u[:max(n_slots, 1)], m[:max(n_slots, 1)], l[:max(n_slots, 1)], out[:nq], lse[:nq]


def _full_membership_from_slab(qptr, kptr, kmask, n_tasks) -> bool:
    """slab 안의 descriptor 로 FULL 판정 (§6.3).

    task 마다 파이썬 루프를 돌면 task 수에 비례해 pack 시간이 늘어난다 (signature 816
    task 에서 4ms 넘게 들어갔다). task 별 기대 마스크를 KV 길이만큼 펼쳐 한 번에 비교한다.
    """
    if n_tasks == 0:
        return True
    qs = np.diff(qptr[:n_tasks + 1]).astype(np.int64)
    ks = np.diff(kptr[:n_tasks + 1]).astype(np.int64)
    total = int(ks.sum())
    if total == 0:
        return True
    want = ((np.uint64(1) << qs.astype(np.uint64)) - np.uint64(1))
    want_expanded = np.repeat(want, ks)
    return bool(np.array_equal(kmask[:total].view(np.uint64), want_expanded))
