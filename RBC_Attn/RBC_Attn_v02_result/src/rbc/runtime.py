"""descriptor 업로드와 CUDA workspace 수명 관리.

v0.2 변경:
  P2  슬롯 출력 모드(direct/partial)와 multi-owner 행, 빈 행 목록을 함께 올린다.
      partial workspace 는 실제 partial 슬롯 수만큼만 필요하다.
  P3  FULL membership 판정을 계획에서 계산해 커널 variant 선택에 쓴다.
  P5  (1단계) 고정 용량 arena + 단일 연속 slab 전송. descriptor 를 하나의 host pinned
      버퍼에 담아 한 번의 H2D 로 보낸다. hot path 에서 pin·device 할당 요청을 없앤다.

`fresh_wall` 측정에 들어가는 비용이 여기 있으므로 pack / submit / wait 를 나눠 잰다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

# slab 안의 배열 순서 (헤더 뒤에 이 순서로 연속 배치)
_ARRAYS = ("task_qptr", "task_q", "task_kptr", "task_k",
           "red_ptr", "red_slot", "slot_mode", "multi_rows", "empty_rows")


def _next_pow2(x: int) -> int:
    v = 1
    while v < x:
        v <<= 1
    return v


def compute_full_membership(plan) -> bool:
    """모든 task 의 모든 KV block 이 그 task 의 모든 query 에 연결됐는지 (P3-c).

    FULL 이면 커널이 membership 비트 검사를 생략한다. causal·NK 경계·padding 은 계속 적용.
    """
    if plan.n_tasks == 0:
        return True
    qs = plan.task_qptr[1:] - plan.task_qptr[:-1]
    for t in range(plan.n_tasks):
        nq_t = int(qs[t])
        want = (1 << nq_t) - 1
        k0, k1 = int(plan.task_kptr[t]), int(plan.task_kptr[t + 1])
        if k1 > k0 and int(plan.task_kmask[k0:k1].min()) != want:
            return False
        if k1 > k0 and int(plan.task_kmask[k0:k1].max()) != want:
            return False
    return True


def empty_row_ids(plan, nq: int) -> np.ndarray:
    """degree==0 인 query 행 번호."""
    if plan.empty_rows == 0:
        return np.zeros(0, dtype=np.int32)
    seen = np.zeros(nq, dtype=bool)
    if plan.task_q.size:
        seen[plan.task_q] = True
    return np.nonzero(~seen)[0].astype(np.int32)


@dataclass
class DevicePlan:
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
    h2d_ms: float = 0.0
    pack_ms: float = 0.0
    h2d_bytes: int = 0
    h2d_copy_count: int = 0
    stats: dict = field(default_factory=dict)


class Arena:
    """고정 용량 pinned/device slab. hot path 에서 할당·pin 을 하지 않는다 (P5).

    용량이 부족하면 조용히 재할당하지 않고 `CapacityExceeded` 를 올린다.
    """

    class CapacityExceeded(RuntimeError):
        pass

    def __init__(self, capacity_i32=1 << 22, capacity_i64=1 << 20, device="cuda"):
        self.device = device
        self.cap_i32 = int(capacity_i32)
        self.cap_i64 = int(capacity_i64)
        self.host_i32 = torch.empty(self.cap_i32, dtype=torch.int32).pin_memory()
        self.host_i64 = torch.empty(self.cap_i64, dtype=torch.int64).pin_memory()
        self.dev_i32 = torch.empty(self.cap_i32, dtype=torch.int32, device=device)
        self.dev_i64 = torch.empty(self.cap_i64, dtype=torch.int64, device=device)
        self.pin_calls = 2          # 초기화 때만
        self.dev_allocs = 2
        self.hot_pin_calls = 0
        self.hot_dev_allocs = 0

    def info(self) -> dict:
        return dict(capacity_i32=self.cap_i32, capacity_i64=self.cap_i64,
                    host_pin_calls_hotpath=self.hot_pin_calls,
                    device_allocation_requests_hotpath=self.hot_dev_allocs)


def upload_plan(plan, device="cuda", block_m=None, arena=None, nq=None,
                full_membership=None) -> DevicePlan:
    """계획 descriptor 를 GPU 로 올린다.

    arena 를 주면 **하나의 연속 slab 으로 한 번만** H2D 한다 (P5).
    없으면 v0.1 방식(배열별 개별 전송)으로 동작해 대조에 쓸 수 있다.
    """
    import time
    qs = plan.task_qptr[1:] - plan.task_qptr[:-1]
    max_q = int(qs.max()) if qs.size else 1
    bm = _next_pow2(max(1, max_q)) if block_m is None else int(block_m)
    max_qtiles = max(1, (max_q + bm - 1) // bm)
    if nq is None:
        nq = int(plan.red_ptr.size and (plan.task_q.max() + 1) or 1)
    empties = empty_row_ids(plan, nq)
    full = compute_full_membership(plan) if full_membership is None else bool(full_membership)

    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)

    if arena is None:
        def up(a, dtype):
            t = torch.from_numpy(np.ascontiguousarray(a, dtype=dtype))
            return t.pin_memory().to(device, non_blocking=True)
        t0 = time.perf_counter()
        ev0.record()
        parts = dict(
            task_qptr=up(plan.task_qptr, np.int32), task_q=up(plan.task_q, np.int32),
            task_kptr=up(plan.task_kptr, np.int32), task_k=up(plan.task_k, np.int32),
            task_kmask=up(plan.task_kmask.astype(np.int64), np.int64),
            red_ptr=up(plan.red_ptr, np.int32), red_slot=up(plan.red_slot, np.int32),
            slot_mode=up(plan.slot_mode, np.int32), multi_rows=up(plan.multi_rows, np.int32),
            empty_row_ids=up(empties, np.int32))
        ev1.record(); torch.cuda.synchronize()
        pack_ms = 0.0
        h2d_ms = ev0.elapsed_time(ev1)
        nbytes = sum(int(v.numel()) * v.element_size() for v in parts.values())
        copies = len(parts)
        _ = time.perf_counter() - t0
    else:
        # --- P5: 단일 slab pack + 한 번의 H2D ---
        t0 = time.perf_counter()
        segs32, off = [], 0
        for name, arr in (("task_qptr", plan.task_qptr), ("task_q", plan.task_q),
                          ("task_kptr", plan.task_kptr), ("task_k", plan.task_k),
                          ("red_ptr", plan.red_ptr), ("red_slot", plan.red_slot),
                          ("slot_mode", plan.slot_mode), ("multi_rows", plan.multi_rows),
                          ("empty_row_ids", empties)):
            a = np.ascontiguousarray(arr, dtype=np.int32)
            n = a.size
            if off + n > arena.cap_i32:
                raise Arena.CapacityExceeded(
                    f"int32 slab 용량 초과: 필요 {off + n} > 용량 {arena.cap_i32}")
            if n:
                arena.host_i32[off:off + n].copy_(torch.from_numpy(a))
            segs32.append((name, off, n)); off += n
        live32 = off
        km = np.ascontiguousarray(plan.task_kmask, dtype=np.uint64).view(np.int64)
        if km.size > arena.cap_i64:
            raise Arena.CapacityExceeded(
                f"int64 slab 용량 초과: 필요 {km.size} > 용량 {arena.cap_i64}")
        if km.size:
            arena.host_i64[:km.size].copy_(torch.from_numpy(km))
        pack_ms = (time.perf_counter() - t0) * 1e3

        ev0.record()
        arena.dev_i32[:live32].copy_(arena.host_i32[:live32], non_blocking=True)
        if km.size:
            arena.dev_i64[:km.size].copy_(arena.host_i64[:km.size], non_blocking=True)
        ev1.record(); torch.cuda.synchronize()
        h2d_ms = ev0.elapsed_time(ev1)
        nbytes = live32 * 4 + km.size * 8
        copies = 1 + (1 if km.size else 0)
        parts = {name: arena.dev_i32[o:o + n] for name, o, n in segs32}
        parts["task_kmask"] = arena.dev_i64[:km.size]

    n_partial = int((plan.slot_mode == 0).sum()) if plan.slot_mode.size else 0
    return DevicePlan(
        task_qptr=parts["task_qptr"], task_q=parts["task_q"],
        task_kptr=parts["task_kptr"], task_k=parts["task_k"],
        task_kmask=parts["task_kmask"], red_ptr=parts["red_ptr"],
        red_slot=parts["red_slot"], slot_mode=parts["slot_mode"],
        multi_rows=parts["multi_rows"], empty_row_ids=parts["empty_row_ids"],
        n_tasks=plan.n_tasks, n_slots=int(plan.task_qptr[-1]) if plan.n_tasks else 0,
        n_partial_slots=n_partial, n_multi_rows=int(len(plan.multi_rows)),
        n_empty_rows=int(empties.size), block_m=bm, max_qtiles=max_qtiles,
        full_membership=full, h2d_ms=h2d_ms, pack_ms=pack_ms,
        h2d_bytes=int(nbytes), h2d_copy_count=int(copies))


class Workspace:
    """partial/출력 버퍼 재사용. run-only 반복에서 할당 비용을 뺀다."""

    def __init__(self, device="cuda"):
        self.device = device
        self._buf = {}

    def get(self, n_slots, nq, G, D):
        key = (n_slots, nq, G, D)
        if key not in self._buf:
            dev = self.device
            self._buf[key] = (
                torch.empty((n_slots, G, D), device=dev, dtype=torch.float32),
                torch.empty((n_slots, G), device=dev, dtype=torch.float32),
                torch.empty((n_slots, G), device=dev, dtype=torch.float32),
                torch.empty((nq, G, D), device=dev, dtype=torch.float32),
                torch.empty((nq, G), device=dev, dtype=torch.float32),
            )
        return self._buf[key]

    def clear(self):
        self._buf.clear()
