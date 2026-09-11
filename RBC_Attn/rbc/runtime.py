"""descriptor 업로드와 CUDA workspace 수명 관리.

`fresh_wall_ms` 측정에 들어가는 비용이 여기 있다: pinned staging + H2D + 워크스페이스 확보.
설계 §8 의 계약대로 CPU plan / H2D / GPU 본체 / merge 를 분리해 잴 수 있게 한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class DevicePlan:
    task_qptr: torch.Tensor
    task_q: torch.Tensor
    task_kptr: torch.Tensor
    task_k: torch.Tensor
    task_kmask: torch.Tensor
    red_ptr: torch.Tensor
    red_slot: torch.Tensor
    n_tasks: int
    n_slots: int
    block_m: int
    max_qtiles: int
    h2d_ms: float = 0.0


def _next_pow2(x: int) -> int:
    v = 1
    while v < x:
        v <<= 1
    return v


def upload_plan(plan, device="cuda", stream=None, pinned_cache=None,
                block_m=None) -> DevicePlan:
    """계획 descriptor 를 GPU 로 올린다.

    block_m 은 task 당 query 위치 수의 최대값에 맞춘 2의 거듭제곱이다.
    상한을 주면 (예: 16) 큰 task 는 여러 CTA 로 쪼개 실행한다.
    """
    qs = plan.task_qptr[1:] - plan.task_qptr[:-1]
    max_q = int(qs.max()) if qs.size else 1
    bm = _next_pow2(max(1, max_q)) if block_m is None else int(block_m)
    max_qtiles = max(1, (max_q + bm - 1) // bm)

    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()

    def up(a, dtype):
        t = torch.from_numpy(np.ascontiguousarray(a, dtype=dtype))
        return t.pin_memory().to(device, non_blocking=True)

    d = DevicePlan(
        task_qptr=up(plan.task_qptr, np.int32),
        task_q=up(plan.task_q, np.int32),
        task_kptr=up(plan.task_kptr, np.int32),
        task_k=up(plan.task_k, np.int32),
        task_kmask=up(plan.task_kmask.astype(np.int64), np.int64),
        red_ptr=up(plan.red_ptr, np.int32),
        red_slot=up(plan.red_slot, np.int32),
        n_tasks=plan.n_tasks,
        n_slots=int(plan.task_qptr[-1]),
        block_m=bm,
        max_qtiles=max_qtiles,
    )
    ev1.record()
    torch.cuda.synchronize()
    d.h2d_ms = ev0.elapsed_time(ev1)
    return d


class Workspace:
    """partial/출력 버퍼 재사용. run-only 반복 측정에서 할당 비용을 뺀다."""

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
