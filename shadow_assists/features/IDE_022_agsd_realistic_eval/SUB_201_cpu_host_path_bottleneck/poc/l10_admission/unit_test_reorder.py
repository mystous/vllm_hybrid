"""[SUB_201/L10] Unit test: burst-aware reorder picks shortest job
when the FCFS waiting queue exceeds the trigger depth.

Doesn't require GPU; only constructs Request objects and runs the
reorder helper in isolation.
"""
from __future__ import annotations

import os
import sys
import time

# Ensure burst-aware on before importing scheduler module.
os.environ["VLLM_BURST_AWARE_ADMISSION"] = "1"
os.environ.setdefault("VLLM_BURST_TRIGGER_DEPTH", "4")
os.environ.setdefault("VLLM_BURST_HEAD_WINDOW", "16")
os.environ.setdefault("VLLM_BURST_AGE_CAP_S", "60.0")

from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm.v1.core.sched import scheduler as _sched
from vllm.v1.request import Request
from vllm.sampling_params import SamplingParams


def make_req(req_id: str, prompt_len: int, max_tokens: int, t0: float) -> Request:
    sp = SamplingParams(max_tokens=max_tokens)
    return Request(
        request_id=req_id,
        prompt_token_ids=[0] * prompt_len,
        sampling_params=sp,
        pooling_params=None,
        arrival_time=t0,
    )


class _StubScheduler:
    """Minimal scheduler-shaped stub to call the bound method."""

    _burst_aware_reorder_waiting = _sched.Scheduler._burst_aware_reorder_waiting

    def __init__(self, waiting: FCFSRequestQueue) -> None:
        self.waiting = waiting


def test_reorder_picks_shortest():
    now = time.time()
    wq = FCFSRequestQueue()
    # Long job first, then 3 short jobs.
    wq.append(make_req("R0_long", 512, 2048, now))
    wq.append(make_req("R1_short", 64, 32, now + 0.001))
    wq.append(make_req("R2_mid", 256, 256, now + 0.002))
    wq.append(make_req("R3_tiny", 8, 8, now + 0.003))

    head_before = [r.request_id for r in list(wq)]
    print("before:", head_before)

    stub = _StubScheduler(wq)
    stub._burst_aware_reorder_waiting()

    head_after = [r.request_id for r in list(wq)]
    print("after :", head_after)

    assert head_after[0] == "R3_tiny", f"expected R3_tiny at head, got {head_after[0]}"
    # rest preserves relative order
    assert head_after[1:] == ["R0_long", "R1_short", "R2_mid"], head_after
    print("PASS: shortest-job-first reorder")


def test_no_reorder_below_trigger_depth():
    now = time.time()
    wq = FCFSRequestQueue()
    wq.append(make_req("R0_long", 512, 2048, now))
    wq.append(make_req("R1_short", 64, 32, now + 0.001))
    head_before = [r.request_id for r in list(wq)]
    stub = _StubScheduler(wq)
    stub._burst_aware_reorder_waiting()
    head_after = [r.request_id for r in list(wq)]
    assert head_before == head_after, (head_before, head_after)
    print("PASS: below trigger depth -> no reorder")


def test_starvation_guard():
    # one ancient request in the window -> fallback to FCFS
    now = time.time()
    wq = FCFSRequestQueue()
    wq.append(make_req("R0_long_old", 512, 2048, now - 999.0))
    wq.append(make_req("R1_short", 64, 32, now))
    wq.append(make_req("R2_short", 64, 32, now))
    wq.append(make_req("R3_short", 64, 32, now))
    head_before = [r.request_id for r in list(wq)]

    # tighten age cap to force guard
    _sched._BURST_AGE_CAP_S_CACHED = 2.0

    stub = _StubScheduler(wq)
    stub._burst_aware_reorder_waiting()
    head_after = [r.request_id for r in list(wq)]
    assert head_before == head_after, (head_before, head_after)
    print("PASS: starvation guard fallback")

    # restore cap
    _sched._BURST_AGE_CAP_S_CACHED = 60.0


if __name__ == "__main__":
    test_reorder_picks_shortest()
    test_no_reorder_below_trigger_depth()
    test_starvation_guard()
    print("ALL OK")
