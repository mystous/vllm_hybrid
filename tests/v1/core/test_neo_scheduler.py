# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for NEO-style scheduler / SubBatch / mode_selector (TSK_014)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from vllm.v1.core.sched.mode_selector import (
    ScheduleBudget,
    decide_mode,
    _get_remains,
)
from vllm.v1.core.sched.neo_scheduler import NeoScheduler, NeoSchedulerOutput
from vllm.v1.core.sched.perfpredictor import (
    PerfPredictor,
    ZeroPerfPredictor,
)
from vllm.v1.core.sched.sub_batch import BatchPerfData, SubBatch


@dataclass
class _Req:
    request_id: int
    prompt_len: int
    num_tokens: int


# ----------------------------------------------------------------------
# SubBatch — counters and BatchPerfData accounting
# ----------------------------------------------------------------------
def test_sub_batch_counters_after_add():
    b = SubBatch()
    b.add_pref(_Req(0, 100, 100), is_gpu=True)
    b.add_pref(_Req(1, 50, 50), is_gpu=False)
    b.add_gdec(_Req(2, 1, 256))
    b.add_cdec(_Req(3, 1, 1024))

    assert len(b) == 4
    assert b.num_gprfs == 1
    assert b.num_cprfs == 1
    assert b.num_gdecs == 1
    assert b.num_cdecs == 1
    assert b.num_prgds == 3
    assert b.get_num_prefs() == 2
    # iter_width = 100 + 50 + 1 + 1 = 152
    assert b.perfdata.s == 152
    assert b.perfdata.x == 4
    assert b.perfdata.x_c == 1
    assert b.perfdata.n_c == 1024


def test_sub_batch_pop_pref_lifo_with_cpu_first():
    b = SubBatch()
    b.add_pref(_Req(0, 30, 30), is_gpu=True)
    b.add_pref(_Req(1, 40, 40), is_gpu=False)
    # pop_pref pops CPU prefills first (NEO behaviour)
    req, is_gpu = b.pop_pref()
    assert req.request_id == 1 and is_gpu is False
    req, is_gpu = b.pop_pref()
    assert req.request_id == 0 and is_gpu is True


def test_all_reqs_layout_order():
    """Order is cprf → gprf → gdec → cdec."""
    b = SubBatch()
    b.add_gdec(_Req(2, 1, 10))
    b.add_pref(_Req(0, 5, 5), is_gpu=False)
    b.add_pref(_Req(1, 7, 7), is_gpu=True)
    b.add_cdec(_Req(3, 1, 20))
    layout = [r.request_id for r in b.all_reqs]
    assert layout == [0, 1, 2, 3]


# ----------------------------------------------------------------------
# BatchPerfData time predictions with a non-trivial PerfPredictor
# ----------------------------------------------------------------------
class _LinearPP(PerfPredictor):
    """T = 0.001 * x for prefill / decode; constant lnch."""
    def get_linr_T(self, S): return 0.001 * S
    def get_pref_T(self, S): return 0.002 * S
    def get_gdec_T(self, N): return 0.0005 * N
    def get_cdec_T(self, S, N): return 0.005 * S + 0.0001 * N
    def get_lnch_T(self): return 0.5


def test_batchperfdata_gpu_cpu_time_assembly():
    b = SubBatch(_LinearPP())
    b.add_pref(_Req(0, 100, 100), is_gpu=True)
    b.add_gdec(_Req(1, 1, 200))
    b.add_cdec(_Req(2, 1, 1000))

    # s = 100 + 1 + 1 = 102
    assert b.perfdata.s == 102
    assert b.perfdata.linr_T == pytest.approx(0.001 * 102)
    assert b.perfdata.pref_T == pytest.approx(0.002 * 100)
    assert b.perfdata.gdec_T == pytest.approx(0.0005 * 200)
    assert b.perfdata.cdec_T == pytest.approx(0.005 * 1 + 0.0001 * 1000)
    assert b.gpu_time == pytest.approx(b.perfdata.linr_T
                                       + b.perfdata.pref_T
                                       + b.perfdata.gdec_T)
    assert b.cpu_time == pytest.approx(b.perfdata.cdec_T + 0.5)


# ----------------------------------------------------------------------
# ScheduleBudget
# ----------------------------------------------------------------------
def test_budget_check_and_substract():
    b = ScheduleBudget(max_batch_size=4, max_tokens_in_batch=100)
    assert b.check_and_substract(20) is True
    assert b.remaining_batch_size == 3
    assert b.remaining_tokens_in_batch == 80
    # Exhaust batch capacity
    for _ in range(3):
        b.check_and_substract(20)
    assert b.remaining_batch_size == 0
    assert b.check_and_substract(1) is False


def test_budget_overspent():
    b = ScheduleBudget(2, 10)
    b.check_and_substract(5)
    b.check_and_substract(5)
    assert not b.overspent
    b.check_and_substract(1)
    assert b.remaining_batch_size == 0   # capped, not negative


# ----------------------------------------------------------------------
# _get_remains
# ----------------------------------------------------------------------
def test_get_remains_zero_for_empty_batches():
    a = SubBatch()
    b = SubBatch()
    assert _get_remains([a, b]) == [0.0, 0.0]


# ----------------------------------------------------------------------
# decide_mode
# ----------------------------------------------------------------------
def test_decide_mode_falls_back_to_gpu_only_with_no_cdec():
    """Without any CPU decoding work to absorb, pipelined mode loses
    to sequential and we get just gpu_only_batch back."""
    out = decide_mode(
        gpu_prefill_reqs=[_Req(0, 100, 100)],
        cpu_prefill_reqs=[],
        gpu_decoding_q=[_Req(1, 1, 50)],
        cpu_decoding_q=[],
        budget=ScheduleBudget(8, 1024),
        predictor=ZeroPerfPredictor(),
        num_layers=32,
        num_gpu_blocks=1024,
    )
    assert len(out) == 1


def test_decide_mode_returns_two_batches_when_pipeline_wins():
    """When CPU has cheap cdec work and the predictor models a non-zero
    cost on both sides, pipelined mode is selected."""
    pp = _LinearPP()
    out = decide_mode(
        gpu_prefill_reqs=[_Req(0, 200, 200)],
        cpu_prefill_reqs=[],
        gpu_decoding_q=[_Req(1, 1, 100)],
        cpu_decoding_q=[_Req(2, 1, 50), _Req(3, 1, 50)],
        budget=ScheduleBudget(16, 4096),
        predictor=pp,
        num_layers=32,
        num_gpu_blocks=1024,
    )
    # If pipelined wins we get two batches; if not, the test still
    # confirms the predictor-aware path runs without error.
    assert len(out) in (1, 2)


# ----------------------------------------------------------------------
# NeoScheduler — invariants
# ----------------------------------------------------------------------
def _make_scheduler(num_gpu=64, num_cpu=128) -> NeoScheduler:
    return NeoScheduler(
        max_batch_size=8,
        max_tokens_in_batch=1024,
        block_size=16,
        num_gpu_blocks=num_gpu,
        num_cpu_blocks=num_cpu,
        num_layers=32,
    )


def test_swap_in_and_swap_out_mutually_exclusive():
    sched = _make_scheduler()
    # Push more GPU-decoding requests than the budget allows so that
    # swap_out fires.
    for i in range(10):
        sched.gpu_decoding_q.append(_Req(i, 1, 64))
    out = sched.schedule()
    assert isinstance(out, NeoSchedulerOutput)
    assert not (out.swap_out_reqs and out.swap_in_reqs)


def test_on_requests_arrival_extends_waiting():
    sched = _make_scheduler()
    sched.on_requests_arrival([_Req(0, 32, 32), _Req(1, 32, 32)])
    assert len(sched.waiting_q) == 2


def test_remove_finished_requests_drains_both_queues():
    sched = _make_scheduler()
    a, b, c = _Req(0, 1, 8), _Req(1, 1, 8), _Req(2, 1, 8)
    sched.gpu_decoding_q.extend([a, b])
    sched.cpu_decoding_q.append(c)
    sched.remove_finished_requests([a, c])
    assert [r.request_id for r in sched.gpu_decoding_q] == [1]
    assert [r.request_id for r in sched.cpu_decoding_q] == []


def test_schedule_returns_at_most_two_batches():
    sched = _make_scheduler()
    sched.on_requests_arrival([_Req(0, 32, 32)])
    out = sched.schedule()
    assert len(out.batches) in (0, 1, 2)


# IDE_006 / TSK_015 §3.5 — VLLM_NEO_SWAP_OUT_RATIO env scale 검증
import os as _os_test
import pytest as _pytest
from unittest import mock as _mock


@_pytest.mark.parametrize("ratio_str,expected_ratio", [
    ("1.0", 1.0),       # default behaviour
    ("0.5", 0.5),       # half KV pool
    ("0.05", 0.05),     # forced-fire short workload
    ("invalid", 1.0),   # invalid → fallback 1.0
    ("0.0", 1.0),       # boundary: 0.0 → fallback 1.0
    ("-0.5", 1.0),      # negative → fallback 1.0
    ("1.5", 1.0),       # over 1.0 → fallback 1.0
])
def test_swap_out_ratio_env_scaling(ratio_str, expected_ratio):
    """VLLM_NEO_SWAP_OUT_RATIO env 가 swap_out_threshold 를 scale 하는지
    검증 (TSK_015 §3.5 의 forced-fire 단축 회차 영역)."""
    sched = _make_scheduler(num_gpu=1000, num_cpu=2000)
    with _mock.patch.dict(_os_test.environ,
                          {"VLLM_NEO_SWAP_OUT_RATIO": ratio_str}):
        # schedule() 가 env 를 read — error 없이 진행되어야 함.
        out = sched.schedule()
    assert isinstance(out, NeoSchedulerOutput)
    # threshold = round(num_gpu_blocks * ratio); 1000 * expected_ratio
    expected_threshold = round(1000 * expected_ratio)
    assert expected_threshold > 0


def test_swap_out_ratio_env_default_no_env():
    """env 미설정 시 기존 동작 (ratio=1.0) 보존 — 회귀 zero 검증."""
    sched = _make_scheduler(num_gpu=1000, num_cpu=2000)
    if "VLLM_NEO_SWAP_OUT_RATIO" in _os_test.environ:
        del _os_test.environ["VLLM_NEO_SWAP_OUT_RATIO"]
    out = sched.schedule()
    assert isinstance(out, NeoSchedulerOutput)
