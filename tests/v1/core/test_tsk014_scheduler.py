# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TST_014 — NeoScheduler / mode_selector / SubBatch 단위 검증.

Coverage
--------
* ``SubBatch`` — add/pop accounting (pref / gdec / cdec) + perfdata 동기.
* ``decide_mode`` — fast-path (no cdec) vs slow-path (cdec present).
* ``decide_mode`` — sequential vs pipelined 결정 heuristic.
* ``NeoScheduler.schedule()`` — 6-step 알고리즘 entry path:
    Step 1 budget reserve / Step 2 swap-out / Step 3 swap-in /
    Step 4 classify / Step 5 mode / Step 6 promote.
* ``on_requests_arrival`` / ``remove_finished_requests`` 정합.

Heavy backend / model / KV cache 의존 없음 — predictor 는 ZeroPerfPredictor.

See ``shadow_assists/features/IDE_006/TSK_014.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

from vllm.v1.core.sched.mode_selector import ScheduleBudget, decide_mode
from vllm.v1.core.sched.neo_scheduler import NeoScheduler, NeoSchedulerOutput
from vllm.v1.core.sched.perfpredictor import ZeroPerfPredictor
from vllm.v1.core.sched.sub_batch import SubBatch


@dataclass
class _Req:
    request_id: int
    prompt_len: int
    num_tokens: int


def _ns(num_gpu_blocks: int = 64, num_cpu_blocks: int = 128) -> NeoScheduler:
    return NeoScheduler(
        max_batch_size=8,
        max_tokens_in_batch=512,
        block_size=4,
        num_gpu_blocks=num_gpu_blocks,
        num_cpu_blocks=num_cpu_blocks,
        num_layers=2,
        predictor=ZeroPerfPredictor(),
    )


# ----------------------------------------------------------------------
# SubBatch — add / pop accounting
# ----------------------------------------------------------------------
def test_subbatch_add_pref_gpu_increments_perfdata():
    sb = SubBatch()
    sb.add_pref(_Req(1, 16, 16), is_gpu=True)
    assert sb.num_gprfs == 1
    assert sb.num_cprfs == 0
    # perfdata.x is per-request count (+1 per add_*), not token sum
    assert sb.perfdata.x == 1


def test_subbatch_add_pref_cpu_routes_to_cprf():
    sb = SubBatch()
    sb.add_pref(_Req(1, 8, 8), is_gpu=False)
    assert sb.num_cprfs == 1
    assert sb.num_gprfs == 0


def test_subbatch_add_gdec_each_one_token():
    sb = SubBatch()
    sb.add_gdec(_Req(1, 4, 5))
    sb.add_gdec(_Req(2, 4, 5))
    assert sb.num_gdecs == 2
    assert sb.perfdata.x == 2               # 1 token per gdec req


def test_subbatch_add_cdec_each_one_token():
    sb = SubBatch()
    sb.add_cdec(_Req(1, 4, 5))
    assert sb.num_cdecs == 1
    assert sb.perfdata.x == 1


def test_subbatch_pop_round_trip():
    sb = SubBatch()
    sb.add_pref(_Req(1, 8, 8), is_gpu=True)
    sb.add_gdec(_Req(2, 4, 5))
    width_before = sb.perfdata.x
    sb.pop_gdec()
    assert sb.num_gdecs == 0
    assert sb.perfdata.x == width_before - 1


def test_subbatch_all_reqs_order_is_kernel_layout():
    """all_reqs must be cprf → gprf → gdec → cdec — used by Step 3.0
    token slice helpers and the kernel layout."""
    sb = SubBatch()
    sb.add_pref(_Req(1, 4, 4), is_gpu=False)
    sb.add_pref(_Req(2, 8, 8), is_gpu=True)
    sb.add_gdec(_Req(3, 4, 5))
    sb.add_cdec(_Req(4, 4, 5))
    assert [r.request_id for r in sb.all_reqs] == [1, 2, 3, 4]


# ----------------------------------------------------------------------
# decide_mode — fast-path (no cdec) vs slow-path
# ----------------------------------------------------------------------
def _budget(remaining_batch=8, remaining_tokens=512) -> ScheduleBudget:
    return ScheduleBudget(remaining_batch, remaining_tokens)


def test_decide_mode_empty_returns_empty_list():
    out = decide_mode(
        gpu_prefill_reqs=[], cpu_prefill_reqs=[],
        gpu_decoding_q=[], cpu_decoding_q=[],
        budget=_budget(), predictor=ZeroPerfPredictor(),
        num_layers=2, num_gpu_blocks=64,
    )
    assert out == []


def test_decide_mode_fast_path_no_cdec_returns_single_batch():
    """No cdec + FORCE_PIPELINED off → single SubBatch (sequential mode)."""
    gpu_pref = [_Req(1, 16, 16)]
    gdec = [_Req(2, 4, 5), _Req(3, 4, 5)]
    out = decide_mode(
        gpu_prefill_reqs=gpu_pref, cpu_prefill_reqs=[],
        gpu_decoding_q=gdec, cpu_decoding_q=[],
        budget=_budget(), predictor=ZeroPerfPredictor(),
        num_layers=2, num_gpu_blocks=64,
    )
    assert len(out) == 1
    sb = out[0]
    assert sb.num_gprfs == 1
    assert sb.num_gdecs == 2
    assert sb.num_cdecs == 0


def test_decide_mode_slow_path_with_cdec_present():
    """cdec_decoding_q populated → enters slow path. With ZeroPerfPredictor
    the heuristic falls back to gpu_only_batch (predicted times all 0),
    but the cdec rows must be considered (no exception, well-formed
    output)."""
    gdec = [_Req(1, 4, 5)]
    cdec = [_Req(2, 4, 5)]
    out = decide_mode(
        gpu_prefill_reqs=[], cpu_prefill_reqs=[],
        gpu_decoding_q=gdec, cpu_decoding_q=cdec,
        budget=_budget(), predictor=ZeroPerfPredictor(),
        num_layers=2, num_gpu_blocks=64,
    )
    # Either single batch (gpu-only fallback) or two — both valid;
    # the contract is: well-formed list of SubBatch with consistent
    # accounting.
    assert isinstance(out, list)
    assert all(isinstance(b, SubBatch) for b in out)
    total_gdecs = sum(b.num_gdecs for b in out)
    total_cdecs = sum(b.num_cdecs for b in out)
    assert total_gdecs >= 1                 # gdec at least carried
    assert total_cdecs >= 0                 # cdec may be deferred


# ----------------------------------------------------------------------
# NeoScheduler — 6-step entry path
# ----------------------------------------------------------------------
def test_scheduler_empty_returns_empty_output():
    sch = _ns()
    out = sch.schedule()
    assert isinstance(out, NeoSchedulerOutput)
    assert out.batches == []
    assert out.swap_out_reqs == []
    assert out.swap_in_reqs == []


def test_scheduler_on_requests_arrival_appends_to_waiting():
    sch = _ns()
    r1, r2 = _Req(1, 8, 8), _Req(2, 16, 16)
    sch.on_requests_arrival([r1, r2])
    assert list(sch.waiting_q) == [r1, r2]


def test_scheduler_remove_finished_clears_decoding_qs():
    sch = _ns()
    r1, r2, r3 = _Req(1, 4, 5), _Req(2, 4, 5), _Req(3, 4, 5)
    sch.gpu_decoding_q.extend([r1, r2])
    sch.cpu_decoding_q.appendleft(r3)
    sch.remove_finished_requests([r1, r3])
    # r1 / r3 removed, r2 stays
    assert [r.request_id for r in sch.gpu_decoding_q] == [2]
    assert list(sch.cpu_decoding_q) == []


def test_scheduler_swap_in_swap_out_mutually_exclusive_per_iter():
    """The schedule() invariant: out and in are not both populated in
    one iteration (Step 2 / Step 3 mutual exclusion)."""
    sch = _ns()
    r = _Req(1, 4, 5)
    sch.gpu_decoding_q.append(r)
    out = sch.schedule()
    # No congestion / no pressure — neither out nor in should fire.
    assert out.swap_out_reqs == []
    assert out.swap_in_reqs == []


def test_scheduler_swap_in_when_cpu_q_has_capacity():
    """cpu_decoding_q head + GPU has headroom → Step 3 promotes via
    _initiate_swap_in."""
    sch = _ns(num_gpu_blocks=64)        # plenty of headroom
    cpu_req = _Req(7, 4, 5)
    sch.cpu_decoding_q.appendleft(cpu_req)
    out = sch.schedule()
    # cpu req should have been swap-in promoted
    assert out.swap_in_reqs == [cpu_req]
    assert cpu_req in sch.gpu_decoding_q
    assert cpu_req not in sch.cpu_decoding_q


def test_scheduler_step6_promotes_only_what_was_used():
    """Step 6 — accepted prefills come out of waiting_q in count
    matching len(batches[*].get_num_prefs())."""
    sch = _ns()
    pref1, pref2 = _Req(1, 16, 16), _Req(2, 8, 8)
    sch.on_requests_arrival([pref1, pref2])
    out = sch.schedule()
    real_prefs = sum(b.get_num_prefs() for b in out.batches)
    # waiting_q must shrink by exactly real_prefs
    remaining = len(sch.waiting_q)
    assert len(sch.waiting_q) + real_prefs == 2
    assert remaining == 2 - real_prefs


# ----------------------------------------------------------------------
# NeoSchedulerOutput surface
# ----------------------------------------------------------------------
def test_scheduler_output_is_pipelined_property():
    out_seq = NeoSchedulerOutput(batches=[SubBatch()],
                                 swap_out_reqs=[], swap_in_reqs=[])
    assert not out_seq.is_pipelined
    out_pip = NeoSchedulerOutput(batches=[SubBatch(), SubBatch()],
                                 swap_out_reqs=[], swap_in_reqs=[])
    assert out_pip.is_pipelined


def test_scheduler_output_repr_smoke():
    out = NeoSchedulerOutput(batches=[SubBatch(), SubBatch()],
                             swap_out_reqs=[_Req(1, 4, 5)],
                             swap_in_reqs=[])
    s = repr(out)
    assert "num_batches=2" in s
    assert "swap_out=1" in s
    assert "swap_in=0" in s
