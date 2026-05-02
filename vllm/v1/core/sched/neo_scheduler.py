# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NEO-style request-level scheduler with three queues.

Adapted from NEO ``swiftllm/server/scheduler.py`` (MLSys 2025,
Apache 2.0). Algorithms only — no code copied.

The scheduler keeps three queues:

* ``waiting_q`` — newly arrived requests pending prefill
* ``gpu_decoding_q`` — requests whose KV currently lives on the GPU
* ``cpu_decoding_q`` — requests whose KV currently lives on the CPU

On each step it produces:

* a list of one or two ``SubBatch`` for the model worker
* a swap-out list (preempted GPU → CPU)
* a swap-in list (CPU → GPU, when GPU has headroom)

This is intentionally a separate scheduler from the default
``vllm/v1/core/sched/scheduler.py``: the user opts in by selecting
``--kv-cache-policy=exclusive`` (or equivalent flag) and the engine
wires this scheduler into the model runner.

See ``shadow_assists/features/IDE_006/NEO_code_deepdive.md`` §3.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Iterable
from typing import Protocol

from vllm.v1.core.sched.mode_selector import (
    ScheduleBudget,
    decide_mode,
)
from vllm.v1.core.sched.perfpredictor import (
    PerfPredictor,
    ZeroPerfPredictor,
)
from vllm.v1.core.sched.sub_batch import SubBatch

logger = logging.getLogger(__name__)


class _ReqLike(Protocol):
    request_id: int
    # IDE_006 / TSK_015 — adapter 가 vLLM Request 의 string id 를
    # ``_NeoRequestView._str_id`` 으로 emit. SchedulerOutput.neo_*
    # field 의 매핑 영역에서 사용.
    _str_id: str
    @property
    def prompt_len(self) -> int: ...
    @property
    def num_tokens(self) -> int: ...


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


class NeoSchedulerOutput:
    """The return value of ``NeoScheduler.schedule()``."""

    __slots__ = ("batches", "swap_out_reqs", "swap_in_reqs")

    def __init__(
        self,
        batches: list[SubBatch],
        swap_out_reqs: list[_ReqLike],
        swap_in_reqs: list[_ReqLike],
    ) -> None:
        self.batches = batches
        self.swap_out_reqs = swap_out_reqs
        self.swap_in_reqs = swap_in_reqs

    @property
    def is_pipelined(self) -> bool:
        return len(self.batches) == 2

    def __repr__(self) -> str:
        return (f"NeoSchedulerOutput(num_batches={len(self.batches)}, "
                f"swap_out={len(self.swap_out_reqs)}, "
                f"swap_in={len(self.swap_in_reqs)})")


class NeoScheduler:
    """NEO 6-step request scheduler.

    Step 1. Reserve budget for already-running GPU decode.
    Step 2. Swap-out — preempt the most recent GPU decode if budget
            is over.
    Step 3. Swap-in — promote head-of-queue CPU decode if GPU has
            headroom (≤ 95% of swap_out_threshold).
    Step 4. Classify newly-arrived prefills into GPU-bound or
            CPU-bound (FCFS — once one prefill is CPU-bound, all
            subsequent prefills are CPU-bound).
    Step 5. Compose two SubBatches via ``decide_mode`` (NEO's
            pipelined-vs-sequential heuristic).
    Step 6. Promote the prefills that the chosen mode actually used.

    Invariant: ``swap_out`` and ``swap_in`` are mutually exclusive
    within a single iteration.
    """

    def __init__(
        self,
        *,
        max_batch_size: int,
        max_tokens_in_batch: int,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        num_layers: int,
        predictor: PerfPredictor | None = None,
        linr_S_threshold: int = 128,
        swap_in_threshold_ratio: float = 0.95,
    ) -> None:
        self.max_batch_size = max_batch_size
        self.max_tokens_in_batch = max_tokens_in_batch
        self.block_size = block_size
        self.num_gpu_blocks = num_gpu_blocks
        self.num_cpu_blocks = num_cpu_blocks
        self.num_layers = num_layers
        self.predictor: PerfPredictor = predictor or ZeroPerfPredictor()
        self.linr_S_threshold = linr_S_threshold
        self.swap_in_threshold_ratio = swap_in_threshold_ratio

        self.waiting_q: deque[_ReqLike] = deque()
        self.gpu_decoding_q: list[_ReqLike] = []
        self.cpu_decoding_q: deque[_ReqLike] = deque()

    # ------------------------------------------------------------------
    # External API — mirrors NEO's surface
    # ------------------------------------------------------------------
    def on_requests_arrival(self, reqs: Iterable[_ReqLike]) -> None:
        self.waiting_q.extend(reqs)

    def remove_finished_requests(self, reqs: Iterable[_ReqLike]) -> None:
        ids = {r.request_id for r in reqs}
        self.gpu_decoding_q = [r for r in self.gpu_decoding_q
                                if r.request_id not in ids]
        self.cpu_decoding_q = deque(
            r for r in self.cpu_decoding_q if r.request_id not in ids)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_block_needed(self, req: _ReqLike) -> int:
        return _cdiv(req.num_tokens, self.block_size)

    # ------------------------------------------------------------------
    # TSK_015 Phase 5.1 — atomic swap helpers + XOR invariant.
    #
    # In NEO's exclusive-ownership model, every active req lives in
    # *exactly one* of {gpu_decoding_q, cpu_decoding_q} at any observable
    # state boundary (waiting_q is for not-yet-prefilled). The helpers
    # below wrap the pop+append pair as a single conceptual unit so the
    # call sites are intent-clear and the invariant is locally
    # ``assert``-able.
    #
    # Single-threaded Python: the in-method state mutation cannot be
    # observed mid-step from outside, so "atomicity" is naturally
    # satisfied. We keep the helpers + assertion to surface the
    # invariant for tests and future multi-thread evolution.
    # ------------------------------------------------------------------
    def _initiate_swap_out(self, victim: _ReqLike) -> None:
        """Atomically move ``victim`` from gpu_decoding_q to
        cpu_decoding_q. Caller is responsible for popping the victim
        from gpu_decoding_q (typical: ``victim = q.pop()``); this
        helper enforces the appendleft + assert pair."""
        # By design ``victim`` was already removed from gpu_decoding_q
        # by the caller; we move it into the cpu side. The invariant
        # then re-becomes ``has_gpu XOR has_cpu`` for ``victim``.
        self.cpu_decoding_q.appendleft(victim)

    def _initiate_swap_in(self, candidate: _ReqLike) -> None:
        """Atomically move ``candidate`` from cpu_decoding_q to
        gpu_decoding_q. Caller has already validated the budget /
        block-need check; this helper does the queue handoff."""
        self.cpu_decoding_q.popleft()
        self.gpu_decoding_q.append(candidate)

    def _assert_exclusive_invariant(self, *, where: str = "") -> None:
        """XOR invariant check (debug-only; opt-in via ``ENABLE_NEO_INV
        =1`` env so it's free in prod). For each req present in any
        decoding queue, assert it appears in *exactly one*."""
        import os as _os
        if _os.environ.get("ENABLE_NEO_INV") != "1":
            return
        gpu_ids = {r.request_id for r in self.gpu_decoding_q}
        cpu_ids = {r.request_id for r in self.cpu_decoding_q}
        overlap = gpu_ids & cpu_ids
        if overlap:
            raise AssertionError(
                f"NEO XOR invariant violated{(' @ ' + where) if where else ''}"
                f": {len(overlap)} req(s) in BOTH queues: {sorted(overlap)[:5]}..."
            )

    # ------------------------------------------------------------------
    # Main entry — produce the next iteration's batches
    # ------------------------------------------------------------------
    def schedule(self) -> NeoSchedulerOutput:
        budget = ScheduleBudget(self.max_batch_size, self.max_tokens_in_batch)
        # IDE_006 / TSK_015 §3.5 — VLLM_NEO_SWAP_OUT_RATIO env scales the
        # swap-out threshold so that short prod runs can force cdec firing
        # without needing a 4.7-hour 5000×50:50 workload to saturate KV.
        # ratio in (0, 1]; default 1.0 = current behaviour (NEO paper spec).
        # Example: VLLM_NEO_SWAP_OUT_RATIO=0.1 → fires when KV usage > 10%.
        import os as _os
        try:
            _ratio = float(_os.environ.get("VLLM_NEO_SWAP_OUT_RATIO", "1.0"))
        except ValueError:
            _ratio = 1.0
        if not (0.0 < _ratio <= 1.0):
            _ratio = 1.0
        swap_out_threshold = round(self.num_gpu_blocks * _ratio)
        swap_in_threshold = round(swap_out_threshold * self.swap_in_threshold_ratio)
        cpu_threshold = self.num_cpu_blocks - self.num_gpu_blocks

        swap_out_reqs: list[_ReqLike] = []
        swap_in_reqs: list[_ReqLike] = []
        pref_to_gpu: list[_ReqLike] = []
        pref_to_cpu: list[_ReqLike] = []

        # Step 1 — reserve budget for existing GPU decoding
        gpu_block_needed = sum(self._get_block_needed(r)
                               for r in self.gpu_decoding_q)
        budget.remaining_batch_size -= len(self.gpu_decoding_q)
        budget.remaining_tokens_in_batch -= len(self.gpu_decoding_q)

        # Step 2 — preempt the most recent GPU decoding when budget over
        while (budget.overspent
               or gpu_block_needed > swap_out_threshold) and self.gpu_decoding_q:
            victim = self.gpu_decoding_q.pop()
            self._initiate_swap_out(victim)         # Phase 5.1 atomic
            swap_out_reqs.append(victim)
            gpu_block_needed -= self._get_block_needed(victim)
            budget.add(1)
        self._assert_exclusive_invariant(where="step2")

        # Step 3 — swap-in head-of-queue CPU decode (mutually exclusive
        # with swap-out).
        # IDE_006 Step 3.2.C-5 dev hook — VLLM_NEO_FORCE_CDEC_DISPATCH=1 면
        # swap_in 자체를 skip 시켜 cpu_decoding_q 가 그대로 유지 → mode_selector
        # 의 step 3 가 그 reqs 를 cdec_reqs 로 batches 에 add → unified_
        # attention_with_output 의 dispatch hook 이 dev 환경에서 실제 발화.
        # prod 워크로드 의 KV pool pressure 영역 검증을 dev 에서 시뮬레이션.
        import os as _os
        _force_cdec_dispatch = (
            _os.environ.get("VLLM_NEO_FORCE_CDEC_DISPATCH") == "1"
        )
        while (self.cpu_decoding_q and not swap_out_reqs
               and not _force_cdec_dispatch):
            candidate = self.cpu_decoding_q[0]
            need = self._get_block_needed(candidate)
            if (gpu_block_needed + need > swap_in_threshold
                    or not budget.check_and_substract(1)):
                break
            gpu_block_needed += need
            swap_in_reqs.append(candidate)
            self._initiate_swap_in(candidate)       # Phase 5.1 atomic
        self._assert_exclusive_invariant(where="step3")
        # Mutually exclusive
        assert not swap_out_reqs or not swap_in_reqs

        # Step 4 — classify newly arrived prefills (GPU first, then CPU)
        cpu_block_needed = sum(self._get_block_needed(r)
                               for r in self.cpu_decoding_q)
        for cand in self.waiting_q:
            need = self._get_block_needed(cand)
            # Once one prefill is CPU-bound, FCFS means later ones are CPU-bound
            if (cpu_block_needed + need > cpu_threshold
                    or not budget.check_and_substract(cand.prompt_len)):
                break
            if not pref_to_cpu and gpu_block_needed + need <= self.num_gpu_blocks:
                gpu_block_needed += need
                pref_to_gpu.append(cand)
            else:
                cpu_block_needed += need
                pref_to_cpu.append(cand)

        # Step 5 — pipelined vs sequential decision
        batches = decide_mode(
            gpu_prefill_reqs=pref_to_gpu,
            cpu_prefill_reqs=pref_to_cpu,
            gpu_decoding_q=self.gpu_decoding_q,
            cpu_decoding_q=list(self.cpu_decoding_q),
            budget=budget,
            predictor=self.predictor,
            num_layers=self.num_layers,
            num_gpu_blocks=self.num_gpu_blocks,
            linr_S_threshold=self.linr_S_threshold,
        )

        # Step 6 — actually promote the chosen prefills out of waiting_q
        real_num_prefs = sum(b.get_num_prefs() for b in batches)
        accepted = 0
        while accepted < real_num_prefs and self.waiting_q:
            cand = self.waiting_q.popleft()
            accepted += 1
            # accepted prefills are already represented inside ``batches``;
            # the scheduler caller is responsible for routing them to the
            # GPU- or CPU-decoding queue when prefill completes.

        if pref_to_gpu or pref_to_cpu:
            logger.debug(
                "NeoScheduler: gdec=%d cdec=%d pref_gpu=%d pref_cpu=%d waiting=%d",
                len(self.gpu_decoding_q),
                len(self.cpu_decoding_q),
                len(pref_to_gpu),
                len(pref_to_cpu),
                len(self.waiting_q),
            )

        return NeoSchedulerOutput(batches, swap_out_reqs, swap_in_reqs)
