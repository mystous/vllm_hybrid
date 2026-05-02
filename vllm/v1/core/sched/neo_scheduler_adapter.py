# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SchedulerInterface adapter for the NEO-style asymmetric scheduler.

The full NEO design (TSK_014) returns *two* sub-batches per step,
which is incompatible with vLLM's ``SchedulerOutput`` (single batch).
A faithful integration requires changes throughout ``EngineCore.step``
and ``GPUModelRunner.execute_model``.

This adapter is the *first wiring stage*: it inherits the vLLM
default ``Scheduler`` so all ``SchedulerInterface`` methods continue
to work (default behaviour). On top of it, every ``schedule()`` call
also drives a sibling ``NeoScheduler`` instance, recording — but not
yet acting on — its mode-selection decisions in
``self.last_neo_output``. Subsequent stages (Llama forward stage
split + GPU runner sub-batch hook) will switch the data path to
those decisions.

Activated by ``--enable-neo-asymmetric`` (i.e.
``SchedulerConfig.enable_neo_asymmetric=True``). When the flag is
off, ``vllm/config/scheduler.py:get_scheduler_cls`` returns the
default scheduler and this adapter is never imported.

See ``shadow_assists/features/IDE_006/NEO_redesign.md``,
``TSK_014`` for the full design.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from vllm.v1.core.sched.scheduler import Scheduler

if TYPE_CHECKING:
    from vllm.v1.core.sched.neo_scheduler import (
        NeoSchedulerOutput,
    )
    from vllm.v1.core.sched.output import SchedulerOutput

logger = logging.getLogger(__name__)


class _NeoRequestView:
    """Lightweight adapter so that vLLM's ``Request`` satisfies
    NeoScheduler's ``_ReqLike`` protocol.

    NEO uses ``request_id`` as a numeric pool index in
    ``NeoBlockManager``; vLLM uses an arbitrary string. This view
    hashes the string to a stable integer slot and exposes the token
    counters NEO's mode-selection needs."""

    __slots__ = ("_req", "request_id", "prompt_len", "_str_id")

    def __init__(self, request) -> None:
        self._req = request
        # vLLM Request id is a string; NEO indexes are integers. We
        # hash the string into a 31-bit slot — collisions inside a
        # single inference run are vanishingly unlikely.
        rid_str = getattr(request, "request_id", str(id(request)))
        self._str_id = rid_str
        self.request_id = abs(hash(rid_str)) & 0x7FFFFFFF
        self.prompt_len = int(getattr(request, "num_prompt_tokens", 0)) or 1

    @classmethod
    def from_id(cls, rid_str: str) -> _NeoRequestView:
        """Build a stub view from just an id (for finish_requests)."""
        stub = object.__new__(cls)
        stub._req = None
        stub._str_id = rid_str
        stub.request_id = abs(hash(rid_str)) & 0x7FFFFFFF
        stub.prompt_len = 1
        return stub

    @property
    def num_tokens(self) -> int:
        if self._req is None:
            return 1
        try:
            return int(self._req.num_tokens)
        except (AttributeError, TypeError):
            return self.prompt_len


class NeoSchedulerAdapter(Scheduler):
    """First-stage NEO wrapper around the default vLLM scheduler.

    Behaviour
    ---------
    * All ``SchedulerInterface`` methods inherited from ``Scheduler``
      retain their default semantics — vanilla path is unchanged.
    * On each ``schedule()`` call, a sibling ``NeoScheduler`` is
      driven with the same arrival/finish events (mirrored from the
      default scheduler's queues) and its decision is recorded in
      ``self.last_neo_output``.
    * Future stages will replace the default ``schedule()`` body
      with one that consumes ``last_neo_output`` and produces a
      ``SchedulerOutput`` representing two sub-batches.
    """

    def __init__(self, *args, **kwargs) -> None:  # noqa: D401
        super().__init__(*args, **kwargs)

        # Lazily build the NEO scheduler — needs config values that
        # are accessible via ``self.vllm_config`` after super().__init__.
        from vllm.v1.core.sched.neo_scheduler import NeoScheduler
        from vllm.v1.core.sched.perfpredictor import (
            TablePerfPredictor,
            ZeroPerfPredictor,
        )

        sched_cfg = self.vllm_config.scheduler_config
        cache_cfg = self.vllm_config.cache_config
        model_cfg = self.vllm_config.model_config

        # Until ModelProfiler runs, fall back to the zero predictor so
        # that mode selection always picks sequential (vanilla
        # behaviour). ``TablePerfPredictor`` is wired here for later
        # stages but its tables remain unfilled.
        self.predictor = ZeroPerfPredictor()
        try:
            self.table_predictor = TablePerfPredictor(self.vllm_config)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "NeoSchedulerAdapter: TablePerfPredictor init failed (%s)."
                " Falling back to ZeroPerfPredictor only.", e,
            )
            self.table_predictor = None

        # KV pool sizes are not finalised when the scheduler is built —
        # ``cache_cfg.num_gpu_blocks`` is set by a worker profile run
        # *after* this point. Until then we estimate from the scheduler
        # / model config so the NEO sibling has a realistic budget for
        # its mode-selection arithmetic. The estimate is intentionally
        # generous so swap-out doesn't fire on every iteration.
        block_size = max(getattr(cache_cfg, "block_size", 16), 1)
        max_model_len = max(getattr(model_cfg, "max_model_len", 0), 1)
        max_num_seqs = max(getattr(sched_cfg, "max_num_seqs", 1), 1)
        estimated_blocks = (max_num_seqs * max_model_len) // block_size + 1

        num_gpu_blocks = (
            getattr(cache_cfg, "num_gpu_blocks", None)
            or getattr(cache_cfg, "num_gpu_blocks_override", None)
            or estimated_blocks
        )
        num_gpu_blocks = max(int(num_gpu_blocks), 1)

        # CPU block count is determined by ``--swap-space`` × layer
        # count; until that's resolved, mirror the GPU estimate so the
        # cpu_threshold = num_cpu_blocks - num_gpu_blocks heuristic
        # doesn't go negative.
        num_cpu_blocks = max(int(num_gpu_blocks * 2), num_gpu_blocks + 1)

        try:
            num_layers = model_cfg.hf_config.num_hidden_layers
        except AttributeError:
            num_layers = 1

        self.neo_scheduler = NeoScheduler(
            max_batch_size=sched_cfg.max_num_seqs,
            max_tokens_in_batch=sched_cfg.max_num_batched_tokens,
            block_size=cache_cfg.block_size,
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            num_layers=num_layers,
            predictor=self.predictor,
        )
        self.last_neo_output: NeoSchedulerOutput | None = None
        # add_request → finish_requests 사이에 같은 view 객체를 재사용
        # 하기 위한 cache. 매 finish 마다 hash 재계산을 피한다 + int-pool
        # 인덱스 stability 도 보장.
        self._neo_view_cache: dict[str, _NeoRequestView] = {}

        # TSK_015 Phase 4.1 — CPU KV buffer 의 *데이터* 는 worker (
        # GPUModelRunner) 에 둠 — TP shard 별로 분산. Adapter (engine
        # 측) 는 *결정* 만 (swap_out_req_ids attach via SchedulerOutput).
        # 본 adapter 는 이제 buffer instance 를 hold 하지 않음.

        # Sibling skip 판정 — predictor 가 ZeroPerfPredictor 면 항상
        # sequential 결정. FORCE flag 도 비활성이면 sibling 작업 100%
        # 폐기 → schedule() 마다 호출 자체를 skip. TSK_017 PerfPredictor
        # 적재 후 predictor swap 시점에 본 flag 도 갱신 필요.
        from vllm.v1.core.sched.mode_selector import _FORCE_PIPELINED
        from vllm.v1.core.sched.perfpredictor import ZeroPerfPredictor
        self._neo_sibling_meaningful: bool = (
            _FORCE_PIPELINED
            or not isinstance(self.predictor, ZeroPerfPredictor)
        )

        # TSK_015 Phase 4.6 — FORCE-cdec dev flag. ``VLLM_NEO_FORCE_CDEC=1``
        # bypass Step 2 (budget overflow) and artificially move 1 req
        # gdec→cdec each schedule, so cdec_reqs path activates on smoke.
        # Cached at import (env var doesn't change at runtime).
        import os as _os
        self._neo_force_cdec: bool = (
            _os.environ.get("VLLM_NEO_FORCE_CDEC") == "1"
        )

        logger.info(
            "NeoSchedulerAdapter: enable_neo_asymmetric activated. "
            "First-stage wiring — vanilla data path retained, NEO "
            "decisions are recorded but not yet executed."
        )

    # ------------------------------------------------------------------
    # SchedulerInterface overrides — keep the NEO sibling in lock-step
    # with the default scheduler's request set so its mode-selection
    # has a non-empty workload to reason about.
    # ------------------------------------------------------------------
    def add_request(self, request) -> None:  # type: ignore[override]
        super().add_request(request)
        # Cache the view by its (string) request_id so finish_requests
        # can retrieve the *same* object — avoids re-hashing the rid
        # and keeps the int-pool index stable between add/finish.
        view = _NeoRequestView(request)
        self._neo_view_cache[view._str_id] = view
        self.neo_scheduler.on_requests_arrival([view])

    # ------------------------------------------------------------------
    # TSK_015 Phase 2 — NeoScheduler queue tracking. NEO 가 prefill→decode
    # transition 을 자체 추적할 hook 이 없으므로 (NEO upstream 는 자체
    # engine 의 이벤트로 추적), vLLM Scheduler 의 ``self.running`` 을
    # 매 schedule 후 mirror 해서 NEO sibling 이 보는 gpu_decoding_q 를
    # 채운다. exclusive policy (kv_cache_policy=exclusive) 가 활성된
    # 후에는 Phase 3/4 가 *진짜 swap* 을 추가; 본 Phase 는 *관찰* 만.
    # ------------------------------------------------------------------
    def _sync_neo_gpu_decoding_q(self) -> None:
        """Mirror vLLM ``self.running`` (decoding subset) into the NEO
        sibling's ``gpu_decoding_q``. Run after ``super().schedule()``.

        The sibling uses this to (a) reserve budget for active decoders
        in Step 1, (b) consider preempting under pressure in Step 2,
        (c) drive ``decide_mode`` 's ``_get_remains`` analysis.
        """
        cache = self._neo_view_cache
        new_gdec: list[_NeoRequestView] = []
        for req in self.running:
            # Decode phase = num_computed_tokens >= num_prompt_tokens.
            # During chunked prefill, num_computed_tokens may equal
            # num_prompt_tokens during the *last* chunk's iteration —
            # treat as decoding from the next step onwards (correct for
            # NEO's decode-budget reasoning).
            if req.num_computed_tokens < req.num_prompt_tokens:
                continue
            view = cache.get(req.request_id)
            if view is not None:
                new_gdec.append(view)
        # NEO 의 gpu_decoding_q 는 ``list`` (not deque) — 직접 교체.
        self.neo_scheduler.gpu_decoding_q = new_gdec

    # ------------------------------------------------------------------
    # NEO PerfPredictor — populate from worker-side profile (TSK_017
    # Step 1.6). Called by EngineCore once after warmup.
    # ------------------------------------------------------------------
    def populate_predictor_from_profile(self, profile_data: dict) -> None:
        """Replace ``ZeroPerfPredictor`` with measurements collected by
        ``GPUModelRunner.profile_neo_predictor`` on worker_0.

        ``profile_data`` is a dict with ``{linr,pref,gdec}_T_pairs`` of
        ``(S_or_N, ms)`` tuples plus ``lnch_T`` ms. We rebuild the
        ``TablePerfPredictor`` 's lists from these pairs (the sampling
        sub-set the worker measured) and atomically swap.
        """
        from vllm.v1.core.sched.perfpredictor import TablePerfPredictor

        if not profile_data:
            logger.warning(
                "NEO: profile_data empty — predictor stays ZeroPerfPredictor."
            )
            return

        try:
            # TSK_015 Phase 3.1 — sync NEO scheduler's num_gpu_blocks
            # with the *real* value (set during warmup). Adapter init
            # used an estimate before warmup; now cache_config.num_gpu_blocks
            # is finalized.
            real_blocks = getattr(
                self.vllm_config.cache_config, "num_gpu_blocks", None
            )
            if real_blocks and real_blocks > 0:
                old = self.neo_scheduler.num_gpu_blocks
                self.neo_scheduler.num_gpu_blocks = int(real_blocks)
                logger.info(
                    "NEO: num_gpu_blocks sync %d → %d (Phase 3.1 — Step 2 "
                    "swap-out threshold reflects real KV pool).",
                    old, real_blocks,
                )

            pred = TablePerfPredictor(self.vllm_config)
            # Replace S/T list pairs with the *measured* sub-set.
            linr_pairs = profile_data.get("linr_T_pairs", [])
            pref_pairs = profile_data.get("pref_T_pairs", [])
            gdec_pairs = profile_data.get("gdec_T_pairs", [])
            if linr_pairs:
                pred.linr_S_list = [p[0] for p in linr_pairs]
                pred.linr_T_list = [p[1] for p in linr_pairs]
                pred.linr_S_lb_idx = pred._get_lb_idx_list(pred.linr_S_list)
            if pref_pairs:
                pred.pref_S_list = [p[0] for p in pref_pairs]
                pred.pref_T_list = [p[1] for p in pref_pairs]
                pred.pref_S_lb_idx = pred._get_lb_idx_list(pred.pref_S_list)
            if gdec_pairs:
                pred.gdec_N_list = [p[0] for p in gdec_pairs]
                pred.gdec_T_list = [p[1] for p in gdec_pairs]
                pred.gdec_N_lb_idx = pred._get_lb_idx_list(pred.gdec_N_list)
            pred.lnch_T = float(profile_data.get("lnch_T", 0.8))

            # Atomic swap. NeoScheduler 도 같은 ref 를 사용하므로 한 번만.
            self.predictor = pred
            self.table_predictor = pred
            self.neo_scheduler.predictor = pred

            # Sibling 의 hot-path skip 영역 갱신 — 진짜 prediction table
            # 보유했으므로 더 이상 항상 sequential 만 결정 안 함.
            self._neo_sibling_meaningful = True

            logger.info(
                "NEO: PerfPredictor populated (linr=%d/pref=%d/gdec=%d points,"
                " lnch_T=%.2fms). NEO scheduler now makes real "
                "sequential vs pipelined decisions.",
                len(linr_pairs), len(pref_pairs), len(gdec_pairs),
                pred.lnch_T,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "NEO: populate_predictor_from_profile failed (%s) — "
                "predictor stays ZeroPerfPredictor.", e,
            )

    def finish_requests(self, request_ids, finished_status) -> None:  # type: ignore[override]
        super().finish_requests(request_ids, finished_status)
        if not request_ids:
            return
        # Pull the *cached* view first; only fall back to ``from_id``
        # (which re-hashes) if a request was never seen by add_request
        # (e.g. cancelled before scheduling).
        cache = self._neo_view_cache
        wrappers = [
            cache.pop(rid, None) or _NeoRequestView.from_id(rid)
            for rid in request_ids
        ]
        self.neo_scheduler.remove_finished_requests(wrappers)

    # ------------------------------------------------------------------
    # schedule — drive both the default and NEO schedulers, attach the
    # NEO decision to the SchedulerOutput for the runner to consume.
    # ------------------------------------------------------------------
    def schedule(self) -> SchedulerOutput:
        # Drive the default scheduler — this is the data path the engine
        # actually consumes for vanilla operation.
        output = super().schedule()

        # Sibling skip — NEO scheduler 가 *의미 있는 결정* 을 내릴 수
        # 없는 환경에서는 sibling 의 schedule() 호출 자체를 skip 해서
        # 매 step 의 SubBatch/BatchPerfData/ScheduleBudget 할당 + 6 단계
        # 알고리즘 통째로 회피. 의미 있는 결정 = pipelined return 가능
        # 한 환경 = (a) FORCE flag 활성 또는 (b) PerfPredictor 가
        # ZeroPerfPredictor 가 아닌 진짜 측정 table 보유 (TSK_017 이후).
        # 둘 다 아니면 항상 sequential mode → sibling 작업 100% 폐기.
        if not self._neo_sibling_meaningful:
            self.last_neo_output = None
            return output

        # TSK_015 Phase 2 — sync NEO sibling's gpu_decoding_q with the
        # *actual* set of decoding requests in vLLM's running list.
        # Without this, NEO's gpu_decoding_q never populates (its only
        # source is Step 3 swap-in from cpu_decoding_q — itself empty)
        # so Step 2 preempt + Step 3 swap-in are dead. With sync, NEO
        # sees the same decode workload vLLM does and can make
        # capacity-related decisions.
        self._sync_neo_gpu_decoding_q()

        # TSK_015 Phase 4.6 — FORCE-cdec dev hook. With ``VLLM_NEO_FORCE_CDEC=1``
        # we artificially move 1 decoding req from gpu_decoding_q to
        # cpu_decoding_q so NEO 's Step 3 / decide_mode / sub-batch
        # populating with cdec_reqs path actually fires on smoke. Real
        # production trigger is Step 2 budget overflow.
        # 동시에 ``_force_swap_out_reqs`` 에 누적해서 본 schedule 의
        # ``last_neo_output.swap_out_reqs`` 에 append — runner 의
        # `_neo_handle_kv_swap` 가 실제 GPU→CPU KV move 를 수행.
        self._force_swap_out_pending: list = []
        if self._neo_force_cdec and self.neo_scheduler.gpu_decoding_q:
            victim = self.neo_scheduler.gpu_decoding_q.pop()
            self.neo_scheduler.cpu_decoding_q.appendleft(victim)
            self._force_swap_out_pending.append(victim)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[NEO FORCE-CDEC] artificially moved req %s to "
                    "cpu_decoding_q (gdec→cdec) for path activation test",
                    victim._str_id,
                )

        # Drive the NEO sibling. ``try/except`` is intentionally *narrow*
        # — the sibling schedule itself runs without try-overhead on the
        # hot path.
        n_wait_before = len(self.neo_scheduler.waiting_q)
        n_gpu_dec_before = len(self.neo_scheduler.gpu_decoding_q)
        n_cpu_dec_before = len(self.neo_scheduler.cpu_decoding_q)
        self.last_neo_output = self.neo_scheduler.schedule()
        # Per-iteration diagnostic — DEBUG only (no-op at default INFO
        # level, so zero hot-path cost). Re-enable with ``--log-level
        # DEBUG`` when investigating.
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[NEO] schedule(): pre-queues "
                "wait=%d gdec=%d cdec=%d → batches=%d swap_in=%d swap_out=%d",
                n_wait_before, n_gpu_dec_before, n_cpu_dec_before,
                len(self.last_neo_output.batches),
                len(self.last_neo_output.swap_in_reqs),
                len(self.last_neo_output.swap_out_reqs),
            )
            for i, b in enumerate(self.last_neo_output.batches):
                logger.debug(
                    "[NEO]   batch[%d]: gprf=%d cprf=%d gdec=%d cdec=%d total=%d",
                    i, b.num_gprfs, b.num_cprfs, b.num_gdecs, b.num_cdecs,
                    len(b),
                )

        # Attach NEO sub-batch decision to SchedulerOutput *only* when
        # the runner actually needs it. Sequential mode (1 sub-batch)
        # is functionally vanilla — the runner's NEO branch checks
        # ``len(pending) == 2``. Attaching a 200+ string list every step
        # for sequential mode just bloats IPC (pickle of SchedulerOutput
        # broadcast to all TP workers) without any benefit. Skip it.
        batches = self.last_neo_output.batches
        if len(batches) >= 2:
            try:
                output.neo_sub_batches = [
                    [r._str_id for r in batch.all_reqs]
                    for batch in batches
                ]
                # IDE_006 / TSK_015 4.5 / TSK_018 3.1 — per-sub-batch
                # cdec token row slice attach. Backend reads this via
                # ``CommonAttentionMetadata.neo_cdec_token_slice`` and
                # dispatches cdec rows to the CPU pacpu kernel. Each
                # tuple is half-open ``(start, end)`` within the
                # sub-batch's contiguous token tensor.
                output.neo_sub_batch_cdec_slices = [
                    batch.cdec_token_slice for batch in batches
                ]
                output.neo_sub_batch_cdec_seq_slices = [
                    batch.cdec_seq_slice for batch in batches
                ]
                output.neo_sub_batch_cdec_req_ids = [
                    [r._str_id for r in batch.cdec_reqs]
                    for batch in batches
                ]
            except (AttributeError, TypeError) as e:
                logger.debug("NEO output attach failed: %s", e)

        # Swap lists rarely populated (only when KV exclusive ownership
        # is active — TSK_015). Skip attach when empty.
        # FORCE-CDEC 의 swap_out 도 합산해서 runner 가 처리.
        swap_in = self.last_neo_output.swap_in_reqs
        swap_out = list(self.last_neo_output.swap_out_reqs)
        if self._force_swap_out_pending:
            swap_out.extend(self._force_swap_out_pending)
        # IDE_006 / TSK_015.B-3.a — finish ↔ swap_out mutex.
        # cdec_req 가 prefill 끝나는 step 에 vLLM finish_requests 가
        # 발화하면서 동시에 NEO swap_out 도 발화하면 EngineCore fatal.
        # finish 우선 — 같은 step 의 finished_req_ids 에 들어간 req 는
        # swap_out 에서 제거.
        finished_set = set(getattr(output, "finished_req_ids", ()) or ())
        if finished_set:
            swap_out = [r for r in swap_out if r._str_id not in finished_set]
            swap_in = [r for r in swap_in if r._str_id not in finished_set]
        if swap_in:
            output.neo_swap_in_req_ids = [r._str_id for r in swap_in]
        if swap_out:
            output.neo_swap_out_req_ids = [r._str_id for r in swap_out]

        return output
