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
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.neo_scheduler import (
        NeoScheduler,
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
    def from_id(cls, rid_str: str) -> "_NeoRequestView":
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
        self.last_neo_output: "NeoSchedulerOutput | None" = None
        # add_request → finish_requests 사이에 같은 view 객체를 재사용
        # 하기 위한 cache. 매 finish 마다 hash 재계산을 피한다 + int-pool
        # 인덱스 stability 도 보장.
        self._neo_view_cache: dict[str, _NeoRequestView] = {}

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
    def schedule(self) -> "SchedulerOutput":
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
            except (AttributeError, TypeError) as e:
                logger.debug("NEO output attach failed: %s", e)

        # Swap lists rarely populated (only when KV exclusive ownership
        # is active — TSK_015). Skip attach when empty.
        swap_in = self.last_neo_output.swap_in_reqs
        swap_out = self.last_neo_output.swap_out_reqs
        if swap_in:
            output.neo_swap_in_req_ids = [r._str_id for r in swap_in]
        if swap_out:
            output.neo_swap_out_req_ids = [r._str_id for r in swap_out]

        return output
