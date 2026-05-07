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
import time
from typing import TYPE_CHECKING

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler  # noqa: F401

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

    __slots__ = ("_req", "request_id", "prompt_len", "_str_id", "_is_decode")

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


class NeoSchedulerAdapter(AsyncScheduler):
    # IDE_006 / TSK_015 Phase B (2026-05-03 root cause fix) — AsyncScheduler
    # 상속으로 변경. config/scheduler.py:get_scheduler_cls 가
    # enable_neo_asymmetric=True 시 본 adapter 를 return — 본 fix 전에는
    # *Scheduler* 직접 상속이라 vLLM 의 async pipeline (schedule + forward
    # overlap) 영역이 통째로 우회됨 → step 수 2× → wall 2× regression.
    # AsyncScheduler 의 _update_after_schedule / _update_request_with_output
    # 자동 상속으로 NEO ON 시도 vanilla 의 async pipeline 정합.
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
        """Mirror vLLM ``self.running`` (all active reqs) into the NEO
        sibling's ``gpu_decoding_q``. Run after ``super().schedule()``.

        The sibling uses this to (a) reserve budget for active reqs
        in Step 1, (b) consider preempting under pressure in Step 2,
        (c) drive ``decide_mode`` 's ``_get_remains`` analysis.

        2026-05-03 fix — *prefill skip 제거*. 이전 버전은 decode 단계
        (num_computed_tokens >= num_prompt_tokens) reqs 만 매핑 →
        prefill 단계 KV pressure 가 NEO 측에 추적 안 됨 →
        ``gpu_block_needed > swap_out_threshold`` 영역 미진입 →
        cdec dispatch 자연 발화 zero. 본 fix 로 prefill + decode
        모든 active reqs 매핑 → 진짜 KV pressure 인식 → swap_out
        결정 path 정상 활성.
        """
        from vllm.v1.request import RequestStatus as _RS
        cache = self._neo_view_cache
        new_gdec: list[_NeoRequestView] = []
        for req in self.running:
            # IDE_006 — NEO 가 이미 swap_out 한 SWAPPED_OUT req 는 *gpu_q
            # 매핑 제외*. 본 영역 누락 시 매 step *같은 req 가 또
            # swap_out 후보* → cpu_decoding_q 에 *중복 무한 누적* →
            # schedule() line 283 의 sum() 영역 quadratic blow → deadlock.
            if req.status == _RS.SWAPPED_OUT:
                continue
            view = cache.get(req.request_id)
            if view is not None:
                # prefill 단계 req 는 swap_out 후보 제외 (decode-only fix).
                view._is_decode = (
                    req.num_computed_tokens >= req.num_prompt_tokens
                )
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
            # TSK_019 SUB_016 — cdec_T 2D table (S × N) populate.
            cdec_pairs = profile_data.get("cdec_T_pairs", [])
            if cdec_pairs:
                cdec_grouped: dict[int, list[tuple[int, float]]] = {}
                for entry in cdec_pairs:
                    S, N, T = entry  # tuple/list both work
                    cdec_grouped.setdefault(int(S), []).append(
                        (int(N), float(T))
                    )
                sorted_S = sorted(cdec_grouped.keys())
                pred.cdec_S_list = sorted_S
                pred.cdec_N_lists = []
                pred.cdec_T_lists = []
                for S in sorted_S:
                    pairs = sorted(cdec_grouped[S], key=lambda p: p[0])
                    pred.cdec_N_lists.append([p[0] for p in pairs])
                    pred.cdec_T_lists.append([p[1] for p in pairs])
                pred.cdec_N_list_agg = sorted({
                    n for lst in pred.cdec_N_lists for n in lst
                })
                pred.cdec_S_lb_idx = pred._get_lb_idx_list(pred.cdec_S_list)
                pred.cdec_N_lb_idx = pred._get_lb_idx_list(
                    pred.cdec_N_list_agg
                )
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

    def _neo_swap_out(self, request, timestamp=None):  # type: ignore[override]
        """IDE_006 / TSK_015 4.5.2.c architectural fix —
        ``_preempt_request`` 의 NEO hook 이 호출 시 vLLM standard 의
        ``_neo_swap_out`` (KV CPU copy + status SWAPPED_OUT + RUNNING
        잔류) 후 NEO scheduler 의 ``cpu_decoding_q`` 에 view 추가.
        mode_selector 가 그것을 cdec_reqs 로 인식 → adapter 가 fork
        sub_batches attach → worker fork branch 진입 + cdec dispatch
        hook 발화.

        13 단계 progressive 우회 fix 의 *진정한 대체*: NEO 의 cdec
        queue 가 vLLM 의 standard preempt path 와 *single source*
        로 통합.
        """
        # vLLM standard _neo_swap_out (KV free + status + RUNNING 잔류)
        super()._neo_swap_out(request, timestamp)
        # NEO scheduler 의 cdec queue 에 view 추가 (mode_selector 가
        # cdec_reqs 로 활용).
        rid = request.request_id
        view = self._neo_view_cache.get(rid)
        if view is not None:
            # 이미 cpu_decoding_q 에 있으면 중복 추가 회피.
            if not any(r._str_id == rid
                       for r in self.neo_scheduler.cpu_decoding_q):
                self.neo_scheduler.cpu_decoding_q.appendleft(view)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "[NEO ARCH-FIX] _preempt → _neo_swap_out → "
                        "cpu_decoding_q 추가: rid=%s queue_size=%d",
                        rid, len(self.neo_scheduler.cpu_decoding_q),
                    )

    # ------------------------------------------------------------------
    # schedule — drive both the default and NEO schedulers, attach the
    # NEO decision to the SchedulerOutput for the runner to consume.
    # ------------------------------------------------------------------
    def _get_kv_pool_usage(self) -> float:
        """TSK_019 plan A1 — current GPU KV pool usage ratio (0.0~1.0).
        Returns 0.0 if attributes unavailable. Logger 보강 — 첫 5 회
        호출 + Exception 시 traceback INFO log.
        """
        try:
            block_pool = self.kv_cache_manager.block_pool
            total = block_pool.num_gpu_blocks
            free = block_pool.get_num_free_blocks()
            if total > 0:
                usage = 1.0 - (free / total)
                if not getattr(self, "_neo_kv_pool_first_logged", False):
                    self._neo_kv_pool_first_log_count = (
                        getattr(self, "_neo_kv_pool_first_log_count", 0) + 1
                    )
                    if self._neo_kv_pool_first_log_count <= 5:
                        logger.info(
                            "[NEO KV POOL] usage=%.4f total=%d free=%d "
                            "(call %d/5)",
                            usage, total, free,
                            self._neo_kv_pool_first_log_count,
                        )
                    if self._neo_kv_pool_first_log_count >= 5:
                        self._neo_kv_pool_first_logged = True
                return usage
            else:
                logger.warning(
                    "[NEO KV POOL] total blocks 0 — block_pool not init?"
                )
        except Exception as _e:  # noqa: BLE001
            import traceback as _tb
            logger.warning(
                "[NEO KV POOL] exception (%s): %s\n%s",
                type(_e).__name__, _e, _tb.format_exc(),
            )
        return 0.0

    def schedule(self) -> SchedulerOutput:
        # TSK_019 plan A1 + F2 — predictive swap_out trigger.
        # NEO 원본 (`swiftllm/server/scheduler.py:265-270`) 의 100%
        # threshold + 95% hysteresis 동등. F2 fix — 0.95 → 1.0 (NEO 원본).
        # try4 thrashing root (95% 가 너무 적극적 — sustained 영역 매 step
        # swap_out 발화) 회피. env override 가능.
        from vllm.v1.request import RequestStatus as _RS_pre
        import os as _os_th
        try:
            _SWAP_OUT_THRESHOLD = float(
                _os_th.environ.get("VLLM_NEO_PREDICTIVE_THRESHOLD", "1.0")
            )
        except ValueError:
            _SWAP_OUT_THRESHOLD = 1.0
        if not (0.5 <= _SWAP_OUT_THRESHOLD <= 1.0):
            _SWAP_OUT_THRESHOLD = 1.0
        _SWAP_IN_THRESHOLD = _SWAP_OUT_THRESHOLD * 0.95  # hysteresis 5%
        # TSK_019 plan F2 — cooldown. swap_out 후 N step 동안 추가 swap_out
        # 회피 (thrashing 회피). module-level counter.
        _COOLDOWN_STEPS = int(
            _os_th.environ.get("VLLM_NEO_SWAP_COOLDOWN", "5")
        )
        if not hasattr(self, "_neo_swap_cooldown_remaining"):
            self._neo_swap_cooldown_remaining = 0
        _swap_out_predictive_ids: list[str] = []
        _swap_out_predictive_block_ids: dict[str, list[int]] = {}
        # cooldown 처리 — swap_out 후 N step 동안 추가 swap_out 회피.
        if self._neo_swap_cooldown_remaining > 0:
            self._neo_swap_cooldown_remaining -= 1
        # TSK_019 try21 deadlock fix — spurious trigger 회피 + RUNNING 드레인
        # 한계.
        # Fix #1 (spurious trigger): kv_usage > threshold sustained + 모든
        # RUNNING SWAPPED_OUT (running_decode=0) → 매 step 진입 → log 12k/min
        # + schedule cost. cooldown set 후 break 로 무한 진입 회피.
        # Fix #2 (RUNNING 드레인): swap_out 가 RUNNING list 의 모든 reqs 를
        # SWAPPED_OUT 으로 비우면 vllm scheduler 가 num_scheduled_tokens=0
        # 만 emit → forward progress 0. cdec 부재 + swap_in 부재 영역에서
        # min_running_decode 보장.
        _MIN_RUNNING_DECODE = int(
            _os_th.environ.get("VLLM_NEO_MIN_RUNNING_DECODE", "32")
        )
        try:
            kv_usage = self._get_kv_pool_usage()
            if (kv_usage > _SWAP_OUT_THRESHOLD
                    and self._neo_swap_cooldown_remaining <= 0):
                running_decode = [
                    r for r in self.running
                    if r.status == _RS_pre.RUNNING
                    and r.num_computed_tokens >= r.num_prompt_tokens
                ]
                # Fix #1 — running_decode empty 면 cooldown 설정 후 즉시
                # return (무한 spurious 진입 차단).
                if not running_decode:
                    self._neo_swap_cooldown_remaining = _COOLDOWN_STEPS
                    if not getattr(self, "_neo_spurious_logged", False):
                        logger.info(
                            "[NEO predictive] running_decode=0 "
                            "(all SWAPPED_OUT or prefill) — cooldown %d steps "
                            "set to suppress spurious trigger.",
                            _COOLDOWN_STEPS,
                        )
                        self._neo_spurious_logged = True
                    raise StopIteration  # exit try block to skip swap_out
                logger.info(
                    "[NEO predictive trigger entered] "
                    "kv_usage=%.4f threshold=%.3f running_decode=%d",
                    kv_usage, _SWAP_OUT_THRESHOLD, len(running_decode),
                )
                _now_ts = time.time()
                # Fix #2 — running_decode 의 마지막 _MIN_RUNNING_DECODE 개는
                # 보존. swap_out 후보를 (len - MIN) 까지만.
                _max_swap_out = max(
                    0, len(running_decode) - _MIN_RUNNING_DECODE
                )
                _victim_count = 0
                while (kv_usage > _SWAP_IN_THRESHOLD
                       and running_decode
                       and _victim_count < _max_swap_out):
                    victim = running_decode.pop()
                    rid = victim.request_id
                    # TSK_019 plan A4 fix — _preempt_request 가 KV cache_
                    # manager.free 호출 *전* 에 GPU block_ids 추출. free 후
                    # 호출하면 input_batch.block_table 의 row 정리됨 → worker
                    # 가 None 받음 → silent skip (chain break root).
                    try:
                        block_groups = (
                            self.kv_cache_manager.get_block_ids(rid)
                        )
                        if block_groups and len(block_groups) > 0:
                            _swap_out_predictive_block_ids[rid] = list(
                                block_groups[0]
                            )
                    except Exception as _be:  # noqa: BLE001
                        logger.warning(
                            "[NEO predictive] get_block_ids failed for %s: %s",
                            rid, _be,
                        )
                    self._preempt_request(victim, _now_ts)
                    _swap_out_predictive_ids.append(rid)
                    _victim_count += 1
                    kv_usage = self._get_kv_pool_usage()
                logger.info(
                    "[NEO predictive swap_out] count=%d "
                    "block_ids_captured=%d kv_usage_final=%.4f "
                    "running_decode_kept=%d (min=%d)",
                    len(_swap_out_predictive_ids),
                    len(_swap_out_predictive_block_ids),
                    kv_usage,
                    len(running_decode), _MIN_RUNNING_DECODE,
                )
                # F2 cooldown — swap_out 발화 시 cooldown 적용
                if _swap_out_predictive_ids:
                    self._neo_swap_cooldown_remaining = _COOLDOWN_STEPS
        except StopIteration:
            pass  # spurious-trigger short-circuit — 정상 path
        except Exception as _e:  # noqa: BLE001
            import traceback as _tb
            logger.warning(
                "NEO predictive swap_out exception (%s): %s\n%s",
                type(_e).__name__, _e, _tb.format_exc(),
            )

        # Drive the default scheduler — this is the data path the engine
        # actually consumes for vanilla operation.
        output = super().schedule()

        # IDE_006 4.5.2.c v32 architectural simplification — NEO sibling
        # schedule() 호출 *완전 제거*.
        #
        # 이전: 매 step NEO sibling schedule (mode_selector / decide_mode /
        # perfpredictor / SubBatch / BatchPerfData / ScheduleBudget 등) 호출
        # → 64 reqs 환경 schedule cost 가 vanilla 의 ~10x 로 키워서 multiproc
        # broadcast deadlock 야기 (v31 진단: hot path 에 perfpredictor._interp_1d).
        #
        # cdec dispatch fork 결정은 NEO decide_mode *없이* 도 가능 — vllm
        # 의 SWAPPED_OUT 상태 reqs 를 cdec_ids 로 직접 attach (single
        # source of truth = vllm scheduler.requests). NEO 의 forward-time
        # wiring (worker fork branch + cdec dispatch hook) 만 살리고 schedule
        # -time decision 은 vllm standard 로 위임.
        self.last_neo_output = None

        # cdec_ids 직접 추출: vllm 의 scheduled reqs 중 SWAPPED_OUT 상태.
        from vllm.v1.request import RequestStatus as _RS
        vllm_ids = list(output.num_scheduled_tokens.keys())
        cdec_ids = [
            rid for rid in vllm_ids
            if (_req := self.requests.get(rid)) is not None
            and _req.status == _RS.SWAPPED_OUT
        ]
        if cdec_ids and len(cdec_ids) < len(vllm_ids):
            try:
                cdec_id_set = set(cdec_ids)
                b0_ids = [rid for rid in vllm_ids
                          if rid not in cdec_id_set]
                b1_ids = cdec_ids
                output.neo_sub_batches = [b0_ids, b1_ids]
                # cdec_token_slice — batches[0] 의 cdec rows zero,
                # batches[1] 의 *모든 reqs* 의 num_scheduled_tokens 합.
                _b0_tokens = sum(
                    output.num_scheduled_tokens.get(_id, 0)
                    for _id in b0_ids
                )
                _b1_tokens = sum(
                    output.num_scheduled_tokens.get(_id, 0)
                    for _id in b1_ids
                )
                output.neo_sub_batch_cdec_slices = [
                    (_b0_tokens, _b0_tokens),  # b0: zero cdec
                    (0, _b1_tokens),           # b1: 전체 cdec
                ]
                output.neo_sub_batch_cdec_seq_slices = [
                    (len(b0_ids), len(b0_ids)),
                    (0, len(b1_ids)),
                ]
                output.neo_sub_batch_cdec_req_ids = [
                    [],
                    list(b1_ids),
                ]
            except (AttributeError, TypeError) as e:
                logger.debug("NEO output attach failed: %s", e)

        # TSK_019 plan A2 — swap_out_req_ids attach 활성.
        # 본 turn 의 predictive trigger (A1) 로 _preempt_request 호출 한
        # req 들을 SchedulerOutput 에 attach → engine/core 의
        # _handle_neo_swaps (A3 gate 해제) → worker _neo_handle_kv_swap
        # (A4 stream wrap) → KV bytes per-layer GPU→CPU 진짜 이동.
        if _swap_out_predictive_ids:
            try:
                output.neo_swap_out_req_ids = list(_swap_out_predictive_ids)
                # TSK_019 plan A4 fix — block_ids dict attach 도 함께.
                if _swap_out_predictive_block_ids:
                    output.neo_swap_out_block_ids = dict(
                        _swap_out_predictive_block_ids
                    )
            except (AttributeError, TypeError) as _ae:
                logger.debug(
                    "NEO swap_out attach failed: %s", _ae,
                )

        # TSK_019 plan B-NEW 진짜 fix — predictive swap_in 도 비활성.
        # 기존: _neo_swap_in() 호출 → 새 GPU block 할당 → status RUNNING.
        # 그러나 새 block_ids 가 input_batch.block_table.np[req_idx] 에
        # 갱신 안 됨 (req_to_new_blocks populate 안 함, worker 의
        # _get_req_gpu_block_ids 가 stale 읽음) → cross-req KV
        # contamination → CUDA device-side assert → silent worker crash.
        # Fix: 모든 swap_in path 제거. SWAPPED_OUT req 는 cdec dispatch
        # (NeoCpuKvBuffer 의 KV 로 CPU pacpu attention) 만 통해 처리.
        # NEO 본가 의 *bidirectional cycle* 는 별도 phase (req_to_new_blocks
        # populate + block_ids dict attach + worker swap_in_one_req 의
        # block_ids 사용 모두 적재 후 활성).

        return output
