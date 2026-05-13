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
import os
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
            make_predictor_from_env,
        )

        sched_cfg = self.vllm_config.scheduler_config
        cache_cfg = self.vllm_config.cache_config
        model_cfg = self.vllm_config.model_config

        # [Plan v4 D14] env 기반 predictor 선택 — default ``heuristic``
        # (interp-free 상수 시간). v31 measurement 의 deadlock root 였던
        # ``TablePerfPredictor._interp_1d`` hot path 회피 + load-aware
        # _get_remains 가 *비제로* 값 반환 (ZeroPerfPredictor 의 항상-0
        # 한계 해소 → ``decide_mode`` sequential 폴백 회피).
        # env: VLLM_NEO_PREDICTOR ∈ {heuristic, zero, table}.
        self.predictor = make_predictor_from_env(self.vllm_config)
        try:
            self.table_predictor_legacy = TablePerfPredictor(self.vllm_config)
        except Exception as _e:  # noqa: BLE001
            self.table_predictor_legacy = None
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
            # [TSK_019 v3 / Phase A-0] CDEC default 활성 옵션 plumb.
            # mode_selector 가 fast-path 의 sequential 폴백 시 NEO 비활성
            # 화하는 default 행동 우회. config 미명시 시 False (안전).
            force_pipelined=getattr(
                sched_cfg, "enable_neo_force_pipelined", False
            ),
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
            # [Plan v4 D14] env VLLM_NEO_PREDICTOR != "table" 이면 self.predictor
            # 는 *덮어쓰지 않음* — adapter __init__ 의 make_predictor_from_env
            # 결과 (Heuristic 또는 Zero) 보호. table_predictor 와
            # neo_scheduler.predictor 만 갱신 (legacy compat).
            import os as _os_d14
            _pred_choice_d14 = _os_d14.environ.get(
                "VLLM_NEO_PREDICTOR", "heuristic"
            ).lower()
            if _pred_choice_d14 == "table":
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
        # [Plan v4 / Phase I] CPU-resident mirror set cleanup. req finish 시
        # 워커 측 _neo_cpu_resident_reqs + buf 도 자연 free 되지만 adapter
        # 의 mirror set 은 별도 cleanup 필요.
        mirror = getattr(self, "_neo_cpu_resident_mirror", None)
        if mirror:
            for rid in request_ids:
                mirror.discard(rid)

    def _neo_swap_out(self, request, timestamp=None):  # type: ignore[override]
        """IDE_006 / TSK_015 4.5.2.c architectural fix —
        ``_preempt_request`` 의 NEO hook 이 호출 시 vLLM standard 의
        ``_neo_swap_out`` (KV CPU copy + status SWAPPED_OUT + RUNNING
        잔류) 후 NEO scheduler 의 ``cpu_decoding_q`` 에 view 추가.

        [Plan v4 / Phase G] block_ids 추출 *before* super() — KV 가 아직
        GPU 에 살아있는 시점에 추출 후 self._neo_natural_swap_block_ids
        dict 에 저장. adapter.schedule() 종료 시 output.neo_swap_out_*
        attach 에 사용 → 워커 _neo_handle_kv_swap 가 GPU→CPU per-layer
        copy 발화.
        """
        # [Plan v4 / Phase G] block_ids 추출 — super() 가 status 만 SWAPPED_OUT
        # 으로 set 하지만 KV 는 deferred 처리 시점까지 GPU 잔류. 본 시점에
        # kv_cache_manager.coordinator.get_blocks(rid) 로 block_ids 추출
        # 후 워커 _neo_handle_kv_swap 에 전달.
        try:
            rid = request.request_id
            blocks = None
            if hasattr(self, "kv_cache_manager"):
                coord = getattr(self.kv_cache_manager, "coordinator", None)
                if coord is not None and hasattr(coord, "get_blocks"):
                    # get_blocks 는 group 별 tuple. 첫 group (Llama 등 single
                    # group 모델) 의 blocks 를 사용.
                    groups = coord.get_blocks(rid)
                    if groups and len(groups) > 0:
                        first_group_blocks = groups[0]
                        if first_group_blocks:
                            blocks = [
                                int(b.block_id) for b in first_group_blocks
                                if hasattr(b, "block_id")
                            ]
            if blocks:
                if not hasattr(self, "_neo_natural_swap_block_ids"):
                    self._neo_natural_swap_block_ids: dict[str, list[int]] = {}
                self._neo_natural_swap_block_ids[rid] = blocks
                # [Plan v4 D8] block_count 안전 영역 stash — try60-γ/62/63
                # SEGV root 회피용. swap-out 시점의 num_computed_tokens 와
                # block_count 로 결정되는 *block_pos 안전 영역 상한선* 을
                # request 에 attach. scheduler.py 의 D5 fix 분기가 본 값과
                # request.num_computed_tokens 를 비교해 num_new_tokens=1
                # 부여 여부 결정. 너머면 decode 보류 → seq_len 동결 →
                # pacpu store_kv 의 block_table OOB 도달 차단.
                # [Plan v4 D12 v3] try70 측정 결과 — D12 의 *어떤* margin
                # 이든 (1 block 또는 8 token) chain firing 을 cascade-
                # deactivate. 정적으로는 동일 reqs 통과 영역인데 동적
                # 으로는 NEO 메커니즘 자체 비활성. D11 dynamic precheck
                # 가 *진짜* root (async lookahead) 를 이미 처리.
                # 따라서 D12 default margin=0 (D8 v1 동일) + env 로
                # 조정 가능 (디버깅용 보존). D11 가 잔존 OOB catch.
                _block_size_d8 = getattr(self, "block_size", 16)
                _d12_margin = int(os.environ.get(
                    "VLLM_NEO_D12_TOKEN_MARGIN", "0"))
                _safe_max = (
                    len(blocks) * _block_size_d8 - 1 - _d12_margin
                )
                request._neo_swap_out_safe_max_computed = _safe_max
                if not getattr(self, "_neo_first_blockid_logged", False):
                    logger.info(
                        "[Plan v4 G+D8] first natural-preempt block_id capture: "
                        "rid=%s nblocks=%d sample=%s safe_max=%d",
                        rid, len(blocks), blocks[:5], _safe_max,
                    )
                    self._neo_first_blockid_logged = True
        except Exception as _e:  # noqa: BLE001
            logger.warning(
                "[Plan v4 G] block_ids 추출 실패 rid=%s: %s",
                getattr(request, "request_id", "?"), _e,
            )

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

        # [Plan v4 D15+D16] Load-aware active swap_out — KV pressure 와
        # *무관* 하게 NEO paper (MLSys 2025) 의 진짜 design 정합. heuristic
        # predictor 의 _get_remains 식 직접 계산 → GPU 시간 - CPU 시간 양수
        # ⇒ CPU slack 가능 → 일부 RUNNING decode reqs 를 active swap_out.
        # 다음 step 에 status=SWAPPED_OUT → 기존 cdec_ids 추출 path 자연
        # 포함 → cdec dispatch 발화. paper 의 *load-aware sub-batch
        # decision* 의 경량 reproduction.
        #
        # v31 deadlock root 회피: SubBatch / decide_mode / _interp_1d 호출
        # X. 매 step 산술 비용 < 5µs (interp-free heuristic 만).
        #
        # KV pressure trigger 가 *이미 fire 한* 회차에서는 load-aware skip
        # (thrashing 회피). cooldown 중에도 skip.
        try:
            from vllm.v1.core.sched.perfpredictor import HeuristicPerfPredictor
            if (isinstance(self.predictor, HeuristicPerfPredictor)
                    and self._neo_swap_cooldown_remaining <= 0
                    and len(_swap_out_predictive_ids) == 0):
                _running_decode_la = [
                    r for r in self.running
                    if r.status == _RS_pre.RUNNING
                    and r.num_computed_tokens >= r.num_prompt_tokens
                ]
                _n_running_la = len(_running_decode_la)
                _min_running_la = int(_os_th.environ.get(
                    "VLLM_NEO_LOAD_AWARE_MIN_RUNNING", "32"
                ))
                if _n_running_la >= _min_running_la:
                    _total_kv_la = sum(
                        r.num_computed_tokens for r in _running_decode_la
                    )
                    _gdec_T_la = self.predictor.get_gdec_T(_total_kv_la)
                    _linr_T_la = self.predictor.get_linr_T(_n_running_la)
                    _gpu_total_T_la = _gdec_T_la + _linr_T_la

                    _cap_la = int(_os_th.environ.get(
                        "VLLM_NEO_LOAD_AWARE_SWAP_OUT_CAP_PER_STEP", "2"
                    ))
                    # short KV first — cdec_T 작음, 안전.
                    _sorted_la = sorted(
                        _running_decode_la,
                        key=lambda r: r.num_computed_tokens,
                    )

                    _cdec_kv_la = 0
                    _cdec_cnt_la = 0
                    _now_ts_la = time.time()
                    for _victim_la in _sorted_la:
                        if _cdec_cnt_la >= _cap_la:
                            break
                        _trial_cnt = _cdec_cnt_la + 1
                        _trial_kv = (_cdec_kv_la
                                     + _victim_la.num_computed_tokens)
                        _cdec_T_la = self.predictor.get_cdec_T(
                            _trial_cnt, _trial_kv
                        )
                        # _get_remains 식: GPU 시간 - CPU 시간. positive ⇒
                        # CPU slack 가능. negative ⇒ CPU bottleneck.
                        if _gpu_total_T_la - _cdec_T_la <= 0:
                            break
                        _rid_la = _victim_la.request_id
                        try:
                            _bg_la = (
                                self.kv_cache_manager.get_block_ids(_rid_la)
                            )
                            if _bg_la and len(_bg_la) > 0:
                                _swap_out_predictive_block_ids[_rid_la] = (
                                    list(_bg_la[0])
                                )
                        except Exception:  # noqa: BLE001
                            pass
                        self._preempt_request(_victim_la, _now_ts_la)
                        _swap_out_predictive_ids.append(_rid_la)
                        _cdec_kv_la = _trial_kv
                        _cdec_cnt_la = _trial_cnt

                    if _cdec_cnt_la > 0:
                        self._neo_swap_cooldown_remaining = _COOLDOWN_STEPS
                        if not getattr(self, "_neo_d15_logged", False):
                            logger.info(
                                "[Plan v4 D15+D16] load-aware active "
                                "swap_out first fire: count=%d "
                                "running_decode=%d gpu_T=%.2fms "
                                "cdec_T=%.2fms",
                                _cdec_cnt_la, _n_running_la,
                                _gpu_total_T_la,
                                self.predictor.get_cdec_T(
                                    _cdec_cnt_la, _cdec_kv_la
                                ),
                            )
                            self._neo_d15_logged = True
        except Exception as _le:  # noqa: BLE001
            logger.debug(
                "Plan v4 D15+D16 load-aware exception: %s", _le,
            )

        # Drive the default scheduler — this is the data path the engine
        # actually consumes for vanilla operation.
        output = super().schedule()

        # IDE_006 Phase 4.2 — NEO 정통 Step 2/3 (swap-out + swap-in)
        # 호출. v31 deadlock root (perfpredictor hot-path) 회피용으로
        # step_2_3_only() 만 호출 — Step 4-6 (decide_mode, prefill 분류)
        # 는 vLLM default 가 이미 처리. env-gated:
        # VLLM_NEO_NEOSCHED_STEP23=1 + VLLM_NEO_SWAP_OUT_RATIO 로 threshold
        # scaling (default 1.0 = NEO paper spec). Adapter 의 기존 predictive
        # / Plan v4 D15+D16 path 와 OR 결합 — Step 2/3 발화 victim 은
        # _swap_out_predictive_ids 에 추가되어 기존 attach 경로 재사용.
        if _os_th.environ.get("VLLM_NEO_NEOSCHED_STEP23", "0") == "1":
            try:
                # gpu_decoding_q sync — neo_scheduler 는 매 step super().schedule()
                # 후 self.running 영역만 알고 있으므로 step_2_3_only() 호출 전
                # mirror 필수. 미호출 시 gpu_q 비어있어 swap_out_threshold 영역
                # 진입 자체가 안 됨 (noop).
                self._sync_neo_gpu_decoding_q()
                _step23_so, _step23_si = self.neo_scheduler.step_2_3_only()
                for _v in _step23_so:
                    _rid_so = getattr(_v, "request_id", None) or \
                              getattr(_v, "_str_id", None)
                    if _rid_so and _rid_so not in _swap_out_predictive_ids:
                        _swap_out_predictive_ids.append(_rid_so)
                if not getattr(self, "_neo_step23_logged", False):
                    logger.info(
                        "[NEO Step2/3] first fire: swap_out=%d "
                        "swap_in=%d cpu_q_size=%d gpu_q_size=%d",
                        len(_step23_so), len(_step23_si),
                        len(self.neo_scheduler.cpu_decoding_q),
                        len(self.neo_scheduler.gpu_decoding_q),
                    )
                    self._neo_step23_logged = True
            except Exception as _step23_e:  # noqa: BLE001
                logger.warning(
                    "[NEO Step2/3] failed: %s: %s",
                    type(_step23_e).__name__, _step23_e,
                )

        # IDE_006 — NEO Scheduler 6-step 전체 driving 통합. NeoScheduler.
        # schedule() 직접 호출 후 swap_out_reqs / swap_in_reqs 영역 vllm
        # path 와 통합. v31 deadlock root (perfpredictor._interp_1d) 영역은
        # HeuristicPerfPredictor 사용 시 미발화 — 본 통합 안전 가능성 검증.
        # env VLLM_NEO_DRIVE_6STEP=1 활성 (default OFF).
        # safety: default = dry-run (VLLM_NEO_6STEP_DRY_RUN=1) — queue
        # save/restore 후 observe + log 만. apply mode (DRY_RUN=0) 시
        # swap_out_reqs victim 영역 block_ids 추출 + _preempt_request 호출,
        # swap_in_reqs 영역 _neo_swap_in 호출. batches 영역은 Option C
        # decide_mode path 와 중복 방지 위해 본 영역에서는 skip (별도 fix).
        if _os_th.environ.get("VLLM_NEO_DRIVE_6STEP", "0") == "1":
            try:
                from collections import deque as _deque_6s
                _dry_run = _os_th.environ.get(
                    "VLLM_NEO_6STEP_DRY_RUN", "1"
                ) == "1"
                self._sync_neo_gpu_decoding_q()
                # queue snapshot — dry-run 시 schedule() 의 mutation 복원용.
                _orig_w = list(self.neo_scheduler.waiting_q)
                _orig_g = list(self.neo_scheduler.gpu_decoding_q)
                _orig_c = list(self.neo_scheduler.cpu_decoding_q)
                _neo_out = self.neo_scheduler.schedule()
                _so_count = len(_neo_out.swap_out_reqs)
                _si_count = len(_neo_out.swap_in_reqs)
                _b_count = len(_neo_out.batches)
                # log + counter — 매 step counter, 50회 마다 INFO log.
                _drv_cnt = getattr(self, "_neo_6step_call_count", 0) + 1
                self._neo_6step_call_count = _drv_cnt
                if (not getattr(self, "_neo_6step_logged", False)
                        or _drv_cnt % 200 == 0):
                    logger.info(
                        "[NEO 6step] call=%d dry_run=%s swap_out=%d "
                        "swap_in=%d batches=%d (waiting=%d gpu_q=%d "
                        "cpu_q=%d)",
                        _drv_cnt, _dry_run, _so_count, _si_count, _b_count,
                        len(_orig_w), len(_orig_g), len(_orig_c),
                    )
                    self._neo_6step_logged = True
                if _dry_run:
                    # mutation 복원. waiting_q / gpu_decoding_q / cpu_decoding_q
                    # 의 schedule() 영역 mutation 영역 무효화.
                    self.neo_scheduler.waiting_q = _deque_6s(_orig_w)
                    self.neo_scheduler.gpu_decoding_q = list(_orig_g)
                    self.neo_scheduler.cpu_decoding_q = _deque_6s(_orig_c)
                else:
                    # apply mode — swap_out_reqs / swap_in_reqs 영역 vllm
                    # path 와 통합.
                    _now_6s = time.time()
                    for _v in _neo_out.swap_out_reqs:
                        _rid_6s = getattr(_v, "_str_id", None) or \
                                  str(getattr(_v, "request_id", ""))
                        if not _rid_6s:
                            continue
                        _req_6s = self.requests.get(_rid_6s)
                        if (_req_6s is None
                                or _req_6s.status != _RS_pre.RUNNING):
                            continue
                        try:
                            _bg_6s = (
                                self.kv_cache_manager.get_block_ids(_rid_6s)
                            )
                            if _bg_6s and len(_bg_6s) > 0:
                                _swap_out_predictive_block_ids[_rid_6s] = (
                                    list(_bg_6s[0])
                                )
                        except Exception:  # noqa: BLE001
                            pass
                        self._preempt_request(_req_6s, _now_6s)
                        if _rid_6s not in _swap_out_predictive_ids:
                            _swap_out_predictive_ids.append(_rid_6s)
                    # swap_in_reqs 영역은 별도 변수 (_neo_6step_swap_in_ids)
                    # 에 누적, 본 schedule() 종료 영역의 swap_in attach path
                    # 영역에서 (line ~1199) output.neo_swap_in_req_ids 와 merge.
                    _si_ids_6s: list[str] = []
                    for _v in _neo_out.swap_in_reqs:
                        _rid_6s = getattr(_v, "_str_id", None) or \
                                  str(getattr(_v, "request_id", ""))
                        if not _rid_6s:
                            continue
                        _req_6s = self.requests.get(_rid_6s)
                        if (_req_6s is None
                                or _req_6s.status != _RS_pre.SWAPPED_OUT):
                            continue
                        if self._neo_swap_in(_req_6s, _now_6s):
                            _si_ids_6s.append(_rid_6s)
                            if _req_6s not in self.running:
                                self.running.append(_req_6s)
                            # mirror set 도 cleanup
                            _mset_6s = getattr(
                                self, "_neo_cpu_resident_mirror", None,
                            )
                            if _mset_6s:
                                _mset_6s.discard(_rid_6s)
                    if _si_ids_6s:
                        # 본 함수 종료 직전 output.neo_swap_in_req_ids 영역
                        # 합치기 (기존 attach path 가 같은 attribute 사용).
                        # attribute 가 없으면 set.
                        _existing_si = getattr(
                            output, "neo_swap_in_req_ids", None,
                        )
                        if _existing_si:
                            output.neo_swap_in_req_ids = list(
                                _existing_si
                            ) + _si_ids_6s
                        else:
                            output.neo_swap_in_req_ids = list(_si_ids_6s)
            except Exception as _drv_e:  # noqa: BLE001
                logger.warning(
                    "[NEO 6step] failed: %s: %s",
                    type(_drv_e).__name__, _drv_e,
                )

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

        # [Option C / D17C] mirror 의 reqs → cdec 후보 + decide_mode
        # 호출하여 load-balanced 배포 (NEO _decide_mode_and_gen_batch Step
        # 3 등가). batches[1].cdec_reqs → cdec_ids.
        # env VLLM_NEO_OPTION_C=1 활성. Option A 와 둘 중 하나만 활성.
        # [Option A / D19] cdec_ids = SWAPPED_OUT ∪ mirror (brute force).
        # env VLLM_NEO_OPTION_A=1 활성. budget check / load balance X.
        _option_c_d17c = _os_th.environ.get("VLLM_NEO_OPTION_C", "0") == "1"
        _option_a_d19 = _os_th.environ.get("VLLM_NEO_OPTION_A", "0") == "1"
        _mirror_oc = getattr(self, "_neo_cpu_resident_mirror", set()) or set()
        cdec_ids: list[str] = []
        if _option_c_d17c:
            _vid_set_oc = set(vllm_ids)
            # [Option C v2 — try99 fact 기반 fix]
            # 직전 (try99) 측정 결과: cdec batch_size=1 → store_kv (Step 0)
            # 가 tid=0 만 work → 13 thread Barrier 1 wait = 370us (41% wall).
            # root: decide_mode 의 alternate 분배 (batches[0]/[1]) 가 mirror
            # 의 reqs 를 절반씩 분배 → batches[1] (cdec) batch_size = mirror/2.
            # mirror=4 인 영역에서는 batch=1-2.
            # v2: decide_mode 호출 제거. mirror ∩ vllm_ids 의 *전체* 를
            # cdec_ids 로. batch_size = full mirror size (~14+) → store_kv
            # 도 14 thread 분배 → Bar 1 wait ~0 → wall 2.4× 단축 기대.
            # env VLLM_NEO_OPTION_C_FULL_MIRROR=0 시 v1 (decide_mode) 복원.
            _full_mirror = (_os_th.environ.get(
                "VLLM_NEO_OPTION_C_FULL_MIRROR", "1") == "1")
            _cdec_cands_oc: list[_NeoRequestView] = []
            for _rid in _mirror_oc:
                if _rid not in _vid_set_oc:
                    continue
                _r = self.requests.get(_rid)
                if _r is None:
                    continue
                _cdec_cands_oc.append(_NeoRequestView(_r))
            if _full_mirror and _cdec_cands_oc:
                # Direct path — mirror 전체 cdec_ids (decide_mode 우회).
                cdec_ids = [v._str_id for v in _cdec_cands_oc]
                if not getattr(self, "_neo_option_c_logged", False):
                    logger.info(
                        "[Option C / D17C v2 — full mirror] cdec_ids=%d "
                        "(mirror 전체 ∩ vllm_ids)",
                        len(cdec_ids),
                    )
                    self._neo_option_c_logged = True
                # decide_mode 분기 skip (cands 보존 위해 empty 로)
                _cdec_cands_oc = []
            _gpu_dec_cands_oc: list[_NeoRequestView] = []
            _gpu_pref_cands_oc: list[_NeoRequestView] = []
            for _rid in vllm_ids:
                if _rid in _mirror_oc:
                    continue
                _r = self.requests.get(_rid)
                if _r is None:
                    continue
                if _r.num_computed_tokens >= _r.num_prompt_tokens:
                    _gpu_dec_cands_oc.append(_NeoRequestView(_r))
                else:
                    _gpu_pref_cands_oc.append(_NeoRequestView(_r))
            if _cdec_cands_oc:
                try:
                    from vllm.v1.core.sched.mode_selector import (
                        decide_mode as _decide_mode_oc,
                        ScheduleBudget as _SB_oc,
                    )
                    _budget_oc = _SB_oc(
                        max(self.scheduler_config.max_num_seqs * 2, 256),
                        max(self.scheduler_config.max_num_batched_tokens * 2,
                            16384),
                    )
                    try:
                        _num_layers_oc = (
                            self.vllm_config.model_config.get_num_layers(
                                self.vllm_config.parallel_config,
                            )
                        )
                    except Exception:  # noqa: BLE001
                        _num_layers_oc = 32
                    _num_gpu_blocks_oc = 1
                    try:
                        _coord = getattr(self.kv_cache_manager,
                                         "coordinator", None)
                        if _coord is not None:
                            _num_gpu_blocks_oc = getattr(
                                _coord, "num_gpu_blocks", 1) or 1
                    except Exception:  # noqa: BLE001
                        pass
                    _batches_oc = _decide_mode_oc(
                        gpu_prefill_reqs=_gpu_pref_cands_oc,
                        cpu_prefill_reqs=[],
                        gpu_decoding_q=_gpu_dec_cands_oc,
                        cpu_decoding_q=_cdec_cands_oc,
                        budget=_budget_oc,
                        predictor=self.predictor,
                        num_layers=_num_layers_oc,
                        num_gpu_blocks=_num_gpu_blocks_oc,
                    )
                    if len(_batches_oc) == 2:
                        cdec_ids = [v._str_id
                                    for v in _batches_oc[1].cdec_reqs]
                    if not getattr(self, "_neo_option_c_logged", False):
                        logger.info(
                            "[Option C / D17C] first fire: mirror=%d "
                            "cdec_cands=%d gpu_dec=%d gpu_pref=%d "
                            "cdec_ids=%d batches_len=%d",
                            len(_mirror_oc), len(_cdec_cands_oc),
                            len(_gpu_dec_cands_oc), len(_gpu_pref_cands_oc),
                            len(cdec_ids), len(_batches_oc),
                        )
                        self._neo_option_c_logged = True
                except Exception as _oce:  # noqa: BLE001
                    logger.warning(
                        "[Option C / D17C] exception (%s): %s",
                        type(_oce).__name__, _oce,
                    )
                    cdec_ids = [
                        rid for rid in vllm_ids
                        if (_r := self.requests.get(rid)) is not None
                        and _r.status == _RS.SWAPPED_OUT
                    ]
        elif _option_a_d19:
            _swapped_out_a = set()
            for rid in vllm_ids:
                _r = self.requests.get(rid)
                if _r is not None and _r.status == _RS.SWAPPED_OUT:
                    _swapped_out_a.add(rid)
            _cdec_set_a = _swapped_out_a | (_mirror_oc & set(vllm_ids))
            cdec_ids = [rid for rid in vllm_ids if rid in _cdec_set_a]
            if cdec_ids and not getattr(self, "_neo_option_a_logged", False):
                logger.info(
                    "[Option A / D19] first fire: mirror=%d "
                    "swapped_out=%d cdec_ids=%d",
                    len(_mirror_oc), len(_swapped_out_a), len(cdec_ids),
                )
                self._neo_option_a_logged = True
        else:
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
        # [Plan v4 / Phase G] *natural-preempt* path (vLLM scheduler 의
        # KV pressure preempt) 도 같은 attach. _neo_swap_out override
        # 가 captured _neo_natural_swap_block_ids 의 reqs 를 합산.
        _all_swap_out_ids = list(_swap_out_predictive_ids)
        _all_swap_out_block_ids = dict(_swap_out_predictive_block_ids)
        _natural_blocks = getattr(self, "_neo_natural_swap_block_ids", None)
        if _natural_blocks:
            for _rid, _blocks in _natural_blocks.items():
                if _rid not in _all_swap_out_block_ids:
                    _all_swap_out_ids.append(_rid)
                    _all_swap_out_block_ids[_rid] = _blocks
            # 추출된 dict 비우기 (다음 step 에 새로 채움).
            self._neo_natural_swap_block_ids = {}

        if _all_swap_out_ids:
            try:
                # [Plan v4 G/H v3] mirror set 에 bound 적용 — CPU pool
                # capacity 보다 작게 유지. 초과 시 새 reqs 는 mirror 에 추가
                # 안 함 → vanilla preempt path 로 흘러감 (hang 회피).
                # Cap: max_cpu_resident_reqs (default 64) × 0.9 = ~57 reqs.
                # env override: VLLM_NEO_MIRROR_MAX (default 56).
                _MIRROR_MAX = int(_os_th.environ.get(
                    "VLLM_NEO_MIRROR_MAX", "56"
                ))
                if not hasattr(self, "_neo_cpu_resident_mirror"):
                    self._neo_cpu_resident_mirror: set[str] = set()

                # 추가 가능한 슬롯 계산.
                _slots_available = max(
                    0, _MIRROR_MAX - len(self._neo_cpu_resident_mirror)
                )
                _attached_ids = []
                _attached_block_ids = {}
                _mirror_added = 0
                for _rid in _all_swap_out_ids:
                    # block_ids 미캡처 → 워커 copy 불가 → mirror 에 안 넣고
                    # attach 도 skip. vanilla preempt 가 처리.
                    if _rid not in _all_swap_out_block_ids:
                        continue
                    # mirror 슬롯 부족 → attach skip (vanilla preempt 로).
                    if _slots_available <= 0:
                        continue
                    _attached_ids.append(_rid)
                    _attached_block_ids[_rid] = _all_swap_out_block_ids[_rid]
                    self._neo_cpu_resident_mirror.add(_rid)
                    _mirror_added += 1
                    _slots_available -= 1

                if _attached_ids:
                    output.neo_swap_out_req_ids = list(_attached_ids)
                    output.neo_swap_out_block_ids = dict(_attached_block_ids)

                logger.info(
                    "[Plan v4 G/H] swap_out attach: predictive=%d natural=%d "
                    "candidates=%d attached=%d mirror+=%d "
                    "(mirror_set_size=%d / cap=%d)",
                    len(_swap_out_predictive_ids),
                    len(_natural_blocks or {}),
                    len(_all_swap_out_ids),
                    len(_attached_ids),
                    _mirror_added,
                    len(self._neo_cpu_resident_mirror),
                    _MIRROR_MAX,
                )
            except (AttributeError, TypeError) as _ae:
                logger.debug(
                    "NEO swap_out attach failed: %s", _ae,
                )

        # [TSK_019 v3 / Phase B-1] Lightweight swap_in candidate evaluation.
        # 기존 (B-NEW fix): predictive swap_in 완전 비활성 — block_table 정합
        # 이슈 (try10~15 root) 회피.
        # 본 fix: deferred preempt 가 *fire 하기 전* 의 SWAPPED_OUT reqs 를
        # 대상으로 swap_in. 이 reqs 의 KV 는 *아직 GPU 에 살아있음* (deferred
        # 단계는 _handle_neo_swaps 에서 처리). 따라서 status 만 SWAPPED_OUT →
        # RUNNING 복원하면 block_table 갱신 불필요 (KV 위치 변동 없음).
        # NEO 의 진짜 bidirectional cycle 의 *근본 안전 path*.
        #
        # GPU KV usage < swap_in_threshold (hysteresis 5% deadband) 이고
        # deferred 큐에 reqs 가 있으면 가장 최근 추가된 reqs 부터 FIFO 로
        # swap_in. C5 에서 LRU 로 교체 가능.
        #
        # Kill switch: VLLM_NEO_DISABLE_SWAP_IN=1 → 본 path 비활성 (B-NEW
        # 등가, vanilla preempt 만 사용).
        # [Plan v4 / Phase I D4] swap_in 의 *진짜* CPU→GPU restore.
        # mirror set 의 reqs 중 oldest pick → self._neo_swap_in 호출
        # (kv_cache_manager.neo_swap_in_alloc 으로 GPU blocks 새 alloc
        # + status SWAPPED_OUT → RUNNING) → output.neo_swap_in_req_ids
        # attach → 워커 _neo_handle_kv_swap 가 KV CPU→GPU copy + buf.free.
        # NEO 의 *진짜 bidirectional migration loop* 활성화.
        # GPU KV usage < threshold 시 + mirror non-empty 시 fire.
        _mirror_d4 = getattr(self, "_neo_cpu_resident_mirror", None)
        if (_os_th.environ.get("VLLM_NEO_DISABLE_SWAP_IN") != "1"
                and _mirror_d4):
            try:
                from vllm.v1.request import RequestStatus as _RS_d4
                import time as _time_d4
                _now_d4 = _time_d4.time()
                _kv_usage_d4 = self._get_kv_pool_usage()
                # [Plan v4 D6] try60-γ pacpu store_kv segfault root fix.
                # D4 의 KV usage threshold 가 0.95 인데 KV 한계 영역
                # workload (try60: 99.7% sustained) 에서는 영영 미충족 →
                # mirror reqs 무한 잔류 → SWAPPED_OUT decode 누적 →
                # seq_len 이 CPU buffer block_count 초과 → pacpu kernel
                # 의 block_table OOB → SIGSEGV.
                # 본 fix: forced mode 에서 KV threshold 무시. mirror
                # non-empty 면 매 step swap_in 시도. GPU pool 부족 시
                # _neo_swap_in 가 False → loop break 로 자연 안전화.
                # per-step cap 도 작게 (기본 2) — KV 한계 영역에서 한
                # 번에 많이 시도하면 pool 부족으로 거의 다 fail.
                # Kill switch: VLLM_NEO_FORCE_SWAP_IN=0 → D4 동작 복원.
                _force_swap_in = _os_th.environ.get(
                    "VLLM_NEO_FORCE_SWAP_IN", "1"
                ) == "1"
                if _force_swap_in or _kv_usage_d4 < _SWAP_IN_THRESHOLD:
                    _max_swap_in = int(_os_th.environ.get(
                        "VLLM_NEO_MAX_SWAP_IN_PER_STEP",
                        "2" if _force_swap_in else "8",
                    ))
                    # [Plan v4 Option I] MIN_BUFFER guard — mirror size 가
                    # MIN_BUFFER 너머일 때만 (len - MIN_BUFFER) 만큼 swap_in.
                    # 그 미만이면 _max_swap_in=0 으로 swap_in skip.
                    # NEO 의 cpu_decoding_q 영구 큐 시간 확보 (chain firing
                    # 활성화 prerequisite). D17C/D19 가 보는 mirror size 가
                    # *항상 MIN_BUFFER 부근* 안정 영역 유지.
                    # env: VLLM_NEO_MIRROR_MIN_BUFFER (default 8).
                    _min_buffer = int(_os_th.environ.get(
                        "VLLM_NEO_MIRROR_MIN_BUFFER", "8"
                    ))
                    _excess = max(0, len(_mirror_d4) - _min_buffer)
                    _max_swap_in = min(_max_swap_in, _excess)
                    # Deadlock escape: GPU-active seq=0 + waiting=0 이고
                    # mirror만 남은 경우, MIN_BUFFER 가드가 모든 swap-in을
                    # 막아 영구 deadlock. 이 조건에서는 가드 무시하고
                    # max_swap_in 복원 (최대 원래 cap 까지).
                    _gpu_active_d4 = sum(
                        1 for _r in self.running
                        if _r.status == _RS_d4.RUNNING
                    )
                    if (_max_swap_in == 0
                            and _gpu_active_d4 == 0
                            and not self.waiting
                            and _mirror_d4):
                        _max_swap_in = min(
                            int(_os_th.environ.get(
                                "VLLM_NEO_MAX_SWAP_IN_PER_STEP",
                                "2" if _force_swap_in else "8",
                            )),
                            len(_mirror_d4),
                        )
                        if not getattr(
                            self, "_neo_escape_logged", False
                        ):
                            logger.info(
                                "[NEO deadlock escape] GPU-active=0 "
                                "waiting=0 mirror=%d — MIN_BUFFER 가드 "
                                "bypass, max_swap_in=%d",
                                len(_mirror_d4), _max_swap_in,
                            )
                            self._neo_escape_logged = True
                    # [Plan v5 Option O2 v2] NEO budget coupling 정합 — D4
                    # 가 self.running 안의 *RUNNING 상태* 슬롯만 카운트.
                    # try85 의 O2 v1 결함: self.running 이 SWAPPED_OUT reqs
                    # 도 포함 → 항상 max_num_seqs (256) 도달 → swap_in 영구
                    # silent → throughput 67 tps 추가 cliff.
                    # v2 fix: RUNNING 상태만 카운트하여 GPU 실제 활성 슬롯
                    # 측정. mirror (SWAPPED_OUT) 반환된 reqs 가 D4 swap_in
                    # 으로 RUNNING 복귀 가능 영역 보장.
                    # NEO swiftllm/server/scheduler.py:278 의
                    # `budget.check_and_substract(1)` 의 *intent 정합*
                    # — NEO 의 budget 은 한 iteration 의 batch 안의 새 슬롯,
                    # 우리는 *현재 GPU 측 active* 측정.
                    _max_batch_size_o2 = self.scheduler_config.max_num_seqs
                    _running_only_o2 = sum(
                        1 for _r in self.running
                        if _r.status == _RS_d4.RUNNING
                    )
                    _remaining_slots_o2 = max(
                        0, _max_batch_size_o2 - _running_only_o2
                    )
                    _max_swap_in = min(_max_swap_in, _remaining_slots_o2)
                    if (_max_swap_in == 0 and _remaining_slots_o2 == 0
                            and not getattr(
                                self, "_neo_option_o2_logged", False)):
                        logger.info(
                            "[Option O2 v2] D4 budget guard fire — "
                            "remaining_slots=%d running_only=%d "
                            "max_seqs=%d mirror=%d total_running=%d",
                            _remaining_slots_o2, _running_only_o2,
                            _max_batch_size_o2, len(_mirror_d4),
                            len(self.running),
                        )
                        self._neo_option_o2_logged = True
                    if _max_swap_in == 0 and not getattr(
                            self, "_neo_option_i_skip_logged", False):
                        logger.info(
                            "[Option I] mirror buffer 유지 — swap_in skip "
                            "first fire (mirror_size=%d MIN_BUFFER=%d)",
                            len(_mirror_d4), _min_buffer,
                        )
                        self._neo_option_i_skip_logged = True
                    _mirror_order = _os_th.environ.get(
                        "VLLM_NEO_SWAP_IN_ORDER", "oldest"
                    ).lower()
                    if _max_swap_in == 0:
                        _candidates = []
                    elif _mirror_order == "newest":
                        _candidates = list(_mirror_d4)[-_max_swap_in:]
                    else:
                        _candidates = list(_mirror_d4)[:_max_swap_in]
                    _swap_in_ids: list[str] = []
                    _failed_pool = 0
                    for _rid in _candidates:
                        _req = self.requests.get(_rid)
                        if _req is None:
                            _mirror_d4.discard(_rid)
                            continue
                        if _req.status != _RS_d4.SWAPPED_OUT:
                            _mirror_d4.discard(_rid)
                            continue
                        if self._neo_swap_in(_req, _now_d4):
                            _mirror_d4.discard(_rid)
                            _swap_in_ids.append(_rid)
                            if _req not in self.running:
                                self.running.append(_req)
                        else:
                            _failed_pool += 1
                            break
                    if _swap_in_ids:
                        try:
                            output.neo_swap_in_req_ids = list(_swap_in_ids)
                            if not getattr(self, "_neo_swap_in_attached_logged", False):
                                logger.info(
                                    "[Plan v4 D4] swap_in attach first fire: "
                                    "ids=%d (pool_failed=%d, kv_usage=%.3f, "
                                    "threshold=%.3f, mirror_remaining=%d)",
                                    len(_swap_in_ids), _failed_pool,
                                    _kv_usage_d4, _SWAP_IN_THRESHOLD,
                                    len(_mirror_d4),
                                )
                                self._neo_swap_in_attached_logged = True
                        except (AttributeError, TypeError) as _ae:
                            logger.debug(
                                "Plan v4 D4 swap_in attach fail: %s", _ae,
                            )
                        # [Plan v4 D7] try61 race fix — swap_in 발화한 reqs 는
                        # 같은 step 의 cdec dispatch 대상에서 차감. 동일 req
                        # 가 swap_in (KV CPU→GPU) + cdec (CPU buffer 읽기) 동시
                        # 진행 시 KV partial state race → pacpu store_kv SEGV
                        # (try61 root). swap_in 후 status=RUNNING 으로 전환됐으니
                        # 다음 step 에서는 cdec_ids 자연 미포함 — 본 fix 는
                        # 본 step 의 이미 attach 된 sub_batches 만 patch.
                        try:
                            _swap_in_set_d7 = set(_swap_in_ids)
                            _sb = getattr(output, "neo_sub_batches", None)
                            if _sb and len(_sb) >= 2:
                                _new_b1 = [
                                    rid for rid in _sb[1]
                                    if rid not in _swap_in_set_d7
                                ]
                                _removed_d7 = len(_sb[1]) - len(_new_b1)
                                if _removed_d7 > 0:
                                    output.neo_sub_batches = [_sb[0], _new_b1]
                                    _ci = getattr(
                                        output, "neo_sub_batch_cdec_req_ids", None
                                    )
                                    if _ci and len(_ci) >= 2:
                                        output.neo_sub_batch_cdec_req_ids = [
                                            _ci[0],
                                            [rid for rid in _ci[1]
                                             if rid not in _swap_in_set_d7],
                                        ]
                                    _cs = getattr(
                                        output, "neo_sub_batch_cdec_slices", None
                                    )
                                    if _cs and len(_cs) >= 2:
                                        _b1cs = _cs[1]
                                        output.neo_sub_batch_cdec_slices = [
                                            _cs[0],
                                            (_b1cs[0],
                                             max(_b1cs[0], _b1cs[1] - _removed_d7)),
                                        ]
                                    _ss = getattr(
                                        output,
                                        "neo_sub_batch_cdec_seq_slices",
                                        None,
                                    )
                                    if _ss and len(_ss) >= 2:
                                        _b1ss = _ss[1]
                                        output.neo_sub_batch_cdec_seq_slices = [
                                            _ss[0],
                                            (_b1ss[0],
                                             max(_b1ss[0], _b1ss[1] - _removed_d7)),
                                        ]
                                    if not getattr(
                                            self,
                                            "_neo_d7_subtract_logged",
                                            False):
                                        logger.info(
                                            "[Plan v4 D7] swap_in/cdec race "
                                            "guard first fire: subtracted=%d "
                                            "swap_in_ids from sub_batch[1]/"
                                            "cdec_req_ids/slices",
                                            _removed_d7,
                                        )
                                        self._neo_d7_subtract_logged = True
                        except (AttributeError, TypeError, IndexError) as _de:
                            logger.debug(
                                "Plan v4 D7 cdec subtract exception: %s", _de,
                            )
            except Exception as _e:  # noqa: BLE001
                logger.debug(
                    "Plan v4 D4 swap_in eval exception: %s", _e,
                )

        if False and (_os_th.environ.get("VLLM_NEO_DISABLE_SWAP_IN") != "1"
                and hasattr(self, "_neo_deferred_free_reqs")
                and self._neo_deferred_free_reqs):
            try:
                # [Phase B-1 fix v2] KV usage check 제거. vllm preempt path
                # 가 deferred 에 append 시 KV 는 *아직 GPU 에 살아있음* —
                # _handle_neo_swaps 의 deferred preempt loop 가 free 함.
                # 따라서 본 시점의 kv_usage 는 *pre-free* 수치 (높음).
                # threshold check 부여 시 swap_in 영영 미발화.
                # 본 path 의 안전성 = 매 step max N reqs 만 rescue (per-step cap)
                # → KV 보존 우선 + 나머지는 vanilla preempt.
                if True:
                    # Cap per-step swap_in to avoid oscillation.
                    _max_swap_in = int(_os_th.environ.get(
                        "VLLM_NEO_MAX_SWAP_IN_PER_STEP", "8"
                    ))
                    _deferred = self._neo_deferred_free_reqs
                    _swap_in_count = min(len(_deferred), _max_swap_in)
                    if _swap_in_count > 0:
                        # [TSK_019 v3 / Phase B-3] LRU policy stub —
                        # ordering 선택. NEO 표준은 LRU based bidirectional
                        # 이지만 본 plan 의 swap_in 은 deferred-only (KV 보존)
                        # 영역이라 ordering 의 의미가 다름:
                        # - 'newest' (default): 가장 최근 swap_out → freshest
                        #   KV → 안전. NEO 의 stack 기반 안전 path.
                        # - 'oldest': head from deferred → 가장 오래 기다린
                        #   reqs 우선 → fairness. NEO 표준 LRU 와 가까움.
                        # Kill switch / fallback: VLLM_NEO_LRU_FALLBACK_FIFO=1
                        # 시 무조건 newest (기존 동작).
                        _swap_in_order = _os_th.environ.get(
                            "VLLM_NEO_SWAP_IN_ORDER", "newest"
                        ).lower()
                        if (_swap_in_order == "oldest"
                                and _os_th.environ.get(
                                    "VLLM_NEO_LRU_FALLBACK_FIFO"
                                ) != "1"):
                            # Pop from head — oldest first (NEO LRU 정합).
                            _swap_in_candidates = _deferred[:_swap_in_count]
                        else:
                            # Default newest first — freshest KV / 가장 안전.
                            _swap_in_candidates = _deferred[-_swap_in_count:]
                        _swap_in_ids = [
                            r.request_id for r in _swap_in_candidates
                        ]
                        try:
                            output.neo_swap_in_req_ids = list(_swap_in_ids)
                            logger.info(
                                "[NEO SWAP_IN] candidates=%d kv_usage=%.3f "
                                "threshold=%.3f ids=%s",
                                _swap_in_count, _kv_usage_now,
                                _SWAP_IN_THRESHOLD,
                                _swap_in_ids[:5],
                            )
                        except (AttributeError, TypeError) as _ae:
                            logger.debug(
                                "NEO swap_in attach failed: %s", _ae,
                            )
            except Exception as _e:  # noqa: BLE001
                logger.debug(
                    "NEO swap_in evaluation exception: %s", _e,
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
