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
        try:
            self.neo_scheduler.on_requests_arrival([
                _NeoRequestView(request)
            ])
        except Exception as e:  # noqa: BLE001
            logger.debug("NEO sibling add_request failed: %s", e)

    def finish_requests(self, request_ids, finished_status) -> None:  # type: ignore[override]
        super().finish_requests(request_ids, finished_status)
        try:
            wrappers = [
                _NeoRequestView.from_id(rid) for rid in request_ids
            ]
            self.neo_scheduler.remove_finished_requests(wrappers)
        except Exception as e:  # noqa: BLE001
            logger.debug("NEO sibling finish_requests failed: %s", e)

    # ------------------------------------------------------------------
    # schedule — drive both the default and NEO schedulers, attach the
    # NEO decision to the SchedulerOutput for the runner to consume.
    # ------------------------------------------------------------------
    def schedule(self) -> "SchedulerOutput":
        # Drive the default scheduler — this is the data path the engine
        # actually consumes for vanilla operation.
        output = super().schedule()

        # Drive the NEO sibling. The decision is *attached* to the
        # SchedulerOutput so that the GPU model runner can consume it
        # in subsequent stages (Step 5.3+).
        try:
            self.last_neo_output = self.neo_scheduler.schedule()
        except Exception as e:  # noqa: BLE001
            logger.debug("NeoScheduler sibling raised: %s", e)
            self.last_neo_output = None

        if self.last_neo_output is not None:
            try:
                output.neo_sub_batches = [
                    [r.request_id for r in batch.all_reqs]
                    for batch in self.last_neo_output.batches
                ]
                output.neo_swap_in_req_ids = [
                    r.request_id for r in self.last_neo_output.swap_in_reqs
                ]
                output.neo_swap_out_req_ids = [
                    r.request_id for r in self.last_neo_output.swap_out_reqs
                ]
            except (AttributeError, TypeError) as e:
                # Defensive: never break the data path on attachment failure.
                logger.debug("NEO output attach failed: %s", e)

        return output
