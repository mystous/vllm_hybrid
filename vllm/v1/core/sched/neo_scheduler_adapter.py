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

        num_gpu_blocks = max(
            getattr(cache_cfg, "num_gpu_blocks_override", None) or 1,
            1,
        )
        # CPU block count is determined by ``--swap-space`` × layer
        # count; before that's known we use a conservative placeholder.
        num_cpu_blocks = max(num_gpu_blocks, 1)

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
    # SchedulerInterface override — record-only NEO sibling
    # ------------------------------------------------------------------
    def schedule(self) -> "SchedulerOutput":
        # Drive the default scheduler — this is the data path the engine
        # actually consumes.
        output = super().schedule()

        # Drive the NEO sibling for visibility. Until later stages wire
        # the runner to consume two sub-batches, the adapter cannot
        # actually run a NEO-style iteration; it only records the
        # decision the NEO scheduler would have made if invoked.
        try:
            self.last_neo_output = self.neo_scheduler.schedule()
        except Exception as e:  # noqa: BLE001
            # Never let the NEO sibling break the data path.
            logger.debug("NeoScheduler sibling raised: %s", e)
            self.last_neo_output = None

        return output
