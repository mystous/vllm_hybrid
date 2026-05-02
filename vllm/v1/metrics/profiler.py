# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model profiler for the NEO-style PerfPredictor.

Algorithms adapted from NEO (https://github.com/NEO-MLSys25/NEO,
MLSys 2025, Apache 2.0). Only the algorithms are reused.

At engine startup, ``ModelProfiler.run()`` runs a small number of
artificial forward passes to measure ``linr_T(S)``, ``pref_T(S)``,
``gdec_T(N)``, and ``cdec_T(S, N)``. The resulting values are written
into a ``TablePerfPredictor`` which the scheduler uses to decide
between sequential and pipelined sub-batch modes.

NEO reference: ``swiftllm/server/profiler.py``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.sched.perfpredictor import TablePerfPredictor

logger = logging.getLogger(__name__)


class ModelProfiler:
    """Measure ``linr_T`` / ``pref_T`` / ``gdec_T`` / ``cdec_T`` at startup.

    ``measure_fn`` is a callback supplied by the engine that takes a
    single integer ``probe`` and a ``probe_kind`` string and returns
    the wall time (in milliseconds) of one forward iteration with the
    requested workload shape. The measurements are populated into the
    supplied ``TablePerfPredictor`` instance in place.

    For unit tests, ``measure_fn`` may be a stub that returns
    deterministic values; ``run()`` then fills the predictor without
    any actual GPU/CPU work.
    """

    def __init__(
        self,
        predictor: TablePerfPredictor,
        measure_fn: Callable[[str, int, int], float],
        *,
        nwarmup: int = 2,
        nrepeat: int = 3,
    ) -> None:
        self.predictor = predictor
        self.measure_fn = measure_fn
        self.nwarmup = nwarmup
        self.nrepeat = nrepeat

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _measure_avg(self, kind: str, p1: int, p2: int = 0) -> float:
        """Run ``nwarmup`` warmup iterations followed by ``nrepeat``
        timed iterations and return the average wall time (ms)."""
        for _ in range(self.nwarmup):
            self.measure_fn(kind, p1, p2)
        total = 0.0
        for _ in range(self.nrepeat):
            total += self.measure_fn(kind, p1, p2)
        return total / max(1, self.nrepeat)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Populate the predictor's four ``*_T_list`` arrays."""
        start = time.perf_counter()

        self.predictor.linr_T_list = [
            self._measure_avg("linr", S) for S in self.predictor.linr_S_list
        ]
        self.predictor.pref_T_list = [
            self._measure_avg("pref", S) for S in self.predictor.pref_S_list
        ]
        self.predictor.gdec_T_list = [
            self._measure_avg("gdec", N) for N in self.predictor.gdec_N_list
        ]

        agg = self.predictor.cdec_N_list_agg
        self.predictor.cdec_T_lists = []
        for S in self.predictor.cdec_S_list:
            row = [self._measure_avg("cdec", S, N) for N in agg]
            self.predictor.cdec_T_lists.append(row)

        # Launch overhead — measure with a 1-token linear probe.
        try:
            self.predictor.lnch_T = self._measure_avg("lnch", 1)
        except NotImplementedError:
            # Keep default of 0.8 ms.
            pass

        elapsed = time.perf_counter() - start
        logger.info(
            "ModelProfiler.run completed in %.2fs (linr=%d, pref=%d, "
            "gdec=%d, cdec=%dx%d)",
            elapsed,
            len(self.predictor.linr_S_list),
            len(self.predictor.pref_S_list),
            len(self.predictor.gdec_N_list),
            len(self.predictor.cdec_S_list),
            len(self.predictor.cdec_N_list_agg),
        )
