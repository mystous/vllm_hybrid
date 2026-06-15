# SPDX-License-Identifier: Apache-2.0
"""METRONOME-LHC — five-stage orchestrator over LHC infrastructure.

Stages (mapped to paper § 06):
  - TEMPO     : PMU-driven lane load monitor (LLC.MISS, MEM_BW, DSA queue,
                AMX thread util) at 100 Hz, ring-buffered for downstream
                stages.
  - CHORD     : Cross-lane producer-consumer scheduling — GPU producer
                (logits, KV) chained to CPU consumer (sampler, scatter,
                detok) one step ahead.
  - METER     : NUMA-topology-aware pinning — TP rank ↔ NUMA node ↔ DSA
                WQ ↔ AMX thread fan-out, NVLink-aware.
  - ACCENT    : Adaptive lane allocation under memory pressure — DSA
                bandwidth budget shrunk / expanded based on LLC-miss
                spike + queue depth.
  - RITARDANDO: Graceful fallback on lane saturation — switches DSA off
                / AMX off without engine crash, restores when load
                normalizes.

The orchestrator is process-local (per worker). Activate with
``VLLM_LHC_METRONOME=1`` env. When inactive, the orchestrator is a no-op
and each lane behaves like Phase 3 (DSA + AMX C3 stand-alone).
"""

from vllm.v1.lhc.metronome.orchestrator import (
    metronome_active,
    metronome_start,
    metronome_step_end,
    metronome_stop,
    metronome_stats,
)

__all__ = [
    "metronome_active",
    "metronome_start",
    "metronome_step_end",
    "metronome_stop",
    "metronome_stats",
]
