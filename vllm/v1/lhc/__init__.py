"""LHC (Lane-Separated Host Coprocessor) — Phase 2 PoC module.

Provides the DSA host-memcpy lane via ``dsa_lane`` and a runtime gate
controlled by ``VLLM_LHC_DSA`` / ``VLLM_LHC_AMX_LOGITS``.

Phase 2 status (2026-06-08):
  - DSA lane: PoC integrated, single-engine 31 GB/s, CPU stall 0%.
    Used for selected host-side memcpy paths (KV staging fan-out).
  - AMX logits-head lane: gate FAIL (~70-100x vs B200). Disabled.
"""

from vllm.v1.lhc.dsa_lane import (
    dsa_lane_available,
    dsa_memcpy,
    dsa_lane_stats,
    dsa_lane_close,
)
from vllm.v1.lhc.regime_detector import (
    WorkloadRegime,
    classify,
    current_regime,
    note_kv_usage,
    note_step,
    note_swap_event,
    should_use_amx_c3,
    should_use_dsa,
    stats_snapshot,
)

__all__ = [
    "dsa_lane_available",
    "dsa_memcpy",
    "dsa_lane_stats",
    "dsa_lane_close",
    # Option C regime detector
    "WorkloadRegime",
    "classify",
    "current_regime",
    "note_kv_usage",
    "note_step",
    "note_swap_event",
    "should_use_amx_c3",
    "should_use_dsa",
    "stats_snapshot",
]
