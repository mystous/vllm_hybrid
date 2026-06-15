# SPDX-License-Identifier: Apache-2.0
"""RITARDANDO — graceful fallback on lane saturation (stage 5).

Saturation detection (any of the following for ``_SAT_WINDOW`` consecutive
samples):
  - DSA queue depth ≥ ``_DSA_SAT_THRESH`` (queue not draining)
  - AMX thread util ≥ ``_AMX_SAT_THRESH`` (worker pool fully busy)
  - CPU total ≥ ``_CPU_SAT_THRESH`` (host compute headroom gone)

On saturation, lanes are *paused* (a sticky off-bit). Pause auto-clears
after ``_PAUSE_TTL_SEC``. This is intentionally coarser than ACCENT:
ACCENT throttles per call, RITARDANDO halts a whole lane for a budgeted
TTL. Both can fire — RITARDANDO wins (overrides ACCENT).
"""

from __future__ import annotations

import logging
import os
import threading
import time

from vllm.v1.lhc.metronome.tempo import tempo_recent_avg

logger = logging.getLogger(__name__)

_DSA_SAT_THRESH = int(os.environ.get("VLLM_LHC_RIT_DSA_SAT", "32"))
_AMX_SAT_THRESH = float(os.environ.get("VLLM_LHC_RIT_AMX_SAT", "90.0"))
_CPU_SAT_THRESH = float(os.environ.get("VLLM_LHC_RIT_CPU_SAT", "95.0"))
_SAT_WINDOW = int(os.environ.get("VLLM_LHC_RIT_WINDOW", "8"))
_PAUSE_TTL_SEC = float(os.environ.get("VLLM_LHC_RIT_TTL", "2.0"))

_lock = threading.RLock()
_dsa_pause_until: float = 0.0
_amx_pause_until: float = 0.0
_pause_events: int = 0


def ritardando_check() -> None:
    """Evaluate saturation and possibly pause lanes. Called from
    orchestrator's step_end after ACCENT."""
    global _dsa_pause_until, _amx_pause_until, _pause_events
    snap = tempo_recent_avg(window=_SAT_WINDOW)
    now = time.time()
    with _lock:
        if snap.dsa_queue_depth >= _DSA_SAT_THRESH and now >= _dsa_pause_until:
            _dsa_pause_until = now + _PAUSE_TTL_SEC
            _pause_events += 1
            logger.warning(
                "RITARDANDO DSA pause: queue_depth=%d ≥ %d, ttl=%.1fs",
                snap.dsa_queue_depth, _DSA_SAT_THRESH, _PAUSE_TTL_SEC,
            )
        amx_sat = (
            snap.amx_thread_util_pct >= _AMX_SAT_THRESH
            or snap.cpu_total_pct >= _CPU_SAT_THRESH
        )
        if amx_sat and now >= _amx_pause_until:
            _amx_pause_until = now + _PAUSE_TTL_SEC
            _pause_events += 1
            logger.warning(
                "RITARDANDO AMX pause: amx_util=%.1f cpu=%.1f ttl=%.1fs",
                snap.amx_thread_util_pct, snap.cpu_total_pct, _PAUSE_TTL_SEC,
            )


def ritardando_dsa_paused() -> bool:
    return time.time() < _dsa_pause_until


def ritardando_amx_paused() -> bool:
    return time.time() < _amx_pause_until


def ritardando_stats() -> dict:
    return {
        "dsa_paused": ritardando_dsa_paused(),
        "amx_paused": ritardando_amx_paused(),
        "pause_events": _pause_events,
    }
