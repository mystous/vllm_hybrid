# SPDX-License-Identifier: Apache-2.0
"""ACCENT — adaptive lane allocation under memory pressure (stage 4).

Reads TEMPO snapshot + decides:
  - if DSA queue depth > thresh AND LLC miss rate spike → throttle DSA
    (return ``False`` from ``accent_dsa_allowed``), force CPU memcpy fallback.
  - if memory bandwidth headroom restored → re-enable DSA.

The decision is per-step (called at scheduler step end) and re-evaluated
every K steps to avoid hysteresis. The signal is consumed by ACCENT-aware
swap-out code (NEO buffer) via :func:`accent_dsa_allowed`.
"""

from __future__ import annotations

import logging
import os
import threading

from vllm.v1.lhc.metronome.tempo import tempo_recent_avg

logger = logging.getLogger(__name__)

# Thresholds (env-tunable). Defaults selected to be quiet on idle systems.
_DSA_QUEUE_THRESH = int(os.environ.get("VLLM_LHC_ACCENT_DSA_Q", "12"))
_DSA_RECOVERY_THRESH = int(os.environ.get("VLLM_LHC_ACCENT_DSA_REC", "4"))
_LLC_SPIKE_THRESH = float(os.environ.get("VLLM_LHC_ACCENT_LLC", "1e9"))
_MEM_BW_HEAVY_MBS = float(os.environ.get("VLLM_LHC_ACCENT_BW", "200000"))

_lock = threading.RLock()
_dsa_allowed: bool = True
_eval_counter: int = 0
_EVAL_EVERY = 16   # re-evaluate every N step_end calls

# Aggregate counters (paper-§08 fodder).
_dsa_throttle_events: int = 0
_dsa_recovery_events: int = 0


def accent_step_end() -> None:
    """Called at the end of each scheduler step. Updates ``_dsa_allowed``
    every ``_EVAL_EVERY`` steps based on a smoothed TEMPO sample."""
    global _eval_counter, _dsa_allowed
    global _dsa_throttle_events, _dsa_recovery_events
    with _lock:
        _eval_counter += 1
        if _eval_counter % _EVAL_EVERY != 0:
            return
        snap = tempo_recent_avg(window=16)
        prev = _dsa_allowed
        if _dsa_allowed:
            # Throttle when queue deep AND host memory bus pressure spikes.
            if (
                snap.dsa_queue_depth >= _DSA_QUEUE_THRESH
                and (snap.llc_miss_rate >= _LLC_SPIKE_THRESH
                     or snap.mem_bw_mbs >= _MEM_BW_HEAVY_MBS)
            ):
                _dsa_allowed = False
                _dsa_throttle_events += 1
        else:
            # Recover when queue clears.
            if snap.dsa_queue_depth <= _DSA_RECOVERY_THRESH:
                _dsa_allowed = True
                _dsa_recovery_events += 1
        if prev != _dsa_allowed:
            logger.info(
                "ACCENT lane switch: dsa_allowed %s → %s (q=%d, llc=%.2g, "
                "bw=%.0f MB/s)",
                prev, _dsa_allowed, snap.dsa_queue_depth,
                snap.llc_miss_rate, snap.mem_bw_mbs,
            )


def accent_dsa_allowed() -> bool:
    """Returns True if DSA lane should be used. Hot path — single load."""
    return _dsa_allowed


def accent_stats() -> dict:
    return {
        "dsa_allowed": _dsa_allowed,
        "throttle_events": _dsa_throttle_events,
        "recovery_events": _dsa_recovery_events,
        "eval_count": _eval_counter,
    }
