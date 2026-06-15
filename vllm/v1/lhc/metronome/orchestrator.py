# SPDX-License-Identifier: Apache-2.0
"""METRONOME-LHC orchestrator — wires the five stages.

Lifecycle:
  - ``metronome_start(rank, tp_size)`` — once at worker init.
      * TEMPO sampler daemon started.
      * METER NUMA pin applied.
  - ``metronome_step_end()`` — once per scheduler step (end-of-step hook).
      * ACCENT re-evaluates DSA budget.
      * RITARDANDO checks for saturation, may pause lanes.
  - ``metronome_stop()`` — at worker shutdown.

Activation: ``VLLM_LHC_METRONOME=1``. When unset, all stages return None /
no-op and behaviour is identical to Phase 3 (lanes available independently).
"""

from __future__ import annotations

import logging
import os
import threading

from vllm.v1.lhc.metronome import accent as _accent
from vllm.v1.lhc.metronome import chord as _chord
from vllm.v1.lhc.metronome import meter as _meter
from vllm.v1.lhc.metronome import ritardando as _ritardando
from vllm.v1.lhc.metronome import tempo as _tempo

logger = logging.getLogger(__name__)

_lock = threading.RLock()
_active = False
_started_rank: int | None = None


def metronome_active() -> bool:
    """Hot-path check for whether the orchestrator is running. Called
    from swap-out / scatter sites to decide between metronome-aware
    behaviour and standalone-lane behaviour."""
    return _active


def metronome_start(rank: int = 0, tp_size: int = 1) -> bool:
    """Idempotent. Returns True if active after the call. Env gate
    ``VLLM_LHC_METRONOME=1`` must be set."""
    global _active, _started_rank
    if os.environ.get("VLLM_LHC_METRONOME", "0") != "1":
        return False
    with _lock:
        if _active:
            return True
        _tempo.tempo_start()
        ok, info = _meter.meter_pin_rank(rank, tp_size)
        logger.info(
            "METRONOME-LHC start: rank=%d tp=%d meter=%s (%s) chord=%s",
            rank, tp_size, ok, info, _chord.chord_enabled(),
        )
        _active = True
        _started_rank = rank
        return True


def metronome_step_end() -> None:
    """Per-step hook. Cheap (lock-free reads, single atomic counter)."""
    if not _active:
        return
    try:
        _accent.accent_step_end()
        _ritardando.ritardando_check()
    except Exception as e:  # noqa: BLE001
        logger.debug("metronome_step_end error: %s", e)


def metronome_stop() -> None:
    """Stop sampler + clear active flag. Safe to call multiple times."""
    global _active
    with _lock:
        if not _active:
            return
        _tempo.tempo_stop()
        _active = False


def metronome_stats() -> dict:
    """Aggregated diagnostics — feeds paper §08 table."""
    if not _active:
        return {"active": False}
    snap = _tempo.tempo_snapshot()
    return {
        "active": True,
        "rank": _started_rank,
        "tempo": {
            "ts": snap.ts,
            "llc_miss_rate": snap.llc_miss_rate,
            "mem_bw_mbs": snap.mem_bw_mbs,
            "dsa_queue_depth": snap.dsa_queue_depth,
            "amx_thread_util_pct": snap.amx_thread_util_pct,
            "cpu_total_pct": snap.cpu_total_pct,
        },
        "accent": _accent.accent_stats(),
        "ritardando": _ritardando.ritardando_stats(),
        "chord": _chord.chord_stats(),
    }
