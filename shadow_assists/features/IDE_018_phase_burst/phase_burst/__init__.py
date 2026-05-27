"""IDE_018 — Sub-Layer Phase-Aware CPU Burst (paper main contribution).

High-level Python API over the C++ scheduler core.

Typical usage in vllm/v1/worker/gpu_model_runner.py forward path:

    from phase_burst import PhaseBurstRuntime, Phase

    rt = PhaseBurstRuntime.global_instance(num_workers=20, cpu_base=80)

    # at phase boundaries (CUDA stream synchronized):
    rt.mark_phase(Phase.ATTENTION, step_id)
    ...
    rt.mark_phase(Phase.LINEAR, step_id)
    ...
    rt.mark_phase(Phase.SAMPLE, step_id)

Tasks (A-J from paper Table 1a/1b) are enqueued via:

    rt.enqueue_python(TaskKind.B_DETOKENIZE, step_id, MASK_ATTN, fn)

The C++ workers pinned to cpu_base..cpu_base+num_workers-1 pick up tasks
whose applicable_phases bitmask matches the current PhaseSignal.
"""

from __future__ import annotations

import os
import threading
from typing import Callable, Optional

try:
    from . import _core  # type: ignore[attr-defined]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "phase_burst._core not built. "
        "Run cmake --build in IDE_018_phase_burst/build first."
    ) from exc

Phase = _core.Phase
TaskKind = _core.TaskKind
PhaseSignal = _core.PhaseSignal
PhaseBurstScheduler = _core.PhaseBurstScheduler
PhaseBurstStats = _core.PhaseBurstStats

MASK_ATTN = _core.MASK_ATTN
MASK_LINEAR = _core.MASK_LINEAR
MASK_SAMPLE = _core.MASK_SAMPLE
MASK_TP_AR = _core.MASK_TP_AR
MASK_IDLE = _core.MASK_IDLE
MASK_POST = _core.MASK_POST
MASK_ANY = _core.MASK_ANY


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "1" if default else "0")
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


class PhaseBurstRuntime:
    """Global per-process runtime.

    ENV controls:
      VLLM_USE_PHASE_BURST=1            enable
      VLLM_PHASE_BURST_NUM_WORKERS=20   worker count (default 20)
      VLLM_PHASE_BURST_CPU_BASE=80      first pinned cpu (default 80)
    """

    _instance: Optional["PhaseBurstRuntime"] = None
    _lock = threading.Lock()

    def __init__(self, num_workers: int = 20, cpu_base: int = 80) -> None:
        self.num_workers = num_workers
        self.cpu_base = cpu_base
        self.signal: PhaseSignal = _core.get_global_signal()
        self.scheduler = PhaseBurstScheduler(self.signal, num_workers, cpu_base)
        self.started = False
        # SUB_184 dummy-fill config (ENV-gated)
        self._dummy_fill_enabled = _env_bool("VLLM_PHASE_BURST_DUMMY_FILL", False)
        self._dummy_count = _env_int("VLLM_PHASE_BURST_DUMMY_COUNT", num_workers)
        self._dummy_iters = _env_int("VLLM_PHASE_BURST_DUMMY_ITERS", 8)

    # ── lifecycle ───────────────────────────────────────────────────
    def start(self) -> None:
        if self.started:
            return
        self.scheduler.start()
        self.started = True

    def stop(self) -> None:
        if not self.started:
            return
        self.scheduler.stop()
        self.started = False

    # ── singleton ───────────────────────────────────────────────────
    @classmethod
    def global_instance(
        cls,
        num_workers: Optional[int] = None,
        cpu_base: Optional[int] = None,
    ) -> "PhaseBurstRuntime":
        with cls._lock:
            if cls._instance is None:
                nw = num_workers if num_workers is not None \
                    else _env_int("VLLM_PHASE_BURST_NUM_WORKERS", 20)
                cb = cpu_base if cpu_base is not None \
                    else _env_int("VLLM_PHASE_BURST_CPU_BASE", 80)
                cls._instance = cls(nw, cb)
                cls._instance.start()
            return cls._instance

    @classmethod
    def shutdown_global(cls) -> None:
        with cls._lock:
            if cls._instance is not None:
                cls._instance.stop()
                cls._instance = None
                _core.release_global_signal()

    # ── phase marking ───────────────────────────────────────────────
    def mark_phase(self, phase: int, step_id: int = 0) -> None:
        """Update phase signal. Called from vLLM forward thread *after*
        cuda stream sync (so phase reflects what GPU is actually doing).

        SUB_184 dummy-fill: when VLLM_PHASE_BURST_DUMMY_FILL=1, the runtime
        also enqueues N heavy C++ dummy tasks on each ATTENTION / LINEAR
        phase mark — to validate the paper §4 overlap hypothesis (CPU util
        ↑ without throughput loss) before wiring real AVX-512 kernels.
        """
        self.signal.update(int(phase), int(step_id))
        if self._dummy_fill_enabled:
            self._maybe_enqueue_dummy(int(phase), int(step_id))

    def _maybe_enqueue_dummy(self, phase: int, step_id: int) -> None:
        # ATTENTION (=PHASE_ATTENTION=1) → attention burst
        # LINEAR    (=PHASE_LINEAR=2)    → linear burst
        try:
            if phase == int(Phase.ATTENTION):
                _core.enqueue_dummy_attention_burst(
                    self.scheduler, step_id,
                    self._dummy_count, self._dummy_iters)
            elif phase == int(Phase.LINEAR):
                _core.enqueue_dummy_linear_burst(
                    self.scheduler, step_id,
                    self._dummy_count, self._dummy_iters)
        except Exception:
            # never let dummy-fill break the forward path.
            pass

    # ── enqueue ────────────────────────────────────────────────────
    def enqueue_python(
        self,
        kind,
        step_id: int,
        applicable_phases: int,
        fn: Callable[[], None],
    ) -> None:
        # pybind11 enum (TaskKind) accepts itself, not int. coerce ints to enum.
        if not isinstance(kind, TaskKind):
            kind = TaskKind(int(kind))
        self.scheduler.enqueue_python_callable(
            kind, int(step_id), int(applicable_phases), fn
        )

    # ── stats ──────────────────────────────────────────────────────
    def snapshot_stats(self) -> PhaseBurstStats:
        return self.scheduler.snapshot_stats()


def is_enabled() -> bool:
    return _env_bool("VLLM_USE_PHASE_BURST", False)


__all__ = [
    "Phase",
    "TaskKind",
    "PhaseSignal",
    "PhaseBurstScheduler",
    "PhaseBurstStats",
    "PhaseBurstRuntime",
    "MASK_ATTN",
    "MASK_LINEAR",
    "MASK_SAMPLE",
    "MASK_TP_AR",
    "MASK_IDLE",
    "MASK_POST",
    "MASK_ANY",
    "is_enabled",
]
