"""LHC Phase 4 — Option C: workload regime detector + dynamic LHC on/off gate.

Background
----------
Option A (always-on LHC) measured noise band (-0.34% mean, 95% CI containing 0)
on the sharegpt + conc=64 + max-tok=2048 baseline because that baseline is
GPU-saturated (GPU 96%+, KV cache <50%, NEO swap rate ~0/s). In that regime
the DSA hooks are never called, so the lane has no productive work and any
sampling/setup cost shows up as small negative regression.

Option C reverses this: classify the workload regime at runtime, and gate
LHC accordingly:

  GPU_SATURATED -> LHC OFF (no overhead, identical to vanilla)
  KV_HEAVY      -> LHC ON  (DSA lane handles host KV staging)
  BALANCED      -> LHC partial (AMX C3 only; DSA off)
  PREFIX_HOT    -> Path 1 ON (AMX C3 prefix hash chain; DSA off)
                   — Phase 4 conc256 follow-up regime for workloads with
                   sustained prefix-cache hit rate (code workload +10.67%).

Signals (sampled every N scheduler steps, default N=20):

  gpu_util          : torch.cuda fallback (nvidia-smi-style polling via
                      torch.cuda.utilization where available).
  kv_cache_usage    : provided by the scheduler hook (block pool free / total).
  neo_swap_rate     : exponentially-weighted swap_out events / second.
  prefix_hit_rate   : EWMA over per-request local prefix-cache hit ratio
                      (alpha=0.2; sampled at request admission).
  step_time_p50_ms  : reserved for future use (e.g. detect prefill-bound).

Threshold defaults are calibrated against the 100+ lever runs in Phase 3 +
Option A baselines. Tunable via env:

  VLLM_LHC_REGIME_GPU_THR        (default 0.90)  -> GPU util above => candidate sat
  VLLM_LHC_REGIME_KV_THR         (default 0.50)  -> KV pct below   => candidate sat
  VLLM_LHC_REGIME_SWAP_THR       (default 1.0)   -> swap/s below   => candidate sat
  VLLM_LHC_REGIME_KV_HEAVY       (default 0.75)  -> KV pct above   => kv_heavy
  VLLM_LHC_REGIME_SWAP_HEAVY     (default 10.0)  -> swap/s above   => kv_heavy
  VLLM_LHC_REGIME_PREFIX_HOT_THR (default 0.60)  -> hit rate above => prefix_hot
  VLLM_LHC_REGIME_INTERVAL       (default 20)    -> classify every N steps

Public API
----------
  WorkloadRegime          enum of regimes.
  classify(...)           pure function (testable, no globals).
  RegimeDetector          singleton — manages signal accumulators + cross-proc
                          shared state via ``multiprocessing.Value``.
  current_regime()        -> WorkloadRegime
  should_use_dsa()        -> bool   (LHC DSA lane gate)
  should_use_amx_c3()     -> bool   (AMX C3 standalone gate)
  note_swap_event(n=1)    -> None   (called from NEO swap hook)
  note_kv_usage(pct)      -> None   (called from scheduler step end)
  note_step(...)          -> None   (called once per scheduler step)
  stats_snapshot()        -> dict   (for paper figures / logging)

Gate semantics
--------------
``VLLM_LHC_REGIME_ADAPTIVE=1`` enables dynamic gating. When unset, the
``should_use_*()`` functions return the static value derived from the legacy
``VLLM_LHC_DSA`` / ``VLLM_LHC_AMX_C3`` envs (Option A behaviour preserved for
back-compat).
"""

from __future__ import annotations

import enum
import logging
import math
import multiprocessing as mp
import os
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class WorkloadRegime(enum.IntEnum):
    """Coarse workload classification used by the LHC gate.

    Encoded as IntEnum so the singleton state can be carried by a
    ``multiprocessing.Value('i', ...)`` shared across TP worker procs.

    PREFIX_HOT was added in Phase 4 conc256 follow-up after the +10.67% code
    workload finding: AMX C3 prefix hash chain wins iff the prefix-cache hit
    rate is high enough that the path 1 hook fires frequently. On other
    workloads (sonnet / chat / balanced) the rate is low, the hook does not
    fire, and the extra setup cost shows up as noise/negative.
    """

    UNKNOWN = 0
    GPU_SATURATED = 1   # LHC OFF
    KV_HEAVY = 2        # LHC ON  (DSA + AMX C3)
    BALANCED = 3        # LHC partial (AMX C3 only)
    PREFIX_HOT = 4      # LHC Path 1 ON  (AMX C3 prefix hash chain)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw not in ("0", "", "false", "False", "no", "NO")


# ----------------------------------------------------------------------------
# Pure classification function (testable; no globals).
# ----------------------------------------------------------------------------


def classify(
    gpu_util: float,
    kv_pct: float,
    swap_rate: float,
    prefix_hit_rate: float = 0.0,
    gpu_thr: float | None = None,
    kv_thr: float | None = None,
    swap_thr: float | None = None,
    kv_heavy: float | None = None,
    swap_heavy: float | None = None,
    prefix_hot_thr: float | None = None,
) -> WorkloadRegime:
    """Classify a workload regime from instantaneous runtime signals.

    Args follow the order documented at module level. Threshold args default
    to env (or hard-coded defaults) when None.

    The rule lattice (in order — first match wins):

      1. KV_HEAVY        kv_pct > KV_HEAVY or swap_rate > SWAP_HEAVY
      2. PREFIX_HOT      gpu_util > GPU_THR and prefix_hit_rate > PREFIX_HOT_THR
                         (GPU saturated AND the prefix-cache hook will fire
                         every iteration → AMX C3 Path 1 has productive work)
      3. GPU_SATURATED   gpu_util > GPU_THR and kv_pct < KV_THR
                                            and swap_rate < SWAP_THR
      4. BALANCED        (default)

    Notes:
      - KV_HEAVY takes priority because once spill starts the DSA lane has
        productive work even when GPU util is still high (overlap regime).
      - PREFIX_HOT is layered between KV_HEAVY and GPU_SATURATED: it requires
        GPU saturation (otherwise CPU offload via AMX has no GPU bottleneck to
        relieve) plus a prefix-cache hit rate above the threshold (otherwise
        the AMX C3 hook is idle and adds noise). Phase 4 conc256: code workload
        sits here (+10.67% with Path 1); sonnet/chat sit in GPU_SATURATED.
      - GPU_SATURATED requires ALL three conditions so a noisy moment of
        high GPU util does not flip the gate off mid-spill.
    """
    if gpu_thr is None:
        gpu_thr = _env_float("VLLM_LHC_REGIME_GPU_THR", 0.90)
    if kv_thr is None:
        kv_thr = _env_float("VLLM_LHC_REGIME_KV_THR", 0.50)
    if swap_thr is None:
        swap_thr = _env_float("VLLM_LHC_REGIME_SWAP_THR", 1.0)
    if kv_heavy is None:
        kv_heavy = _env_float("VLLM_LHC_REGIME_KV_HEAVY", 0.75)
    if swap_heavy is None:
        swap_heavy = _env_float("VLLM_LHC_REGIME_SWAP_HEAVY", 10.0)
    if prefix_hot_thr is None:
        prefix_hot_thr = _env_float("VLLM_LHC_REGIME_PREFIX_HOT_THR", 0.60)

    if kv_pct > kv_heavy or swap_rate > swap_heavy:
        return WorkloadRegime.KV_HEAVY
    if gpu_util > gpu_thr and prefix_hit_rate > prefix_hot_thr:
        return WorkloadRegime.PREFIX_HOT
    if gpu_util > gpu_thr and kv_pct < kv_thr and swap_rate < swap_thr:
        return WorkloadRegime.GPU_SATURATED
    return WorkloadRegime.BALANCED


# ----------------------------------------------------------------------------
# Singleton detector — owns the EWMA accumulators + the mp-shared regime int.
# ----------------------------------------------------------------------------


@dataclass
class _State:
    last_step_ts: float = 0.0
    step_count: int = 0
    # EWMA(swap events / sec) with alpha=0.3.
    swap_rate_ewma: float = 0.0
    # Last instantaneous samples (for stats_snapshot).
    last_gpu_util: float = 0.0
    last_kv_pct: float = 0.0
    last_classify_step: int = -1
    # Prefix-cache hit rate EWMA (alpha=0.2 — slower because hit rate is more
    # of a workload property than a per-step burst). Updated via
    # note_prefix_hit(local_hits, total_prompt_tokens) or
    # note_prefix_hit_rate(rate).
    prefix_hit_rate_ewma: float = 0.0
    prefix_hit_sample_count: int = 0
    # Histogram of classifications over the run (for regime accuracy report).
    hist: dict[int, int] = field(default_factory=dict)


class RegimeDetector:
    """Process-singleton (per worker proc); cross-proc state via mp.Value."""

    _instance: "RegimeDetector | None" = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._state = _State(last_step_ts=time.monotonic())
        self._lock = threading.Lock()
        # Cross-proc gate state — mp.Value('i') so TP workers see updates
        # the EngineCore proc writes. Encoded as WorkloadRegime int value.
        self._shared = mp.Value("i", int(WorkloadRegime.UNKNOWN))
        self._interval = max(1, _env_int("VLLM_LHC_REGIME_INTERVAL", 20))
        self._adaptive = _env_flag("VLLM_LHC_REGIME_ADAPTIVE", False)
        logger.info(
            "[LHC Regime] init: adaptive=%s, interval=%d steps",
            self._adaptive, self._interval,
        )
        # Dump the final regime distribution at interpreter shutdown so the
        # measurement wrapper can extract PREFIX_HOT vs GPU_SATURATED ratios
        # post-mortem from the boot log.
        import atexit
        atexit.register(self._dump_atexit)

    def _dump_atexit(self) -> None:
        try:
            snap = self.stats_snapshot()
            logger.info("[LHC Regime] final stats_snapshot=%s", snap)
        except Exception:  # noqa: BLE001
            pass

    @classmethod
    def instance(cls) -> "RegimeDetector":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = RegimeDetector()
            return cls._instance

    # --- signal accumulators ------------------------------------------------

    def note_swap_event(self, n: int = 1) -> None:
        """Record N swap_out events. Updates EWMA rate."""
        if n <= 0:
            return
        with self._lock:
            now = time.monotonic()
            dt = max(now - self._state.last_step_ts, 1e-3)
            rate = n / dt
            # EWMA alpha=0.3
            self._state.swap_rate_ewma = (
                0.3 * rate + 0.7 * self._state.swap_rate_ewma
            )

    def note_kv_usage(self, kv_pct: float) -> None:
        with self._lock:
            self._state.last_kv_pct = max(0.0, min(1.0, float(kv_pct)))

    def note_prefix_hit(self, local_hits: int, total_prompt_tokens: int) -> None:
        """Record a per-request prefix-cache outcome.

        Called from the scheduler whenever a new request enters with a known
        ``num_new_local_computed_tokens`` (local hits) and
        ``request.num_prompt_tokens``. The detector tracks an EWMA over per-
        request hit ratios (alpha=0.2 — workload-level signal, not bursty).
        """
        if total_prompt_tokens <= 0:
            return
        rate = max(0.0, min(1.0, local_hits / float(total_prompt_tokens)))
        with self._lock:
            self._state.prefix_hit_rate_ewma = (
                0.2 * rate + 0.8 * self._state.prefix_hit_rate_ewma
            )
            self._state.prefix_hit_sample_count += 1

    def note_prefix_hit_rate(self, rate: float) -> None:
        """Inject an externally-sampled aggregate prefix-cache hit rate."""
        rate = max(0.0, min(1.0, float(rate)))
        with self._lock:
            self._state.prefix_hit_rate_ewma = (
                0.2 * rate + 0.8 * self._state.prefix_hit_rate_ewma
            )
            self._state.prefix_hit_sample_count += 1

    def note_step(self, gpu_util: float | None = None) -> None:
        """Called from scheduler at end of step. Triggers classify every
        ``VLLM_LHC_REGIME_INTERVAL`` steps."""
        with self._lock:
            self._state.step_count += 1
            self._state.last_step_ts = time.monotonic()
            if gpu_util is not None:
                self._state.last_gpu_util = max(0.0, min(1.0, float(gpu_util)))
            # Decay swap rate over time even when no swap events arrive
            # (otherwise EWMA stays high after the spill subsides).
            self._state.swap_rate_ewma *= 0.95
            if self._state.step_count - self._state.last_classify_step \
                    >= self._interval:
                regime = self._classify_unlocked()
                self._state.last_classify_step = self._state.step_count
                self._shared.value = int(regime)
                self._state.hist[int(regime)] = (
                    self._state.hist.get(int(regime), 0) + 1
                )

    def _classify_unlocked(self) -> WorkloadRegime:
        gpu_util = self._state.last_gpu_util
        # If gpu_util was never reported, attempt cheap probe (torch.cuda
        # utilization() works on most CUDA driver versions; on failure
        # leave at 0 which routes us to BALANCED in absence of swap).
        if gpu_util == 0.0:
            gpu_util = _probe_gpu_util_safe()
            self._state.last_gpu_util = gpu_util
        return classify(
            gpu_util=gpu_util,
            kv_pct=self._state.last_kv_pct,
            swap_rate=self._state.swap_rate_ewma,
            prefix_hit_rate=self._state.prefix_hit_rate_ewma,
        )

    # --- gate queries -------------------------------------------------------

    def current_regime(self) -> WorkloadRegime:
        return WorkloadRegime(self._shared.value)

    def should_use_dsa(self) -> bool:
        """DSA gate. In adaptive mode: on iff KV_HEAVY.
        In static mode (default): obey VLLM_LHC_DSA.
        """
        if not self._adaptive:
            return _env_flag("VLLM_LHC_DSA", False)
        return self.current_regime() == WorkloadRegime.KV_HEAVY

    def should_use_amx_c3(self) -> bool:
        """AMX C3 gate. In adaptive mode: on iff KV_HEAVY or BALANCED
        (deliberately NOT in PREFIX_HOT — PREFIX_HOT routes through the
        prefix hash chain hook, not the C3 KV path).
        In static mode (default): obey VLLM_LHC_AMX_C3.
        """
        if not self._adaptive:
            return _env_flag("VLLM_LHC_AMX_C3", False)
        regime = self.current_regime()
        return regime in (
            WorkloadRegime.KV_HEAVY, WorkloadRegime.BALANCED,
        )

    def should_use_amx_c3_prefix(self) -> bool:
        """AMX C3 prefix-hash gate (Path 1).

        In adaptive mode: on iff PREFIX_HOT (workload exhibits sustained
        prefix-cache hit rate above threshold). This is the gate added in
        Phase 4 conc256 after observing that the +10.67% Path 1 win
        on code workloads disappears or inverts on sonnet/chat where the
        hash hook fires only on cold prefills.

        In static mode (default): obey VLLM_LHC_AMX_C3_PREFIX. This preserves
        the existing Phase 4 / step3_path1.sh behaviour.
        """
        if not self._adaptive:
            return _env_flag("VLLM_LHC_AMX_C3_PREFIX", False)
        return self.current_regime() == WorkloadRegime.PREFIX_HOT

    def stats_snapshot(self) -> dict:
        with self._lock:
            total = sum(self._state.hist.values()) or 1
            dist = {
                WorkloadRegime(k).name: v / total
                for k, v in self._state.hist.items()
            }
            return {
                "adaptive": self._adaptive,
                "interval": self._interval,
                "step_count": self._state.step_count,
                "last_gpu_util": self._state.last_gpu_util,
                "last_kv_pct": self._state.last_kv_pct,
                "swap_rate_ewma": self._state.swap_rate_ewma,
                "prefix_hit_rate_ewma": self._state.prefix_hit_rate_ewma,
                "prefix_hit_sample_count": self._state.prefix_hit_sample_count,
                "current_regime": self.current_regime().name,
                "classify_count": sum(self._state.hist.values()),
                "regime_distribution": dist,
            }


def _probe_gpu_util_safe() -> float:
    """Best-effort GPU util probe via torch.cuda. Returns 0.0 on failure.

    Costs ~ a few hundred microseconds; called at most every
    VLLM_LHC_REGIME_INTERVAL steps (default 20) so amortized overhead is
    well below the per-step budget.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        # torch.cuda.utilization() returns a percent int (0-100); on driver
        # versions where pynvml is missing it returns -1 / raises.
        u = torch.cuda.utilization()
        if u < 0:
            return 0.0
        return float(u) / 100.0
    except Exception as e:  # noqa: BLE001
        if not hasattr(_probe_gpu_util_safe, "_warned"):
            logger.info("[LHC Regime] gpu_util probe failed: %r — fallback 0",
                        e)
            _probe_gpu_util_safe._warned = True  # type: ignore[attr-defined]
        return 0.0


# Convenience top-level functions ---------------------------------------------


def current_regime() -> WorkloadRegime:
    return RegimeDetector.instance().current_regime()


def should_use_dsa() -> bool:
    return RegimeDetector.instance().should_use_dsa()


def should_use_amx_c3() -> bool:
    return RegimeDetector.instance().should_use_amx_c3()


def should_use_amx_c3_prefix() -> bool:
    return RegimeDetector.instance().should_use_amx_c3_prefix()


def note_swap_event(n: int = 1) -> None:
    RegimeDetector.instance().note_swap_event(n)


def note_kv_usage(kv_pct: float) -> None:
    RegimeDetector.instance().note_kv_usage(kv_pct)


def note_prefix_hit(local_hits: int, total_prompt_tokens: int) -> None:
    RegimeDetector.instance().note_prefix_hit(local_hits, total_prompt_tokens)


def note_prefix_hit_rate(rate: float) -> None:
    RegimeDetector.instance().note_prefix_hit_rate(rate)


def note_step(gpu_util: float | None = None) -> None:
    RegimeDetector.instance().note_step(gpu_util)


def stats_snapshot() -> dict:
    return RegimeDetector.instance().stats_snapshot()


__all__ = [
    "WorkloadRegime",
    "classify",
    "RegimeDetector",
    "current_regime",
    "should_use_dsa",
    "should_use_amx_c3",
    "should_use_amx_c3_prefix",
    "note_swap_event",
    "note_kv_usage",
    "note_prefix_hit",
    "note_prefix_hit_rate",
    "note_step",
    "stats_snapshot",
]
