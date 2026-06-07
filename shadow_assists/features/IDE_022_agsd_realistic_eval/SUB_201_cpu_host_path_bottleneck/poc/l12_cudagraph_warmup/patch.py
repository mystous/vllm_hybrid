"""L12 — instrumentation hooks for ``GPUModelRunner``.

This module **does not** monkey-patch vLLM at import time.  Instead it
exposes two helpers that are wired into ``gpu_model_runner.py`` via a
**single ~10 line in-place edit** (see ``apply_patch.sh``).  Keeping the
hook surface tiny minimises the risk of merge conflicts and makes the
delta auditable.

Modes (env-gated):

* ``VLLM_CUDAGRAPH_PREDICTIVE_WARMUP=1``  – enable CPU-side observation
  + predictor.  No GPU work.  Logs distribution + prediction stats every
  ``VLLM_CUDAGRAPH_PREDICTIVE_LOG_EVERY`` (default 1000) steps.
* ``VLLM_CUDAGRAPH_PREDICTIVE_WARMUP=2``  – also call
  ``cudagraph_dispatcher.dispatch`` on the *predicted* size to time the
  CPU lookup cost.  Still no GPU forward.  Useful for upper-bound
  overhead measurement.

Note: a real GPU pre-replay would require either (a) running ``_dummy_run``
on a side stream — which clobbers persistent buffers and breaks
correctness — or (b) holding a private copy of the cudagraph's buffers,
which costs O(K * sizeof(buf)) GPU memory and is not justified by the
data this PoC collected.  See ``MEASUREMENTS.md`` for the reasoning.
"""

from __future__ import annotations

import os
import time
from typing import Any

# Imported lazily because predictor.py lives next to this file (not on
# sys.path when GPUModelRunner imports us). The patched runner injects
# the absolute path before importing.
from predictor import CompositePredictor  # noqa: E402  (deferred path)


_LOG_EVERY = int(os.environ.get("VLLM_CUDAGRAPH_PREDICTIVE_LOG_EVERY", "1000"))
_MODE = int(os.environ.get("VLLM_CUDAGRAPH_PREDICTIVE_WARMUP", "0"))
_LOG_PATH = os.environ.get(
    "VLLM_CUDAGRAPH_PREDICTIVE_LOG_PATH",
    "/tmp/l12_predictor_stats.jsonl",
)


class PredictiveWarmupHook:
    """Per-rank observation + prediction state.

    Attached as ``runner.l12_hook`` in ``apply_patch.sh``.
    """

    def __init__(self, captured_sizes: list[int], rank: int = 0):
        self.enabled = _MODE > 0
        self.rank = rank
        self.log_path = f"{_LOG_PATH}.rank{rank}"
        self._predictor = CompositePredictor(
            captured_sizes,
            window=int(os.environ.get("VLLM_CUDAGRAPH_PREDICTIVE_WINDOW", "8")),
            trend_window=4,
        )
        self._size_hist: dict[int, int] = {}
        self._predict_hit_distance: list[int] = []  # |pred - actual|
        self._pred_correct: int = 0
        self._n_steps: int = 0
        self._last_predicted: int | None = None
        # CPU overhead instrumentation
        self._obs_total_ns: int = 0
        self._pred_total_ns: int = 0
        # Optional second-stage dispatch (mode == 2)
        self._dispatch_total_ns: int = 0
        # File open once
        self._fp: Any = None
        if self.enabled:
            try:
                self._fp = open(self.log_path, "w", buffering=1)
            except OSError:
                self._fp = None

    def observe_and_predict(
        self,
        actual_size: int,
        cudagraph_dispatcher=None,
    ) -> int | None:
        """Hot path. Called once per forward step from ``_prepare_cudagraph_dispatch``.

        Always returns the predicted next size (or None).  Caller may use
        it for warmup; in this PoC the prediction is only logged.
        """
        if not self.enabled:
            return None
        self._n_steps += 1

        # Score prediction accuracy from previous step
        if self._last_predicted is not None:
            if self._last_predicted == actual_size:
                self._pred_correct += 1
            self._predict_hit_distance.append(
                abs(self._last_predicted - actual_size)
            )

        # Observe
        t0 = time.perf_counter_ns()
        self._predictor.observe(actual_size)
        self._size_hist[actual_size] = self._size_hist.get(actual_size, 0) + 1
        t1 = time.perf_counter_ns()
        self._obs_total_ns += t1 - t0

        # Predict
        t2 = time.perf_counter_ns()
        predicted, mode = self._predictor.predict()
        t3 = time.perf_counter_ns()
        self._pred_total_ns += t3 - t2
        self._last_predicted = predicted

        # Optional: time the dispatch lookup on the predicted size
        # (mode 2). Still no GPU work — pure CPU upper-bound.
        if _MODE >= 2 and cudagraph_dispatcher is not None and predicted is not None:
            try:
                t4 = time.perf_counter_ns()
                cudagraph_dispatcher.dispatch(num_tokens=predicted)
                t5 = time.perf_counter_ns()
                self._dispatch_total_ns += t5 - t4
            except Exception:  # noqa: BLE001
                pass  # never break the real step

        if self._n_steps % _LOG_EVERY == 0:
            self._emit_stats(mode)
        return predicted

    def _emit_stats(self, last_mode: str) -> None:
        if self._fp is None:
            return
        import json

        # mean / p50 / p99 of |pred - actual|
        dists = sorted(self._predict_hit_distance[-_LOG_EVERY:])

        def _pct(xs, q):
            if not xs:
                return None
            i = min(len(xs) - 1, int(round(q * (len(xs) - 1))))
            return xs[i]

        rec = {
            "ts": round(time.time(), 3),
            "rank": self.rank,
            "n_steps": self._n_steps,
            "n_observed": self._predictor.n_observed,
            "n_predicted_ramp": self._predictor.n_predicted_ramp,
            "n_predicted_steady": self._predictor.n_predicted_steady,
            "pred_exact_hits": self._pred_correct,
            "pred_exact_rate": (
                round(self._pred_correct / max(1, self._n_steps - 1), 4)
            ),
            "pred_dist_p50": _pct(dists, 0.5),
            "pred_dist_p99": _pct(dists, 0.99),
            "size_hist_top10": dict(
                sorted(
                    self._size_hist.items(),
                    key=lambda kv: -kv[1],
                )[:10]
            ),
            "last_mode": last_mode,
            "obs_ns_per_step": (
                round(self._obs_total_ns / self._n_steps, 1)
            ),
            "pred_ns_per_step": (
                round(self._pred_total_ns / self._n_steps, 1)
            ),
            "dispatch_ns_per_step": (
                round(self._dispatch_total_ns / self._n_steps, 1)
                if _MODE >= 2
                else 0
            ),
        }
        try:
            self._fp.write(json.dumps(rec) + "\n")
        except OSError:
            pass


def make_hook(runner) -> PredictiveWarmupHook | None:
    """Factory invoked from the patched ``GPUModelRunner.__init__``."""
    if _MODE <= 0:
        return None
    sizes = list(runner.cudagraph_batch_sizes) or [1]
    rank = getattr(runner, "rank", 0) or 0
    return PredictiveWarmupHook(sizes, rank=rank)
