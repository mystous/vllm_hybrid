"""L12 — CPU-side cudagraph batch-size predictor (standalone).

This module implements a small Python class that observes the *padded*
cudagraph batch size used at each forward-step and predicts the next
likely size.  It is imported by the patched ``GPUModelRunner`` (see
``patch.py``) and can also be exercised standalone for unit tests.

Two predictors are bundled:

* ``SlidingWindowMode`` – majority vote over the last ``W`` observations.
  O(1) update, O(W) predict (default W=8).  Very fast.
* ``LastSeenPlusOne`` – returns the next-larger captured size from a
  pre-computed sorted list (a simple "burst growing" heuristic).

The predictor itself is *pure CPU* and *never* touches GPU memory.  The
hot path (``observe`` + ``predict``) measures sub-microsecond on a single
core (see ``MEASUREMENTS.md``).  Therefore wiring it into the dispatch
path is safe even in steady state.
"""

from __future__ import annotations

import bisect
import collections
import time


class SlidingWindowMode:
    """Majority-vote predictor over the last ``window`` observations."""

    __slots__ = ("_window", "_buf", "_counts", "_last_emit_ns")

    def __init__(self, window: int = 8):
        if window < 1:
            raise ValueError("window must be >= 1")
        self._window = window
        self._buf: collections.deque[int] = collections.deque(maxlen=window)
        self._counts: collections.Counter[int] = collections.Counter()
        self._last_emit_ns: int = 0

    def observe(self, size: int) -> None:
        if len(self._buf) == self._window:
            popped = self._buf[0]
            self._counts[popped] -= 1
            if self._counts[popped] <= 0:
                del self._counts[popped]
        self._buf.append(size)
        self._counts[size] += 1

    def predict(self) -> int | None:
        if not self._counts:
            return None
        # most_common(1) is O(W) but W is small (~8); ~0.3us in CPython.
        return self._counts.most_common(1)[0][0]

    def window_size(self) -> int:
        return len(self._buf)


class LastSeenPlusOne:
    """Returns the next-larger captured size (graded warmup heuristic).

    On a burst pattern (low → high concurrency) the padded batch size
    walks monotonically upward; pre-warming the *next* size in the
    captured list keeps its cudagraph nodes resident in L2.
    """

    __slots__ = ("_sizes", "_last")

    def __init__(self, captured_sizes: list[int]):
        self._sizes = sorted(set(captured_sizes))
        if not self._sizes:
            raise ValueError("captured_sizes must be non-empty")
        self._last: int | None = None

    def observe(self, size: int) -> None:
        self._last = size

    def predict(self) -> int | None:
        if self._last is None:
            return None
        idx = bisect.bisect_right(self._sizes, self._last)
        if idx >= len(self._sizes):
            return self._sizes[-1]  # already at top — predict largest
        return self._sizes[idx]


class CompositePredictor:
    """SlidingWindow for steady state + LastSeenPlusOne for growth.

    Heuristic: if the last ``trend_window`` observations are strictly
    non-decreasing → trust ``LastSeenPlusOne`` (we are in a ramp);
    otherwise trust majority vote (steady state).
    """

    def __init__(self, captured_sizes: list[int], window: int = 8,
                 trend_window: int = 4):
        self._sw = SlidingWindowMode(window=window)
        self._lsp = LastSeenPlusOne(captured_sizes)
        self._trend_window = trend_window
        self._recent: collections.deque[int] = collections.deque(
            maxlen=trend_window
        )
        # counters for stats
        self.n_observed = 0
        self.n_predicted = 0
        self.n_predicted_ramp = 0
        self.n_predicted_steady = 0

    def observe(self, size: int) -> None:
        self._sw.observe(size)
        self._lsp.observe(size)
        self._recent.append(size)
        self.n_observed += 1

    def predict(self) -> tuple[int | None, str]:
        if not self._recent:
            return None, "cold"
        self.n_predicted += 1
        # Trend check: is the most recent burst strictly non-decreasing
        # AND showing real growth (not all equal)?
        if len(self._recent) == self._trend_window:
            recent = list(self._recent)
            non_dec = all(
                recent[i] <= recent[i + 1] for i in range(len(recent) - 1)
            )
            growing = recent[-1] > recent[0]
            if non_dec and growing:
                self.n_predicted_ramp += 1
                return self._lsp.predict(), "ramp"
        self.n_predicted_steady += 1
        return self._sw.predict(), "steady"


# ─── microbench (standalone) ──────────────────────────────────────────────
def _bench(n_iter: int = 200_000, window: int = 8) -> dict:
    """Measure observe + predict cost for the composite predictor."""
    import random

    captured = (
        [1, 2, 4]
        + list(range(8, 256, 8))
        + list(range(256, 513, 16))
    )
    p = CompositePredictor(captured, window=window)
    rng = random.Random(0)
    samples = [rng.choice(captured) for _ in range(n_iter)]

    # warm
    for s in samples[:1024]:
        p.observe(s)
        p.predict()

    t0 = time.perf_counter()
    for s in samples:
        p.observe(s)
        p.predict()
    dt = time.perf_counter() - t0
    return {
        "n_iter": n_iter,
        "elapsed_s": round(dt, 4),
        "ns_per_call": round(dt * 1e9 / n_iter, 1),
        "calls_per_s": int(n_iter / dt),
        "stats": {
            "observed": p.n_observed,
            "predicted_ramp": p.n_predicted_ramp,
            "predicted_steady": p.n_predicted_steady,
        },
    }


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--window", type=int, default=8)
    args = ap.parse_args()
    print(json.dumps(_bench(args.n, args.window), indent=2))
