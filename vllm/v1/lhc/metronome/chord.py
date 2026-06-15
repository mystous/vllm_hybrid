# SPDX-License-Identifier: Apache-2.0
"""CHORD — cross-lane producer-consumer scheduling (stage 2).

Chains GPU-side producers (logits, freshly freed KV blocks) to host-side
consumers (sampler, scatter, detok) so that **step N's host work runs
during step N-1's GPU forward**. Implemented as a small bounded FIFO
queue per consumer kind; producers ``enqueue`` and consumers ``drain``.

Active when ``VLLM_LHC_METRONOME_CHORD=1``; otherwise the enqueue/drain
calls degrade to no-ops (callers fall back to inline execution).
"""

from __future__ import annotations

import logging
import os
import queue
import threading

logger = logging.getLogger(__name__)

_QUEUE_CAP = int(os.environ.get("VLLM_LHC_METRONOME_CHORD_QCAP", "8"))


class _ChordQueue:
    """Bounded FIFO with non-blocking enqueue (drops on full) and best-
    effort drain. ``maxsize=0`` means disabled.
    """

    def __init__(self, cap: int) -> None:
        self.cap = cap
        self.q: queue.Queue = queue.Queue(maxsize=max(cap, 1))
        self.dropped = 0
        self.enqueued = 0
        self.drained = 0
        self._lock = threading.Lock()

    def enqueue(self, item) -> bool:
        try:
            self.q.put_nowait(item)
            self.enqueued += 1
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def drain(self) -> list:
        out: list = []
        while True:
            try:
                out.append(self.q.get_nowait())
            except queue.Empty:
                break
        self.drained += len(out)
        return out

    def size(self) -> int:
        return self.q.qsize()


# Three logical chains we support out of the box.
_chains: dict[str, _ChordQueue] = {}
_lock = threading.RLock()


def _ensure_chain(name: str) -> _ChordQueue:
    with _lock:
        c = _chains.get(name)
        if c is None:
            c = _ChordQueue(_QUEUE_CAP)
            _chains[name] = c
        return c


def chord_enabled() -> bool:
    return os.environ.get("VLLM_LHC_METRONOME_CHORD", "1") == "1"


def chord_enqueue(chain: str, item) -> bool:
    """Enqueue an item for the named chain (e.g. ``sampler``, ``scatter``,
    ``detok``). Returns True on success, False if dropped (callers should
    fall back to inline)."""
    if not chord_enabled():
        return False
    return _ensure_chain(chain).enqueue(item)


def chord_drain(chain: str) -> list:
    """Drain pending items for the named chain. Returns empty list if
    disabled / empty."""
    if not chord_enabled():
        return []
    return _ensure_chain(chain).drain()


def chord_size(chain: str) -> int:
    return _ensure_chain(chain).size()


def chord_stats() -> dict:
    out = {}
    with _lock:
        for k, v in _chains.items():
            out[k] = {
                "size": v.size(),
                "enqueued": v.enqueued,
                "drained": v.drained,
                "dropped": v.dropped,
            }
    return out
