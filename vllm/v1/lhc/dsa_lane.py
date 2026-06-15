"""LHC Phase 2 — Python wrapper around ``libdsa_lane.so``.

Provides a single-thread-safe ``dsa_memcpy(dst_ptr, src_ptr, n_bytes)``
that submits a MEMMOVE descriptor to an Intel DSA dedicated work queue
(default ``/dev/dsa/wq0.0``). On any failure (no device, no library,
runtime status != SUCCESS) the lane reports unavailable and callers
should fall back to plain memcpy / torch copy paths.

Design constraints:
  * the .so binds to one mm via cdev open at first ``dsa_lane_available``
    call. The fd / mmap'd portal are held for the process lifetime.
  * synchronous submit + poll. For pipelining the C lib can be extended
    (Phase 3); the integration site here issues one descriptor per call.
  * thread-safe: the .so guards init with a pthread mutex; the descriptor
    + completion record are stack-local per call so concurrent threads
    can submit safely against the same WQ portal.

Env gates (read once at first invocation):
  VLLM_LHC_DSA      = "1" : enable lane attempt (default off)
  VLLM_LHC_DSA_DEV  = path: override device path (default /dev/dsa/wq0.0)
  VLLM_LHC_DSA_MIN  = int : minimum byte count to use DSA (default 65536).
                            Smaller copies use plain memcpy — the per-call
                            descriptor cost (~1 us) dominates at < 64 KB.
  VLLM_LHC_DSA_WQ_PER_RANK = "1" : (Phase 3) pick WQ from TP rank →
                            rank R → /dev/dsa/wq{R//4}.{R%4}. Avoids
                            PASID conflict observed on TP=8 EBUSY.
  VLLM_LHC_DSA_RANK = int : explicit rank override (otherwise read from
                            LOCAL_RANK / RANK / 0).
  VLLM_LHC_DSA_FORCE_REMOTE_NUMA = "1" : (Phase 4 anti-pattern) deliberately
                            invert the rank → DSA device mapping so rank 0-3
                            land on dsa1 (NUMA node 1) and rank 4-7 land on
                            dsa0 (NUMA node 0). Used to quantify cross-socket
                            penalty for LHC misuse measurement. Requires
                            VLLM_LHC_DSA_WQ_PER_RANK=1.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_LIB_PATH = Path(__file__).with_name("libdsa_lane.so")

_lib: ctypes.CDLL | None = None
_available: bool | None = None
_lock = threading.Lock()
_min_bytes = 65536


def _resolve_dev_path() -> str:
    """Pick the DSA WQ this process should use.

    Default: VLLM_LHC_DSA_DEV (single WQ for the whole process).

    WQ-per-rank mode (VLLM_LHC_DSA_WQ_PER_RANK=1):
      We have 2 DSA devices × 4 dedicated WQs each = 8 WQs.
      rank R maps to /dev/dsa/wq{R//4}.{R%4} → rank 0–3 → dsa0/wq0.0–0.3,
      rank 4–7 → dsa1/wq1.0–1.3. Each rank holds its own PASID, avoiding
      the TP=8 EBUSY observed in Phase 2 (all ranks sharing wq0.0).

    Override hooks:
      VLLM_LHC_DSA_RANK   — explicit rank (otherwise LOCAL_RANK | RANK | 0).
      VLLM_LHC_DSA_DEV    — explicit full path; bypasses per-rank logic.
    """
    explicit = os.environ.get("VLLM_LHC_DSA_DEV")
    if explicit:
        return explicit
    if os.environ.get("VLLM_LHC_DSA_WQ_PER_RANK", "0") != "1":
        return "/dev/dsa/wq0.0"
    try:
        rank = int(
            os.environ.get("VLLM_LHC_DSA_RANK")
            or os.environ.get("LOCAL_RANK")
            or os.environ.get("RANK")
            or "0"
        )
    except ValueError:
        rank = 0
    dev_idx = (rank // 4) % 2
    wq_idx = rank % 4
    # Anti-pattern: cross-socket NUMA forced mapping (LHC Phase 4 misuse
    # measurement). Inverts dev_idx so rank 0-3 → dsa1 (remote NUMA) and
    # rank 4-7 → dsa0. The WQ index inside each device is kept so PASID
    # is still per-rank (no contention) — we are only stressing the
    # cross-socket data path, not the queue arbitration.
    if os.environ.get("VLLM_LHC_DSA_FORCE_REMOTE_NUMA", "0") == "1":
        dev_idx = 1 - dev_idx
    return f"/dev/dsa/wq{dev_idx}.{wq_idx}"


def _try_load() -> bool:
    global _lib, _available, _min_bytes
    if _available is not None:
        return _available
    with _lock:
        if _available is not None:
            return _available
        # Gate: env must be enabled.
        if os.environ.get("VLLM_LHC_DSA", "0") != "1":
            _available = False
            return False
        if not _LIB_PATH.exists():
            logger.info("[LHC DSA] %s missing — lane disabled", _LIB_PATH)
            _available = False
            return False
        try:
            lib = ctypes.CDLL(str(_LIB_PATH))
        except OSError as e:
            logger.info("[LHC DSA] dlopen %s failed: %s — lane disabled",
                        _LIB_PATH, e)
            _available = False
            return False
        # Signatures.
        lib.dsa_lane_init.argtypes = [ctypes.c_char_p]
        lib.dsa_lane_init.restype = ctypes.c_int
        lib.dsa_lane_memcpy.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
        ]
        lib.dsa_lane_memcpy.restype = ctypes.c_int
        lib.dsa_lane_stats.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        lib.dsa_lane_stats.restype = None
        lib.dsa_lane_close.argtypes = []
        lib.dsa_lane_close.restype = None
        dev_path = _resolve_dev_path()
        rc = lib.dsa_lane_init(dev_path.encode())
        if rc != 0:
            logger.info("[LHC DSA] init rc=%d for %s — lane disabled",
                        rc, dev_path)
            _available = False
            return False
        # Self-test: 4 KB roundtrip.
        try:
            src = (ctypes.c_uint8 * 4096)(*([0xA5] * 4096))
            dst = (ctypes.c_uint8 * 4096)()
            rc = lib.dsa_lane_memcpy(
                ctypes.cast(dst, ctypes.c_void_p),
                ctypes.cast(src, ctypes.c_void_p),
                4096,
            )
            ok = rc == 0 and all(b == 0xA5 for b in dst)
            if not ok:
                logger.info("[LHC DSA] self-test failed rc=%d — lane disabled",
                            rc)
                _available = False
                return False
        except Exception as e:  # noqa: BLE001
            logger.info("[LHC DSA] self-test exception: %r — lane disabled", e)
            _available = False
            return False
        try:
            _min_bytes = int(os.environ.get("VLLM_LHC_DSA_MIN", "65536"))
        except ValueError:
            _min_bytes = 65536
        _lib = lib
        _available = True
        logger.info(
            "[LHC DSA] lane ENABLED (%s, min_bytes=%d)", dev_path, _min_bytes,
        )
        return True


def dsa_lane_available() -> bool:
    return _try_load()


def dsa_memcpy(dst_ptr: int, src_ptr: int, n: int) -> bool:
    """Submit one MEMMOVE via DSA. Returns True on success.

    Caller passes raw integer pointers (e.g. from ``tensor.data_ptr()``
    or ctypes ``addressof``) and a byte count. On any failure the caller
    must fall back to a plain memcpy.

    Bypasses DSA and returns False when n < the configured minimum so
    that the descriptor overhead does not regress short copies.

    Option C (LHC Phase 4) — when ``VLLM_LHC_REGIME_ADAPTIVE=1`` we consult
    the regime detector: only KV_HEAVY routes through DSA. In static mode
    (legacy / Option A) behaviour is unchanged.
    """
    if n < _min_bytes:
        return False
    if not _try_load():
        return False
    # Adaptive gate (Option C). Lazy import to avoid circular import at
    # module load (regime_detector itself imports nothing from lhc).
    if os.environ.get("VLLM_LHC_REGIME_ADAPTIVE", "0") == "1":
        try:
            from vllm.v1.lhc.regime_detector import should_use_dsa
            if not should_use_dsa():
                return False
        except Exception:  # noqa: BLE001
            pass  # detector failure must not break the lane
    rc = _lib.dsa_lane_memcpy(  # type: ignore[union-attr]
        ctypes.c_void_p(dst_ptr),
        ctypes.c_void_p(src_ptr),
        ctypes.c_size_t(n),
    )
    return rc == 0


def dsa_lane_stats() -> dict[str, int]:
    if not _try_load():
        return {"available": 0, "ops": 0, "bytes": 0, "fails": 0}
    o = ctypes.c_uint64(0)
    b = ctypes.c_uint64(0)
    f = ctypes.c_uint64(0)
    _lib.dsa_lane_stats(ctypes.byref(o), ctypes.byref(b), ctypes.byref(f))  # type: ignore[union-attr]
    return {"available": 1, "ops": o.value, "bytes": b.value, "fails": f.value}


def dsa_lane_close() -> None:
    global _lib, _available
    with _lock:
        if _lib is not None:
            _lib.dsa_lane_close()
            _lib = None
            _available = False
