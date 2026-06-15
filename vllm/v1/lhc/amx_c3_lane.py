# SPDX-License-Identifier: Apache-2.0
"""AMX C3 lane — prefix-cache radix-tree byte scan + AVX-512 rolling hash.

Phase 3 microbench result (lhc_phase3/amx_sub_lane_microbench.md):
  - 1 MB scan: 130 µs on Xeon Platinum 8570 AMX, vs 265 µs on B200.
  - Gate ≤ 3× GPU latency: PASS (2.04×).

Public API:
  - ``amx_c3_available()`` — runtime check (CPU features + lib).
  - ``amx_c3_prefix_scan(buf, granule)`` — returns rolling hash array.

The implementation is a NumPy-based fallback that mimics the C3 byte-scan
shape so the integration path is exercised on dev machines that lack
AMX (e.g. Alder Lake / RTX 3090). On AMX-capable Sapphire Rapids / Xeon
hardware the production library (built from
``shadow_assists/.../poc/cpu_heavy_C3_cpusampling``) is loaded via dlopen.

METRONOME-LHC integration:
  - When ``metronome_active()`` and ``ritardando_amx_paused()``, callers
    should fall back to the GPU radix-tree hasher.
  - ACCENT does not directly throttle AMX (compute lane is bound to
    physical cores, not bandwidth-limited).
"""

from __future__ import annotations

import ctypes
import logging
import os
import threading

logger = logging.getLogger(__name__)

_lock = threading.RLock()
_lib: ctypes.CDLL | None = None
_probed = False


def _try_load_amx_lib() -> ctypes.CDLL | None:
    """Locate the AMX prefix-scan shared library. Honours
    ``VLLM_LHC_AMX_C3_LIB`` env override. Returns None if not available.
    """
    candidates = []
    env = os.environ.get("VLLM_LHC_AMX_C3_LIB")
    if env:
        candidates.append(env)
    # Standard install locations.
    candidates.append(
        os.path.join(os.path.dirname(__file__), "libamx_c3.so")
    )
    candidates.append("/usr/local/lib/libamx_c3.so")
    for path in candidates:
        try:
            lib = ctypes.CDLL(path)
            return lib
        except OSError:
            continue
    return None


def _have_amx_isa() -> bool:
    """Best-effort AMX_TILE flag check from /proc/cpuinfo."""
    try:
        with open("/proc/cpuinfo") as f:
            txt = f.read()
        return "amx_tile" in txt
    except OSError:
        return False


def amx_c3_available() -> bool:
    """Returns True when the AMX C3 lane can be used. Caches the answer.

    Option C (LHC Phase 4) — when ``VLLM_LHC_REGIME_ADAPTIVE=1`` we also
    gate on the regime detector. The hardware-probe answer is still cached
    so we do not redlich /proc/cpuinfo each call; the regime gate runs on
    top.

    Anti-pattern override — ``VLLM_LHC_AMX_C3_FORCE_EVERY_STEP=1`` bypasses
    both the regime gate AND the standard prefix-cache-hit gate by simply
    short-circuiting any "skip" branch. Used in LHC Phase 4 misuse
    measurement to quantify the penalty of running the AMX C3 prefix scan
    on every scheduler step (no prefix-hit filter, no regime filter).
    """
    if os.environ.get("VLLM_LHC_AMX_C3_FORCE_EVERY_STEP", "0") == "1":
        # Skip the regime gate so we always probe / load the lib.
        pass
    elif os.environ.get("VLLM_LHC_REGIME_ADAPTIVE", "0") == "1":
        try:
            from vllm.v1.lhc.regime_detector import should_use_amx_c3
            if not should_use_amx_c3():
                return False
        except Exception:  # noqa: BLE001
            pass
    global _lib, _probed
    with _lock:
        if _probed:
            return _lib is not None
        _probed = True
        if not _have_amx_isa():
            logger.info("AMX C3 lane: amx_tile flag absent — disabled")
            return False
        _lib = _try_load_amx_lib()
        if _lib is None:
            logger.info(
                "AMX C3 lane: shared lib not found — disabled "
                "(set VLLM_LHC_AMX_C3_LIB=<path> to enable)"
            )
            return False
        try:
            # Expected ABI: void amx_c3_scan(const void* buf, size_t n,
            #                                size_t granule, uint64_t* out);
            _lib.amx_c3_scan.argtypes = [
                ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
                ctypes.c_void_p,
            ]
            _lib.amx_c3_scan.restype = None
        except AttributeError:
            logger.warning(
                "AMX C3 lane: lib missing amx_c3_scan symbol — disabled"
            )
            _lib = None
            return False
        logger.info("AMX C3 lane: enabled")
        return True


def amx_c3_prefix_scan(buf_ptr: int, n_bytes: int, granule: int):
    """Run the byte-scan + rolling hash over ``n_bytes`` of memory starting
    at ``buf_ptr``. Returns a numpy array of uint64 hashes of length
    ``n_bytes // granule``. Falls back to a pure-Python implementation
    when the AMX lib is not available — keeps the integration path
    exercise-able on dev (RTX 3090) machines.
    """
    import numpy as np
    n_out = max(n_bytes // max(granule, 1), 1)
    out = np.zeros(n_out, dtype=np.uint64)
    if amx_c3_available():
        _lib.amx_c3_scan(  # type: ignore[union-attr]
            ctypes.c_void_p(buf_ptr),
            ctypes.c_size_t(n_bytes),
            ctypes.c_size_t(granule),
            out.ctypes.data_as(ctypes.c_void_p),
        )
        return out
    # Fallback — slow, but functional. Uses a rolling 64-bit FNV-1a so
    # tests can validate the integration glue without AMX hardware.
    arr = (ctypes.c_uint8 * n_bytes).from_address(buf_ptr)
    h = np.uint64(0xcbf29ce484222325)
    p = np.uint64(0x100000001b3)
    idx = 0
    for i in range(0, n_bytes, granule):
        h_local = h
        end = min(i + granule, n_bytes)
        for j in range(i, end):
            h_local = (h_local ^ np.uint64(arr[j])) * p
        out[idx] = h_local
        idx += 1
        if idx >= n_out:
            break
    return out
