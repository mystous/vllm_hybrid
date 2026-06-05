# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ctypes binding for libamx_draft_qwen05b.so (SUB_187 AMX kernel).

Wraps the C ABI exposed by `libamx_draft_qwen05b.so`:

  int    amx_draft_qwen05b_init(void);
  void   amx_draft_qwen05b_free(void);
  double amx_draft_qwen05b_step_ms(int B, int K);
  double amx_draft_qwen05b_single_ms(int B);
  double amx_draft_qwen05b_mlp_ms(int B);
  int    amx_draft_qwen05b_hw_amx(void);
  void   amx_draft_qwen05b_forward(const uint16_t* input_bf16, int B,
                                   uint16_t* logits_out, int32_t* ids_out,
                                   int K);  # Phase A3 (SUB_198)

See `shadow_assists/features/IDE_019_multi_source_drafter/
SUB_198_amx_real_integration/AMX_INTEGRATION_DESIGN.md` for full design,
ABI rationale, and vocab-mismatch handling plan.

The current .so is a *latency microbench*: it does NOT accept input token
ids; its `step_ms(B, K)` runs K LM-head matmul iterations on an
init-time random activation buffer. This module establishes the binding
+ safety gating (hw_amx check) needed for the next-step real-forward
extension (SUB_198 §3 (d)).

Safety:
  * On dev hosts without AMX (Alder Lake, Skylake, etc.) `hw_amx()`
    returns 0 → callers MUST NOT call `step_ms()` (SIGILL risk).
  * `is_available()` performs both the dlopen and the `hw_amx()` check
    so callers can route around the kernel cleanly on dev boxes.
  * Library handle is released on interpreter exit via `atexit`.
"""
from __future__ import annotations

import atexit
import ctypes
import os
import sys
import threading
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────────
# Kernel constants (mirror DraftState in src/amx_draft_qwen05b.cpp)
# ─────────────────────────────────────────────────────────────────────
KERNEL_HIDDEN: int = 896
KERNEL_INTERMEDIATE: int = 4864
# Padded vocab compiled into the .so (config Qwen2.5-0.5B-Instruct is
# 151,936; kernel rounds to 152,064 = 128×1188 for 16-multiple tile
# alignment). See AMX_INTEGRATION_DESIGN.md §4.
KERNEL_VOCAB: int = 152_064
CONFIG_VOCAB_QWEN_05B: int = 151_936
KERNEL_LAYERS: int = 24
KERNEL_B_MAX: int = 16


# ─────────────────────────────────────────────────────────────────────
# Default library search
# ─────────────────────────────────────────────────────────────────────
# Resolve repo root by walking up from this file
# (vllm/v1/spec_decode/cpu_amx_kernel.py -> repo_root).
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[3]

_DEFAULT_LIB_REL = (
    "shadow_assists/features/IDE_019_multi_source_drafter/"
    "SUB_187_amx_draft_head/build/libamx_draft_qwen05b.so"
)


def default_library_path() -> Path:
    env = os.environ.get("VLLM_CPU_DRAFT_KERNEL_PATH")
    if env:
        return Path(env).expanduser().resolve()
    return (_REPO_ROOT / _DEFAULT_LIB_REL).resolve()


# ─────────────────────────────────────────────────────────────────────
# Module-level singleton — share lib handle across CpuAmxProposer
# instances (kernel init allocates ~265 MB of weights).
# ─────────────────────────────────────────────────────────────────────
_LIB_LOCK = threading.Lock()
_LIB_CACHE: dict[str, "AmxDraftKernel"] = {}
_WARNED: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        print(f"[cpu_amx_kernel] {msg}", file=sys.stderr, flush=True)


class AmxDraftKernel:
    """ctypes binding for the SUB_187 AMX draft kernel.

    Lifecycle:
      k = AmxDraftKernel.get()          # cached per (library path)
      if k.is_available():               # dlopen OK + hw_amx() == 1
          k.ensure_init()                # idempotent C-side init
          ms = k.step_ms(B=1, K=7)       # PROD-only — SIGILL on no-AMX CPU

    On dev hosts (no AMX), `is_available()` returns False; callers must
    NOT call `step_ms`. See AMX_INTEGRATION_DESIGN.md §5.3.
    """

    # ─── public construction ──────────────────────────────────────
    @classmethod
    def get(cls, lib_path: Optional[Path] = None) -> "AmxDraftKernel":
        """Return cached instance for the resolved library path.

        Re-uses a single handle across calls / proposers. Caches per
        absolute library path so an explicit override creates a separate
        binding.
        """
        path = (lib_path or default_library_path()).resolve()
        key = str(path)
        with _LIB_LOCK:
            if key in _LIB_CACHE:
                return _LIB_CACHE[key]
            inst = cls(path)
            _LIB_CACHE[key] = inst
            return inst

    def __init__(self, lib_path: Path):
        self.lib_path: Path = lib_path
        self._lib: Optional[ctypes.CDLL] = None
        self._load_error: Optional[str] = None
        self._initialized: bool = False
        self._init_rc: Optional[int] = None
        self._hw_amx: Optional[int] = None
        self._dlopen()

    # ─── dlopen + signature bind ──────────────────────────────────
    def _dlopen(self) -> None:
        if not self.lib_path.exists():
            self._load_error = f"library not found: {self.lib_path}"
            _warn_once(
                f"lib_missing::{self.lib_path}",
                f"{self._load_error} — AMX path disabled.",
            )
            return
        try:
            lib = ctypes.CDLL(str(self.lib_path), mode=ctypes.RTLD_GLOBAL)
        except OSError as e:
            self._load_error = f"dlopen failed: {e}"
            _warn_once(
                f"lib_dlopen::{self.lib_path}",
                f"{self._load_error} — AMX path disabled.",
            )
            return

        try:
            lib.amx_draft_qwen05b_init.restype = ctypes.c_int
            lib.amx_draft_qwen05b_init.argtypes = []

            lib.amx_draft_qwen05b_free.restype = None
            lib.amx_draft_qwen05b_free.argtypes = []

            lib.amx_draft_qwen05b_step_ms.restype = ctypes.c_double
            lib.amx_draft_qwen05b_step_ms.argtypes = [
                ctypes.c_int, ctypes.c_int
            ]

            lib.amx_draft_qwen05b_single_ms.restype = ctypes.c_double
            lib.amx_draft_qwen05b_single_ms.argtypes = [ctypes.c_int]

            lib.amx_draft_qwen05b_mlp_ms.restype = ctypes.c_double
            lib.amx_draft_qwen05b_mlp_ms.argtypes = [ctypes.c_int]

            lib.amx_draft_qwen05b_hw_amx.restype = ctypes.c_int
            lib.amx_draft_qwen05b_hw_amx.argtypes = []

            # Phase A3 (SUB_198) — real-forward ABI
            lib.amx_draft_qwen05b_forward.restype = None
            lib.amx_draft_qwen05b_forward.argtypes = [
                ctypes.POINTER(ctypes.c_uint16),  # input_bf16 (may be NULL)
                ctypes.c_int,                     # B
                ctypes.POINTER(ctypes.c_uint16),  # logits_out [B,K,VOCAB]
                ctypes.POINTER(ctypes.c_int32),   # ids_out    [B,K]
                ctypes.c_int,                     # K
            ]
        except AttributeError as e:
            self._load_error = f"missing symbol: {e}"
            _warn_once(
                f"lib_symbol::{self.lib_path}",
                f"{self._load_error} — AMX path disabled.",
            )
            return

        self._lib = lib
        # Probe hw_amx — CPUID-only, safe on any x86.
        try:
            self._hw_amx = int(lib.amx_draft_qwen05b_hw_amx())
        except Exception as e:  # pragma: no cover — defensive
            self._load_error = f"hw_amx probe failed: {e}"
            self._hw_amx = 0

        # Register cleanup once we have a working lib handle.
        atexit.register(self._safe_free)

    # ─── public predicates ────────────────────────────────────────
    @property
    def loaded(self) -> bool:
        """True iff dlopen + symbol bind succeeded (NOT AMX-required)."""
        return self._lib is not None

    @property
    def hw_amx(self) -> int:
        """1 iff CPUID reports AMX_TILE + AMX_BF16. 0 otherwise.

        Note: still 0 on Sapphire Rapids if `amx_request_permission()`
        was never made — `hw_amx()` in the .so is CPUID-only so this
        only reflects the *hardware* bits.
        """
        return int(self._hw_amx or 0)

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def is_available(self) -> bool:
        """Safe gate for callers: True iff lib loaded AND AMX present."""
        return self.loaded and self.hw_amx == 1

    # ─── lifecycle ────────────────────────────────────────────────
    def ensure_init(self) -> int:
        """Call C-side init once. Returns the C return code (0 = OK)."""
        if not self.loaded:
            raise RuntimeError(
                f"AmxDraftKernel not loaded: {self._load_error}"
            )
        if self._initialized:
            return self._init_rc or 0
        assert self._lib is not None
        rc = int(self._lib.amx_draft_qwen05b_init())
        self._init_rc = rc
        if rc == 0:
            self._initialized = True
        else:
            _warn_once(
                f"init_fail::{self.lib_path}::{rc}",
                f"amx_draft_qwen05b_init returned {rc} — "
                "AMX path will not be usable. "
                "(-1=no AMX, -2=perm denied, -3/-4/-5=alloc fail)",
            )
        return rc

    def _safe_free(self) -> None:
        """atexit hook — release kernel-side buffers."""
        if self._initialized and self._lib is not None:
            try:
                self._lib.amx_draft_qwen05b_free()
            except Exception:  # pragma: no cover — interpreter teardown
                pass
            self._initialized = False

    # ─── hot-path entries ─────────────────────────────────────────
    def step_ms(self, B: int, K: int) -> float:
        """Run K LM-head matmul iterations, return wall ms.

        WARNING — calling on a non-AMX CPU will raise SIGILL (the kernel
        executes `tdpbf16ps` directly). Callers MUST gate on
        `is_available()`. Dev-host smoke tests should NOT invoke this.
        """
        if not self.is_available():
            raise RuntimeError(
                "AMX kernel not available "
                f"(loaded={self.loaded}, hw_amx={self.hw_amx}, "
                f"err={self._load_error}). Refusing to call step_ms — "
                "SIGILL risk."
            )
        if self._init_rc != 0:
            raise RuntimeError(
                f"AMX kernel init returned {self._init_rc}; "
                "step_ms not callable."
            )
        if B < 1 or B > KERNEL_B_MAX:
            raise ValueError(
                f"B must be 1..{KERNEL_B_MAX}, got {B}"
            )
        if K < 1:
            raise ValueError(f"K must be >=1, got {K}")
        assert self._lib is not None
        return float(self._lib.amx_draft_qwen05b_step_ms(int(B), int(K)))

    def single_ms(self, B: int) -> float:
        """One LM-head matmul (K=1). Same SIGILL gating as step_ms."""
        if not self.is_available():
            raise RuntimeError("AMX kernel not available")
        assert self._lib is not None
        return float(self._lib.amx_draft_qwen05b_single_ms(int(B)))

    def mlp_ms(self, B: int) -> float:
        """One MLP-gate matmul (per-layer linear cost probe)."""
        if not self.is_available():
            raise RuntimeError("AMX kernel not available")
        assert self._lib is not None
        return float(self._lib.amx_draft_qwen05b_mlp_ms(int(B)))

    # ─── Phase A3 (SUB_198) — real forward (shape/dtype only) ─────
    def forward(self, input_bf16, B: int, K: int):
        """Run K LM-head matmuls and return (logits_bf16, ids_int32).

        Args:
          input_bf16: numpy ndarray of shape (B, KERNEL_HIDDEN), dtype
            uint16 (BF16 bit-pattern) OR None to reuse the kernel's
            internal init-time random activation buffer.
          B: batch (1..KERNEL_B_MAX).
          K: number of draft steps (>=1).

        Returns:
          logits: numpy uint16 (BF16) ndarray of shape (B, K,
            KERNEL_VOCAB). Padded vocab columns ≥ CONFIG_VOCAB_QWEN_05B
            are filled with BF16 of the FP32 accumulator output (they
            are NOT valid Qwen tokens — see AMX_INTEGRATION_DESIGN
            §4 on padding).
          ids: numpy int32 ndarray of shape (B, K) — argmax restricted
            to [0, CONFIG_VOCAB_QWEN_05B). Caller can therefore trust
            that each id is a valid Qwen 0.5B vocab token.

        NOTE: real LM-head weights are NOT loaded in this PoC — the
        kernel still holds random BF16 init from `init`. The returned
        ids are therefore NOT semantically meaningful. This entry
        validates SHAPE / DTYPE / ARGMAX-RANGE only.
        """
        import numpy as np
        if not self.is_available():
            raise RuntimeError(
                "AMX kernel not available — refusing to call forward.")
        if self._init_rc != 0:
            self.ensure_init()
        if self._init_rc != 0:
            raise RuntimeError(
                f"AMX kernel init returned {self._init_rc}; forward "
                "not callable.")
        if B < 1 or B > KERNEL_B_MAX:
            raise ValueError(
                f"B must be 1..{KERNEL_B_MAX}, got {B}")
        if K < 1:
            raise ValueError(f"K must be >=1, got {K}")

        # Output buffers (caller-owned, C-contiguous, aligned by numpy).
        logits = np.empty((B, K, KERNEL_VOCAB), dtype=np.uint16)
        ids = np.empty((B, K), dtype=np.int32)

        # Input pointer — None ⇒ pass nullptr ⇒ kernel reuses act_in.
        if input_bf16 is None:
            in_ptr = ctypes.cast(0, ctypes.POINTER(ctypes.c_uint16))
        else:
            if input_bf16.dtype != np.uint16:
                raise TypeError(
                    "input_bf16 must be numpy uint16 (BF16 bit-pattern),"
                    f" got dtype={input_bf16.dtype}")
            if input_bf16.shape != (B, KERNEL_HIDDEN):
                raise ValueError(
                    f"input_bf16 shape must be ({B},{KERNEL_HIDDEN}), "
                    f"got {tuple(input_bf16.shape)}")
            if not input_bf16.flags["C_CONTIGUOUS"]:
                input_bf16 = np.ascontiguousarray(input_bf16)
            in_ptr = input_bf16.ctypes.data_as(
                ctypes.POINTER(ctypes.c_uint16))

        logits_ptr = logits.ctypes.data_as(
            ctypes.POINTER(ctypes.c_uint16))
        ids_ptr = ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

        assert self._lib is not None
        self._lib.amx_draft_qwen05b_forward(
            in_ptr, int(B), logits_ptr, ids_ptr, int(K))
        return logits, ids


# ─────────────────────────────────────────────────────────────────────
# Convenience top-level helpers
# ─────────────────────────────────────────────────────────────────────

def load_default() -> AmxDraftKernel:
    """Load the default library (env override aware)."""
    return AmxDraftKernel.get()


def kernel_is_available() -> bool:
    """One-liner gate suitable for `if kernel_is_available(): ...`."""
    try:
        return load_default().is_available()
    except Exception:  # pragma: no cover — defensive
        return False


__all__ = [
    "AmxDraftKernel",
    "CONFIG_VOCAB_QWEN_05B",
    "KERNEL_B_MAX",
    "KERNEL_HIDDEN",
    "KERNEL_INTERMEDIATE",
    "KERNEL_LAYERS",
    "KERNEL_VOCAB",
    "default_library_path",
    "kernel_is_available",
    "load_default",
]
