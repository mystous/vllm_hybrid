# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase A3 (SUB_198) — forward ABI smoke.

Validates:
  • symbol `amx_draft_qwen05b_forward` is bound and callable
  • output shapes are (B, K, KERNEL_VOCAB) BF16 + (B, K) int32
  • argmax ids are within [0, CONFIG_VOCAB_QWEN_05B) — padded vocab
    columns are NOT selected even though they exist in the logits buffer
  • two call modes: input_bf16=None (reuse kernel act_in) AND explicit
    numpy uint16 (B, HIDDEN) buffer.

Real-weight semantic correctness is OUT OF SCOPE (weights are random
BF16 from `init`); only shape / dtype / range are asserted.

Run (worktree-isolated, OMP=16):
  OMP_NUM_THREADS=16 /workspace/vllm_dev_prj/bin/python \
    tests/v1/spec_decode/test_cpu_amx_kernel_forward.py
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

WORKTREE_ROOT = Path("/workspace/poc_worktrees/wt_a1_cpudraft")
WORKTREE_KERNEL_PATH = (
    WORKTREE_ROOT / "vllm" / "v1" / "spec_decode" / "cpu_amx_kernel.py"
)


def _load_kernel_module():
    for mod_name in ("vllm", "vllm.v1", "vllm.v1.spec_decode"):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)
    spec = importlib.util.spec_from_file_location(
        "wt_cpu_amx_kernel", WORKTREE_KERNEL_PATH
    )
    assert spec and spec.loader, f"cannot load {WORKTREE_KERNEL_PATH}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def case_forward_symbol_bound() -> None:
    print("=" * 72)
    print("CASE — forward symbol bound + signature set")
    print("=" * 72)
    sys.modules.pop("wt_cpu_amx_kernel", None)
    mod = _load_kernel_module()
    kern = mod.AmxDraftKernel.get()
    print(f"  lib: {kern.lib_path}")
    print(f"  hw_amx={kern.hw_amx} loaded={kern.loaded}")
    assert kern.loaded, f"dlopen failed: {kern.load_error}"
    assert hasattr(kern._lib, "amx_draft_qwen05b_forward"), \
        "amx_draft_qwen05b_forward not bound — rebuild the .so?"
    # Signature check
    sig = kern._lib.amx_draft_qwen05b_forward
    assert sig.restype is None, f"forward restype should be void, got {sig.restype}"
    assert len(sig.argtypes) == 5, \
        f"forward should have 5 argtypes, got {len(sig.argtypes)}"
    print("  [PASS] forward symbol bound + 5 argtypes set")


def case_forward_shape_dtype_argmax() -> None:
    print("=" * 72)
    print("CASE — forward shape / dtype / argmax range")
    print("=" * 72)
    sys.modules.pop("wt_cpu_amx_kernel", None)
    mod = _load_kernel_module()
    kern = mod.AmxDraftKernel.get()
    if not kern.is_available():
        print("  [SKIP] AMX not available")
        return
    rc = kern.ensure_init()
    assert rc == 0, f"init rc={rc}"

    Vc = mod.CONFIG_VOCAB_QWEN_05B
    V = mod.KERNEL_VOCAB
    H = mod.KERNEL_HIDDEN

    for B, K, mode in (
        (1, 7, "act_in"),  # input=None, kernel reuses init act_in
        (4, 7, "user"),    # input=user buffer
        (8, 3, "user"),
    ):
        if mode == "act_in":
            inp = None
        else:
            # Random BF16 input — values uniform in [-0.05, 0.05] (matches
            # kernel's init seed range so dynamic-range of logits is sane).
            rng = np.random.default_rng(0xCAFE + B + K)
            fp32 = rng.uniform(-0.05, 0.05, size=(B, H)).astype(np.float32)
            # Cast fp32 → bf16 via bit-shift truncation (round-to-nearest).
            u32 = fp32.view(np.uint32)
            inp = ((u32 + 0x8000) >> 16).astype(np.uint16)

        logits, ids = kern.forward(inp, B, K)

        # ── shape / dtype assertions ──
        assert logits.shape == (B, K, V), \
            f"B={B} K={K}: logits shape {logits.shape} != ({B},{K},{V})"
        assert logits.dtype == np.uint16, \
            f"B={B} K={K}: logits dtype {logits.dtype} != uint16(BF16)"
        assert ids.shape == (B, K), \
            f"B={B} K={K}: ids shape {ids.shape} != ({B},{K})"
        assert ids.dtype == np.int32, \
            f"B={B} K={K}: ids dtype {ids.dtype} != int32"

        # ── argmax range — IMPORTANT: must be in [0, Vc) NOT [0, V) ──
        id_min = int(ids.min())
        id_max = int(ids.max())
        assert 0 <= id_min, f"ids min {id_min} < 0"
        assert id_max < Vc, (
            f"B={B} K={K} mode={mode}: argmax id {id_max} >= Vc={Vc} "
            "— argmax must exclude padded vocab cols!"
        )

        # ── logits range sanity (BF16 of FP32 accumulator, random
        #     weights ⇒ values must be finite). ──
        # Reinterpret BF16 as FP32 by zero-extending into the high half.
        lo_fp32 = (logits.astype(np.uint32) << 16).view(np.float32)
        assert np.all(np.isfinite(lo_fp32)), \
            f"B={B} K={K}: non-finite logits detected"

        print(f"  [PASS] B={B} K={K} mode={mode}: "
              f"logits.shape={logits.shape} dtype={logits.dtype} "
              f"ids.shape={ids.shape} dtype={ids.dtype} "
              f"ids in [{id_min}, {id_max}] "
              f"(Vc={Vc}, V_padded={V})")


def case_forward_input_validation() -> None:
    print("=" * 72)
    print("CASE — forward input validation")
    print("=" * 72)
    sys.modules.pop("wt_cpu_amx_kernel", None)
    mod = _load_kernel_module()
    kern = mod.AmxDraftKernel.get()
    if not kern.is_available():
        print("  [SKIP] AMX not available")
        return
    kern.ensure_init()

    H = mod.KERNEL_HIDDEN
    # Bad B
    try:
        kern.forward(None, 0, 1)
        raise AssertionError("B=0 should raise ValueError")
    except ValueError as e:
        print(f"  B=0 -> ValueError: {e}")
    # Bad K
    try:
        kern.forward(None, 1, 0)
        raise AssertionError("K=0 should raise ValueError")
    except ValueError as e:
        print(f"  K=0 -> ValueError: {e}")
    # Wrong dtype
    try:
        bad = np.zeros((1, H), dtype=np.float32)
        kern.forward(bad, 1, 1)
        raise AssertionError("float32 input should raise TypeError")
    except TypeError as e:
        print(f"  fp32 input -> TypeError: {e}")
    # Wrong shape
    try:
        bad = np.zeros((1, H // 2), dtype=np.uint16)
        kern.forward(bad, 1, 1)
        raise AssertionError("wrong-shape input should raise ValueError")
    except ValueError as e:
        print(f"  bad shape -> ValueError: {e}")
    print("  [PASS] input validation")


# pytest entry points
def test_forward_symbol_bound():
    case_forward_symbol_bound()


def test_forward_shape_dtype_argmax():
    case_forward_shape_dtype_argmax()


def test_forward_input_validation():
    case_forward_input_validation()


def main() -> int:
    cases = [
        case_forward_symbol_bound,
        case_forward_shape_dtype_argmax,
        case_forward_input_validation,
    ]
    failures: list[tuple[str, BaseException]] = []
    for fn in cases:
        try:
            fn()
        except BaseException as e:
            failures.append((fn.__name__, e))
            print(f"  [FAIL] {fn.__name__}: {type(e).__name__}: {e}")
    print("=" * 72)
    if failures:
        print(f"FAILED: {len(failures)} case(s)")
        return 1
    print(f"ALL {len(cases)} FORWARD SMOKE CASES PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
