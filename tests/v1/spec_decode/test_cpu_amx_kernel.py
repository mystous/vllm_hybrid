# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Import-only smoke tests for cpu_amx_kernel ctypes binding.

These tests verify the *binding* — dlopen, symbol resolution, hw_amx
probe, safety gating — WITHOUT executing the AMX intrinsics on a CPU
that does not support them (which would SIGILL the test process).

`step_ms()` / `single_ms()` / `mlp_ms()` are only called when
`AmxDraftKernel.is_available()` returns True (CPUID reports AMX_TILE +
AMX_BF16). On the dev host (B200 / Alder Lake) this is False, and the
test asserts that calling those entries raises RuntimeError instead of
crashing.

Run (worktree-isolated):
  /workspace/vllm_dev_prj/bin/python -m pytest \
    tests/v1/spec_decode/test_cpu_amx_kernel.py -v

Or directly (does not need pytest):
  /workspace/vllm_dev_prj/bin/python \
    tests/v1/spec_decode/test_cpu_amx_kernel.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────
# Load the WORKTREE copy of cpu_amx_kernel.py directly. The installed
# vllm package may have a different / no copy, and the full vllm
# package needs the vllm._C native ext which we want to avoid in this
# smoke harness.
# ─────────────────────────────────────────────────────────────────────
WORKTREE_ROOT = Path("/workspace/poc_worktrees/wt_a1_cpudraft")
WORKTREE_KERNEL_PATH = (
    WORKTREE_ROOT / "vllm" / "v1" / "spec_decode" / "cpu_amx_kernel.py"
)


def _load_kernel_module():
    """Load cpu_amx_kernel.py without going through the vllm package."""
    # Stub vllm.* parents so any future inter-module imports do not
    # pull in the heavy vllm engine.
    for mod_name in (
        "vllm",
        "vllm.v1",
        "vllm.v1.spec_decode",
    ):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)

    spec = importlib.util.spec_from_file_location(
        "wt_cpu_amx_kernel", WORKTREE_KERNEL_PATH
    )
    assert spec and spec.loader, f"cannot load {WORKTREE_KERNEL_PATH}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────────────
# Test cases (procedural — runnable with or without pytest)
# ─────────────────────────────────────────────────────────────────────

def case_constants() -> None:
    """Kernel constants reflect the .so DraftState shape."""
    print("=" * 60)
    print("CASE 1 — kernel constants present + correct")
    print("=" * 60)
    mod = _load_kernel_module()
    assert mod.KERNEL_HIDDEN == 896
    assert mod.KERNEL_INTERMEDIATE == 4864
    assert mod.KERNEL_VOCAB == 152_064
    assert mod.CONFIG_VOCAB_QWEN_05B == 151_936
    assert mod.KERNEL_LAYERS == 24
    assert mod.KERNEL_B_MAX == 16
    # Padding sanity: kernel must be >= config and aligned to 128.
    assert mod.KERNEL_VOCAB >= mod.CONFIG_VOCAB_QWEN_05B
    assert mod.KERNEL_VOCAB % 128 == 0
    print(f"  KERNEL_VOCAB={mod.KERNEL_VOCAB} "
          f"CONFIG={mod.CONFIG_VOCAB_QWEN_05B} "
          f"Δ={mod.KERNEL_VOCAB - mod.CONFIG_VOCAB_QWEN_05B}")
    print("  [PASS] constants")


def case_default_library_path() -> None:
    """default_library_path resolves to the SUB_187 build artifact."""
    print("=" * 60)
    print("CASE 2 — default library path resolves to SUB_187/build/")
    print("=" * 60)
    os.environ.pop("VLLM_CPU_DRAFT_KERNEL_PATH", None)
    mod = _load_kernel_module()
    p = mod.default_library_path()
    print(f"  path: {p}")
    assert str(p).endswith("libamx_draft_qwen05b.so"), \
        f"unexpected default path: {p}"
    assert "SUB_187_amx_draft_head/build" in str(p), \
        f"default path not in SUB_187 build dir: {p}"
    assert p.exists(), f"library not present at expected path: {p}"
    print(f"  exists={p.exists()} size={p.stat().st_size} bytes")
    print("  [PASS] default path")


def case_env_override() -> None:
    """VLLM_CPU_DRAFT_KERNEL_PATH env wins over default."""
    print("=" * 60)
    print("CASE 3 — VLLM_CPU_DRAFT_KERNEL_PATH env override")
    print("=" * 60)
    override = "/tmp/this_path_does_not_exist_for_test.so"
    os.environ["VLLM_CPU_DRAFT_KERNEL_PATH"] = override
    try:
        mod = _load_kernel_module()
        p = mod.default_library_path()
        assert str(p) == str(Path(override).resolve()), \
            f"env override ignored: got {p}"
        print(f"  override path: {p}")
        print("  [PASS] env override")
    finally:
        os.environ.pop("VLLM_CPU_DRAFT_KERNEL_PATH", None)


def case_load_and_probe() -> None:
    """AmxDraftKernel.get() must dlopen + bind symbols + probe hw_amx.

    Whether AMX is available depends on the host CPU. We assert on the
    dlopen + symbol resolve side and only conditionally on hw_amx.
    """
    print("=" * 60)
    print("CASE 4 — AmxDraftKernel.get() dlopen + symbol + hw_amx probe")
    print("=" * 60)
    os.environ.pop("VLLM_CPU_DRAFT_KERNEL_PATH", None)
    # Drop any prior cached module so the lib cache starts fresh.
    sys.modules.pop("wt_cpu_amx_kernel", None)
    mod = _load_kernel_module()
    kern = mod.AmxDraftKernel.get()
    print(f"  lib_path: {kern.lib_path}")
    print(f"  loaded:   {kern.loaded}")
    print(f"  hw_amx:   {kern.hw_amx}")
    print(f"  error:    {kern.load_error}")
    assert kern.loaded, \
        f"dlopen / symbol resolve failed: {kern.load_error}"
    # All 6 entry points must have been bound.
    for sym in (
        "amx_draft_qwen05b_init",
        "amx_draft_qwen05b_free",
        "amx_draft_qwen05b_step_ms",
        "amx_draft_qwen05b_single_ms",
        "amx_draft_qwen05b_mlp_ms",
        "amx_draft_qwen05b_hw_amx",
    ):
        assert hasattr(kern._lib, sym), f"missing symbol: {sym}"
    print("  [PASS] dlopen + 6 symbols bound + hw_amx probed")


def case_safety_gate_on_dev_host() -> None:
    """On a non-AMX CPU, step_ms / single_ms / mlp_ms must raise — NOT
    SIGILL.

    On a Sapphire Rapids prod host, this case is a no-op and prints
    a [SKIP] marker (the actual prod-side execution is in the next
    PoC turn).
    """
    print("=" * 60)
    print("CASE 5 — safety gate: no-AMX CPU → RuntimeError (no SIGILL)")
    print("=" * 60)
    sys.modules.pop("wt_cpu_amx_kernel", None)
    mod = _load_kernel_module()
    kern = mod.AmxDraftKernel.get()
    if kern.hw_amx == 1:
        print("  [SKIP] CPU reports AMX; safety-gate test does not apply "
              "(prod-side execution test belongs to next turn).")
        return
    assert not kern.is_available(), \
        "is_available() must be False when hw_amx == 0"
    raised = 0
    for name, args in (
        ("step_ms", (1, 1)),
        ("single_ms", (1,)),
        ("mlp_ms", (1,)),
    ):
        try:
            getattr(kern, name)(*args)
            print(f"  [FAIL] {name}{args} did NOT raise — would SIGILL!")
        except RuntimeError as e:
            raised += 1
            print(f"  {name}{args} → RuntimeError: {e}")
    assert raised == 3, f"expected 3 RuntimeError raises, got {raised}"
    print("  [PASS] safety gate active — no kernel intrinsics executed")


def case_kernel_is_available_helper() -> None:
    """Top-level helper returns False on dev host, True on prod."""
    print("=" * 60)
    print("CASE 6 — kernel_is_available() top-level helper")
    print("=" * 60)
    sys.modules.pop("wt_cpu_amx_kernel", None)
    mod = _load_kernel_module()
    avail = mod.kernel_is_available()
    print(f"  kernel_is_available(): {avail}")
    # Type assertion: must always be bool, never raises.
    assert isinstance(avail, bool)
    print("  [PASS] helper returns bool without raising")


# ─────────────────────────────────────────────────────────────────────
# Optional pytest entry points (so `pytest -v` also runs the suite)
# ─────────────────────────────────────────────────────────────────────

def test_kernel_constants():
    case_constants()


def test_default_library_path():
    case_default_library_path()


def test_env_override():
    case_env_override()


def test_load_and_probe():
    case_load_and_probe()


def test_safety_gate_on_dev_host():
    case_safety_gate_on_dev_host()


def test_kernel_is_available_helper():
    case_kernel_is_available_helper()


def main() -> int:
    cases = [
        case_constants,
        case_default_library_path,
        case_env_override,
        case_load_and_probe,
        case_safety_gate_on_dev_host,
        case_kernel_is_available_helper,
    ]
    failures: list[tuple[str, BaseException]] = []
    for fn in cases:
        try:
            fn()
        except BaseException as e:
            failures.append((fn.__name__, e))
            print(f"  [FAIL] {fn.__name__}: {type(e).__name__}: {e}")
    print("=" * 60)
    if failures:
        print(f"FAILED: {len(failures)} case(s)")
        return 1
    print(f"ALL {len(cases)} SMOKE CASES PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
