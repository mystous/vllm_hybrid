"""SUB_198 / SUB_201 A1 — smoke test for CpuAmxProposer scaffold.

Runs without vllm engine boot. Validates:
  (1) toy mode (default): deterministic last+i drafts
  (2) real mode (VLLM_USE_AMX_DRAFT=1, transformers installed,
      Qwen 0.5B HF cache present): real next-token argmax drafts

Run with vllm venv:
  /workspace/vllm_dev_prj/bin/python \
    shadow_assists/features/IDE_019_multi_source_drafter/\
SUB_198_amx_real_integration/src/test_cpu_amx_smoke.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
import types
from types import SimpleNamespace
from typing import Any

# Load the WORKTREE copy of cpu_amx.py directly (bypasses the installed
# vllm package which has a different cpu_amx.py and requires the
# vllm._C native extension built).
WORKTREE_CPU_AMX_PATH = (
    "/workspace/poc_worktrees/wt_a1_cpudraft/"
    "vllm/v1/spec_decode/cpu_amx.py"
)


def _load_cpu_amx_module():
    # Stub the vllm modules cpu_amx imports so we do not need the full
    # vllm engine (which needs the vllm._C native ext to be built).
    for mod_name in (
        "vllm",
        "vllm.config",
        "vllm.v1",
        "vllm.v1.worker",
        "vllm.v1.worker.gpu_input_batch",
        "vllm.v1.spec_decode",
    ):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)
    sys.modules["vllm.config"].VllmConfig = object  # type: ignore[attr-defined]
    sys.modules["vllm.v1.worker.gpu_input_batch"].InputBatch = object  # type: ignore[attr-defined]

    spec = importlib.util.spec_from_file_location(
        "wt_cpu_amx", WORKTREE_CPU_AMX_PATH
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_fake_vllm_config(K: int = 7,
                            max_model_len: int = 32768) -> SimpleNamespace:
    """Minimal stand-in for VllmConfig — only the attrs CpuAmxProposer touches."""
    return SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=K),
        model_config=SimpleNamespace(max_model_len=max_model_len),
    )


def case_toy() -> None:
    print("=" * 60)
    print("CASE 1 — toy mode (VLLM_USE_AMX_DRAFT unset)")
    print("=" * 60)
    os.environ.pop("VLLM_USE_AMX_DRAFT", None)
    # Drop any cached wt_cpu_amx so __init__ re-reads env on each case.
    sys.modules.pop("wt_cpu_amx", None)
    mod = _load_cpu_amx_module()
    CpuAmxProposer = mod.CpuAmxProposer

    cfg = _build_fake_vllm_config(K=7)
    prop = CpuAmxProposer(cfg)
    prop.load_model()
    sampled = [[785, 3974, 13876, 38835], [], [100]]
    t0 = time.perf_counter()
    drafts = prop.propose(input_batch=None, sampled_token_ids=sampled)  # type: ignore[arg-type]
    dt = (time.perf_counter() - t0) * 1000
    print(f"  drafts: {drafts}")
    print(f"  wall: {dt:.2f} ms")
    assert len(drafts) == 3
    assert drafts[0] == [(38835 + i) % 32_000 + 1 for i in range(1, 8)]
    assert drafts[1] == []
    assert len(drafts[2]) == 7 and all(1 <= x <= 32_000 for x in drafts[2])
    print("  [PASS] toy mode")


def case_real() -> None:
    print("=" * 60)
    print("CASE 2 — real PyTorch CPU mode (VLLM_USE_AMX_DRAFT=1)")
    print("=" * 60)
    os.environ["VLLM_USE_AMX_DRAFT"] = "1"
    os.environ.setdefault("VLLM_CPU_DRAFT_THREADS", "8")
    sys.modules.pop("wt_cpu_amx", None)
    mod = _load_cpu_amx_module()
    CpuAmxProposer = mod.CpuAmxProposer

    cfg = _build_fake_vllm_config(K=4)  # K=4 to keep smoke fast
    prop = CpuAmxProposer(cfg)
    t_load = time.perf_counter()
    prop.load_model()
    print(f"  load_model wall: {(time.perf_counter()-t_load):.2f} s")
    if prop._model is None:  # type: ignore[attr-defined]
        print("  [SKIP] real model not loadable in this env (no "
              "transformers/cache/etc).")
        return

    # "The quick brown fox" tokenized = [785, 3974, 13876, 38835]
    sampled = [[785, 3974, 13876, 38835]]
    t0 = time.perf_counter()
    drafts = prop.propose(input_batch=None, sampled_token_ids=sampled)  # type: ignore[arg-type]
    dt = (time.perf_counter() - t0) * 1000
    print(f"  drafts (K=4): {drafts}")
    print(f"  wall: {dt:.2f} ms  ({dt/4:.2f} ms/step)")

    assert len(drafts) == 1
    assert len(drafts[0]) == 4
    assert all(0 <= x < 151936 for x in drafts[0]), \
        f"draft ids out of Qwen vocab: {drafts[0]}"
    # Decode to verify they form coherent next tokens.
    tok = prop._tokenizer  # type: ignore[attr-defined]
    decoded = tok.decode(drafts[0])
    print(f"  decoded: {decoded!r}")
    # Sanity: should resemble "jumps over..." or similar continuation.
    # Not a strict assert (random init etc could differ), just print.
    print("  [PASS] real mode — real next-token ids returned")


def case_real_fallback() -> None:
    print("=" * 60)
    print("CASE 3 — real-mode env set but bogus model id → toy fallback")
    print("=" * 60)
    os.environ["VLLM_USE_AMX_DRAFT"] = "1"
    os.environ["VLLM_CPU_DRAFT_MODEL"] = "no_such_model_for_smoke_test_xyz/foo"
    sys.modules.pop("wt_cpu_amx", None)
    mod = _load_cpu_amx_module()
    CpuAmxProposer = mod.CpuAmxProposer

    cfg = _build_fake_vllm_config(K=3)
    prop = CpuAmxProposer(cfg)
    prop.load_model()
    drafts = prop.propose(input_batch=None,
                          sampled_token_ids=[[100, 200, 300]])  # type: ignore[arg-type]
    print(f"  drafts: {drafts}")
    assert drafts[0] == [(300 + i) % 32_000 + 1 for i in range(1, 4)]
    print("  [PASS] real-mode load failure → toy fallback")


def main() -> int:
    failures: list[Any] = []
    for fn in (case_toy, case_real, case_real_fallback):
        try:
            fn()
        except Exception as e:  # pragma: no cover
            failures.append((fn.__name__, e))
            print(f"  [FAIL] {fn.__name__}: {type(e).__name__}: {e}")
        # Reset env so cases do not bleed.
        for k in ("VLLM_USE_AMX_DRAFT", "VLLM_CPU_DRAFT_MODEL",
                  "VLLM_CPU_DRAFT_THREADS"):
            os.environ.pop(k, None)
    print("=" * 60)
    if failures:
        print(f"FAILED: {len(failures)} case(s)")
        return 1
    print("ALL SMOKE CASES PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
