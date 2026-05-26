"""IDE_018 / TSK_034 — End-to-end paper main result test scaffolding.

Protocol:
  canonical AGSD-gated balanced 500p, Qwen 32B TP=4×2 (prod) or scaled-down
  on dev (Qwen 1.5B TP=1 + RTX 3090).

  Two runs:
    - baseline: VLLM_USE_PHASE_BURST=0
    - phase-burst: VLLM_USE_PHASE_BURST=1, VLLM_PHASE_BURST_NUM_WORKERS=20

  Compare:
    - throughput (tps)         target: +10-20%
    - CPU util avg (monitor.py) target: 4.1% → 30%+
    - GPU util avg              target: 27.7% → 35-40%
    - per-token logprob max abs diff < 1e-3  (accuracy gate)

This file is *scaffolding* — actual benchmark runner is invoked in a separate
turn (vllm-build + GPU avail required).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

FEATURE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FEATURE_DIR))

# Try to load the built C++ module — skip if build hasn't been run.
try:
    import phase_burst  # noqa: F401
    HAS_CORE = True
except ImportError:
    HAS_CORE = False


# ── unit: build/import smoke ────────────────────────────────────────


@pytest.mark.skipif(not HAS_CORE, reason="phase_burst._core not built")
def test_import_and_enum():
    from phase_burst import Phase, TaskKind, MASK_ATTN, MASK_ANY
    assert int(Phase.ATTENTION) == 1
    assert int(TaskKind.A_SCHEDULE) == 0
    assert MASK_ATTN == (1 << 1)
    assert MASK_ANY == 0x3F


@pytest.mark.skipif(not HAS_CORE, reason="phase_burst._core not built")
def test_signal_round_trip():
    from phase_burst import PhaseBurstRuntime, Phase, TaskKind, MASK_ATTN

    rt = PhaseBurstRuntime(num_workers=2, cpu_base=0)
    rt.start()
    try:
        rt.mark_phase(Phase.ATTENTION, step_id=42)
        assert rt.signal.current() == int(Phase.ATTENTION)
        assert rt.signal.current_step() == 42

        cnt = {"n": 0}

        def fn():
            cnt["n"] += 1

        for _ in range(8):
            rt.enqueue_python(TaskKind.A_SCHEDULE, 42, MASK_ATTN, fn)

        deadline = time.time() + 3.0
        while cnt["n"] < 8 and time.time() < deadline:
            time.sleep(0.005)
        assert cnt["n"] == 8, f"only {cnt['n']} of 8 tasks fired"
    finally:
        rt.stop()


@pytest.mark.skipif(not HAS_CORE, reason="phase_burst._core not built")
def test_phase_signal_latency_p50():
    """TSK_031 target: p50 < 50 μs."""
    bench_bin = FEATURE_DIR / "build" / "phase_burst_bench"
    if not bench_bin.exists():
        pytest.skip(f"{bench_bin} not built")
    proc = subprocess.run([str(bench_bin), "2000"],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, f"bench failed: {proc.stderr}"
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    p50_us = data["p50_ns"] / 1000.0
    p99_us = data["p99_ns"] / 1000.0
    print(f"phase signal: p50={p50_us:.2f} μs / p99={p99_us:.2f} μs")
    assert p50_us < 50.0, f"p50 {p50_us:.2f} μs > target 50 μs"


# ── e2e: paper main result (heavy, marked separately) ────────────────


@pytest.mark.skipif(
    os.environ.get("RUN_PAPER_E2E", "0") != "1",
    reason="set RUN_PAPER_E2E=1 to run heavy benchmark",
)
def test_paper_main_throughput_delta():
    """Compares baseline vs phase-burst tps.
    target: +10-20% (5,474 → 6,021-6,569)."""
    # Driven by external runner script (paper main figure protocol).
    # Stub: expect summary json from canonical run.
    summary = Path("/tmp/IDE_018_e2e_summary.json")
    if not summary.exists():
        pytest.skip(f"no summary at {summary} — run runner first")
    data = json.loads(summary.read_text())
    base_tps = data["baseline"]["throughput_tps"]
    burst_tps = data["phase_burst"]["throughput_tps"]
    delta = (burst_tps - base_tps) / base_tps
    print(f"throughput baseline={base_tps:.0f} burst={burst_tps:.0f} "
          f"delta={delta:+.1%}")
    assert delta >= 0.05, f"throughput delta {delta:+.1%} < +5% min gate"

    base_cpu = data["baseline"]["cpu_util_avg"]
    burst_cpu = data["phase_burst"]["cpu_util_avg"]
    print(f"CPU util baseline={base_cpu:.1f}% burst={burst_cpu:.1f}%")
    assert burst_cpu >= 20.0, f"CPU util {burst_cpu:.1f}% < 20% gate"


@pytest.mark.skipif(
    os.environ.get("RUN_PAPER_E2E", "0") != "1",
    reason="set RUN_PAPER_E2E=1 to run heavy benchmark",
)
def test_accuracy_gate_logprob_diff():
    summary = Path("/tmp/IDE_018_e2e_summary.json")
    if not summary.exists():
        pytest.skip("no summary")
    data = json.loads(summary.read_text())
    max_abs_diff = data["accuracy"]["logprob_max_abs_diff"]
    assert max_abs_diff < 1e-3, \
        f"per-token logprob max abs diff {max_abs_diff} >= 1e-3 gate"
