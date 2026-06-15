"""Unit test for VLLM_LHC_DSA_WQ_PER_RANK mapping logic.

Validates the _resolve_dev_path() function for rank 0–7 produces the
expected 2-device × 4-WQ mapping, and that fallback paths work.
"""

import importlib
import os
import sys

# Ensure repo root is on path
sys.path.insert(0, "/workspace/host_vllm_hybrid")


def _reload():
    """Re-import the lane module to pick up env changes."""
    import vllm.v1.lhc.dsa_lane as m

    importlib.reload(m)
    return m


def test_default_path():
    for k in (
        "VLLM_LHC_DSA_DEV",
        "VLLM_LHC_DSA_WQ_PER_RANK",
        "VLLM_LHC_DSA_RANK",
        "LOCAL_RANK",
        "RANK",
    ):
        os.environ.pop(k, None)
    m = _reload()
    assert m._resolve_dev_path() == "/dev/dsa/wq0.0", m._resolve_dev_path()
    print("[T1] default → wq0.0  OK")


def test_explicit_dev():
    os.environ["VLLM_LHC_DSA_DEV"] = "/dev/dsa/wq1.2"
    m = _reload()
    assert m._resolve_dev_path() == "/dev/dsa/wq1.2", m._resolve_dev_path()
    os.environ.pop("VLLM_LHC_DSA_DEV")
    print("[T2] explicit dev override  OK")


def test_per_rank_mapping():
    os.environ.pop("VLLM_LHC_DSA_DEV", None)
    os.environ["VLLM_LHC_DSA_WQ_PER_RANK"] = "1"
    expected = {
        0: "/dev/dsa/wq0.0",
        1: "/dev/dsa/wq0.1",
        2: "/dev/dsa/wq0.2",
        3: "/dev/dsa/wq0.3",
        4: "/dev/dsa/wq1.0",
        5: "/dev/dsa/wq1.1",
        6: "/dev/dsa/wq1.2",
        7: "/dev/dsa/wq1.3",
    }
    for r, exp in expected.items():
        os.environ["VLLM_LHC_DSA_RANK"] = str(r)
        m = _reload()
        got = m._resolve_dev_path()
        assert got == exp, f"rank {r}: expected {exp}, got {got}"
        print(f"[T3] rank={r} → {got}  OK")
    os.environ.pop("VLLM_LHC_DSA_WQ_PER_RANK")
    os.environ.pop("VLLM_LHC_DSA_RANK")


def test_local_rank_fallback():
    os.environ.pop("VLLM_LHC_DSA_DEV", None)
    os.environ["VLLM_LHC_DSA_WQ_PER_RANK"] = "1"
    os.environ["LOCAL_RANK"] = "5"
    os.environ.pop("VLLM_LHC_DSA_RANK", None)
    m = _reload()
    assert m._resolve_dev_path() == "/dev/dsa/wq1.1", m._resolve_dev_path()
    os.environ.pop("LOCAL_RANK")
    os.environ.pop("VLLM_LHC_DSA_WQ_PER_RANK")
    print("[T4] LOCAL_RANK=5 → wq1.1  OK")


def test_lane_gate_off_when_disabled():
    for k in (
        "VLLM_LHC_DSA",
        "VLLM_LHC_DSA_WQ_PER_RANK",
        "VLLM_LHC_DSA_DEV",
        "VLLM_LHC_DSA_RANK",
        "LOCAL_RANK",
        "RANK",
    ):
        os.environ.pop(k, None)
    m = _reload()
    assert m.dsa_lane_available() is False
    assert m.dsa_memcpy(0, 0, 4096) is False
    print("[T5] disabled gate fallthrough  OK")


if __name__ == "__main__":
    test_default_path()
    test_explicit_dev()
    test_per_rank_mapping()
    test_local_rank_fallback()
    test_lane_gate_off_when_disabled()
    print("\nALL PASS")
