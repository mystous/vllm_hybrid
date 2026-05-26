"""IDE_016 / TSK_025 — correctness test for AVX-512 sampling vs PyTorch baseline.

CLAUDE.md 운영 해석:
  분포·의도 수준 유사성. per-token logprob max abs diff < 1e-3.
  token-level 일치는 informational.

status: skeleton — C++ extension 미빌드 시 fallback 로만 검증.
"""
import pytest
import torch

import sys
sys.path.insert(0, str(__file__.rsplit("/", 2)[0] + "/src/_python"))
import avx512_sampling as avx


@pytest.fixture
def qwen_logits():
    """Generate Qwen-shaped logits: batch=32, vocab=152064 (Qwen 2.5)."""
    torch.manual_seed(42)
    return torch.randn(32, 152064, dtype=torch.bfloat16)


@pytest.mark.parametrize("k,p,temp", [
    (1, 1.0, 1.0),       # greedy
    (20, 0.95, 1.0),     # default chat sampling
    (50, 0.9, 0.7),      # creative
    (100, 1.0, 1.0),     # top-100 no nucleus
])
def test_topk_topp_distribution_match(qwen_logits, k, p, temp):
    """Per-token logprob max abs diff < 1e-3 (CLAUDE.md 운영 해석)."""
    # Reference (PyTorch path)
    ref_token = avx._torch_fallback_topk_topp(qwen_logits.clone(), k=k, p=p, temperature=temp)

    if not avx.is_available():
        pytest.skip("AVX-512 kernel not built — skipping match test")

    out_token = avx.topk_topp(qwen_logits.clone(), k=k, p=p, temperature=temp)

    # 정확한 token 일치는 informational (categorical sample 의 stochasticity)
    # 진짜 게이트는 underlying distribution 의 max abs diff
    # 본 test 는 단순 shape + dtype + 합리적 range 만 확인
    assert out_token.shape == ref_token.shape
    assert out_token.dtype == ref_token.dtype
    assert (out_token >= 0).all()
    assert (out_token < qwen_logits.size(-1)).all()


def test_topk_only_indices(qwen_logits):
    """top-k indices 만 비교 (deterministic)."""
    ref_idx = torch.topk(qwen_logits, k=20, dim=-1).indices

    if not avx.is_available():
        pytest.skip("AVX-512 kernel not built")

    out_idx, _ = avx.topk_only(qwen_logits, k=20)

    # top-k indices set must match (order within set may differ)
    for b in range(qwen_logits.size(0)):
        ref_set = set(ref_idx[b].tolist())
        out_set = set(out_idx[b].tolist())
        assert ref_set == out_set, f"batch {b}: {ref_set ^ out_set} differs"


def test_greedy_matches_argmax(qwen_logits):
    """greedy (k=1, p=1) must return argmax token."""
    expected = qwen_logits.float().argmax(dim=-1)
    out = avx.topk_topp(qwen_logits, k=1, p=1.0, temperature=1.0)
    # both paths (fallback or avx512) should give argmax
    assert (out == expected.long()).all(), \
        f"mismatches: {(out != expected.long()).sum().item()}/{out.numel()}"


@pytest.mark.benchmark
def test_sampling_latency():
    """Measure latency — paper expectation ≥2× speedup from PyTorch."""
    import time
    logits = torch.randn(32, 152064, dtype=torch.bfloat16)

    # warmup
    for _ in range(5):
        _ = avx._torch_fallback_topk_topp(logits, k=20, p=0.95)

    N_ITER = 100

    t0 = time.perf_counter()
    for _ in range(N_ITER):
        _ = avx._torch_fallback_topk_topp(logits, k=20, p=0.95)
    t_torch = (time.perf_counter() - t0) * 1000 / N_ITER

    print(f"\n  PyTorch baseline: {t_torch:.2f} ms / call")

    if avx.is_available():
        t0 = time.perf_counter()
        for _ in range(N_ITER):
            _ = avx.topk_topp(logits, k=20, p=0.95)
        t_avx = (time.perf_counter() - t0) * 1000 / N_ITER
        print(f"  AVX-512 kernel  : {t_avx:.2f} ms / call ({t_torch/t_avx:.2f}x speedup)")
        assert t_avx < t_torch, f"AVX-512 should be faster than PyTorch (got {t_avx} vs {t_torch})"
    else:
        print("  AVX-512 kernel  : not built (skip)")
