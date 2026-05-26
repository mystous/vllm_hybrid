"""IDE_016 / TSK_025 — Python wrapper for AVX-512 sampling kernel.

vLLM `vllm/v1/sample/sampler.py:3521 _sample` 의 hook 으로 사용.
ENV `VLLM_USE_AVX512_SAMPLING=1` 시 활성화.

status: skeleton — C++ extension 미빌드 시 fallback to PyTorch path.
"""
from __future__ import annotations

import os
import torch
from typing import Tuple

try:
    from . import _avx512_sampling_cpp  # pybind11 C++ extension
    _AVX512_AVAILABLE = True
except ImportError:
    _avx512_sampling_cpp = None
    _AVX512_AVAILABLE = False


def is_available() -> bool:
    """Return True if AVX-512 sampling kernel is built and loadable."""
    return _AVX512_AVAILABLE and (os.environ.get("VLLM_USE_AVX512_SAMPLING", "0") == "1")


def topk_topp(
    logits: torch.Tensor,
    k: int,
    p: float,
    temperature: float = 1.0,
) -> torch.Tensor:
    """AVX-512 accelerated top-k/top-p sampling.

    Args:
        logits: [B, V] BF16 CPU tensor
        k: top-k value (≥1)
        p: top-p (nucleus) threshold (0 < p ≤ 1)
        temperature: softmax temperature

    Returns:
        [B] int64 sampled token IDs

    Falls back to PyTorch sampler if AVX-512 not available.
    """
    if not is_available():
        return _torch_fallback_topk_topp(logits, k, p, temperature)

    assert logits.device.type == "cpu", "AVX-512 sampling requires CPU tensor"
    assert logits.dtype == torch.bfloat16, "expects BF16 logits"
    assert logits.dim() == 2, f"expect [B, V], got shape {logits.shape}"

    return _avx512_sampling_cpp.fused_sample(
        logits.contiguous(), k, p, temperature
    )


def _torch_fallback_topk_topp(
    logits: torch.Tensor,
    k: int,
    p: float,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Reference implementation using PyTorch — used as correctness baseline + fallback."""
    logits = logits.float() / temperature  # apply temperature
    # top-k
    if k > 0 and k < logits.size(-1):
        values, _ = torch.topk(logits, k=k, dim=-1)
        threshold = values[..., -1:].expand_as(logits)
        logits = torch.where(logits < threshold, float("-inf"), logits)
    probs = torch.softmax(logits, dim=-1)
    # top-p (nucleus)
    if 0 < p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumprob = sorted_probs.cumsum(dim=-1)
        keep_mask = cumprob <= p
        # Always keep at least 1
        keep_mask[..., 0] = True
        # Map back to original index space
        new_probs = torch.zeros_like(probs)
        new_probs.scatter_(-1, sorted_idx, keep_mask.float() * sorted_probs)
        probs = new_probs / new_probs.sum(dim=-1, keepdim=True).clamp_min(1e-10)
    # categorical sample
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def topk_only(logits: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-k indices + values (for debugging / correctness test)."""
    if not is_available():
        v, i = torch.topk(logits, k=k, dim=-1)
        return i, v
    return _avx512_sampling_cpp.topk(logits.contiguous(), k)
