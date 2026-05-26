"""IDE_016 / TSK_025 — Python wrapper for AVX-512 sampling + AMX matmul kernels.

vLLM `vllm/v1/sample/sampler.py:3521 _sample` 의 hook 으로 사용.
ENV `VLLM_USE_AVX512_SAMPLING=1` 시 활성화.

ABI 결정: pybind11 binding 은 numpy ndarray 인터페이스만 사용한다 (torch 2.11
가 GCC 13.3 으로 빌드되어 다른 GCC version 과 torch::Tensor type_caster ABI
가 호환되지 않음). torch.Tensor ↔ numpy 변환은 본 Python wrapper 가
zero-copy 로 처리한다 (torch.from_numpy / tensor.numpy()).

Public APIs:
    is_available()                : bool
    cpu_has_avx512() / amx_is_available(): runtime ISA probes
    topk(logits, k)               : top-k partial sort
    topp_cutoff(sorted_probs, p)  : top-p nucleus cutoff
    fused_sample(logits, k, p, T) : end-to-end sample
    apply_temperature(logits, T)  : in-place
    apply_logit_bias(logits, bias): in-place
    softmax(logits)
    apply_repetition_penalty / apply_frequency_penalty / apply_presence_penalty
    amx_matmul(A, B_packed) / amx_repack_b(B)
"""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch

try:
    import avx512_amx_pool._core as _core   # type: ignore[import-not-found]
    _CORE_LOADED = True
except ImportError:
    _core = None
    _CORE_LOADED = False


# ──────────────────────────────────────────────────────────────────────
# torch ↔ numpy view helpers
# ──────────────────────────────────────────────────────────────────────

def _bf16_tensor_to_uint16_view(t: torch.Tensor) -> np.ndarray:
    """Zero-copy view: torch.bfloat16 → np.uint16 (raw bits)."""
    assert t.dtype == torch.bfloat16
    assert t.is_cpu and t.is_contiguous()
    return t.view(torch.uint16).numpy()


def _uint16_ndarray_to_bf16_tensor(a: np.ndarray) -> torch.Tensor:
    """Zero-copy view: np.uint16 → torch.bfloat16."""
    t = torch.from_numpy(a)
    return t.view(torch.bfloat16)


def _to_contig(t: torch.Tensor) -> torch.Tensor:
    return t if t.is_contiguous() else t.contiguous()


# ──────────────────────────────────────────────────────────────────────
# Capability + env
# ──────────────────────────────────────────────────────────────────────

def is_available() -> bool:
    """True iff kernel built AND env enabled AND CPU has AVX-512."""
    if not _CORE_LOADED:
        return False
    if os.environ.get("VLLM_USE_AVX512_SAMPLING", "0") != "1":
        return False
    try:
        return bool(_core.cpu_has_avx512())
    except Exception:
        return False


def amx_is_available() -> bool:
    if not _CORE_LOADED:
        return False
    try:
        return bool(_core.cpu_has_amx())
    except Exception:
        return False


def request_amx_permission() -> bool:
    if not _CORE_LOADED:
        return False
    try:
        return _core.amx_request_permission() == 0
    except Exception:
        return False


def cpu_has_avx512() -> bool:
    if not _CORE_LOADED:
        return False
    try:
        return bool(_core.cpu_has_avx512())
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────
# Sampling APIs
# ──────────────────────────────────────────────────────────────────────

def topk(logits: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """AVX-512 top-k. Returns (indices [B,K] int64, values [B,K] float32)."""
    if not is_available():
        v, i = torch.topk(logits.float(), k=k, dim=-1)
        return i.long(), v.float()
    t = _to_contig(logits)
    if t.dtype == torch.bfloat16:
        idx_np, val_np = _core.topk_bf16(_bf16_tensor_to_uint16_view(t), k)
    elif t.dtype == torch.float32:
        idx_np, val_np = _core.topk_fp32(t.numpy(), k)
    else:
        # convert FP16 → FP32
        idx_np, val_np = _core.topk_fp32(t.float().numpy(), k)
    return torch.from_numpy(idx_np), torch.from_numpy(val_np)


def topp_cutoff(sorted_probs: torch.Tensor, p: float) -> torch.Tensor:
    """Top-p cutoff on already sorted descending probs (FP32). Returns [B] int64."""
    if not is_available():
        cum = sorted_probs.cumsum(dim=-1)
        keep = (cum < p).sum(dim=-1) + 1
        return keep.clamp(min=1, max=sorted_probs.size(-1)).long()
    t = _to_contig(sorted_probs.float())
    cut_np = _core.topp_cutoff(t.numpy(), p)
    return torch.from_numpy(cut_np)


def fused_sample(logits: torch.Tensor, k: int, p: float,
                temperature: float = 1.0, rng_seed: int = 0) -> torch.Tensor:
    """AVX-512 fused softmax + top-k + top-p + categorical sample."""
    if not is_available():
        return _torch_fallback_fused_sample(logits, k, p, temperature, rng_seed)
    t = _to_contig(logits)
    if t.dtype == torch.bfloat16:
        out_np = _core.fused_sample_bf16(
            _bf16_tensor_to_uint16_view(t), k, p, temperature, rng_seed)
    elif t.dtype == torch.float32:
        out_np = _core.fused_sample_fp32(t.numpy(), k, p, temperature, rng_seed)
    else:
        out_np = _core.fused_sample_fp32(t.float().numpy(), k, p, temperature, rng_seed)
    return torch.from_numpy(out_np)


def topk_topp(logits: torch.Tensor, k: int, p: float,
              temperature: float = 1.0, rng_seed: int = 0) -> torch.Tensor:
    """Alias for fused_sample."""
    return fused_sample(logits, k, p, temperature, rng_seed)


# ──────────────────────────────────────────────────────────────────────
# Logit processor APIs
# ──────────────────────────────────────────────────────────────────────

def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if not is_available():
        if temperature in (0.0, 1.0):
            return logits
        logits.div_(temperature)
        return logits
    t = _to_contig(logits)
    if t.dtype == torch.float32:
        _core.apply_temperature_fp32(t.numpy(), temperature)
    elif t.dtype == torch.bfloat16:
        _core.apply_temperature_bf16(_bf16_tensor_to_uint16_view(t), temperature)
    else:
        logits.div_(temperature)
    return t


def apply_logit_bias(logits: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    if not is_available() or logits.dtype != torch.float32:
        logits.add_(bias)
        return logits
    _core.apply_logit_bias(_to_contig(logits).numpy(),
                          _to_contig(bias.float()).numpy())
    return logits


def softmax(logits: torch.Tensor) -> torch.Tensor:
    if not is_available() or logits.dtype != torch.float32:
        return torch.softmax(logits.float(), dim=-1)
    out_np = _core.softmax(_to_contig(logits).numpy())
    return torch.from_numpy(out_np)


def apply_repetition_penalty(logits: torch.Tensor,
                            token_ids: torch.Tensor,
                            lengths: torch.Tensor,
                            penalty: float) -> torch.Tensor:
    if not is_available() or logits.dtype != torch.float32:
        # Fallback (scalar)
        for b in range(logits.size(0)):
            L = int(lengths[b].item())
            for n in range(L):
                t = int(token_ids[b, n].item())
                if 0 <= t < logits.size(1):
                    v = logits[b, t].item()
                    logits[b, t] = v / penalty if v > 0 else v * penalty
        return logits
    _core.apply_repetition_penalty(
        _to_contig(logits).numpy(),
        _to_contig(token_ids.to(torch.int32)).numpy(),
        _to_contig(lengths.to(torch.int32)).numpy(),
        penalty)
    return logits


def apply_frequency_penalty(logits: torch.Tensor, freq: torch.Tensor,
                           alpha: float) -> torch.Tensor:
    if not is_available() or logits.dtype != torch.float32:
        logits.sub_(freq.float() * alpha)
        return logits
    _core.apply_frequency_penalty(
        _to_contig(logits).numpy(),
        _to_contig(freq.to(torch.int32)).numpy(),
        alpha)
    return logits


def apply_presence_penalty(logits: torch.Tensor, freq: torch.Tensor,
                          alpha: float) -> torch.Tensor:
    if not is_available() or logits.dtype != torch.float32:
        mask = (freq > 0).float()
        logits.sub_(mask * alpha)
        return logits
    _core.apply_presence_penalty(
        _to_contig(logits).numpy(),
        _to_contig(freq.to(torch.int32)).numpy(),
        alpha)
    return logits


# ──────────────────────────────────────────────────────────────────────
# AMX matmul
# ──────────────────────────────────────────────────────────────────────

def amx_matmul(A: torch.Tensor, B_packed: torch.Tensor) -> torch.Tensor:
    """AMX BF16 matmul. A [M,K] bf16, B_packed [K/2, N*2] bf16 → C [M,N] fp32."""
    if not amx_is_available() or not _CORE_LOADED:
        # Fallback PyTorch: unpack and run float matmul
        K_pair, N2 = B_packed.shape
        N = N2 // 2
        K = K_pair * 2
        B_unpacked = (B_packed.view(K_pair, N, 2)
                     .permute(0, 2, 1).reshape(K, N).contiguous())
        return (A.float() @ B_unpacked.float())
    A_np = _bf16_tensor_to_uint16_view(_to_contig(A))
    B_np = _bf16_tensor_to_uint16_view(_to_contig(B_packed))
    C_np = _core.amx_matmul(A_np, B_np)
    return torch.from_numpy(C_np)


def amx_repack_b(B: torch.Tensor) -> torch.Tensor:
    """Repack BF16 [K, N] row-major → AMX [K/2, N*2]."""
    if not _CORE_LOADED:
        K, N = B.shape
        K_eff = K - (K % 2)
        out = torch.zeros((K_eff // 2 + (K % 2), N * 2), dtype=torch.bfloat16)
        for k in range(0, K_eff, 2):
            kp = k // 2
            for n in range(N):
                out[kp, n * 2 + 0] = B[k, n]
                out[kp, n * 2 + 1] = B[k + 1, n]
        if K % 2:
            kp = K_eff // 2
            for n in range(N):
                out[kp, n * 2 + 0] = B[K_eff, n]
        return out
    B_np = _bf16_tensor_to_uint16_view(_to_contig(B))
    out_np = _core.amx_repack_b(B_np)
    return _uint16_ndarray_to_bf16_tensor(out_np)


# ──────────────────────────────────────────────────────────────────────
# Reference / fallback
# ──────────────────────────────────────────────────────────────────────

def _torch_fallback_fused_sample(logits: torch.Tensor, k: int, p: float,
                                 temperature: float = 1.0,
                                 rng_seed: int = 0) -> torch.Tensor:
    """Pure PyTorch reference — fallback and correctness baseline."""
    x = logits.float()
    if temperature > 0 and temperature != 1.0:
        x = x / temperature
    if 0 < k < x.size(-1):
        topv, _ = torch.topk(x, k=k, dim=-1)
        thr = topv[..., -1:].expand_as(x)
        x = torch.where(x < thr, torch.full_like(x, float("-inf")), x)
    probs = torch.softmax(x, dim=-1)
    if 0 < p < 1.0:
        sp, si = torch.sort(probs, descending=True, dim=-1)
        cum = sp.cumsum(dim=-1)
        keep = (cum - sp) < p
        keep[..., 0] = True
        new_p = torch.zeros_like(probs)
        new_p.scatter_(-1, si, keep.float() * sp)
        probs = new_p / new_p.sum(dim=-1, keepdim=True).clamp_min(1e-10)
    if rng_seed:
        g = torch.Generator()
        g.manual_seed(rng_seed)
        return torch.multinomial(probs, num_samples=1, generator=g).squeeze(-1).long()
    return torch.multinomial(probs, num_samples=1).squeeze(-1).long()


def _torch_fallback_topk_topp(logits: torch.Tensor, k: int, p: float,
                              temperature: float = 1.0) -> torch.Tensor:
    return _torch_fallback_fused_sample(logits, k, p, temperature)


def topk_only(logits: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    return topk(logits, k)
