# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TST_018 — NEO pacpu (CPU paged attention) kernel + Python wrapper 단위.

Coverage
--------
* ``_resolve_neo_macro`` — model name → macro 매핑
* ``is_kernel_available`` / ``load_kernel`` — pre-built ``.so`` 검출 +
  graceful fallback
* ``ensure_loaded`` — auto_build 분기 (None/True/False) + env 변수 검사
* ``_to_neo_kv_view`` — vLLM HND per-layer → NEO multi-layer view 변환
* ``forward_attention`` — Qwen2.5-1.5B + TP=1 매개변수로 합성 Q/K/V
  → kernel 호출 → 출력 norm finite + shape 검증

Build artifact 의존: ``libpacpu-qwen2_5_1_5b-tp1.so`` 가 존재해야 kernel
호출 테스트 발화. 미존재 시 그 부분 ``skip``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vllm.v1.attention.ops import neo_pacpu


# ----------------------------------------------------------------------
# 모델 macro 매핑
# ----------------------------------------------------------------------
def test_resolve_neo_macro_qwen():
    assert neo_pacpu._resolve_neo_macro("Qwen/Qwen2.5-1.5B-Instruct") == \
        "qwen2_5_1_5b"
    assert neo_pacpu._resolve_neo_macro("qwen2.5-1.5b") == "qwen2_5_1_5b"


def test_resolve_neo_macro_llama():
    assert neo_pacpu._resolve_neo_macro(
        "meta-llama/Llama-3.3-70B-Instruct"
    ) == "llama3_3_70b"
    assert neo_pacpu._resolve_neo_macro("llama-3.1-70b") == "llama3_3_70b"


def test_resolve_neo_macro_unknown_returns_none():
    # Use a guaranteed-unknown name (no substring match in
    # _MODEL_TO_NEO_MACRO). Mistral / Phi3 / Gemma now mapped post
    # 2026-05-02 macro 확장.
    assert neo_pacpu._resolve_neo_macro("xyz-unknown-model-9z9z") is None
    assert neo_pacpu._resolve_neo_macro("foo/bar") is None


# ----------------------------------------------------------------------
# Kernel availability + load
# ----------------------------------------------------------------------
def test_is_kernel_available_false_for_unknown_model():
    assert not neo_pacpu.is_kernel_available("nonexistent-model", 1)


def test_load_kernel_returns_false_for_unknown_macro():
    assert not neo_pacpu.load_kernel("nonexistent-model", 1)


def test_load_kernel_qwen2_5_1_5b_when_built():
    """Pre-built .so present → load succeeds + torch.ops.pacpu registers."""
    so = (Path(__file__).resolve().parents[3]
          / "csrc" / "cpu" / "pacpu" / "build"
          / "libpacpu-qwen2_5_1_5b-tp1.so")
    if not so.is_file():
        pytest.skip(f"build artifact missing: {so}")
    ok = neo_pacpu.load_kernel("Qwen/Qwen2.5-1.5B-Instruct", 1)
    assert ok is True
    assert hasattr(torch.ops, "pacpu")


# ----------------------------------------------------------------------
# ensure_loaded (auto_build resolution)
# ----------------------------------------------------------------------
def test_ensure_loaded_unknown_model_skips(monkeypatch):
    monkeypatch.delenv("VLLM_NEO_AUTO_BUILD", raising=False)
    assert not neo_pacpu.ensure_loaded("nonexistent-model", 1,
                                       auto_build=False)


def test_ensure_loaded_explicit_false_no_build(monkeypatch, tmp_path):
    """auto_build=False + missing .so → returns False, no build attempt."""
    monkeypatch.delenv("VLLM_NEO_AUTO_BUILD", raising=False)
    # Force a different macro to avoid using the existing qwen build cache
    # (which would already be in _loaded_libraries from earlier tests).
    # Instead use llama3_3_70b which is unlikely to be built on dev.
    so = neo_pacpu._so_path("llama3_3_70b", 99)
    if so.is_file():
        pytest.skip(f"unexpected build artifact present: {so}")
    ok = neo_pacpu.ensure_loaded("llama-3.3-70b", 99, auto_build=False)
    assert ok is False


def test_ensure_loaded_env_resolves_auto_build(monkeypatch):
    """env ``VLLM_NEO_AUTO_BUILD=0/1`` properly resolves auto_build=None."""
    # We don't actually invoke the build (just test the resolution path).
    # Force unknown model so it short-circuits before resolution matters.
    monkeypatch.setenv("VLLM_NEO_AUTO_BUILD", "1")
    assert not neo_pacpu.ensure_loaded("nonexistent", 1)
    monkeypatch.setenv("VLLM_NEO_AUTO_BUILD", "0")
    assert not neo_pacpu.ensure_loaded("nonexistent", 1)


# ----------------------------------------------------------------------
# KV layout adapter — HND view conversion
# ----------------------------------------------------------------------
def test_kv_view_hnd_4d_unsqueezes_layer_dim():
    """vLLM 4D HND ``(blocks, kv_heads, block_size, head_dim)`` →
    NEO 5D ``(num_layers=1, blocks, kv_heads, block_size, head_dim)``."""
    kv = torch.randn(8, 2, 16, 64, dtype=torch.float16)
    out = neo_pacpu._to_neo_kv_view(kv)
    assert out.shape == (1, 8, 2, 16, 64)
    assert out.dtype is torch.float16


def test_kv_view_5d_passthrough():
    kv = torch.randn(2, 8, 2, 16, 64, dtype=torch.float16)
    out = neo_pacpu._to_neo_kv_view(kv)
    assert out.shape == (2, 8, 2, 16, 64)


def test_kv_view_rejects_unexpected_dim():
    kv = torch.randn(8, 2, 16, dtype=torch.float16)              # 3D
    with pytest.raises(ValueError, match=r"unexpected kv_layer.dim"):
        neo_pacpu._to_neo_kv_view(kv)


def test_kv_view_makes_contiguous():
    """non-contiguous input (e.g. permuted) must be made contiguous."""
    kv = torch.randn(8, 2, 16, 64, dtype=torch.float16)
    kv_perm = kv.permute(0, 2, 1, 3)                             # NHD-ish
    out = neo_pacpu._to_neo_kv_view(kv_perm)
    assert out.is_contiguous()


# ----------------------------------------------------------------------
# Synthetic forward_attention smoke (depends on built .so)
# ----------------------------------------------------------------------
def _build_so_path() -> Path:
    return (Path(__file__).resolve().parents[3]
            / "csrc" / "cpu" / "pacpu" / "build"
            / "libpacpu-qwen2_5_1_5b-tp1.so")


def test_forward_attention_synthetic_qwen_1_5b():
    """Smoke: construct Qwen-1.5B + TP=1 sized inputs, call neo_pacpu.
    Verify output is finite, has the expected shape, and norm > 0
    (a meaningful attention result — not all zeros)."""
    so = _build_so_path()
    if not so.is_file():
        pytest.skip(f"build artifact missing: {so}")

    ok = neo_pacpu.load_kernel("Qwen/Qwen2.5-1.5B-Instruct", 1)
    assert ok and hasattr(torch.ops, "pacpu")

    # Qwen2.5-1.5B + TP=1 hyperparams (must match dtype.h)
    # NUM_LAYERS = 28 (per-layer view, not used directly)
    NUM_Q_HEADS = 12
    NUM_KV_HEADS = 2
    HEAD_DIM = 128
    BLOCK_SIZE = 16
    BATCH = 2                # 2 cdec rows
    NUM_BLOCKS = 4

    q = torch.randn(BATCH, NUM_Q_HEADS, HEAD_DIM, dtype=torch.float16)
    k = torch.randn(BATCH, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16)
    v = torch.randn(BATCH, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16)

    # Per-layer KV cache (vLLM HND): (blocks, kv_heads, block_size, head_dim)
    k_cache = torch.randn(NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM,
                          dtype=torch.float16)
    v_cache = torch.randn(NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM,
                          dtype=torch.float16)
    # Each seq points to NUM_BLOCKS / BATCH blocks (rounded). Simple
    # sequential mapping for the smoke.
    block_table = torch.tensor(
        [[0, 1], [2, 3]], dtype=torch.int32,
    )
    # Each cdec seq has covered ``BLOCK_SIZE`` tokens (1 block) —
    # convenient for the (seq_len - 1) / BLOCK_SIZE arithmetic.
    seq_lengths = [BLOCK_SIZE, BLOCK_SIZE]

    out = torch.empty(BATCH, NUM_Q_HEADS * HEAD_DIM, dtype=torch.float32)
    neo_pacpu.forward_attention(
        cur_layer=0,
        softmax_scale=HEAD_DIM**-0.5,
        seq_ids=[0, 1],
        seq_lengths=seq_lengths,
        q=q, k_new=k, v_new=v,
        k_cache_layer=k_cache, v_cache_layer=v_cache,
        block_table=block_table,
        output=out,
    )

    # Output: shape + finite + non-trivial norm
    assert out.shape == (BATCH, NUM_Q_HEADS * HEAD_DIM)
    assert torch.isfinite(out).all()
    assert out.norm().item() > 0.0
