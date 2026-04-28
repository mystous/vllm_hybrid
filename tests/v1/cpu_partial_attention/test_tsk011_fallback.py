# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TSK_011 — deadline-aware cold path + GPU full-FA fallback (opt-F).

Covers:
  - ``_resolve_cold_deadline_s`` env parsing (unset / 0 / negative / invalid /
    positive).
  - ``_get_async_executor`` single-pool reuse.
  - ``forward_partial_with_lse_async`` Future contract — submit returns Future,
    .result(timeout=...) raises ``concurrent.futures.TimeoutError`` on slow
    work.
  - ``_record_cold_fallback_breadcrumb`` bounded firing log.
  - ``_fallback_full_fa_sdpa`` numerical equivalence vs hand-built reference
    SDPA on the same hot-prefix-on-GPU + cold-prefix-on-CPU layout.
"""

from __future__ import annotations

import concurrent.futures
import time
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from vllm.v1.attention.ops.cpu_partial_attention import (
    _get_async_executor,
)
from vllm.v1.attention.ops.kv_view_adapter import KVPageLayout, KVViewAdapter


# --------------------------------------------------------------------------
# _resolve_cold_deadline_s — env parsing
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("", None),
        ("0", None),
        ("0.0", None),
        ("-100", None),
        ("abc", None),
        ("500", 0.5),
        ("1000.0", 1.0),
        ("250.5", 0.2505),
    ],
)
def test_resolve_cold_deadline_s(raw, expected):
    from vllm.v1.attention.backends import flash_attn as fa
    with patch.object(fa, "_COLD_FALLBACK_DEADLINE_MS_RAW", raw):
        got = fa._resolve_cold_deadline_s()
        if expected is None:
            assert got is None
        else:
            assert got == pytest.approx(expected)


# --------------------------------------------------------------------------
# Async executor + future contract
# --------------------------------------------------------------------------

def test_async_executor_reuse():
    e1 = _get_async_executor()
    e2 = _get_async_executor()
    assert e1 is e2


def test_future_returns_result():
    executor = _get_async_executor()
    future = executor.submit(lambda: ("ok_a", "ok_b"))
    a, b = future.result(timeout=5)
    assert (a, b) == ("ok_a", "ok_b")


def test_future_timeout_raises():
    executor = _get_async_executor()

    def slow():
        time.sleep(0.5)
        return "done"

    future = executor.submit(slow)
    with pytest.raises(concurrent.futures.TimeoutError):
        future.result(timeout=0.05)
    # Drain so the single-worker pool is free for subsequent tests.
    future.result(timeout=5)


# --------------------------------------------------------------------------
# Breadcrumb — bounded firing log
# --------------------------------------------------------------------------

def test_fallback_breadcrumb_bounded():
    from vllm.v1.attention.backends import flash_attn as fa
    fa._COLD_FALLBACK_FIRING_COUNT = 0
    fa._COLD_FALLBACK_FIRING_LOG_DONE = False
    n = fa._COLD_FALLBACK_FIRING_LOG_LIMIT + 3
    for _ in range(n):
        fa._record_cold_fallback_breadcrumb(
            n_fallback_seqs=1, deadline_s=0.1, cold_blocks_total=10
        )
    assert fa._COLD_FALLBACK_FIRING_LOG_DONE is True
    assert fa._COLD_FALLBACK_FIRING_COUNT == fa._COLD_FALLBACK_FIRING_LOG_LIMIT


# --------------------------------------------------------------------------
# _fallback_full_fa_sdpa — numerical equivalence
# --------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fallback_full_fa_sdpa_matches_reference_decode(dtype):
    """opt-F fallback (SDPA) result equals a hand-built reference on the
    same hot-on-GPU + cold-on-CPU layout.

    Decode-only scope (q_len == 1). 2 cold blocks + 1 hot block, GQA factor 2.
    """
    from vllm.v1.attention.backends.flash_attn import _fallback_full_fa_sdpa

    device = torch.device("cuda:0")
    num_q_heads = 4
    num_kv_heads = 2  # GQA factor 2
    head_dim = 32
    block_size = 8
    n_cold_blocks = 2
    n_hot_blocks = 1
    seq_len_total = (n_cold_blocks + n_hot_blocks) * block_size  # 24
    n_q = 1

    torch.manual_seed(0)

    # GPU paged KV cache. We allocate 4 blocks; only block 1 is used as
    # hot for our seq.
    num_gpu_blocks = 4
    key_cache = torch.randn(
        num_gpu_blocks, block_size, num_kv_heads, head_dim,
        device=device, dtype=dtype,
    )
    value_cache = torch.randn_like(key_cache)

    # CPU paged KV — combined K+V layout, int8 canonical of dtype-typed pages.
    layout = KVPageLayout(
        head_dim=head_dim, num_kv_heads=num_kv_heads,
        block_size=block_size, dtype=dtype,
    )
    num_cpu_blocks = 4
    cpu_buf = torch.empty(
        num_cpu_blocks, layout.page_size_bytes, dtype=torch.int8
    )
    cpu_adapter = KVViewAdapter(cpu_buf, layout)
    g1 = torch.Generator()
    g1.manual_seed(1)
    g2 = torch.Generator()
    g2.manual_seed(2)
    # In-place fill via the typed view (shares storage with cpu_buf).
    with torch.no_grad():
        cpu_adapter.k_view().copy_(
            torch.randn(
                num_cpu_blocks, block_size, num_kv_heads, head_dim,
                generator=g1, dtype=dtype,
            )
        )
        cpu_adapter.v_view().copy_(
            torch.randn(
                num_cpu_blocks, block_size, num_kv_heads, head_dim,
                generator=g2, dtype=dtype,
            )
        )

    # block_table[0, :] = [cold_dummy, cold_dummy, hot_block_id=1, pad]
    # The function reads only [n_cold_blocks_i :] = [hot_block_id, pad].
    block_table = torch.tensor(
        [[0, 0, 1, 0]], dtype=torch.int32, device=device
    )
    # cold blocks live in CPU pages 2 and 3.
    cold_block_ids = torch.tensor([[2, 3]], dtype=torch.int32)

    softmax_scale = 1.0 / (head_dim ** 0.5)

    query = torch.randn(
        n_q, num_q_heads, head_dim, device=device, dtype=dtype
    )
    output = torch.zeros_like(query)

    cu_q_list = [0, 1]
    n_cold_list = [n_cold_blocks]
    seq_lens_total_cpu = torch.tensor([seq_len_total], dtype=torch.int32)
    fallback_seq_ids = [0]
    cpu_kv_cache = [cpu_buf]

    n_transferred = _fallback_full_fa_sdpa(
        output=output,
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        block_size=block_size,
        cu_q_list=cu_q_list,
        n_cold_list=n_cold_list,
        seq_lens_total_cpu=seq_lens_total_cpu,
        fallback_seq_ids=fallback_seq_ids,
        cpu_kv_cache=cpu_kv_cache,
        cold_kv_layout=layout,
        cold_block_ids=cold_block_ids,
        softmax_scale=softmax_scale,
    )
    assert n_transferred == n_cold_blocks

    # Reference: build full K/V manually from the same source tensors.
    cold_k_view = cpu_adapter.k_view()
    cold_v_view = cpu_adapter.v_view()
    ref_cold_k = (
        cold_k_view[2:4]
        .reshape(-1, num_kv_heads, head_dim)
        .to(device)
    )
    ref_cold_v = (
        cold_v_view[2:4]
        .reshape(-1, num_kv_heads, head_dim)
        .to(device)
    )
    ref_hot_k = key_cache[1].reshape(-1, num_kv_heads, head_dim)
    ref_hot_v = value_cache[1].reshape(-1, num_kv_heads, head_dim)

    ref_full_k = torch.cat([ref_cold_k, ref_hot_k], dim=0)
    ref_full_v = torch.cat([ref_cold_v, ref_hot_v], dim=0)

    q_b = query.transpose(0, 1).unsqueeze(0)
    k_b = ref_full_k.transpose(0, 1).unsqueeze(0)
    v_b = ref_full_v.transpose(0, 1).unsqueeze(0)
    ref_out = F.scaled_dot_product_attention(
        q_b, k_b, v_b,
        attn_mask=None, is_causal=False,
        scale=softmax_scale, enable_gqa=True,
    )
    ref_out = ref_out.squeeze(0).transpose(0, 1)

    # bf16/fp16 — within ~1 ulp.
    atol = 5e-3 if dtype is torch.bfloat16 else 1e-3
    rtol = 5e-2 if dtype is torch.bfloat16 else 1e-2
    torch.testing.assert_close(output, ref_out, atol=atol, rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fallback_full_fa_rejects_prefill():
    """Decode-only scope contract — q_len > 1 raises."""
    from vllm.v1.attention.backends.flash_attn import _fallback_full_fa_sdpa

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    num_q_heads = 2
    num_kv_heads = 1
    head_dim = 16
    block_size = 4

    layout = KVPageLayout(
        head_dim=head_dim, num_kv_heads=num_kv_heads,
        block_size=block_size, dtype=dtype,
    )
    cpu_buf = torch.empty(2, layout.page_size_bytes, dtype=torch.int8)
    cold_block_ids = torch.tensor([[0]], dtype=torch.int32)

    key_cache = torch.zeros(
        2, block_size, num_kv_heads, head_dim, device=device, dtype=dtype
    )
    value_cache = torch.zeros_like(key_cache)
    block_table = torch.tensor([[0, 1]], dtype=torch.int32, device=device)

    # n_q = 3 (prefill) — should reject.
    query = torch.randn(3, num_q_heads, head_dim, device=device, dtype=dtype)
    output = torch.zeros_like(query)

    with pytest.raises(RuntimeError, match="decode-only"):
        _fallback_full_fa_sdpa(
            output=output,
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            block_size=block_size,
            cu_q_list=[0, 3],
            n_cold_list=[1],
            seq_lens_total_cpu=torch.tensor([8], dtype=torch.int32),
            fallback_seq_ids=[0],
            cpu_kv_cache=[cpu_buf],
            cold_kv_layout=layout,
            cold_block_ids=cold_block_ids,
            softmax_scale=1.0,
        )
