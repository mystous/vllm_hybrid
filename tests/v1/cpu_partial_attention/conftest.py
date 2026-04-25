# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared fixtures for IDE_006 (Cold-KV CPU Partial Attention) tests.

See ``shadow_assists/features/IDE_006/TST_001.md`` and ``TST_002.md``
for the test specification.
"""

from __future__ import annotations

import pytest
import torch


# --- Sweep dimensions --------------------------------------------------


@pytest.fixture(params=[torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def kv_dtype(request) -> torch.dtype:
    """KV dtype — Phase 1 scope: BF16 / FP16 only."""
    return request.param


@pytest.fixture(params=[16, 32], ids=["block16", "block32"])
def block_size(request) -> int:
    """Paged KV block size (tokens per block)."""
    return request.param


@pytest.fixture
def head_dim() -> int:
    """Head dim — Qwen2.5-7B baseline (PLN_001 §3 scope lock)."""
    return 128


@pytest.fixture
def num_kv_heads() -> int:
    """KV heads — Qwen2.5-7B GQA baseline (Q=32 / KV=4)."""
    return 4


# --- KV cache synthesis ------------------------------------------------


@pytest.fixture
def make_canonical_kv():
    """Factory for a synthetic canonical int8 KV cache tensor.

    See ``vllm/v1/kv_offload/spec.py:51`` ``CanonicalKVCaches`` —
    shape ``(num_blocks, page_size_in_bytes)``, dtype ``int8``.

    Returns a callable ``(num_blocks, page_size_bytes, *, seed=0)``.
    """

    def _factory(num_blocks: int, page_size_bytes: int, *, seed: int = 0):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randint(
            -128, 128, (num_blocks, page_size_bytes),
            dtype=torch.int8, generator=gen,
        )

    return _factory
