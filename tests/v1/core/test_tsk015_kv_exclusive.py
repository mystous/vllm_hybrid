# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for TSK_015 — KV cache exclusive ownership.

Coverage
--------
* ``NeoCpuKvBuffer`` — alloc/free/get_block_ids, capacity bounds.
* ``copy_layer_in/out`` — BF16 ↔ FP16 cast, roundtrip equality.
* ``NeoScheduler`` atomic swap helpers — XOR invariant on
  gpu_decoding_q / cpu_decoding_q under swap_in/out sequences.
* SchedulerConfig.kv_cache_policy default + literal validation.

See ``shadow_assists/features/IDE_006/TSK_015.md``.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass

import pytest
import torch

from vllm.v1.core.sched.neo_cpu_kv_buffer import (
    NeoCpuKvBuffer,
    NeoCpuKvBufferSpec,
)
from vllm.v1.core.sched.neo_scheduler import NeoScheduler
from vllm.v1.core.sched.perfpredictor import ZeroPerfPredictor


# ----------------------------------------------------------------------
# NeoCpuKvBuffer — alloc / free / capacity
# ----------------------------------------------------------------------
def _mini_spec(num_blocks: int = 16) -> NeoCpuKvBufferSpec:
    return NeoCpuKvBufferSpec(
        num_layers=2,
        num_kv_heads=2,
        block_size=4,
        head_dim=8,
        num_cpu_blocks=num_blocks,
        dtype=torch.float16,
    )


def test_buffer_alloc_consumes_pool_and_free_returns_blocks():
    buf = NeoCpuKvBuffer(_mini_spec(num_blocks=10))
    assert buf.num_free_blocks == 10
    assert buf.num_resident_reqs == 0

    ids_a = buf.alloc("req_a", num_blocks=3)
    assert ids_a is not None and len(ids_a) == 3
    assert buf.num_free_blocks == 7
    assert buf.num_resident_reqs == 1

    ids_b = buf.alloc("req_b", num_blocks=2)
    assert ids_b is not None and set(ids_a).isdisjoint(set(ids_b))
    assert buf.num_free_blocks == 5

    # Freed blocks return to pool, count drops back.
    freed = buf.free("req_a")
    assert freed == ids_a
    assert buf.num_free_blocks == 8
    assert buf.num_resident_reqs == 1
    # Re-alloc same id after free works (no stale state).
    ids_a2 = buf.alloc("req_a", num_blocks=4)
    assert ids_a2 is not None and len(ids_a2) == 4


def test_buffer_alloc_returns_none_when_pool_exhausted():
    buf = NeoCpuKvBuffer(_mini_spec(num_blocks=4))
    assert buf.alloc("req_x", num_blocks=10) is None
    assert buf.num_free_blocks == 4         # pool untouched on failure


def test_buffer_double_alloc_raises():
    buf = NeoCpuKvBuffer(_mini_spec())
    buf.alloc("req_dup", num_blocks=1)
    with pytest.raises(ValueError):
        buf.alloc("req_dup", num_blocks=1)


def test_buffer_get_block_ids_returns_none_for_missing():
    buf = NeoCpuKvBuffer(_mini_spec())
    assert buf.get_block_ids("never_alloc") is None


# ----------------------------------------------------------------------
# NeoCpuKvBuffer — copy_layer_in / copy_layer_out roundtrip
# ----------------------------------------------------------------------
def test_copy_layer_roundtrip_fp16_input():
    spec = _mini_spec(num_blocks=8)
    buf = NeoCpuKvBuffer(spec)
    buf.alloc("req_rt", num_blocks=2)

    k = torch.randn(2, spec.num_kv_heads, spec.block_size, spec.head_dim,
                    dtype=torch.float16)
    v = torch.randn(2, spec.num_kv_heads, spec.block_size, spec.head_dim,
                    dtype=torch.float16)
    buf.copy_layer_in("req_rt", layer_idx=0, k_src=k, v_src=v)

    k_out, v_out = buf.copy_layer_out("req_rt", layer_idx=0)
    # FP16-in / FP16-out → bit-exact.
    assert torch.equal(k_out, k)
    assert torch.equal(v_out, v)


def test_copy_layer_bf16_to_fp16_cast():
    spec = _mini_spec(num_blocks=8)
    buf = NeoCpuKvBuffer(spec)
    buf.alloc("req_cast", num_blocks=1)

    k_bf16 = torch.randn(1, spec.num_kv_heads, spec.block_size, spec.head_dim,
                         dtype=torch.bfloat16)
    v_bf16 = torch.randn(1, spec.num_kv_heads, spec.block_size, spec.head_dim,
                         dtype=torch.bfloat16)
    buf.copy_layer_in("req_cast", layer_idx=0, k_src=k_bf16, v_src=v_bf16)

    k_out, _ = buf.copy_layer_out("req_cast", layer_idx=0)
    assert k_out.dtype is torch.float16
    # Roundtrip via FP16 — equal to FP16 cast of input.
    assert torch.equal(k_out, k_bf16.to(torch.float16))


def test_copy_layer_in_rejects_shape_mismatch():
    spec = _mini_spec(num_blocks=8)
    buf = NeoCpuKvBuffer(spec)
    buf.alloc("req_bad", num_blocks=2)

    bad_k = torch.zeros(1, spec.num_kv_heads, spec.block_size, spec.head_dim,
                        dtype=torch.float16)        # 1 block, expected 2
    bad_v = torch.zeros(1, spec.num_kv_heads, spec.block_size, spec.head_dim,
                        dtype=torch.float16)
    with pytest.raises(ValueError, match=r"src tensors expect 2 blocks"):
        buf.copy_layer_in("req_bad", layer_idx=0, k_src=bad_k, v_src=bad_v)


def test_copy_layer_in_rejects_invalid_layer():
    spec = _mini_spec(num_blocks=4)
    buf = NeoCpuKvBuffer(spec)
    buf.alloc("req_lyr", num_blocks=1)
    k = torch.zeros(1, spec.num_kv_heads, spec.block_size, spec.head_dim,
                    dtype=torch.float16)
    with pytest.raises(ValueError, match=r"layer_idx 99"):
        buf.copy_layer_in("req_lyr", layer_idx=99, k_src=k, v_src=k)


# ----------------------------------------------------------------------
# NeoScheduler atomic swap helpers — XOR invariant (Phase 5.1)
# ----------------------------------------------------------------------
@dataclass
class _Req:
    request_id: int
    prompt_len: int
    num_tokens: int


def _scheduler_under_test() -> NeoScheduler:
    return NeoScheduler(
        max_batch_size=4,
        max_tokens_in_batch=64,
        block_size=4,
        num_gpu_blocks=64,
        num_cpu_blocks=128,
        num_layers=2,
        predictor=ZeroPerfPredictor(),
    )


def test_initiate_swap_out_moves_req_atomically():
    sch = _scheduler_under_test()
    r = _Req(request_id=42, prompt_len=4, num_tokens=4)
    sch.gpu_decoding_q.append(r)
    # caller pops then hands to helper
    victim = sch.gpu_decoding_q.pop()
    sch._initiate_swap_out(victim)
    assert r not in sch.gpu_decoding_q
    assert sch.cpu_decoding_q[0] is r


def test_initiate_swap_in_moves_req_atomically():
    sch = _scheduler_under_test()
    r = _Req(request_id=7, prompt_len=4, num_tokens=4)
    sch.cpu_decoding_q.appendleft(r)
    sch._initiate_swap_in(r)
    assert r not in sch.cpu_decoding_q
    assert sch.gpu_decoding_q[-1] is r


def test_xor_invariant_holds_after_swap_sequence(monkeypatch):
    """Repeated swap-out / swap-in sequence preserves the XOR invariant
    when ``ENABLE_NEO_INV=1`` is set."""
    monkeypatch.setenv("ENABLE_NEO_INV", "1")
    sch = _scheduler_under_test()
    reqs = [_Req(request_id=i, prompt_len=4, num_tokens=4) for i in range(3)]
    for r in reqs:
        sch.gpu_decoding_q.append(r)
    # Out → in → out cycle for each, invariant must hold throughout.
    for r in reqs:
        v = sch.gpu_decoding_q.pop()
        sch._initiate_swap_out(v)
        sch._assert_exclusive_invariant(where="post_out")
        sch._initiate_swap_in(v)
        sch._assert_exclusive_invariant(where="post_in")
        v2 = sch.gpu_decoding_q.pop()
        sch._initiate_swap_out(v2)
        sch._assert_exclusive_invariant(where="post_out2")


def test_xor_invariant_detects_violation(monkeypatch):
    """Manually create a both-queues state to confirm the invariant
    actually fires (so the assertion isn't a no-op)."""
    monkeypatch.setenv("ENABLE_NEO_INV", "1")
    sch = _scheduler_under_test()
    r = _Req(request_id=99, prompt_len=4, num_tokens=4)
    sch.gpu_decoding_q.append(r)
    sch.cpu_decoding_q.appendleft(r)        # corrupt state — both queues
    with pytest.raises(AssertionError, match="XOR invariant"):
        sch._assert_exclusive_invariant(where="manual_violation")


def test_xor_invariant_disabled_when_flag_unset(monkeypatch):
    """The check is opt-in — without the env flag, even a corrupted
    state must not raise (so prod hot path stays free)."""
    monkeypatch.delenv("ENABLE_NEO_INV", raising=False)
    sch = _scheduler_under_test()
    r = _Req(request_id=1, prompt_len=4, num_tokens=4)
    sch.gpu_decoding_q.append(r)
    sch.cpu_decoding_q.appendleft(r)        # corrupt state
    sch._assert_exclusive_invariant()       # no raise, assertion off


# ----------------------------------------------------------------------
# kv_cache_policy config flag
# ----------------------------------------------------------------------
def test_kv_cache_policy_default_is_mirror():
    from vllm.config.scheduler import SchedulerConfig
    cfg = SchedulerConfig.default_factory(max_model_len=128, max_num_seqs=4,
                                           max_num_batched_tokens=128)
    assert cfg.kv_cache_policy == "mirror"


def test_kv_cache_policy_accepts_exclusive():
    from vllm.config.scheduler import SchedulerConfig
    cfg = SchedulerConfig.default_factory(max_model_len=128, max_num_seqs=4,
                                           max_num_batched_tokens=128,
                                           kv_cache_policy="exclusive")
    assert cfg.kv_cache_policy == "exclusive"


def test_kv_cache_policy_rejects_invalid():
    from vllm.config.scheduler import SchedulerConfig
    with pytest.raises(Exception):
        SchedulerConfig.default_factory(max_model_len=128, max_num_seqs=4,
                                         max_num_batched_tokens=128,
                                         kv_cache_policy="bogus_value")


# ----------------------------------------------------------------------
# Buffer free is no-op for unknown req (defensive)
# ----------------------------------------------------------------------
def test_buffer_free_unknown_returns_none():
    buf = NeoCpuKvBuffer(_mini_spec())
    assert buf.free("never_existed") is None


# ----------------------------------------------------------------------
# Phase 5.3 — Buffer capacity stress (TST_015 partial — prod-scale concurrent
# in-flight measurement is deferred until Phase 4.4 real wiring lands; what
# we *can* verify here is the buffer-level claim that exclusive policy never
# under-allocates the CPU pool and recycles correctly under churn).
# ----------------------------------------------------------------------
def test_buffer_fills_pool_to_exact_capacity():
    """50 reqs each taking 2 blocks should fit into a 100-block pool with
    zero waste; the 51st must fail (None)."""
    spec = _mini_spec(num_blocks=100)
    buf = NeoCpuKvBuffer(spec)
    for i in range(50):
        ids = buf.alloc(f"r{i}", num_blocks=2)
        assert ids is not None and len(ids) == 2
    assert buf.num_free_blocks == 0
    assert buf.num_resident_reqs == 50
    # 51st must fail without partial alloc
    assert buf.alloc("r_overflow", num_blocks=1) is None
    assert buf.num_free_blocks == 0           # pool not corrupted on failure


def test_buffer_alloc_free_churn_preserves_pool_size():
    """Repeated alloc/free across many cycles never leaks blocks."""
    spec = _mini_spec(num_blocks=64)
    buf = NeoCpuKvBuffer(spec)
    for cycle in range(20):
        # alloc 8 reqs, 4 blocks each → consumes pool fully
        for i in range(8):
            ids = buf.alloc(f"c{cycle}_r{i}", num_blocks=4)
            assert ids is not None
        assert buf.num_free_blocks == 32      # 64 - 8*4
        for i in range(8):
            buf.free(f"c{cycle}_r{i}")
        assert buf.num_free_blocks == 64      # full restore each cycle


def test_buffer_id_uniqueness_under_full_pool():
    """Disjoint id sets across all resident reqs (no double-handing)."""
    spec = _mini_spec(num_blocks=32)
    buf = NeoCpuKvBuffer(spec)
    seen: set[int] = set()
    for i in range(8):
        ids = buf.alloc(f"u{i}", num_blocks=4)
        assert ids is not None
        s = set(ids)
        assert s.isdisjoint(seen)             # no id reused while alive
        seen.update(s)
    assert len(seen) == 32                    # full pool partitioned
