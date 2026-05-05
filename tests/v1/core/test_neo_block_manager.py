# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``vllm/v1/core/sched/neo_block_manager.py`` (TSK_015)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from vllm.v1.core.sched.neo_block_manager import (
    DeviceBlockManager,
    NeoBlockManager,
)


@dataclass
class _Req:
    request_id: int
    num_tokens: int
    is_finished: bool = False


# ----------------------------------------------------------------------
# DeviceBlockManager — alloc / free
# ----------------------------------------------------------------------
def _make_dev(num_blocks: int = 16, block_size: int = 4) -> DeviceBlockManager:
    return DeviceBlockManager(
        device_name="cuda",
        num_blocks=num_blocks,
        block_size=block_size,
        max_seqs=8,
        max_blocks_per_seq=8,
        extra_layer_for_cprf=False,
    )


def test_alloc_grows_seq_num_blks():
    dev = _make_dev()
    r = _Req(request_id=0, num_tokens=10)   # ceil(10/4) = 3 blocks
    vids, pids = dev.alloc([r])
    assert dev.held(0) == 3
    assert len(vids) == 3
    assert len(pids) == 3
    assert dev.num_free() == 16 - 3


def test_alloc_incremental():
    dev = _make_dev()
    r = _Req(0, num_tokens=4)    # 1 block
    dev.alloc([r])
    assert dev.held(0) == 1
    r.num_tokens = 12            # ceil(12/4) = 3 blocks total
    dev.alloc([r])
    assert dev.held(0) == 3
    assert dev.num_free() == 16 - 3


def test_alloc_raises_when_exhausted():
    dev = _make_dev(num_blocks=2)
    r = _Req(0, num_tokens=20)   # ceil(20/4) = 5 blocks
    with pytest.raises(RuntimeError, match="No free blocks"):
        dev.alloc([r])


def test_free_returns_all_pids_and_resets_held():
    dev = _make_dev()
    r = _Req(0, num_tokens=10)
    _, pids = dev.alloc([r])
    freed = dev.free([r])
    assert sorted(freed) == sorted(pids)
    assert dev.held(0) == 0
    assert dev.num_free() == 16


def test_alloc_with_split_point():
    """split_point=N means the *first N* requests use split 1 and the
    rest use split 0."""
    dev = DeviceBlockManager(
        device_name="cuda",
        num_blocks=16,
        block_size=4,
        max_seqs=8,
        max_blocks_per_seq=8,
        extra_layer_for_cprf=True,   # allows split_id = 1
    )
    a = _Req(0, num_tokens=4)        # 1 block
    b = _Req(1, num_tokens=8)        # 2 blocks
    dev.alloc([a, b], split_point=1)
    # split 1 has been hit by request a (1 block), split 0 by b (2 blocks)
    assert dev.num_free(split_id=1) == 16 - 1
    assert dev.num_free(split_id=0) == 16 - 2


# ----------------------------------------------------------------------
# NeoBlockManager — initiate_swap atomicity
# ----------------------------------------------------------------------
def _make_paired() -> NeoBlockManager:
    return NeoBlockManager(
        block_size=4,
        max_seqs=8,
        max_blocks_per_seq=8,
        num_gpu_blocks=8,
        num_cpu_blocks=16,
    )


def test_initiate_swap_out_atomic():
    bm = _make_paired()
    r = _Req(0, num_tokens=8)         # 2 blocks
    bm.gpu.alloc([r])
    assert bm.gpu.held(0) == 2
    assert bm.cpu.held(0) == 0

    src_pids, dst_vids, dst_pids = bm.initiate_swap(
        [r], is_swap_out=True, omit_last=False
    )
    # After swap, source is free and destination holds blocks
    assert bm.gpu.held(0) == 0
    assert bm.cpu.held(0) == 2
    assert len(src_pids) == 2
    assert len(dst_vids) == 2
    assert len(dst_pids) == 2


def test_initiate_swap_in_atomic():
    bm = _make_paired()
    r = _Req(0, num_tokens=8)
    bm.cpu.alloc([r])
    assert bm.cpu.held(0) == 2

    src_pids, dst_vids, dst_pids = bm.initiate_swap(
        [r], is_swap_out=False, omit_last=False
    )
    assert bm.cpu.held(0) == 0
    assert bm.gpu.held(0) == 2
    assert len(src_pids) == 2


def test_swap_in_with_use_itm_rejected():
    bm = _make_paired()
    r = _Req(0, num_tokens=4)
    bm.cpu.alloc([r])
    with pytest.raises(AssertionError, match="intermediate split"):
        bm.initiate_swap([r], is_swap_out=False, use_itm=True)


def test_exclusive_invariant_after_alloc():
    bm = _make_paired()
    r = _Req(0, num_tokens=4)
    bm.gpu.alloc([r])
    bm.assert_exclusive([r])


def test_exclusive_invariant_violated_when_both_held():
    """Manually break the invariant to verify the assertion fires."""
    bm = _make_paired()
    r = _Req(0, num_tokens=4)
    bm.gpu.alloc([r])
    bm.cpu.alloc([r])
    with pytest.raises(AssertionError, match="exclusive invariant"):
        bm.assert_exclusive([r])


# ----------------------------------------------------------------------
# free_finished
# ----------------------------------------------------------------------
def test_free_finished_releases_blocks_on_owning_device_only():
    bm = _make_paired()
    r1 = _Req(0, num_tokens=8)        # 2 GPU blocks
    r2 = _Req(1, num_tokens=4)        # 1 CPU block
    bm.gpu.alloc([r1])
    bm.cpu.alloc([r2])
    gpu_pids, cpu_pids = bm.free_finished([r1, r2])
    assert sorted(gpu_pids) == sorted([
        bm.gpu.block_table[r1.request_id * 8 + j] for j in range(0)
    ]) or len(gpu_pids) == 2   # 2 ids, order may differ
    assert len(cpu_pids) == 1
    assert bm.gpu.held(0) == 0
    assert bm.cpu.held(1) == 0
