# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NEO-style request-level GPU/CPU exclusive KV block manager.

Algorithms adapted from NEO (https://github.com/NEO-MLSys25/NEO,
MLSys 2025, Apache 2.0). Only the algorithms are reused.

The NEO design contrasts with vLLM's default ``OffloadingConnector``,
which mirrors blocks on both GPU and CPU. NEO instead enforces an
*exclusive* invariant: every request's KV cache lives on either GPU
*or* CPU, never both. ``_initiate_swap`` performs an atomic
``free(src) → alloc(dst)`` so the invariant holds at every observable
point.

This module is intentionally self-contained: it does not import
vLLM's existing ``KVCacheManager`` or ``BlockTable``. The two block
managers coexist; the NEO scheduler activates this manager only when
the user opts into the asymmetric pipelining mode (see TSK_014). The
default ``--kv-cache-policy=mirror`` path is unaffected.

See ``shadow_assists/features/IDE_006/NEO_code_deepdive.md`` §5
for the algorithm reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol


# ----------------------------------------------------------------------
# Minimal request-shaped Protocol — accepts vLLM's Request without
# importing it, to keep this module dependency-free for unit tests.
# ----------------------------------------------------------------------
class _BlockOwner(Protocol):
    request_id: int
    @property
    def num_tokens(self) -> int: ...   # current sequence length

    @property
    def is_finished(self) -> bool: ...


def _cdiv(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


# ----------------------------------------------------------------------
# Per-device block manager
# ----------------------------------------------------------------------
@dataclass
class _DeviceState:
    """Plain-Python state for one (device × split) free pool."""
    is_block_free: list[bool]
    num_free_blocks: int


class DeviceBlockManager:
    """Manage allocated and free blocks on a single device (CPU or GPU).

    Block tables are stored as a flat ``list[int]`` (length
    ``max_seqs * max_blocks_per_seq``); ``block_table[seq_id *
    max_blocks_per_seq + j]`` returns the physical block id.

    ``nsplits`` is ``1 + extra_layer_for_cprf``: when CPU-side
    prefill is enabled, the extra split holds intermediate KVs that
    the CPU prefill writes to a special "intermediate" GPU layer
    before being swapped to the CPU.
    """

    def __init__(
        self,
        *,
        device_name: str,
        num_blocks: int,
        block_size: int,
        max_seqs: int,
        max_blocks_per_seq: int,
        extra_layer_for_cprf: bool = False,
    ) -> None:
        self.device_name = device_name
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.max_seqs = max_seqs
        self.block_table_width = max_blocks_per_seq
        self.nsplits = 1 + int(extra_layer_for_cprf)

        # seq_id -> number of blocks currently held by this seq
        self.seq_num_blks: list[int] = [0] * max_seqs
        # Flat (seq_id, idx) -> physical block id table
        self.block_table: list[int] = [0] * (max_seqs * max_blocks_per_seq)

        # One free pool per split
        self._states = [
            _DeviceState(
                is_block_free=[True] * num_blocks,
                num_free_blocks=num_blocks,
            )
            for _ in range(self.nsplits)
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_new_blk_ids(self, count: int, split_id: int = 0) -> list[int]:
        """Return ``count`` new physical block ids from the requested
        split, marking them as used."""
        if count == 0:
            return []
        state = self._states[split_id]
        if count > state.num_free_blocks:
            raise RuntimeError(
                f"No free blocks on {self.device_name} split {split_id}: "
                f"requested {count}, free {state.num_free_blocks}, "
                f"total {self.num_blocks}"
            )
        result: list[int] = []
        for i, free in enumerate(state.is_block_free):
            if free:
                result.append(i)
                state.is_block_free[i] = False
                if len(result) == count:
                    break
        state.num_free_blocks -= count
        return result

    # ------------------------------------------------------------------
    # Public — alloc / free
    # ------------------------------------------------------------------
    def alloc(
        self,
        reqs: list[_BlockOwner],
        split_point: int = 0,
        omit_last: bool = False,
    ) -> tuple[list[int], list[int]]:
        """Ensure that every request has enough blocks for all its
        currently held tokens.

        Requests at indices ``[0, split_point)`` are allocated from
        split 1, the rest from split 0.

        Returns ``(new_block_vids, new_block_pids)``. The ``vid`` is
        ``seq_id * block_table_width + slot_idx`` so that callers can
        update the flat block table reference.
        """
        if not reqs:
            return [], []

        new_block_vids: list[int] = []
        new_block_pids: list[int] = []

        # Compute new block counts per request.
        per_req_new: list[int] = []
        for r in reqs:
            target = _cdiv(r.num_tokens - int(omit_last), self.block_size)
            held = self.seq_num_blks[r.request_id]
            if target < held:
                raise RuntimeError(
                    f"On {self.device_name}: request {r.request_id} has "
                    f"{held} blocks but only needs {target}"
                )
            per_req_new.append(target - held)

        new_count_split1 = sum(per_req_new[:split_point])
        new_count_split0 = sum(per_req_new[split_point:])
        pids_split1 = self._get_new_blk_ids(new_count_split1, 1)
        pids_split0 = self._get_new_blk_ids(new_count_split0, 0)

        cursor1 = 0
        cursor0 = 0
        for i, r in enumerate(reqs):
            held = self.seq_num_blks[r.request_id]
            new = per_req_new[i]
            if new == 0:
                continue
            if i < split_point:
                pids = pids_split1[cursor1:cursor1 + new]
                cursor1 += new
            else:
                pids = pids_split0[cursor0:cursor0 + new]
                cursor0 += new
            for j, pid in enumerate(pids):
                vid = r.request_id * self.block_table_width + held + j
                self.block_table[vid] = pid
                new_block_vids.append(vid)
                new_block_pids.append(pid)
            self.seq_num_blks[r.request_id] = held + new
        return new_block_vids, new_block_pids

    def free(
        self,
        reqs: list[_BlockOwner],
        split_id: int = 0,
    ) -> list[int]:
        """Return all blocks held by these requests to the free pool of
        ``split_id``. Returns the list of freed physical block ids.
        """
        if not reqs:
            return []
        state = self._states[split_id]
        freed: list[int] = []
        for r in reqs:
            held = self.seq_num_blks[r.request_id]
            for j in range(held):
                vid = r.request_id * self.block_table_width + j
                pid = self.block_table[vid]
                if not state.is_block_free[pid]:
                    state.is_block_free[pid] = True
                    state.num_free_blocks += 1
                    freed.append(pid)
            self.seq_num_blks[r.request_id] = 0
        return freed

    def num_free(self, split_id: int = 0) -> int:
        return self._states[split_id].num_free_blocks

    def held(self, request_id: int) -> int:
        return self.seq_num_blks[request_id]


# ----------------------------------------------------------------------
# Top-level — paired GPU + CPU block manager
# ----------------------------------------------------------------------
class NeoBlockManager:
    """Pair of GPU and CPU ``DeviceBlockManager`` enforcing the NEO
    *exclusive* invariant.

    A request is either *GPU-resident* (its KV is held by the GPU
    block manager) or *CPU-resident* — never both. Use
    ``_initiate_swap`` to atomically migrate a request between the two
    devices: it frees the source-side blocks and allocates fresh
    destination-side blocks in a single call.

    The physical-data PCIe transfer is the responsibility of a
    separate worker layer (``OffloadingConnector`` in vLLM's
    terminology); this manager only tracks *ownership*.
    """

    def __init__(
        self,
        *,
        block_size: int,
        max_seqs: int,
        max_blocks_per_seq: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        extra_layer_for_cprf: bool = False,
    ) -> None:
        self.block_size = block_size
        self.max_seqs = max_seqs
        self.max_blocks_per_seq = max_blocks_per_seq
        self.extra_layer_for_cprf = extra_layer_for_cprf

        self.gpu = DeviceBlockManager(
            device_name="cuda",
            num_blocks=num_gpu_blocks,
            block_size=block_size,
            max_seqs=max_seqs,
            max_blocks_per_seq=max_blocks_per_seq,
            extra_layer_for_cprf=extra_layer_for_cprf,
        )
        self.cpu = DeviceBlockManager(
            device_name="cpu",
            num_blocks=num_cpu_blocks,
            block_size=block_size,
            max_seqs=max_seqs,
            max_blocks_per_seq=max_blocks_per_seq,
            extra_layer_for_cprf=extra_layer_for_cprf,
        )

    # ------------------------------------------------------------------
    # Atomic swap (NEO `_initiate_swap`)
    # ------------------------------------------------------------------
    def initiate_swap(
        self,
        reqs: list[_BlockOwner],
        is_swap_out: bool,
        *,
        use_itm: bool = False,
        omit_last: bool = True,
    ) -> tuple[list[int], list[int], list[int]]:
        """Atomically migrate the given requests' ownership.

        Returns ``(src_pids, dst_vids, dst_pids)`` — the physical
        block ids freed on the source device and the new (vid, pid)
        pairs allocated on the destination device. ``use_itm=True``
        only applies to swap-out and selects the intermediate GPU
        split (the ``extra_layer_for_cprf`` policy).
        """
        if is_swap_out:
            src_mgr, dst_mgr = self.gpu, self.cpu
        else:
            assert not use_itm, "Cannot swap-in to intermediate split"
            src_mgr, dst_mgr = self.cpu, self.gpu

        if not reqs:
            return [], [], []

        src_pids = src_mgr.free(reqs, split_id=int(use_itm))
        dst_vids, dst_pids = dst_mgr.alloc(reqs, omit_last=omit_last)
        return src_pids, dst_vids, dst_pids

    # ------------------------------------------------------------------
    # Exclusive invariant
    # ------------------------------------------------------------------
    def assert_exclusive(self, reqs: Iterable[_BlockOwner]) -> None:
        """Sanity check: each request is held by exactly one device."""
        for r in reqs:
            on_gpu = self.gpu.held(r.request_id) > 0
            on_cpu = self.cpu.held(r.request_id) > 0
            assert on_gpu ^ on_cpu or (not on_gpu and not on_cpu), (
                f"Request {r.request_id} violates exclusive invariant: "
                f"on_gpu={on_gpu} on_cpu={on_cpu}"
            )

    # ------------------------------------------------------------------
    # Free at request finish
    # ------------------------------------------------------------------
    def free_finished(self, reqs: list[_BlockOwner]) -> tuple[list[int], list[int]]:
        """Free all blocks held by finished requests on both devices.
        Returns ``(gpu_pids, cpu_pids)`` — the freed physical ids."""
        # A request will only ever be on one side, but free() is safe
        # to call on the empty side (returns []).
        return self.gpu.free(reqs), self.cpu.free(reqs)
