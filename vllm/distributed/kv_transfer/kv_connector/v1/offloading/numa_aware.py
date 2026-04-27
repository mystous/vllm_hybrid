# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NUMA-aware bindings for the Cold-KV CPU partial attention path
(IDE_006 / TSK_004).

The OffloadingConnector and the partial-attention kernel together form
a hot path where a CPU buffer (allocated per worker) is read by that
worker's partial-attention kernel during decode. On a dual-socket
Sapphire Rapids box the default first-touch policy can land the buffer
on whichever NUMA node hits the page first, so the kernel may end up
reading 50 % of its data over UPI from the wrong socket — a 2~3×
latency penalty that throughput sweeps (TST_002) would attribute to
the algorithm rather than to topology.

This module provides two cooperating bindings, both opt-in via call
sites, both with safe fallbacks:

  * ``bind_worker_to_local_numa()`` calls ``numa_set_preferred(node)``
    on the current thread so that subsequent
    ``torch.zeros(..., pin_memory=True)`` allocations prefer the
    worker's local NUMA node. This addresses TSK_004 step (a) —
    connector buffer NUMA bind.

  * ``pin_threads_to_local_numa()`` calls
    ``os.sched_setaffinity(0, cores_of_local_node)`` on the worker
    process so every thread spawned (including OpenMP threads inside
    the partial-attention kernel) inherits the affinity. This
    addresses step (b) — partial-attention kernel thread bind.

Local node is determined in this priority order:

  1. The NUMA node of the worker's GPU
     (``current_platform.get_device_numa_node(local_rank)``). This
     keeps GPU↔CPU PCIe transfers on the same socket.
  2. ``rank % num_numa_nodes``. Round-robin distribution over nodes.
  3. None. Falls back to default first-touch policy.

Both bindings are idempotent and silent on systems where libnuma is
unavailable / NUMA is disabled / the platform has no NUMA support.
This keeps dev (single-socket Core i9-12900KF) and prod (dual-socket
Xeon 8480+) on the same code path with no behavioural change on dev.
"""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock

from vllm.logger import init_logger

logger = init_logger(__name__)

_lock = Lock()
_bind_done: dict[int, int | None] = {}  # rank -> node (or None if no-op)
_pin_done: dict[int, list[int] | None] = {}  # rank -> cpus (or None if no-op)


def _get_rank_world() -> tuple[int, int]:
    """Best-effort (rank, world_size) lookup. Returns (0, 1) when not
    in a distributed context."""
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            return int(dist.get_rank()), int(dist.get_world_size())
    except Exception:
        pass
    # vLLM stores local rank in this env var on worker subprocess startup.
    rank_env = os.environ.get("LOCAL_RANK")
    if rank_env is not None:
        try:
            return int(rank_env), int(os.environ.get("WORLD_SIZE", 1))
        except ValueError:
            pass
    return 0, 1


def _resolve_local_numa_node(rank: int, world_size: int) -> int | None:
    """Pick the NUMA node this worker should bind to.

    Strategy (in priority order):
      1. GPU NUMA node — keeps PCIe transfer between this worker's GPU
         and its CPU buffer on the same socket.
      2. ``rank % num_numa_nodes`` — round-robin fallback.
      3. None — caller should skip binding.
    """
    try:
        from vllm.platforms import current_platform

        if hasattr(current_platform, "get_device_numa_node"):
            local_rank = int(os.environ.get("LOCAL_RANK", rank))
            node = current_platform.get_device_numa_node(local_rank)
            if node is not None and node >= 0:
                return node
    except Exception as exc:  # pragma: no cover - best-effort
        logger.debug("get_device_numa_node failed: %r", exc)

    try:
        num_nodes = _count_numa_nodes()
        if num_nodes > 1:
            return rank % num_nodes
    except Exception as exc:  # pragma: no cover - best-effort
        logger.debug("_count_numa_nodes failed: %r", exc)

    return None


def _count_numa_nodes() -> int:
    """Count NUMA nodes that have CPUs assigned. Single-node systems
    return 1; missing /sys hierarchy returns 1."""
    node_root = Path("/sys/devices/system/node")
    if not node_root.exists():
        return 1
    count = 0
    for child in node_root.iterdir():
        if not child.name.startswith("node"):
            continue
        cpulist_file = child / "cpulist"
        try:
            if cpulist_file.read_text().strip():
                count += 1
        except OSError:
            continue
    return max(count, 1)


def _read_cpulist(node: int) -> list[int]:
    """Parse /sys/devices/system/node/nodeN/cpulist into an explicit
    list of CPU IDs. ``cpulist`` uses ranges like ``0-3,8-11``."""
    try:
        text = Path(f"/sys/devices/system/node/node{node}/cpulist").read_text()
    except OSError:
        return []
    cpus: list[int] = []
    for part in text.strip().split(","):
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            cpus.extend(range(int(lo_s), int(hi_s) + 1))
        else:
            cpus.append(int(part))
    return cpus


def _partition_node_cpus_for_rank(
    rank: int, world_size: int, node: int, cpus: list[int]
) -> list[int]:
    """Carve out this rank's slice of a NUMA node's cores.

    On a TP=8 / dual-socket prod box, four ranks land on the same NUMA
    node. Pinning every rank to *all* of that node's cores means each
    worker subprocess sees ``CPU_COUNT(affinity) = 56`` and the
    partial-attention kernel asks libgomp for 56 OpenMP threads — four
    workers × 56 = 224 threads on 56 cores, plus torch / flashinfer /
    ray threads, which exhausts ``RLIMIT_NPROC`` /
    ``kernel.threads-max`` and ``pthread_create`` returns ``EAGAIN``
    ("Thread creation failed: Resource temporarily unavailable").

    To avoid this we partition the node's cpulist into one contiguous
    slice per co-located rank. ``CPU_COUNT(affinity)`` then naturally
    returns ``cores_per_node / ranks_on_node``, the partial-attention
    kernel scales to its rightful share, and total OMP thread count
    across all workers stays ≤ physical core count.

    Single-rank-per-node (dev box, TP≤num_numa_nodes) is a no-op — the
    rank gets the full node, exactly as before.
    """
    if not cpus or world_size <= 1:
        return cpus
    same_node_ranks = sorted(
        r
        for r in range(world_size)
        if _resolve_local_numa_node(r, world_size) == node
    )
    if rank not in same_node_ranks:
        return cpus
    n_on_node = len(same_node_ranks)
    if n_on_node <= 1:
        return cpus
    idx = same_node_ranks.index(rank)
    base, rem = divmod(len(cpus), n_on_node)
    if base == 0:
        # More ranks than cores on this node — fall back to full node
        # rather than picking a single core (which would over-serialise
        # OpenMP). Operator can override via VLLM_PARTIAL_ATTN_THREADS.
        return cpus
    start = idx * base + min(idx, rem)
    extra = 1 if idx < rem else 0
    end = start + base + extra
    return cpus[start:end]


def bind_worker_to_local_numa(
    rank: int | None = None, world_size: int | None = None
) -> int | None:
    """Set the calling thread's preferred NUMA node so subsequent CPU
    allocations land on the worker's local socket.

    Idempotent — repeated calls in the same process are a no-op after
    the first successful invocation. Returns the bound node id, or
    ``None`` when binding was skipped (no libnuma, single node, etc.).
    """
    if rank is None or world_size is None:
        rank, world_size = _get_rank_world()

    with _lock:
        if rank in _bind_done:
            return _bind_done[rank]

        node = _resolve_local_numa_node(rank, world_size)
        if node is None:
            _bind_done[rank] = None
            logger.debug(
                "[IDE_006/TSK_004] rank=%d skipping NUMA bind "
                "(no local node resolved)",
                rank,
            )
            return None

        ok = False
        try:
            from vllm.platforms.intel_cpu_utils import NUMAAllocator

            allocator = NUMAAllocator()
            if allocator.is_available:
                ok = allocator.bind_to_node(node)
        except Exception as exc:  # pragma: no cover
            logger.debug("[IDE_006/TSK_004] NUMA allocator unavailable: %r", exc)

        _bind_done[rank] = node if ok else None
        if ok:
            logger.info(
                "[IDE_006/TSK_004] rank=%d bound CPU memory preference "
                "to NUMA node %d",
                rank,
                node,
            )
        else:
            logger.debug(
                "[IDE_006/TSK_004] rank=%d NUMA preferred bind skipped "
                "(libnuma unavailable)",
                rank,
            )
        return _bind_done[rank]


def pin_threads_to_local_numa(
    rank: int | None = None, world_size: int | None = None
) -> list[int] | None:
    """Pin this worker process's CPU affinity to the cores of its local
    NUMA node, so OpenMP / std::thread workers spawned later (notably
    by the CPU partial-attention kernel) inherit the same restriction
    and do not cross sockets.

    Idempotent. Returns the list of pinned CPU IDs, or ``None`` when
    pinning was skipped.
    """
    if rank is None or world_size is None:
        rank, world_size = _get_rank_world()

    with _lock:
        if rank in _pin_done:
            return _pin_done[rank]

        node = _resolve_local_numa_node(rank, world_size)
        if node is None:
            _pin_done[rank] = None
            return None

        node_cpus = _read_cpulist(node)
        if not node_cpus:
            _pin_done[rank] = None
            logger.debug(
                "[IDE_006/TSK_004] rank=%d node=%d empty cpulist; skipping "
                "thread pin",
                rank,
                node,
            )
            return None

        cpus = _partition_node_cpus_for_rank(rank, world_size, node, node_cpus)

        try:
            os.sched_setaffinity(0, cpus)
        except (AttributeError, OSError) as exc:
            _pin_done[rank] = None
            logger.debug(
                "[IDE_006/TSK_004] rank=%d sched_setaffinity failed: %r",
                rank,
                exc,
            )
            return None

        _pin_done[rank] = cpus
        logger.info(
            "[IDE_006/TSK_004] rank=%d pinned threads to NUMA node %d "
            "(%d/%d cores: %d~%d)",
            rank,
            node,
            len(cpus),
            len(node_cpus),
            cpus[0],
            cpus[-1],
        )
        return cpus
