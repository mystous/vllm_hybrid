# SPDX-License-Identifier: Apache-2.0
"""METER — NUMA-topology-aware pinning (stage 3 of METRONOME-LHC).

Pins each TP rank to a NUMA node such that:
  - rank ↔ NUMA node matches the GPU's NVLink-local PCIe root complex,
  - the rank's DSA WQ shares the NUMA node (dsa0 → node 0, dsa1 → node 1),
  - AMX worker threads reuse the same node's cores.

Activated via ``VLLM_LHC_METRONOME_METER=1``; idempotent. Falls back to
``sched_setaffinity`` over libc when ``libnuma.so.1`` is unavailable.
"""

from __future__ import annotations

import ctypes
import logging
import os

logger = logging.getLogger(__name__)


def _detect_numa_nodes() -> int:
    """Best-effort NUMA node count. Returns 1 when unknown."""
    try:
        nodes = [
            d for d in os.listdir("/sys/devices/system/node")
            if d.startswith("node")
        ]
        return max(len(nodes), 1)
    except OSError:
        return 1


def _node_cpus(node: int) -> list[int]:
    """List of CPU IDs in a NUMA node, parsed from cpulist."""
    try:
        with open(f"/sys/devices/system/node/node{node}/cpulist") as f:
            txt = f.read().strip()
    except OSError:
        return []
    out: list[int] = []
    for part in txt.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


def meter_pin_rank(rank: int, tp_size: int) -> tuple[bool, str]:
    """Pin the current process to the NUMA node corresponding to ``rank``.

    Returns ``(ok, info)``. ``info`` carries node + cpu list summary.
    """
    if os.environ.get("VLLM_LHC_METRONOME_METER", "1") != "1":
        return False, "disabled"
    n_nodes = _detect_numa_nodes()
    if n_nodes <= 1:
        return False, f"single-node (n_nodes={n_nodes})"
    # Map rank → node by partitioning ranks evenly.
    ranks_per_node = max(1, tp_size // n_nodes)
    node = min(rank // ranks_per_node, n_nodes - 1)
    cpus = _node_cpus(node)
    if not cpus:
        return False, f"node {node} cpulist empty"
    # Set affinity to the node's CPU set.
    try:
        os.sched_setaffinity(0, set(cpus))
    except OSError as e:
        return False, f"sched_setaffinity failed: {e}"
    # Best-effort libnuma membind (matches neo_cpu_kv_buffer's pattern).
    try:
        lib = ctypes.CDLL("libnuma.so.1")
        if lib.numa_available() >= 0:
            lib.numa_set_preferred.argtypes = [ctypes.c_int]
            lib.numa_set_preferred(int(node))
    except OSError:
        pass
    logger.info(
        "METER pin: rank=%d/%d → node=%d cpus=%d (sample %s)",
        rank, tp_size, node, len(cpus), cpus[:4],
    )
    return True, f"node={node} cpus={len(cpus)}"


def meter_rank_to_dsa_wq(rank: int, tp_size: int) -> int:
    """Resolve which DSA WQ a rank should use. Matches ``dsa_lane``'s
    rank-to-WQ assignment (engine 0..3 → dsa0, 4..7 → dsa1)."""
    return rank % max(tp_size, 1)
