# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TSK_004 — vllm/distributed/.../offloading/numa_aware.py 의 단위 테스트.

prod (Xeon SPR + H100×8, dual-socket) 에서 ``_partition_node_cpus_for_rank``
가 잘못된 슬라이스를 돌려주면 OpenMP thread oversubscription 으로
``pthread_create: EAGAIN`` (libgomp) 이 터졌다 — 이전 회귀 incident. dev
머신은 단일 socket 이라 *자연스럽게는 그 경로를 밟지 않으므로*, monkeypatch
로 가짜 8-rank × 2-node 토폴로지를 만들어 partition 수학을 직접 검증한다.

본 파일이 다루는 contract:

* ``_partition_node_cpus_for_rank``:
    - TP=1 → 전체 cpus 그대로
    - TP=2 같은 node → 2-way 균등 분할
    - TP=8 dual-socket (4 ranks per node) → 4-way 균등 분할 (per node)
    - TP=4 single node (4 ranks per node) → 4-way 균등 분할
    - cpus < ranks (oversubscription) → 전체 cpus (single-core slice 회피)
    - empty cpus → []
    - rank 가 node 에 속하지 않으면 (sanity guard) 전체 cpus
* ``_resolve_local_numa_node`` 우선순위:
    - GPU NUMA node 가 결정되면 그 값
    - GPU 없으면 rank % num_nodes
    - 단일 node 면 None
* ``bind_worker_to_local_numa`` / ``pin_threads_to_local_numa`` idempotency
* ``pin_threads_to_local_numa`` 의 ``_pin_fast_done`` lock-free fast path
* ``_count_numa_nodes`` 와 ``_read_cpulist`` 의 /sys 파싱
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.offloading import (
    numa_aware as na,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_module_state():
    """각 테스트가 깨끗한 module-level 상태로 시작하도록 격리."""
    saved_bind = dict(na._bind_done)
    saved_pin = dict(na._pin_done)
    saved_fast = na._pin_fast_done
    na._bind_done.clear()
    na._pin_done.clear()
    na._pin_fast_done = False
    try:
        yield
    finally:
        na._bind_done.clear()
        na._bind_done.update(saved_bind)
        na._pin_done.clear()
        na._pin_done.update(saved_pin)
        na._pin_fast_done = saved_fast


def _fake_resolver(rank_to_node: dict[int, int | None]):
    """``_resolve_local_numa_node(rank, world_size)`` 를 dict 매핑으로 대체."""

    def resolver(rank: int, world_size: int) -> int | None:
        return rank_to_node.get(rank)

    return resolver


# ---------------------------------------------------------------------------
# _partition_node_cpus_for_rank
# ---------------------------------------------------------------------------


def test_partition_tp1_returns_full_list(monkeypatch):
    monkeypatch.setattr(
        na, "_resolve_local_numa_node", _fake_resolver({0: 0})
    )
    cpus = list(range(8))
    assert na._partition_node_cpus_for_rank(0, world_size=1, node=0, cpus=cpus) == cpus


def test_partition_tp2_same_node_splits_in_half(monkeypatch):
    monkeypatch.setattr(
        na, "_resolve_local_numa_node",
        _fake_resolver({0: 0, 1: 0}),
    )
    cpus = list(range(8))
    rank0 = na._partition_node_cpus_for_rank(0, world_size=2, node=0, cpus=cpus)
    rank1 = na._partition_node_cpus_for_rank(1, world_size=2, node=0, cpus=cpus)
    assert rank0 == [0, 1, 2, 3]
    assert rank1 == [4, 5, 6, 7]
    # 합집합 == 원본 (cpu 누락 없음)
    assert sorted(rank0 + rank1) == cpus


def test_partition_tp8_dual_socket_56_cores(monkeypatch):
    """prod 토폴로지 — 8 rank, node 0/1 각 4 rank, 56 cores/node."""
    rank_to_node = {r: (0 if r < 4 else 1) for r in range(8)}
    monkeypatch.setattr(
        na, "_resolve_local_numa_node", _fake_resolver(rank_to_node)
    )
    node0_cpus = list(range(56))
    node1_cpus = list(range(56, 112))

    slices_node0 = [
        na._partition_node_cpus_for_rank(r, 8, 0, node0_cpus)
        for r in range(4)
    ]
    slices_node1 = [
        na._partition_node_cpus_for_rank(r, 8, 1, node1_cpus)
        for r in range(4, 8)
    ]
    # 각 슬라이스는 14 cores (56/4) — OpenMP 가 14 thread 만 쓰도록 강제
    for s in slices_node0 + slices_node1:
        assert len(s) == 14, (
            "TP=8 dual-socket 에서 한 rank 가 14 코어를 초과 받으면 "
            "총 OMP thread = 4 rank × 14 = 56 ≤ 코어 (정상). 14 보다 크면 oversubscription."
        )
    # 합집합이 원본 — 누락/중복 없음
    flat0 = [c for s in slices_node0 for c in s]
    flat1 = [c for s in slices_node1 for c in s]
    assert sorted(flat0) == node0_cpus
    assert sorted(flat1) == node1_cpus
    # 슬라이스가 contiguous (kernel locality)
    for s in slices_node0 + slices_node1:
        assert s == list(range(s[0], s[0] + len(s)))


def test_partition_tp4_single_node_splits_4_way(monkeypatch):
    """TP=4 가 모두 같은 node 에 — 14 cores/rank (56/4)."""
    monkeypatch.setattr(
        na, "_resolve_local_numa_node",
        _fake_resolver({0: 0, 1: 0, 2: 0, 3: 0}),
    )
    cpus = list(range(56))
    slices = [na._partition_node_cpus_for_rank(r, 4, 0, cpus) for r in range(4)]
    for s in slices:
        assert len(s) == 14
    flat = [c for s in slices for c in s]
    assert sorted(flat) == cpus


def test_partition_uneven_remainder(monkeypatch):
    """50 cores / 4 ranks → 13, 13, 12, 12 (rem=2 → 첫 두 rank 가 +1)."""
    monkeypatch.setattr(
        na, "_resolve_local_numa_node",
        _fake_resolver({0: 0, 1: 0, 2: 0, 3: 0}),
    )
    cpus = list(range(50))
    sizes = [
        len(na._partition_node_cpus_for_rank(r, 4, 0, cpus))
        for r in range(4)
    ]
    assert sizes == [13, 13, 12, 12], (
        f"divmod(50, 4) = (12, 2) → 첫 두 rank +1. 실제: {sizes}"
    )
    # 합 == 50
    assert sum(sizes) == 50


def test_partition_oversubscription_returns_full(monkeypatch):
    """ranks 보다 cpus 가 적으면 single-core slice 가 아니라 전체 반환.
    prod 의 NPROC 한계 회피 의도가 망가지면 대신 fallback — operator 가
    VLLM_PARTIAL_ATTN_THREADS 로 별도 제어."""
    monkeypatch.setattr(
        na, "_resolve_local_numa_node",
        _fake_resolver({0: 0, 1: 0, 2: 0, 3: 0}),
    )
    cpus = [0, 1]  # 4 ranks 보다 적음
    for r in range(4):
        assert na._partition_node_cpus_for_rank(r, 4, 0, cpus) == cpus


def test_partition_empty_cpus(monkeypatch):
    monkeypatch.setattr(na, "_resolve_local_numa_node", _fake_resolver({0: 0}))
    assert na._partition_node_cpus_for_rank(0, 1, 0, []) == []
    # world_size>1 + empty 도 안전
    assert na._partition_node_cpus_for_rank(0, 4, 0, []) == []


def test_partition_rank_not_on_node_returns_full(monkeypatch):
    """sanity guard — rank 가 입력 node 에 속하지 않으면 partition 안 함."""
    monkeypatch.setattr(
        na, "_resolve_local_numa_node",
        _fake_resolver({0: 0, 1: 1}),
    )
    # rank 1 은 node 1 인데 node 0 의 cpu list 로 호출됨 → 전체 반환
    cpus = list(range(8))
    assert na._partition_node_cpus_for_rank(1, 2, 0, cpus) == cpus


# ---------------------------------------------------------------------------
# _resolve_local_numa_node priority
# ---------------------------------------------------------------------------


def test_resolve_uses_gpu_numa_node_when_available(monkeypatch):
    """current_platform.get_device_numa_node 가 정상값을 주면 그게 1순위."""

    class FakePlat:
        @staticmethod
        def get_device_numa_node(local_rank):
            return 1

    fake_module = type("M", (), {"current_platform": FakePlat()})
    monkeypatch.setitem(__import__("sys").modules, "vllm.platforms", fake_module)
    monkeypatch.setenv("LOCAL_RANK", "3")
    assert na._resolve_local_numa_node(3, world_size=8) == 1


def test_resolve_falls_back_to_round_robin(monkeypatch):
    """GPU NUMA 가 None 이면 rank % num_nodes."""

    class FakePlat:
        @staticmethod
        def get_device_numa_node(local_rank):
            return None  # platform 이 모름

    fake_module = type("M", (), {"current_platform": FakePlat()})
    monkeypatch.setitem(__import__("sys").modules, "vllm.platforms", fake_module)
    monkeypatch.setattr(na, "_count_numa_nodes", lambda: 2)
    assert na._resolve_local_numa_node(0, world_size=8) == 0
    assert na._resolve_local_numa_node(1, world_size=8) == 1
    assert na._resolve_local_numa_node(7, world_size=8) == 1


def test_resolve_returns_none_on_single_node(monkeypatch):
    """단일 socket dev 머신 — round-robin 도 의미 없으므로 None."""

    class FakePlat:
        @staticmethod
        def get_device_numa_node(local_rank):
            return None

    fake_module = type("M", (), {"current_platform": FakePlat()})
    monkeypatch.setitem(__import__("sys").modules, "vllm.platforms", fake_module)
    monkeypatch.setattr(na, "_count_numa_nodes", lambda: 1)
    assert na._resolve_local_numa_node(0, world_size=1) is None
    assert na._resolve_local_numa_node(3, world_size=4) is None


# ---------------------------------------------------------------------------
# _read_cpulist / _count_numa_nodes — /sys 파싱
# ---------------------------------------------------------------------------


def test_read_cpulist_parses_ranges(monkeypatch, tmp_path):
    node_dir = tmp_path / "node5"
    node_dir.mkdir()
    (node_dir / "cpulist").write_text("0-3,8-11,15\n")

    real_path = Path

    def fake_path(arg):
        if isinstance(arg, str) and arg.startswith("/sys/devices/system/node/node5/"):
            return tmp_path / arg.replace("/sys/devices/system/node/", "")
        return real_path(arg)

    monkeypatch.setattr(na, "Path", fake_path)
    cpus = na._read_cpulist(5)
    assert cpus == [0, 1, 2, 3, 8, 9, 10, 11, 15]


def test_read_cpulist_missing_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(na, "Path", lambda _: tmp_path / "nope")
    assert na._read_cpulist(99) == []


def test_count_numa_nodes_no_sys_returns_one(monkeypatch, tmp_path):
    """/sys/devices/system/node 가 없는 환경 (container 등) → 1."""
    monkeypatch.setattr(na, "Path", lambda _: tmp_path / "absent")
    assert na._count_numa_nodes() == 1


# ---------------------------------------------------------------------------
# bind_worker_to_local_numa idempotency / no-libnuma silent
# ---------------------------------------------------------------------------


def test_bind_idempotent_returns_cached_value(monkeypatch):
    """같은 rank 로 두 번 부르면 두 번째는 cache hit."""
    monkeypatch.setattr(
        na, "_resolve_local_numa_node", lambda r, ws: 0
    )

    call_count = {"n": 0}

    class FakeAllocator:
        is_available = True

        def bind_to_node(self, node):
            call_count["n"] += 1
            return True

    fake_mod = type("M", (), {"NUMAAllocator": FakeAllocator})
    monkeypatch.setitem(
        __import__("sys").modules, "vllm.platforms.intel_cpu_utils", fake_mod
    )

    n1 = na.bind_worker_to_local_numa(rank=0, world_size=1)
    n2 = na.bind_worker_to_local_numa(rank=0, world_size=1)
    assert n1 == 0 and n2 == 0
    assert call_count["n"] == 1, "idempotent — 두 번째 호출은 allocator 안 부름"


def test_bind_returns_none_when_no_local_node(monkeypatch):
    monkeypatch.setattr(na, "_resolve_local_numa_node", lambda r, ws: None)
    assert na.bind_worker_to_local_numa(rank=0, world_size=1) is None


def test_bind_returns_none_when_libnuma_missing(monkeypatch):
    """import 자체가 실패해도 RuntimeError 없이 None — silent no-op."""
    monkeypatch.setattr(na, "_resolve_local_numa_node", lambda r, ws: 0)

    import sys as _sys

    real_modules = _sys.modules.copy()
    # intel_cpu_utils 가 import 되면 ImportError 흉내
    if "vllm.platforms.intel_cpu_utils" in _sys.modules:
        monkeypatch.delitem(_sys.modules, "vllm.platforms.intel_cpu_utils")

    class _FailModule:
        def __getattr__(self, _):
            raise ImportError("simulated libnuma absent")

    monkeypatch.setitem(
        _sys.modules, "vllm.platforms.intel_cpu_utils", _FailModule()
    )
    assert na.bind_worker_to_local_numa(rank=0, world_size=1) is None
    _ = real_modules  # keep alive


# ---------------------------------------------------------------------------
# pin_threads_to_local_numa fast path / idempotency
# ---------------------------------------------------------------------------


def test_pin_idempotent_and_fast_path(monkeypatch):
    """첫 호출 후 _pin_fast_done=True. 두 번째부터는 lock 없이 cache 반환."""
    monkeypatch.setattr(na, "_resolve_local_numa_node", lambda r, ws: 0)
    monkeypatch.setattr(na, "_read_cpulist", lambda node: [0, 1, 2, 3])

    affinity_calls: list = []
    monkeypatch.setattr(
        na.os, "sched_setaffinity",
        lambda pid, cpus: affinity_calls.append((pid, sorted(cpus))),
    )

    cpus_first = na.pin_threads_to_local_numa(rank=0, world_size=1)
    assert cpus_first == [0, 1, 2, 3]
    assert na._pin_fast_done is True

    cpus_second = na.pin_threads_to_local_numa(rank=0, world_size=1)
    assert cpus_second == [0, 1, 2, 3]
    assert len(affinity_calls) == 1, (
        "두 번째 호출은 cache hit — sched_setaffinity 다시 부르면 안 됨"
    )


def test_pin_returns_none_when_no_node(monkeypatch):
    monkeypatch.setattr(na, "_resolve_local_numa_node", lambda r, ws: None)
    assert na.pin_threads_to_local_numa(rank=0, world_size=1) is None
    assert na._pin_fast_done is True, "no-op 결정도 fast path 로 캐시"


def test_pin_returns_none_when_empty_cpulist(monkeypatch):
    """node 는 결정됐지만 /sys/.../cpulist 가 비어 있음 → silent skip."""
    monkeypatch.setattr(na, "_resolve_local_numa_node", lambda r, ws: 3)
    monkeypatch.setattr(na, "_read_cpulist", lambda node: [])
    assert na.pin_threads_to_local_numa(rank=0, world_size=1) is None


def test_pin_handles_oserror_silently(monkeypatch):
    """sched_setaffinity 가 OSError (e.g. permission denied in container) →
    silent None, raise 없음."""
    monkeypatch.setattr(na, "_resolve_local_numa_node", lambda r, ws: 0)
    monkeypatch.setattr(na, "_read_cpulist", lambda node: [0, 1])

    def _raise(_pid, _cpus):
        raise OSError("EPERM")

    monkeypatch.setattr(na.os, "sched_setaffinity", _raise)
    assert na.pin_threads_to_local_numa(rank=0, world_size=1) is None


def test_pin_partitions_when_multiple_ranks_same_node(monkeypatch):
    """TP=4 single-node 시나리오 — 한 worker 가 자기 슬라이스만 받음."""
    rank_to_node = {0: 0, 1: 0, 2: 0, 3: 0}
    monkeypatch.setattr(
        na, "_resolve_local_numa_node", _fake_resolver(rank_to_node)
    )
    monkeypatch.setattr(na, "_read_cpulist", lambda node: list(range(16)))

    captured: list = []
    monkeypatch.setattr(
        na.os, "sched_setaffinity",
        lambda pid, cpus: captured.append(sorted(cpus)),
    )
    cpus = na.pin_threads_to_local_numa(rank=2, world_size=4)
    assert len(cpus) == 4, "16 cores / 4 ranks = 4 cores per rank"
    assert captured == [cpus]
