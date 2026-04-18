# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
NinjaGap §03 Phase 2 — 1GB HugeTLB Allocator (CPU engine)

hugetlbfs 마운트(1GB 페이지) 를 백업 스토리지로 사용하는 torch CPU tensor
할당자. 2MB THP 가 이미 켠 상태에서 추가로 L2 dTLB coverage 를 100% 로
확장하려는 목적.

적용 범위:
  Commit 1 (현): CPU KV cache tensor (_allocate_kv_cache_tensors hook)
  Commit 2 (예정): 모델 weight tensor (load_model 직후 slab allocator 로 복사)

호스트 준비 (sudo, 서버 기동 전):
    echo 64 > /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages
    mkdir -p /mnt/hugetlb_1g
    mount -t hugetlbfs -o pagesize=1G,size=64G none /mnt/hugetlb_1g

Env flag:
    HYBRID_HUGETLB_1G_ENABLE          master switch (기본 0)
    HYBRID_HUGETLB_1G_PATH            mount dir (기본 /mnt/hugetlb_1g)
    HYBRID_HUGETLB_1G_BIND_WEIGHTS    Commit 2 activation (기본 0, 현재 no-op)

Graceful fallback 원칙: 어떤 실패든 None 반환. 호출자는 None 을 받으면
기본 allocator 로 우회하고 기동을 멈추지 않는다. 이 모듈은 절대 예외를
위로 던지지 않는다.
"""

from __future__ import annotations

import logging
import mmap
import os
import threading
from typing import Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

PAGE_1G_BYTES = 1 << 30  # 1 GiB

# numpy 가 지원하지 않는 dtype 은 raw-byte 뷰로 잡아뒀다가 torch.view 로 reinterpret
_TORCH_TO_NUMPY = {
    torch.int8: np.int8,
    torch.uint8: np.uint8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
}


def is_configured() -> bool:
    return os.environ.get("HYBRID_HUGETLB_1G_ENABLE", "0") == "1"


def bind_weights_enabled() -> bool:
    """Commit 2 opt-in. Commit 1 시점에는 항상 False 처럼 동작."""
    return os.environ.get("HYBRID_HUGETLB_1G_BIND_WEIGHTS", "0") == "1"


def _mount_path() -> str:
    return os.environ.get("HYBRID_HUGETLB_1G_PATH", "/mnt/hugetlb_1g")


def _torch_to_numpy_dtype(dtype: torch.dtype):
    """bfloat16 은 numpy 에 없으므로 int16 버퍼로 잡고 나중에 .view(bf16). """
    if dtype == torch.bfloat16:
        return np.int16
    nd = _TORCH_TO_NUMPY.get(dtype)
    if nd is None:
        raise ValueError(f"Unsupported torch dtype for hugetlb: {dtype}")
    return nd


class _Slab:
    """
    1GB hugetlbfs region 위의 bump allocator.
    작은 파라미터 하나당 1GB 소비를 피하기 위한 sub-allocation 스토리지.
    """

    __slots__ = ("mm", "size", "offset", "numa_node")

    def __init__(self, mm: mmap.mmap, numa_node: int):
        self.mm = mm
        self.size = len(mm)
        self.offset = 0
        self.numa_node = numa_node

    def try_alloc(self, size_bytes: int, align: int = 64) -> Optional[int]:
        """성공 시 offset, 실패 시 None. slab 전체는 불변 (thread-unsafe 주의)."""
        aligned = (self.offset + align - 1) & ~(align - 1)
        if aligned + size_bytes > self.size:
            return None
        self.offset = aligned + size_bytes
        return aligned

    @property
    def free_bytes(self) -> int:
        return max(0, self.size - self.offset)


class HugeTLB1GAllocator:
    """
    프로세스 당 singleton. hugetlbfs 마운트 위에서 1GB 배수 크기의 파일을
    만들어 mmap 한 뒤 NUMA first-touch 로 local node 에 고정한다.

    두 종류의 할당:
    - alloc_region(): KV cache 처럼 수 GB 단위 큰 버퍼용. 1GB 배수 round up.
    - alloc_sub_tensor(): weight 처럼 작은 tensor 다수. _Slab bump allocator
      로 sub-allocate 해서 1GB 단위 낭비를 피함.
    """

    _instance: Optional["HugeTLB1GAllocator"] = None
    _lock = threading.Lock()
    _init_failed: bool = False

    @classmethod
    def get(cls) -> Optional["HugeTLB1GAllocator"]:
        """Singleton factory. env off / setup fail 시 None."""
        if not is_configured():
            return None
        if cls._init_failed:
            return None
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls._try_init()
                if cls._instance is None:
                    cls._init_failed = True
        return cls._instance

    @classmethod
    def _try_init(cls) -> Optional["HugeTLB1GAllocator"]:
        path = _mount_path()
        if not os.path.isdir(path):
            logger.warning(
                "[HYBRID-HUGETLB-1G] mount path not found: %s — allocator disabled",
                path)
            return None
        # writability probe
        probe = os.path.join(path, f".probe_{os.getpid()}")
        try:
            fd = os.open(probe, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
            os.close(fd)
            os.unlink(probe)
        except OSError as e:
            logger.warning(
                "[HYBRID-HUGETLB-1G] mount path not writable: %s (%s)",
                path, e)
            return None
        inst = cls.__new__(cls)
        inst.path = path
        inst._mmaps: list[mmap.mmap] = []
        inst._total_bytes = 0
        # per-NUMA slab pools for sub-allocation (weight bind).
        # numa_node=-1 은 "unspecified" bucket.
        inst._slabs_by_node: dict[int, list[_Slab]] = {}
        logger.info(
            "[HYBRID-HUGETLB-1G] allocator ready at %s "
            "(bind_weights=%s, pid=%d)",
            path, bind_weights_enabled(), os.getpid())
        return inst

    def alloc_region(
        self,
        size_bytes: int,
        numa_node: int = -1,
        tag: str = "kv",
    ) -> Optional[mmap.mmap]:
        """1GB 배수로 round up 후 hugetlbfs mmap 반환. 실패 시 None."""
        pages = (size_bytes + PAGE_1G_BYTES - 1) // PAGE_1G_BYTES
        total = pages * PAGE_1G_BYTES
        fname = f"vllm_{tag}_{os.getpid()}_{len(self._mmaps)}.bin"
        fpath = os.path.join(self.path, fname)
        try:
            fd = os.open(fpath, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        except OSError as e:
            logger.warning(
                "[HYBRID-HUGETLB-1G] create %s failed: %s", fpath, e)
            return None
        try:
            os.ftruncate(fd, total)
        except OSError as e:
            # hugetlb quota 부족 등
            logger.warning(
                "[HYBRID-HUGETLB-1G] ftruncate %dGiB failed: %s",
                total >> 30, e)
            os.close(fd)
            try:
                os.unlink(fpath)
            except OSError:
                pass
            return None

        # NUMA bind before first-touch so pages land on the right node
        if numa_node >= 0:
            try:
                from vllm.platforms.intel_cpu_utils import NUMAAllocator
                alloc = NUMAAllocator()
                if alloc.is_available:
                    alloc.bind_to_node(numa_node)
            except Exception as e:  # noqa: BLE001
                logger.debug(
                    "[HYBRID-HUGETLB-1G] NUMA bind skipped: %s", e)

        try:
            mm = mmap.mmap(
                fd, total,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
        except (OSError, ValueError) as e:
            logger.warning(
                "[HYBRID-HUGETLB-1G] mmap %dGiB failed: %s",
                total >> 30, e)
            os.close(fd)
            try:
                os.unlink(fpath)
            except OSError:
                pass
            return None
        os.close(fd)
        # Unlink now — mapping keeps a reference, file becomes anonymous.
        # Server crash 시 커널이 자동 정리.
        try:
            os.unlink(fpath)
        except OSError:
            pass

        # Force first-touch (one write per 1GB page — NUMA node 확정)
        try:
            for off in range(0, total, PAGE_1G_BYTES):
                mm[off] = 0
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[HYBRID-HUGETLB-1G] first-touch failed: %s", e)
            mm.close()
            return None

        self._mmaps.append(mm)
        self._total_bytes += total
        logger.info(
            "[HYBRID-HUGETLB-1G] alloc tag=%s: %d GiB (pages=%d, node=%d, "
            "cumulative=%d GiB)",
            tag, total >> 30, pages, numa_node, self._total_bytes >> 30)
        return mm

    def alloc_sub_tensor(
        self,
        shape,
        dtype: torch.dtype,
        numa_node: int = -1,
        tag: str = "w",
    ) -> Optional[torch.Tensor]:
        """
        작은 tensor 를 slab 에서 sub-allocate 해 torch.Tensor 로 반환.
        - 요청 크기가 1GB 초과면 자체 region 할당
        - 알맞은 slab 없으면 새 1GB region 하나를 추가
        - 실패 시 None
        """
        elem_size = torch.empty(0, dtype=dtype).element_size()
        n_elem = 1
        for d in shape:
            n_elem *= int(d)
        size_bytes = n_elem * elem_size
        if size_bytes == 0:
            # Empty tensor: 기본 allocator 로 충분
            return None

        # 대형 tensor (> 1GB) 는 자체 region
        if size_bytes > PAGE_1G_BYTES:
            mm = self.alloc_region(size_bytes, numa_node=numa_node, tag=tag)
            if mm is None:
                return None
            return _wrap_tensor(mm, 0, shape, dtype, n_elem)

        # slab 찾기 (해당 numa_node bucket)
        bucket = self._slabs_by_node.setdefault(numa_node, [])
        for slab in bucket:
            off = slab.try_alloc(size_bytes)
            if off is not None:
                return _wrap_tensor(slab.mm, off, shape, dtype, n_elem)

        # 새 slab 필요
        mm = self.alloc_region(PAGE_1G_BYTES, numa_node=numa_node,
                               tag=f"slab_{tag}")
        if mm is None:
            return None
        slab = _Slab(mm, numa_node)
        bucket.append(slab)
        off = slab.try_alloc(size_bytes)
        if off is None:
            # size_bytes <= 1GB 인데도 실패할 이유 없음. 방어적 처리.
            return None
        return _wrap_tensor(slab.mm, off, shape, dtype, n_elem)

    def release_all(self) -> None:
        self._slabs_by_node.clear()
        for mm in self._mmaps:
            try:
                mm.close()
            except Exception:  # noqa: BLE001
                pass
        self._mmaps.clear()
        self._total_bytes = 0
        logger.info("[HYBRID-HUGETLB-1G] released all regions")


def _wrap_tensor(
    mm: mmap.mmap,
    byte_offset: int,
    shape,
    dtype: torch.dtype,
    n_elem: int,
) -> Optional[torch.Tensor]:
    """mmap buffer 의 byte_offset 부터 torch.Tensor wrap. 실패 시 None."""
    try:
        np_dtype = _torch_to_numpy_dtype(dtype)
        arr = np.frombuffer(
            mm, dtype=np_dtype, count=n_elem, offset=byte_offset,
        ).reshape(tuple(int(d) for d in shape))
        t = torch.from_numpy(arr)
        if dtype == torch.bfloat16:
            t = t.view(torch.bfloat16)
        # mmap lifetime 보호 — 여러 tensor 가 같은 mmap 을 나눠 쓸 수 있음
        existing = getattr(t, "_hugetlb_mm", None)
        if existing is None:
            t._hugetlb_mm = mm  # type: ignore[attr-defined]
        return t
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "[HYBRID-HUGETLB-1G] tensor wrap failed (%s) — fallback", e)
        return None


def alloc_tensor_1g(
    shape,
    dtype: torch.dtype,
    numa_node: int = -1,
    tag: str = "kv",
) -> Optional[torch.Tensor]:
    """
    큰 단일 버퍼 (KV cache 등) 용 할당자. 요청 크기를 1GB 배수로 round-up
    해 자체 region 으로 확보한다. 작은 tensor 다수에는 alloc_sub_tensor 를
    쓸 것 (slab 기반).

    실패 시 None. 호출자는 None 받으면 기본 allocator 로 fallback 해야 함.
    반환 tensor 에 _hugetlb_mm attribute 로 mmap lifetime 연결됨.
    """
    alloc = HugeTLB1GAllocator.get()
    if alloc is None:
        return None
    elem_size = torch.empty(0, dtype=dtype).element_size()
    n_elem = 1
    for d in shape:
        n_elem *= int(d)
    size_bytes = n_elem * elem_size
    mm = alloc.alloc_region(size_bytes, numa_node=numa_node, tag=tag)
    if mm is None:
        return None
    return _wrap_tensor(mm, 0, shape, dtype, n_elem)


def bind_params_to_hugetlb(
    model: torch.nn.Module,
    numa_node: int = -1,
) -> Tuple[int, int]:
    """
    NinjaGap §03 Phase 2 — 모델 nn.Parameter.data 를 1GB hugetlbfs slab 으로
    in-place 복사. HYBRID_HUGETLB_1G_BIND_WEIGHTS=1 이고 allocator 사용 가능할
    때만 동작. 한 파라미터씩 copy_ 후 data 교체 → GC 가 orig 해제 → peak 메모리
    는 per-parameter 크기.

    Returns (migrated_count, migrated_bytes). 실패하거나 env off 면 (0, 0).
    어떤 파라미터라도 graceful skip 하고 나머지는 계속 진행 — 부분 적용 허용.
    """
    if not bind_weights_enabled():
        return (0, 0)
    alloc = HugeTLB1GAllocator.get()
    if alloc is None:
        return (0, 0)

    migrated = 0
    skipped = 0
    bytes_total = 0
    bytes_skipped = 0
    with torch.no_grad():
        for name, p in model.named_parameters(recurse=True):
            if not isinstance(p, torch.nn.Parameter):
                continue
            orig = p.data
            if orig.device.type != "cpu":
                continue
            if orig.numel() == 0:
                continue
            try:
                new_t = alloc.alloc_sub_tensor(
                    shape=orig.shape,
                    dtype=orig.dtype,
                    numa_node=numa_node,
                    tag=f"w_{name[:40].replace('.', '_')}",
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "[HYBRID-HUGETLB-1G] alloc_sub_tensor raised for %s: %s "
                    "— skipping", name, e)
                new_t = None
            if new_t is None:
                skipped += 1
                bytes_skipped += orig.numel() * orig.element_size()
                continue
            try:
                new_t.copy_(orig)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "[HYBRID-HUGETLB-1G] copy failed for %s (%s) — skipping",
                    name, e)
                skipped += 1
                bytes_skipped += orig.numel() * orig.element_size()
                continue
            p.data = new_t
            migrated += 1
            bytes_total += orig.numel() * orig.element_size()
    logger.info(
        "[HYBRID-HUGETLB-1G] weight bind: migrated=%d (%.2f GiB), "
        "skipped=%d (%.2f GiB), node=%d",
        migrated, bytes_total / (1024**3),
        skipped, bytes_skipped / (1024**3), numa_node)
    return (migrated, bytes_total)
