# IDE_017 — DMA + Zero-Copy CPU-GPU Data Plane

> **scope**: cudaHostAlloc + cudaMemcpyAsync pool / zero-copy pinned buffer / cold-KV decompress + DMA push.
> **paper angle**: cuda pinned-memory + DMA × spec decode 통합 첫 production case. LMCache KV offload 와 다른 spec-specific data flow.
> **parent**: TSK_020 / IDE_015 Phase A 결과 기반.
> **status**: ✅ design + skeleton 작성 완료 / ⚠ build + 검증 별도 turn 필요.

---

## 1. 이론적 배경

### 1.1 Phase A 측정 input (SUB_166 DMA microbench)

| Block | latency | bandwidth |
|---:|---:|---:|
| 4 KB | 35 μs | 0.12 GB/s |
| **1 MB** | 60 μs | **17.5 GB/s** ← crossover |
| 16 MB | 338 μs | 49.6 GB/s |
| 64 MB | 1251 μs | **53.6 GB/s** (asymptotic) |

→ **fixed overhead 35 μs per transfer / 54 GB/s asymptotic / crossover 1 MB**.
→ small data (< 256 KB): cudaMemcpy 와 차이 없음, batching 필요.
→ medium/large data (≥ 1 MB): DMA bandwidth-bound — 본 IDE 의 main lever.

### 1.2 3 sub-task

| TSK | 영역 | scope | priority |
|---|---|---|---|
| TSK_028 | Pinned memory pool + DMA push primitive | cudaHostAlloc allocator + DMA timing | ★★ |
| TSK_029 | Zero-copy CPU compute path | pinned buffer dual-access (CPU + GPU 동시) | ★ |
| TSK_030 | Cold-KV decompress + DMA push (IDE_006 재정의) | AVX-512 dequant + bandwidth-bound DMA | ★★ |

---

## 2. 구현 방향

### 2.1 Pinned pool allocator (TSK_028)

```cpp
// src/dma_pool/pinned_pool.cpp
class PinnedPool {
public:
    // size-class allocator (4KB, 64KB, 1MB, 16MB, 64MB)
    void* alloc(size_t size);
    void free(void* ptr);

    // batched push to GPU
    cudaEvent_t push_async(void* host_ptr, void* dev_ptr, size_t size, cudaStream_t stream);
};
```

### 2.2 Zero-copy buffer (TSK_029)

```cpp
// src/dma_pool/zero_copy_buffer.cpp
// dual-access region: CPU 가 update, GPU 가 read (또는 vice versa)
// host-mapped (cudaHostAllocMapped) 사용
```

### 2.3 Cold-KV decompress + DMA (TSK_030)

```cpp
// src/dma_pool/cold_kv_decompress.cpp
// CPU AVX-512 decompress (Q8/INT4 → BF16) + DMA push
// IDE_006 (Cold-KV CPU Partial Attention) 의 재정의
```

---

## 3. 측정 결과 referenced

- [SUB_166](../IDE_015_cpu_extreme_util/SUB_166_dma_microbench/RESULTS.md) — DMA microbench
- [SUB_113](../IDE_015_cpu_extreme_util/SUB_113_numa_audit/RESULTS.md) — GPU-NUMA affinity (DMA latency 에 영향)

---

## 4. Hardware target

| target | requirement |
|---|---|
| CUDA + pinned memory | nvidia driver, libcuda, libcudart |
| PCIe 5.0 (H100) | asymptotic 54 GB/s 측정 (SUB_166) |
| NUMA-aware allocation | GPU 0-3 ↔ NUMA 0 / GPU 4-7 ↔ NUMA 1 (SUB_113) |

Build:
```bash
cd shadow_assists/features/IDE_017_dma_zero_copy/src
mkdir build && cd build
cmake .. -DCUDA_ARCH=90  # H100
make -j 16
```

---

## 5. 검증

- latency benchmark: SUB_166 protocol 재현 + per-stream 동시 측정
- correctness: GPU read 후 CPU 의 update 가 일관성 있는지 (zero-copy)
- e2e: SUB_098 canonical AGSD-gated 에 통합 후 +1-3% lift 측정 (가설)
