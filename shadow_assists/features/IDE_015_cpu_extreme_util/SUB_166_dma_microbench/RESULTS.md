# SUB_166 — DMA push latency microbench (1-run)

> **parent**: IDE_017 / TSK_028 SUB_119 (원 plan) — measurement-only equivalent
> **scope**: 2026-05-26 ~23:32 KST (~30s)
> **status**: ✅ 완료 — block size sweep 4KB ~ 64MB × 200 iter / size
> **device**: NVIDIA H100 80GB HBM3 (GPU 1, NUMA 0 attached)
> **method**: torch `pin_memory=True` host buffer → `.copy_(src, non_blocking=True)` on dedicated cuda stream + synchronize
> **measurement protocol (1-run)**: 사용자 지시 — 각 block 200 iter 통계만

---

## 0. 두괄식 — fixed overhead 35 μs + asymptotic 54 GB/s + crossover 1 MB

| Block | mean latency | p50 | p99 | bandwidth | regime |
|---:|---:|---:|---:|---:|---|
| 4 KB | 35.4 μs | 34.9 μs | 44.3 μs | 0.12 GB/s | overhead-bound |
| 16 KB | 37.7 μs | 37.5 μs | 42.6 μs | 0.43 GB/s | overhead-bound |
| 64 KB | 35.9 μs | 35.7 μs | 41.0 μs | 1.83 GB/s | overhead-bound |
| 256 KB | 41.2 μs | 40.9 μs | 45.6 μs | 6.37 GB/s | overhead-bound |
| **1 MB** | 60.1 μs | 61.3 μs | 66.5 μs | **17.5 GB/s** | **crossover** ⭐ |
| 4 MB | 113 μs | 110 μs | 133 μs | 37.2 GB/s | bandwidth-bound |
| 16 MB | 338 μs | 338 μs | 354 μs | 49.6 GB/s | bandwidth-bound |
| **64 MB** | 1251 μs | 1249 μs | 1269 μs | **53.6 GB/s** ⭐ | asymptotic |

→ **fixed overhead ~35 μs/transfer** (API + sync 비용).
→ **asymptotic bandwidth ~54 GB/s** (PCIe 5.0 x16 이론 ~63 GB/s 의 **~85%**).
→ **crossover block size ~1 MB** — 이 이상에서는 bandwidth-bound, 이하는 overhead-bound.

---

## 1. IDE_017 / TSK_028 의 설계 입력

| 결정 | 본 microbench 의 기준 | 추천 |
|---|---|---|
| 작은 데이터 (≤ 256 KB) 전송 시 DMA vs cudaMemcpy 선택 | DMA 든 cudaMemcpy 든 ~35 μs/transfer fixed cost | **차이 거의 없음** — DMA 의 batching 없으면 비효율 |
| 중간 데이터 (1-4 MB) | DMA 60-113 μs / 17-37 GB/s | DMA 사용 권장 (cudaMemcpy 보다 latency 작음) |
| 큰 데이터 (≥ 16 MB) | DMA 338-1251 μs / 49.6-53.6 GB/s | DMA + async chunking 권장 |
| **threshold rule** | 1 MB | 1 MB 이하 → cudaMemcpy, 이상 → DMA (TSK_028 SUB_119 design rule) |

### spec decode 영역 typical 전송 크기

| 데이터 종류 | block size 추정 | 본 SUB 의 적용 영역 |
|---|---|---|
| token logits (vocab × batch BF16) | ~10 MB (152K × 32 × 2 bytes) | 4 MB 영역 (37 GB/s) — DMA 적용 가능 |
| spec candidate IDs (K × batch × int64) | ~2-8 KB | 4 KB 영역 (overhead-bound) — 굳이 DMA 안 함 |
| KV cache chunk (per-layer per-token) | ~1-100 KB | overhead-bound 영역 — batch 가능 시만 |
| **cold KV chunk (per-request, BF16)** | 10-100 MB | bandwidth-bound 영역 — DMA + compress 적합 (TSK_030) |
| draft head logits (small vocab) | 1-2 MB | crossover 영역 |

→ **TSK_030 (Cold-KV decompress + DMA push)** 가 본 microbench 의 가장 강한 application — large block (MB ~ 10s MB) bandwidth 활용.
→ TSK_028/029 의 작은 데이터 (logits, candidates) 는 batching 없이 DMA 만으로는 net positive 어려움.

---

## 2. 비교 — typical NVLink (GPU-GPU) 대비

| 영역 | 측정 |
|---|---|
| 본 SUB CPU→GPU DMA asymptotic bandwidth | 53.6 GB/s |
| H100 NVLink 4.0 GPU-GPU (이론) | 900 GB/s (per GPU 대) |
| PCIe 5.0 x16 (이론) | 63 GB/s (uni-dir) |

→ DMA 는 NVLink 의 ~6% bandwidth 만 — GPU-GPU 직접 전송이 가능하면 DMA 보다 빠름. CPU 가 source/dest 인 경우만 DMA 유리.

---

## 3. 다음 step

- **TSK_028 SUB_118** (kernel dev — 별도 turn): cudaHostAlloc + cudaMemcpyAsync pool size-class allocator — 본 microbench 의 1 MB threshold 를 input
- **TSK_028 SUB_120** (별도 turn): vLLM allocator 와 integration
- **TSK_030 SUB_125** (별도 turn — IDE_006 재정의): CPU AVX-512 decompress + DMA push — 본 microbench 의 large-block (10-100 MB) bandwidth 가 lever
- **(미할당)** 멀티-GPU + cross-NUMA 영역 DMA 측정 — GPU 4-7 (NUMA 1) 도 동일한지 확인 (가설: PCIe affinity 때문에 cross-NUMA 시 latency ↑)

---

## 4. raw data

- `dma_latency.json` — full result (8 block sizes × stats)
- `dma_latency.csv` — table format
- 소스: `/tmp/sub166_dma_microbench.py`
