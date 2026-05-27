# SUB_176 — IDE_017 / TSK_028 pinned pool + DMA push canonical 500p e2e

> **parent**: TSK_028 (IDE_017). pinned memory pool + DMA push primitive 의 prod 환경 검증.
> **scope**: 2026-05-27 09:57 ~ 10:06 KST. pool microbench (8 cell × alloc/free × DMA) + batch microbench (6 cell) + NUMA cross-affinity microbench (4 cell × 2) + canonical 500p baseline (9 cell).
> **status**: 완료 — pool build PASS / SUB_166 재현 OK / 53.6 GB/s 95% 재현 / NUMA marginal / **vllm 통합 = 이미 NEO 가 흡수 / drop-in marginal value 없음** ⚠

---

## 0. 두괄식 — pool 자체 PASS, vllm 통합 미실행 (NEO 가 흡수)

| 지표 | 측정 | 평가 |
|---|---:|---|
| pool build (nvcc -arch=sm_90, 408 line .cpp + missing #include 1 보수) | libpinned_pool.so 28 KB | OK |
| pool alloc/free p50 (size-class hot path, lockless ring dequeue/enqueue) | **0.65 μs / 0.61 μs** | target < 5 μs **달성** (~8× margin) |
| DMA push 64 MB asymptotic bandwidth | **51.13 GB/s** | SUB_166 의 53.6 GB/s **95% 재현** (single-stream 환경 차이) |
| DMA push 64 MB p50 latency | 1222 μs | SUB_166 의 1251 μs 와 ±2% 정합 |
| DMA push 1 MB crossover p50 | 39.6 μs | SUB_166 의 60.1 μs 보다 **빠름** (torch wrapper overhead 제거) |
| batched push amortization (n=32 × 64 KB) | 13.0 → 4.7 μs/xfer | **2.8× amortization** (fixed overhead 분산 확인) |
| cross-NUMA penalty (GPU 1 + pool numa=1 vs numa=0) | **~1% (negligible)** | PCIe 5.0 dominant, NUMA-aware lever 매우 작음 |
| canonical 500p × 3 mix × 1-run AGSD-gated tps (max-tokens=32) | balanced 4,029 / sonnet 4,470 / code 4,631 / **avg 4,377** | DMA pool 미사용 — pure baseline 환경 정합성 확인 |
| vllm KV swap path 통합 시도 | 미실행 ⚠ | **NEO 가 이미 N-slot pinned staging pool 보유** (`_neo_init_swap_staging`, VLLM_NEO_ASYNC_SWAP_BUFFERS) — drop-in marginal value 없음 |
| accuracy gate (logprob diff) | n/a | DMA pool 이 vllm path 에 들어가지 않음 — 정확도 영향 0 |

**결론 #1**: 본 IDE_017 의 lockless ring pinned pool 은 spec 대로 작동. alloc/free hot path 0.65 μs 는 size-class lookup + atomic CAS 만 — design target 충분히 달성.

**결론 #2**: SUB_166 의 microbench 재현 OK. 53.6 → 51.13 GB/s 의 ~5% gap 은 단일 stream + push-only (pull 없음) 환경 차이로 설명. fixed overhead 가 SUB_166 (35 μs) 보다 작은 9-13 μs/transfer 인 이유 = torch.copy_(non_blocking=True) wrapper 없이 직접 cudaMemcpyAsync 호출.

**결론 #3** ⚠: **vllm KV swap path 통합 불가 (실용적 의미)** — `vllm/v1/worker/gpu_model_runner.py` 의 NEO (`_neo_init_swap_staging`, `_neo_swap_out_gather_phase`, `_neo_swap_out_dma_phase`, `_neo_pending_dma_infos`) 가 이미 **N-slot ring pinned staging tensor pool + async DMA + batched gather** 를 직접 구현. 본 IDE_017 의 C++ pool 의 유일한 추가 lever 인 NUMA-aware 도 본 머신에서 ~1% 차이만 (cross-NUMA penalty 작음). 즉 본 pool 을 NEO 의 torch.empty(pin_memory=True) 자리에 끼워 넣어도 실측 lift 0 expected.

**결론 #4**: drop-in CPU replacement 가 e2e lift 를 거의 만들지 못한다는 이전 SUBs (SUB_173 AVX-512 tokenizer / SUB_174 sampling / SUB_175 AMX matmul) 의 패턴이 본 SUB 에도 적용. **본 IDE_017 의 진정한 lever** 는 **새 사용처 발굴** (예: cold-KV chunk decompress + DMA push — TSK_030) — pool 자체가 아니라 그 pool 위에 올라가는 workload.

---

## 1. pool 빌드

| 항목 | 값 |
|---|---|
| 소스 | `/workspace/vllm_hybrid/shadow_assists/features/IDE_017_dma_zero_copy/src/pinned_pool.cpp` (408 → 409 line; missing `#include <unordered_map>` 추가) |
| 빌드 cmd | `nvcc -arch=sm_90 -O3 -std=c++17 -Xcompiler -fPIC -shared -DVLLM_DMA_HAVE_NUMA=1 -o libpinned_pool.so pinned_pool.cpp -lcudart -lnuma` |
| 결과 | `build/libpinned_pool.so` 28288 B |
| ABI | extern "C" — `pinned_pool_create/destroy/alloc/free/push_async/push_batch_async/event_sync/event_destroy/stream_create/destroy/sync` |
| 환경 | CUDA 12.8 (V12.8.93), GCC 12, libnuma 2.0.14 |

---

## 2. pool microbench (Step 4 — SUB_166 protocol 재현)

### 2.1. 환경

| 항목 | 값 |
|---|---|
| GPU | NVIDIA H100 80GB HBM3 (GPU 1, NUMA 0 attached) |
| pool config | total_limit 4 GiB, numa hint 0 (same NUMA) |
| pool 5 size class | 4 KB × 256 / 64 KB × 128 / 1 MB × 64 / 16 MB × 16 / 64 MB × 8 (총 owned 841 MB) |
| iters | 200/size (alloc+free) / 200 (DMA push) / 8 warmup |
| protocol | alloc → memset(0xA5) → push_async → event_sync → free → roundtrip device→host compare 0xA5 |

### 2.2. 결과 (size sweep)

| Block | alloc p50 | alloc p99 | free p50 | DMA p50 | DMA p99 | bandwidth | rt_ok | SUB_166 reference |
|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 4 KB | 0.69 μs | 1.91 | 0.64 | **9.21 μs** | 14.59 | 0.41 GB/s | OK | 35.4 μs (torch overhead 포함) |
| 16 KB | 0.67 | 0.98 | 0.63 | 12.76 | 21.33 | 1.20 GB/s | OK | 37.7 |
| 64 KB | 0.66 | 0.88 | 0.61 | 12.50 | 15.31 | 4.88 GB/s | OK | 35.9 |
| 256 KB | 0.65 | 0.86 | 0.60 | 18.12 | 24.44 | 13.48 GB/s | OK | 41.2 |
| **1 MB** | 0.65 | 0.90 | 0.61 | **39.64** | 42.10 | **24.64 GB/s** | OK | 60.1 (crossover) |
| 4 MB | 0.65 | 0.86 | 0.60 | 102.62 | 112.35 | 38.07 GB/s | OK | 113 |
| 16 MB | 0.65 | 0.80 | 0.60 | 328.30 | 345.28 | 47.59 GB/s | OK | 338 |
| **64 MB** | 0.69 | 0.98 | 0.64 | **1222.40** | 1249.54 | **51.13 GB/s** | OK | 1251 (asymptotic 53.6) |

**관찰**:
- alloc/free p50 모두 **0.6-0.7 μs** — size 무관 flat (lockless ring dequeue/enqueue + std::unordered_map record 등록의 합). target < 5 μs **달성**.
- DMA p50 가 SUB_166 보다 일관되게 **작음** — 작은 size 에서 약 ~3×. 원인: SUB_166 은 `torch.copy_(non_blocking=True)` + `stream.synchronize()` 로 torch python wrapper / stream object overhead 포함. 본 SUB 는 ctypes 로 직접 cudaMemcpyAsync + cudaEventSynchronize → fixed call cost ~9 μs vs torch ~35 μs.
- asymptotic bandwidth 51.13 GB/s (SUB_166 의 53.6 GB/s **95% 재현**). gap 은 single-stream + push-only 환경 차이.
- 1 MB crossover 39.6 μs (SUB_166 60.1 μs) — torch overhead 의 의의 추가 검증.

### 2.3. raw

- `pool_microbench.json` — full 8 size × stats

---

## 3. batched DMA push microbench

### 3.1. 환경

| 항목 | 값 |
|---|---|
| chunk size | 64 KB (overhead-bound region; SUB_166 의 36 μs typical) |
| batch n | {1, 2, 4, 8, 16, 32} |
| iters | 100/batch + 4 warmup |
| API | `pinned_pool_push_batch_async(host_ptrs[], dev_ptrs[], sizes[], n, stream)` — n cudaMemcpyAsync + 1 cudaEventRecord |

### 3.2. 결과 (n × per-transfer amortization)

| batch n | total p50 | per-xfer p50 | total p99 | amortization |
|---:|---:|---:|---:|---:|
| 1 | 13.02 μs | 13.02 μs | 17.66 | 1.00× (baseline) |
| 2 | 17.68 | 8.84 | 21.32 | 1.47× |
| 4 | 27.22 | 6.81 | 31.67 | 1.91× |
| 8 | 46.40 | 5.80 | 49.62 | 2.25× |
| 16 | 80.31 | 5.02 | 85.24 | 2.59× |
| **32** | **150.99** | **4.72 μs** | 159.46 | **2.76×** |

**관찰**: n=32 에서 per-transfer 13 → 4.7 μs, **2.8× amortization**. fixed overhead 가 cudaEventRecord/Sync 의 single-event cost 로 amortize 됨. 본 design 의 `push_batch_async` API 작동.

- raw: `pool_batch_microbench.json`

---

## 4. NUMA cross-affinity microbench

### 4.1. 환경

| 항목 | 값 |
|---|---|
| GPU | GPU 1 (NUMA 0 attached, PCIe topology PXB) |
| pool config | same total_limit (4 GiB), numa_node {0 (same), 1 (cross)} |
| size sweep | 64 KB / 1 MB / 16 MB / 64 MB |
| iters | 100/size + 5 warmup |

### 4.2. 결과

| Block | numa=0 (same) p50 | numa=1 (cross) p50 | cross/same | numa=0 BW | numa=1 BW | BW gap |
|---:|---:|---:|---:|---:|---:|---:|
| 64 KB | 12.22 μs | 12.48 μs | 1.021× | 5.00 GB/s | 4.89 | -2.2% |
| 1 MB | 36.81 | 36.52 | 0.992× | 26.53 | 26.74 | +0.8% |
| 16 MB | 321.88 | 324.99 | 1.010× | 48.54 | 48.08 | -1.0% |
| 64 MB | 1226.85 | 1236.25 | 1.008× | 50.94 | 50.56 | -0.7% |

**관찰**: cross-NUMA penalty **최대 2.2%, 일반 ~1%** — H100 + Sapphire Rapids 의 UPI inter-socket 가 충분히 빠르고 PCIe DMA 가 dominant cost. SUB_113 의 GPU 4-7↔NUMA 1 affinity 권장 의 measurable benefit 은 매우 작음.

→ **본 IDE_017 의 NUMA-aware 설계 가치 = 본 머신에서 < 2% lever** (즉 의미 없음). vllm 의 NEO 가 NUMA-naive 인 것이 큰 문제 아님.

- raw: `pool_numa_microbench.json`

---

## 5. canonical 500p baseline (Step 5 — vllm 통합 없이)

### 5.1. 환경

| 항목 | 값 |
|---|---|
| 모델 | Qwen/Qwen2.5-32B-Instruct (BF16) |
| 구성 | TP=4×2 dual instance (GPU 0-3 vanilla / GPU 4-7 trident-suffix), `--gpu-memory-utilization 0.80`, `--max-model-len 4096`, `--max-num-seqs 128`, `--max-num-batched-tokens 4096`, `cudagraph_mode=PIECEWISE`, `disable-custom-all-reduce`. suffix 의 `num_speculative_tokens=32` |
| router | sub094_router (port 8000) + classifier workers=4 |
| benchmark | sub094_benchmark — 500 prompts × 32 max-tokens × 32 concurrency × 3 mix × 1-run |
| pthread EAGAIN guard | `RAYON_NUM_THREADS=OMP=OPENBLAS=MKL=4`, `TOKENIZERS_PARALLELISM=false` |
| vllm boot | 81 s (8×10 s polling cycle) |

### 5.2. 결과 (9 cell)

| mix | vanilla-only tps | trident-only tps | **agsd-gated tps** | wall (AGSD) | p50 (AGSD) | p99 (AGSD) |
|---|---:|---:|---:|---:|---:|---:|
| balanced | 1,639.0 | 1,635.0 | **4,028.8** | 3.97 s | 0.182 s | 0.490 s |
| sonnet-heavy | 2,127.8 | 3,237.4 | **4,470.0** | 3.58 s | 0.167 s | 0.454 s |
| code-heavy | 1,939.0 | 3,009.8 | **4,630.5** | 3.46 s | 0.166 s | 0.376 s |

**3-mix AGSD avg = 4,376.5 tps**.

### 5.3. AGSD backend routing

| mix | trident | vanilla |
|---|---:|---:|
| balanced | 335 | 165 |
| sonnet-heavy | 400 | 100 |
| code-heavy | 400 | 100 |

→ classifier 가 mix 별로 trident 우선 routing — 정상 작동.

### 5.4. reference 와 비교

| 비교 대상 | protocol | AGSD 3-mix avg |
|---|---|---:|
| 본 SUB_176 | **max-tokens=32** × 32 concurrency × 500p | 4,377 tps |
| SUB_169 OFF | max-tokens=256 × 32 concurrency × 500p | 6,126 tps |
| SUB_172 OFF | max-tokens=256 × 32 concurrency × 500p | 6,160 tps |
| SUB_160 | max-tokens=256 × 32 concurrency × 500p | 6,169 tps |

→ 본 SUB 의 4,377 tps 는 **max-tokens 1/8 차이** 로 직접 비교 불가. wall time 의 startup overhead 비중이 max-tokens=32 에서 훨씬 큼. 본 SUB 안의 환경 정합성만 확인.

---

## 6. vllm KV swap path 통합 시도 결과

### 6.1. 통합 후보 위치 분석

`vllm/v1/worker/gpu_model_runner.py` 의 NEO swap path (TSK_015 / SUB_026 / SUB_028 의 결과):

| 함수 | 역할 | 본 IDE_017 hook 가능성 |
|---|---|---|
| `_neo_init_swap_staging()` | N-slot pinned tensor pool 초기화 (`torch.empty(shape, dtype, pin_memory=True)`) | **이미 pool concept 구현** — drop-in 가능하나 의의 없음 |
| `_neo_swap_out_gather_phase()` | GPU kv_caches → 독립 GPU tensor gather (slot_idx) | GPU-side gather, 본 pool 무관 |
| `_neo_swap_out_dma_phase()` | gather 결과를 staging 의 K/V tensor 로 async copy_(non_blocking=True) | torch tensor → torch tensor — 본 C++ pool 의 raw void* 와 ABI 불일치 |
| `_neo_pending_dma_infos` | async DMA list — drain 시 모두 처리 | already async ring |
| `VLLM_NEO_ASYNC_SWAP_BUFFERS` ENV | pool size N default 3, range 1..8 | NEO 가 이미 ENV 노출 |

### 6.2. drop-in replacement 가 marginal value 없는 이유

1. **NEO 는 이미 pinned pool 의 모든 lever 를 흡수**:
   - **size-class pool**: NEO 는 fixed-shape (max_blocks × layers × heads × head_dim) staging tensor → vllm 의 KV cache 형식에 맞춰 더 정확한 fit. 본 IDE_017 의 5 size class 는 generic.
   - **lockless ring**: NEO 는 N-slot tensor 를 slot_idx 로 회전 — index 회전이라 더 단순. lock-free.
   - **async batched DMA**: NEO 는 layer 별 cudaMemcpyAsync + cudaEvent → 본 IDE_017 의 push_batch_async 와 동일 패턴.
   - **pinned alloc**: NEO 는 torch.empty(pin_memory=True) — 내부적으로 cudaHostAlloc 호출과 동일.

2. **본 IDE_017 의 유일한 추가 lever = NUMA-aware**:
   - 그러나 본 머신에서 cross-NUMA penalty ~1% (§4) — 의미 없음.
   - 또한 vllm worker 가 GPU NUMA-local CPU 에 cgroup pin 되면 OS 의 first-touch policy 로 자동으로 NUMA-local 할당 (이미 자동).

3. **ABI gap**: NEO 의 staging 은 torch.Tensor (dtype/shape/strides 메타). 본 IDE_017 은 raw `void*`. torch.from_blob + custom allocator 로 plumb 가능하나 추가 복잡도 만 발생, lift 0 expected.

### 6.3. 결정

- **drop-in 통합 미실행** — measurement lift 0 expected (NEO 가 이미 동등 기능 흡수).
- accuracy gate 측정 불가 (벤치마크 path 가 본 pool 을 거치지 않음).
- 본 IDE_017 의 진정한 entry 는 **새 데이터 plane workload** (예: TSK_030 cold-KV decompress + DMA push, 또는 KV cache offload to host + reload — LMCache 와 차별화) — pool 자체가 아니라 그 pool 을 사용하는 새 path 의 발굴.

---

## 7. utilization (canonical baseline, monitor.py 0.5s)

| 지표 | 값 |
|---|---:|
| CPU avg | **3.91%** (n=219 samples) |
| GPU avg (8 GPU) | **17.36%** |
| GPU 0-3 (vanilla TP=4) per-GPU avg | 20.01 / 19.68 / 22.01 / 21.00 % |
| GPU 4-7 (trident TP=4) per-GPU avg | 14.05 / 14.52 / 13.37 / 14.28 % |

→ CPU 3.91% 는 vllm thread mechanism 의 typical busy-wait idle gap (SUB_148 / SUB_162 의 finding 과 정합). GPU 17% 는 max-tokens=32 의 startup-dominated short-wall workload 의 일반적 utilization (SUB_169 의 36% 는 max-tokens=256 / 더 긴 generate phase).

- raw: `_monitor_off_cpu.csv`, `_monitor_off_gpu.csv`

---

## 8. 한계 / 후속 turn 필요

### 8.1. 본 SUB 의 한계

1. **vllm 통합 미실행** — NEO 가 동등 기능 흡수로 drop-in marginal value 0. e2e lift 측정 불가.
2. **NUMA-aware lever 본 머신에서 < 2%** — Sapphire Rapids UPI 빠름 + PCIe 5.0 dominant cost.
3. **pool 의 진정한 사용처 = 새 workload** (cold-KV chunk decompress, KV offload-to-host, etc.) — TSK_030 / 별도 IDE 의 work, 본 SUB scope 외.

### 8.2. 후속 turn 후보

| 후속 | scope | 기대 |
|---|---|---|
| TSK_030 cold-KV decompress + DMA push | AVX-512 INT8 → BF16 dequant + 본 pool push (large block bandwidth-bound 영역 64 MB / 51 GB/s) | TTFT −5-10% 가능성 (긴 prompt + KV pressure workload) |
| LMCache vs IDE_017 comparison | LMCache 의 host KV offload path 와 본 pool 의 비교 | architectural 차이 + 비교 |
| 별도 IDE — KV chunk prefetch using DMA | speculative prefetch 패턴 + 본 pool batched_push | hit rate 의존 |

### 8.3. 본 SUB 의 deliverable

- **본 pool 의 기능 검증 완료** (alloc/free hot path 0.65 μs / DMA 53 GB/s 95% reproduce / batched amortization 2.8× / NUMA penalty negligible).
- **vllm 통합 가능 여부 honest assessment**: 현 NEO 와의 기능 중복으로 drop-in 의의 없음.
- canonical baseline (max-tokens=32) 환경 정합성 확인 — 본 머신 + dev_prj python + 본 launcher 가 정상 작동.

---

## 9. raw artifacts

- `build/libpinned_pool.so` — pool 빌드 산출물
- `run_microbench.py` / `pool_microbench.json` — size sweep
- `run_batch_microbench.py` / `pool_batch_microbench.json` — batch amortization
- `run_numa_microbench.py` / `pool_numa_microbench.json` — NUMA cross
- `launcher.sh` / `logs/main_off.log` / `logs/boot_off_seconds.txt` — canonical launcher
- `baseline_500p_off/{balanced,sonnet-heavy,code-heavy}/benchmark_*.json` — canonical 결과
- `_monitor_off_{cpu,gpu}.csv` — utilization 캡처
