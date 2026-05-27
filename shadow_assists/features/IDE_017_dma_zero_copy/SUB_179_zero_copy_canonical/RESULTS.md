# SUB_179 — Zero-Copy CPU compute path RESULTS

**parent**: `IDE_017` / `TSK_029`
**date (KST)**: 2026-05-27 10:44 ~ (진행 중)
**scope**: `cudaHostAllocMapped` dual-access pinned buffer (CPU 와 GPU 가 같은
물리 메모리 page-locked 영역을 동시 read/write) 의 small data 전송 우위 검증.
SUB_166 의 1 MB crossover 와 비교, vLLM small data 경로 integration boundary 분석.

---

## 0. honest scope statement (선언)

이전 패턴:

| SUB | type | result |
|---|---|---|
| SUB_173 tokenizer | drop-in | +0.86% noise |
| SUB_174 sampling | drop-in | −1.10%~−4.80% |
| SUB_175 AMX matmul | drop-in | 12× slower |
| SUB_176 DMA pool | drop-in | NEO 점유 → lift 0 |
| SUB_177 AMX prefill | drop-in | 10⁴-10⁵× gap |
| **SUB_178 cold-KV** | **NEW workload** | **conditional accept ⭐** |

→ drop-in 5/5 failure / NEW workload 1/1 conditional accept.
→ zero-copy 도 본질적으로 **drop-in 성격** (기존 cudaMemcpyAsync 를 mapped pointer 로
교체). 본 SUB 는 (1) zero-copy 의 small data 우위가 실재하는지 microbench 로 확인,
(2) vLLM 안 small data 경로 식별 + integration boundary 분석, (3) canonical 500p
baseline 환경 정합성 까지 honest scope.

**판정 기준**: 6번째 drop-in 실패 (PyTorch 의 `pin_memory=True` + `non_blocking=True`
가 이미 같은 lever 흡수) vs SUB_178 패턴의 NEW workload (zero-copy 만 가능한
새로운 access pattern 존재).

---

## 1. zero-copy candidate 식별 (vLLM small data 경로)

vLLM 안 small CPU → GPU 전송 site 를 grep 으로 식별. 추정 transfer rate 는
vllm v1 의 InputBatch / Sampler 구조 기반.

| site | 파일:line | 데이터 | 크기 (추정) | 빈도 |
|---|---|---:|---:|---:|
| spec candidate IDs | `vllm/v1/spec_decode/ngram_proposer_gpu.py:530` (`_pinned_idx_buf`) | active req indices int64 | n_active × 8 B (≤ 1 KB @ 128 reqs) | per-step |
| spec num_tokens sync | `vllm/v1/spec_decode/ngram_proposer_gpu.py:606` (`_sync_num_tokens`) | int32 seq length per req | n_active × 4 B (≤ 512 B) | per-step |
| sample/spec flattened indices | `vllm/v1/worker/gpu_model_runner.py:1884-1914` | int64 index tensor | n × 8 B (typ. 1-8 KB) | per-step |
| mrope/xdrope positions | `vllm/v1/worker/gpu_model_runner.py:2235-2243` | int64 positions | 3 × scheduled_tokens × 8 B (≤ 96 KB) | per-step (mrope만) |
| sampler internals (`pin_memory=True`) | `vllm/v1/sample/sampler.py:162` | temperature / top_k / top_p / etc per req | n_active × few B (< 4 KB) | per-step |
| logits processors | `vllm/v1/sample/logits_processor/builtin.py:30/158/231` | min_tokens / freq / presence | n_active × n_param × 4 B (< 4 KB) | per-step |
| eagle drafter tensors | `vllm/v1/spec_decode/eagle.py:214,1199` | int input_ids / positions | n_spec × 8 B (< 4 KB) | per-step |
| tree_attn_bias | `vllm/v1/attention/backends/tree_attn.py:93` | causal/tree mask f16/bf16 | tree_size² × 2 B (typ. 2 KB ~ 64 KB) | per-batch |
| pooler output (D2H) | `vllm/v1/worker/gpu_model_runner.py:321/378/386/403` | embeddings → CPU | model_hidden × n_reqs × 2 B (≥ MB) | per-batch (pooling 모델만) |

요약: small data CPU→GPU 의 **개별 transfer 는 4 B ~ 64 KB 범위** —
**SUB_166 의 4-256 KB overhead-bound region 정확히** match. 본 SUB
microbench 가 시뮬레이션할 영역 그대로.

**중요 관찰**: 위 site 들은 **이미 모두 `pin_memory=True` + `non_blocking=True`** 로
PyTorch 의 CUDACachingHostAllocator 가 page-locked staging 을 자동 관리하고 있음.
즉, vLLM v1 의 small-data path 는 cuda runtime 시점에서 이미 "pinned host →
device async copy" 로 구현되어 있어 본 microbench 의 **DMA 모드와 동일**.

zero-copy 의 이론적 lever:
- DMA 모드: CPU 가 pinned src 에 write → cudaMemcpyAsync 가 PCIe 로 device buffer
  로 copy → GPU kernel 이 device buffer 에서 read (2 step, fixed ~35 μs overhead)
- ZC 모드: CPU 가 cudaHostAllocMapped buffer 에 write → GPU kernel 이 같은
  page-locked 메모리를 PCIe BAR 로 직접 read (1 step, fixed overhead 없음, 단
  per-byte read 가 BAR 통과)

이론상 small data (≤ 64 KB) 에서 zero-copy 가 fixed overhead 제거로 net win.
≥ 1 MB 에서는 BAR read latency 가 누적되어 DMA 가 다시 win (SUB_166 crossover).

---

## 2. zero-copy buffer 구현 (`src/zero_copy_buffer.cpp`)

핵심 API:

```cpp
extern "C" int zc_alloc(ZeroCopyBuffer* zcb, size_t size);   // cudaHostAllocMapped
extern "C" int zc_free (ZeroCopyBuffer* zcb);                // cudaFreeHost
extern "C" int zc_check_device_support(int dev_id);          // cudaDevAttrCanMapHostMemory
```

microbench API (3 mode 각 mean ms/iter 반환):

```cpp
extern "C" double zc_bench_mode_zerocopy(size_t bytes, int iters, int gpu_id);
extern "C" double zc_bench_mode_dma     (size_t bytes, int iters, int gpu_id);
extern "C" double zc_bench_mode_devonly (size_t bytes, int iters, int gpu_id);
```

각 모드는 동일 kernel (`copy_u32_kernel<<<32,256>>>`) 의 mean GPU event time
을 측정 — `cudaEventRecord` 가 stream 안에서 정확한 GPU-side timing.

build:
```
nvcc -x cu -arch=sm_90 -O3 -std=c++17 -Xcompiler -fPIC -shared \
     -o libzero_copy.so zero_copy_buffer.cpp -lcudart
```

빌드 PASS (`libzero_copy.so` 39 KB, GPU 1 `cudaDevAttrCanMapHostMemory=1`).

---

## 3. microbench (H100, GPU 1, NUMA0 host)

**환경**: H100 PCIe 5.0 / Sapphire Rapids 8480+ / `taskset 0-7` / 1-run.
warmup 5 iters / measure {5000, 2000, 1000, 500, 200, 100, 50} iters 별 크기.

### 3.1 측정 결과

| size_kb | zc_us | dma_us | dev_us | zc-dev | dma-dev | ratio dma/zc | 해석 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | **7.69** | 11.91 | 6.02 | 1.67 | 5.88 | **1.548×** | ZC win |
| 8 | **8.38** | 16.35 | 6.54 | 1.84 | 9.81 | **1.951×** | ZC win |
| 16 | **9.43** | 18.35 | 5.85 | 3.58 | 12.50 | **1.946×** | ZC win |
| 32 | **10.53** | 19.50 | 5.87 | 4.66 | 13.63 | **1.852×** | ZC win |
| 64 | **12.59** | 21.44 | 6.02 | 6.57 | 15.42 | **1.703×** | ZC win |
| 128 | **16.26** | 24.50 | 5.90 | 10.36 | 18.59 | **1.507×** | ZC win |
| 256 | **24.60** | 28.29 | 6.12 | 18.48 | 22.18 | **1.150×** | ZC win |
| **512** | 42.07 | **41.59** | 6.54 | 35.53 | 35.04 | **0.988×** | **crossover** |
| 1024 | 74.68 | **76.12** | 7.31 | 67.36 | 68.81 | 1.019× | tie |
| 4096 | 175.06 | **160.55** | 12.30 | 162.76 | 148.25 | **0.917×** | DMA win |
| 16384 | 450.53 | **425.35** | 36.68 | 413.85 | 388.66 | **0.944×** | DMA win |
| 65536 | 1699.18 | **1546.29** | 264.31 | 1434.87 | 1281.98 | **0.910×** | DMA win |

자료: `zero_copy_microbench.json`.

### 3.2 해석

- **4 KB ~ 256 KB (overhead-bound)**: zero-copy 가 DMA 대비 **1.15× ~ 1.95× 빠름**.
  특히 8-64 KB 에서 1.7-2.0× 우위 — **vLLM small CPU→GPU site 의 main range** 와
  정확히 일치 (active_idx, sample/spec flattened indices 등).
- **kernel overhead = dev_us ≈ 6 μs** flat (kernel launch + minimal compute). 본
  값을 빼면 pure transfer cost: ZC 4 KB = 1.67 μs / DMA 4 KB = 5.88 μs →
  **DMA overhead 4.21 μs** = SUB_166 의 35 μs 보다 작음 (현재 PCIe 5.0 + Sapphire
  Rapids 환경 + 같은-stream pipelined, vs SUB_166 의 single discrete transfer).
- **512 KB crossover** — ZC = DMA tie. SUB_166 의 1 MB crossover 보다 빠른 영역에서
  나타남 (실측 ratio 0.988 → 1.019 → 0.917). 이유: 본 microbench 는 kernel + transfer
  를 함께 stream 으로 묶었기 때문에 DMA mode 도 pinned pipeline 이 활성화됨.
- **≥ 1 MB (bandwidth-bound)**: DMA 가 9-10% 우위. zero-copy 의 PCIe BAR read 가
  bulk transfer 대비 throughput 손해 (SUB_166 의 53.6 GB/s asymptotic vs ZC 의
  ~38 GB/s 수준).
- **CPU write cost** (`zc_bench_cpu_write` 동일 host-mapped 영역): 4-64 KB 0.1-1.4 μs
  (negligible vs 35 μs DMA overhead) — CPU side 의 mapped write 가 cache write-back
  덕분에 빠름.

### 3.3 SUB_166 1 MB crossover 재검증

| 출처 | crossover | 측정 환경 |
|---|---|---|
| SUB_166 (microbench) | 1 MB | discrete cudaMemcpy, GPU 1, torch wrapper |
| SUB_176 (pool 재현) | 1 MB @ 39.6 μs | torch wrapper 제거 |
| **SUB_179 (zc vs DMA, 본 SUB)** | **512 KB** | same-stream kernel + transfer pipelined |

본 SUB 의 같은-stream pipelined DMA 가 SUB_166 의 discrete DMA 보다 효율적이라
crossover 가 **1 MB → 512 KB 로 앞당겨짐**. zero-copy 의 sweet spot 은 SUB_166
보다 좁아진 **4-256 KB 영역**.

---

## 4. canonical 500p baseline (control measurement)

본 SUB 는 zero-copy 의 vLLM e2e integration 을 수행하지 않음. canonical 500p
baseline 은 환경 정합성 (이전 SUBs 와 같은 conditions 임을 확인) 만 측정.

**환경 (SUB_176 와 동일 protocol)**:
- Qwen 2.5 32B / TP=4×2 / `vllm serve` × 2 (vanilla 8001 / trident suffix 8002)
- AGSD router 8000 / `sub094_router.py` / `sub094_benchmark.py`
- 500p × **max-tokens=32** × concurrency=32 × 3 mix (balanced / sonnet-heavy / code-heavy)
- 1-run / boot=80 s

### 4.1 측정 결과 (AGSD-gated, 본 SUB)

| mix | wall (s) | tps | p50 (ms) | p99 (ms) | backend (trident/vanilla) |
|---|---:|---:|---:|---:|---|
| balanced | 4.0 | **3,978.4** | 176 | 477 | 335 / 165 |
| sonnet-heavy | 3.7 | **4,325.6** | 169 | 454 | 400 / 100 |
| code-heavy | 3.5 | **4,593.8** | 162 | 393 | 400 / 100 |
| **avg** |  | **4,299.3** |  |  |  |

### 4.2 SUB_176/177 와 정합성 비교

| 출처 | balanced | sonnet | code | avg | protocol |
|---|---:|---:|---:|---:|---|
| **SUB_176** (pinned pool baseline) | 4,028.8 | 4,470.0 | 4,630.5 | **4,376.4** | max-tokens=32 |
| **SUB_179** (본 SUB) | 3,978.4 | 4,325.6 | 4,593.8 | **4,299.3** | max-tokens=32 |
| Δ vs SUB_176 | −1.25% | −3.23% | −0.79% | **−1.76%** | (1-run noise 영역 < 3%) |
| SUB_177 (AMX prefill baseline) | 5,224 | 6,198 | 6,909 | 6,110 | max-tokens=**256** |

→ 본 SUB 의 baseline 은 **SUB_176 와 같은 max-tokens=32 protocol** 로 정합. avg
−1.76% 는 1-run noise floor 안 (이전 SUBs 의 |Δ|<3% 정합성 기준 PASS). SUB_177
의 6,110 tps 는 max-tokens=256 long-decode protocol 로 별개 (token-volume 8×
차이) — 직접 비교 불가.

본 SUB 의 **pure baseline 환경 정합성 PASS**.

---

## 5. vLLM 통합 가능여부 + boundary 분석

### 5.1 boundary 명확화

vLLM v1 의 small CPU→GPU path 는 이미 **`pin_memory=True` + `non_blocking=True`** 로
PyTorch 의 CUDACachingHostAllocator + cudaMemcpyAsync 를 사용. 즉:

- 현재 path = 본 microbench 의 **DMA 모드**
- zero-copy 도입 = 본 microbench 의 **ZC 모드**

microbench 가 보여준 1.5-2× 우위는 **per-transfer 1.67-10.36 μs 단위**. vLLM
의 GPU forward 1 step (H100 Qwen 32B TP=4 decode) 가 **35-44 ms** (SUB_177 TTFT
profile) 이므로, 본 small-data transfer 의 **per-step total ≤ 60 μs**:

- 본 site 8-10 개 × (DMA 4-20 μs) = **40-200 μs / step 누적**
- ZC 로 교체 시 절감 = **20-100 μs / step**
- vs step total = 35,000-44,000 μs → 절감 비율 = **0.05-0.3%**

### 5.2 6번째 drop-in 실패 가능성

**예상 e2e lift = 0.05-0.3% (noise floor 이하)** — SUB_173/174/176 의 결과와 동일
규모. drop-in 6/6 실패 패턴 거의 확정.

### 5.3 zero-copy 가 NEW workload 인 시나리오 (있다면)

| 시나리오 | feasibility |
|---|---|
| (a) CPU 가 GPU step 중간에 결과 stream → GPU 가 같은 page 를 sync 없이 read | ✗ — vLLM 의 scheduler/sampler 는 GPU 결과를 받은 후 다음 step 시작 (interlock model). zero-copy 의 비동기 update 채널이 활용될 fork 없음 |
| (b) AGSD router 의 backend dispatch 가 GPU 가 결과 보면서 CPU 가 다음 batch 준비 | ✗ — dispatch 는 < 1 KB JSON 으로 HTTP 통신, CUDA path 와 무관 |
| (c) IDE_018 phase-burst CPU 와 GPU 가 같은 attention bias 공유 | △ — phase task_pool 의 task descriptor 가 zero-copy 영역에 거주 가능, 그러나 task descriptor 는 ≤ 256 B per task — 100 tasks 도 25 KB → SUB_179 의 ZC sweet spot (4-64 KB) 영역. **유일한 conditional NEW workload 후보**. |
| (d) cold-KV (SUB_178) 의 dequant buffer 가 zero-copy → GPU 가 dequant 진행 중에도 read 가능 | △ — coherence model 이 약해 정확도 보장 비자명. SUB_178 의 DMA pipeline 이 안전 |

**conditional positive scenario (c)**: IDE_018 의 phase task_pool 의 descriptor
sharing 가 zero-copy 의 native fit. 본 SUB 는 그 가능성을 식별 만, 실 통합은
별도 SUB 필요.

---

## 6. verdict

| 차원 | 판정 |
|---|---|
| zero-copy buffer 구현 | ✅ PASS (`libzero_copy.so` 빌드 + GPU 1 mapped support 1) |
| microbench (small data 4-256 KB) | ✅ ZC 가 DMA 대비 **1.15-1.95× 빠름** (1-run, 1000-5000 iters) |
| SUB_166 1 MB crossover 재검증 | ✅ 본 SUB 환경에서는 **512 KB** (same-stream pipelined effect) |
| canonical 500p baseline | ✅ avg 4,299.3 tps (3 mix) / vs SUB_176 4,376.4 = **−1.76%** (1-run noise PASS, |Δ|<3%) |
| vLLM 통합 가능여부 (drop-in) | ⚠ 가능하나 **per-step 절감 20-100 μs / step total 35-44 ms = 0.05-0.3% lift** = noise floor 이하 |
| vLLM 통합 가능여부 (NEW workload) | △ IDE_018 phase task_pool descriptor sharing 1 후보 식별 (실 통합 별도 SUB) |
| paper §4 lever 자격 | **drop-in 6번째 noise-floor 실패 (강건한 무동작)**, NEW workload 후보는 IDE_018 phase-burst 통합 별도 검증 시 |

**결론**:
- 본 SUB 는 zero-copy 의 **physics (small-data 1.5-2× 우위)** 는 명확히 입증.
- vLLM v1 안에서 **drop-in 효과는 0.05-0.3% lift** — drop-in 6/6 실패 패턴.
- 단, SUB_178 처럼 **NEW workload 후보 1 개 식별**: IDE_018 phase-burst 의 task
  descriptor zero-copy sharing. 본 후보는 IDE_018 통합 SUB 의 prerequisite 로 기록.

---

## 7. artifacts

- `src/zero_copy_buffer.cpp` — `cudaHostAllocMapped` dual-access buffer + 3 mode microbench
- `build/libzero_copy.so` — built shared lib (nvcc -arch=sm_90)
- `run_microbench.py` — ctypes runner, mean ms/iter
- `zero_copy_microbench.json` — measurement table (12 cells × 3 modes + cpu_write)
- `launcher.sh` — canonical 500p baseline (SUB_176/177/178 pattern)
- `baseline_500p_off/{balanced,sonnet-heavy,code-heavy}/` — 3 mix benchmark output
- `logs/` — vllm boot, monitor, bench logs

---

## 8. cross-reference

- SUB_166: DMA microbench (35 μs overhead / 1 MB crossover / 53.6 GB/s asymptotic) — 본 SUB 의 reference baseline
- SUB_176: pinned pool canonical 500p — 환경 정합성 reference
- SUB_178: cold-KV dequant + DMA pipeline (NEW workload) — 본 SUB 의 verdict template
- IDE_018 phase-burst — 본 SUB 가 식별한 conditional NEW workload 후보의 통합 site
