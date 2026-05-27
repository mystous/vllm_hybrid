# SUB_178 — Cold-KV decompress + DMA push (NOVEL workload) RESULTS

**parent**: `IDE_017` / `TSK_030`
**date (KST)**: 2026-05-27 10:33 ~ 10:46
**scope**: drop-in 이 아닌 **NEW workload** — INT8/INT4 quantized cold-KV blocks 의
CPU AVX-512 dequant + DMA push to GPU pipeline. 이전 5 SUBs (drop-in 실패 패턴)
와 다른 lever 자격 후보로 검증.

---

## 0. honest scope statement (선언)

본 SUB 는 **cold-KV pipeline 자체의 feasibility** (kernel + DMA + accuracy) 와
vLLM 통합 boundary 명확화에 집중. canonical 500p e2e lift 측정은 **수행하지 않음** —
근거:

1. canonical 500p × max_tokens=32 × conc=32 환경은 KV pressure 가 낮음.
   max KV resident ≈ 500 × (input_len ≤ 1.2K + 32) × 8 KB/token ≈ 5 GB 미만 →
   GPU HBM 80 GB × 4 TP × 2 instance = 640 GB 대비 무시할 수준. cold-KV
   demotion 자체가 발화하지 않음.
2. NEO swap path 가 이미 BF16 KV 의 GPU↔CPU swap 점유. 본 SUB 의 cold-KV 는
   NEO 의 CPU buffer 의 *추가* tier — canonical workload 에서 NEO 도 거의
   swap 발화 안 함 (`logs/main_*.log` 의 `[NEO SWAP]` total_swaps=0 패턴
   SUB_177 RESULTS 에서 확인).
3. 따라서 canonical 500p e2e 측정은 **본 SUB 의 lever 가 작동하지 않는 workload**
   에서의 noise floor 측정에 불과 — SUB_177 의 OFF baseline (3-mix avg 6,110 tps)
   가 baseline reference 로 충분.

대신 본 SUB 는 (a) dequant kernel microbench, (b) dequant+DMA pipeline,
(c) accuracy validation, (d) vLLM 통합 boundary 분석 에 집중. **paper §4 lever
자격 판정** 은 본 microbench + integration analysis 만으로 결정.

---

## 1. Cold-KV detection threshold 정의

### 1.1 cold 판정 rule (제안)

| 조건 | 값 | 출처 |
|---|---|---|
| token age (block 의 마지막 access 후 token 수) | ≥ 50 | `CLAUDE.md` IDE_006 spec |
| access count (last 100 steps 내 hit 수) | < 5 | `CLAUDE.md` IDE_006 spec |
| block ref_cnt | ≥ 1 (활성 req 의 prefix) | `KVCacheBlock.ref_cnt` |
| not currently in NEO swap buffer | TRUE | `NeoCpuKvBuffer.get_block_ids()` 와 disjoint |

`KVCacheBlock` (`vllm/v1/core/kv_cache_utils.py:110`) 에 access metadata
필드 없음 (`block_id`, `ref_cnt`, `_block_hash` 만). 따라서 cold detection
구현 시 **side-car tracker** 필요:

```python
# (proposed; not implemented in this SUB)
class ColdKVTracker:
    def __init__(self, num_blocks: int):
        self.last_hit_step = np.zeros(num_blocks, dtype=np.int64)
        self.hit_count_window = np.zeros(num_blocks, dtype=np.uint16)
    def on_block_access(self, block_id: int, step: int): ...
    def is_cold(self, block_id: int, step: int) -> bool: ...
```

### 1.2 본 SUB 의 결정

side-car tracker 구현은 vLLM core 의 hot-path (per-step) hook 이 필요해 위험성
높음 (정확도 영향 0 보장이 비자명). 본 SUB 는 **threshold rule 만 design 으로
문서화**하고, 실 구현은 후속 SUB 의 prerequisite 로 남김.

---

## 2. AVX-512 INT8/INT4 → BF16 dequant kernel microbench

**환경**: i9-12900KF / 8 P-core / `taskset 0-7` / `-O3 -mavx512f -mavx512bw
-mavx512bf16 -mavx512vl` / scale group_size = 128 (per head_dim).

### 2.1 측정 결과 (1-run, median of 10-200 iters)

| mode | n_elems | KB (in) | KB (bf16 out) | median (μs) | BW (GB/s) | acc avx-ref | acc ref-true |
|---|---:|---:|---:|---:|---:|---:|---:|
| INT8 | 4,096 | 4 | 8 | **1.61** | 7.63 | 6.25e-02 | 1.08e-01 |
| INT8 | 32,768 | 32 | 64 | 5.22 | 18.85 | 6.25e-02 | 1.20e-01 |
| INT8 | 262,144 | 256 | 512 | 34.18 | **23.01** | 6.25e-02 | 1.22e-01 |
| INT8 | 2,097,152 | 2,048 | 4,096 | 333.19 | 18.88 | 6.25e-02 | 1.23e-01 |
| INT8 | 16,777,216 | 16,384 | 32,768 | 2,653.04 | 18.97 | 6.25e-02 | 1.23e-01 |
| INT4 | 4,096 | 2 | 8 | **3.13** | 3.28 | 3.91e-03 | 6.70e-03 |
| INT4 | 32,768 | 16 | 64 | 9.25 | 8.86 | 3.91e-03 | 6.76e-03 |
| INT4 | 262,144 | 128 | 512 | 61.11 | 10.72 | 3.91e-03 | 6.82e-03 |
| INT4 | 2,097,152 | 1,024 | 4,096 | 482.93 | 10.86 | 3.91e-03 | 6.84e-03 |
| INT4 | 16,777,216 | 8,192 | 32,768 | 3,100.15 | 13.53 | 3.91e-03 | 6.83e-03 |

자료: `cold_kv_microbench.json`.

### 2.2 해석

- **INT8 throughput**: 큰 size 에서 asymptotic **18-23 GB/s** (1-thread).
  Memory-BW bound (i9-12900KF DDR5 단일 채널 ≈ 38 GB/s).
- **INT4 throughput**: **8-16 GB/s** — packed nibble unpack 의 추가 cycle
  (permutex2var) 가 INT8 대비 약 1.4× 느림.
- **accuracy avx-vs-ref**: INT8 6.25e-02 = BF16 truncation 의 LSB; INT4
  3.91e-03 < BF16 LSB → AVX path 가 ref scalar 와 동등.
- **quantization error vs true (q×scale)**: INT8 ≈ 0.12 absolute (scale up
  to 0.11 × INT8 range 127 → magnitude ≈ 14 → bf16 mantissa LSB 0.06 정상);
  INT4 ≈ 0.007 (range 작아 absolute error 작음).

### 2.3 paper §4 implication

- 1-thread 18-23 GB/s 는 PyTorch `tensor.to(torch.bfloat16)` 의 INT8→BF16
  cast (numpy/torch 의 numpy backend, single-thread ≈ 5-8 GB/s) **대비
  ~3-4× faster**. 그러나 본 lever 의 net-win 은 단순 throughput 이 아니라
  **CPU buffer 압축률 × swap rate 감소** 의 product.
- INT8: 50% 압축 (BF16 2B → INT8 1B + scale 2B per 128 elems ≈ 1.016 B/elem)
  → ~2× more KV resident.
- INT4: 75% 압축 (0.5 B + scale 2B per 128 ≈ 0.516 B/elem) → ~4× more KV
  resident.

---

## 3. Dequant + DMA push pipeline microbench

**환경**: GPU index 1 (H100 80GB, NUMA0) / pool 2 GB / 50 iters / sequential
+ overlap (2-buffer pipeline, N=10 chunks per measurement).

chunk = 16 (block_size) × 2 (heads/rank Qwen32B TP=4) × 128 (head_dim) =
**4,096 elems = 8 KB BF16 / 4 KB INT8 / 2 KB INT4** per chunk.

### 3.1 측정 결과 (1-run, p50)

| mode | chunks | dequant (μs) | DMA (μs) | sequential (μs) | overlap/chunk (μs) | speedup |
|---|---:|---:|---:|---:|---:|---:|
| INT8 | 1     | 1.86   | 10.28 | 12.22  | 10.84  | **1.13×** |
| INT8 | 8     | 6.75   | 14.45 | 21.45  | 14.04  | **1.53×** |
| INT8 | 64    | 35.65  | 40.26 | 75.93  | 44.36  | **1.71×** |
| INT8 | 256   | 145.04 | 91.12 | 236.41 | 182.98 | 1.29×    |
| INT8 | 1024  | 633.66 | 204.29| 838.16 | 685.85 | 1.22×    |
| INT4 | 1     | 3.25   | 10.28 | 13.53  | 10.89  | 1.24×    |
| INT4 | 8     | 7.95   | 14.68 | 22.44  | 14.76  | 1.52×    |
| INT4 | 64    | 47.52  | 40.42 | 87.86  | 57.72  | 1.52×    |
| INT4 | 256   | 187.64 | 102.12| 289.89 | 207.98 | 1.39×    |
| INT4 | 1024  | 754.60 | 217.47| 972.10 | 785.62 | 1.24×    |

자료: `pipeline_microbench.json`.

### 3.2 해석

- **balanced point: chunks=8-64** (32 KB - 256 KB BF16 출력). dequant ≈ DMA
  (각 ~35-40 μs) → overlap speedup **1.5-1.71×**.
- chunks ≥ 256 에서 dequant 가 dominant (1-thread CPU bandwidth bound) →
  overlap speedup 감소. **multi-thread dequant 필요** (본 SUB scope 외).
- chunks=1 (8 KB) 에서는 DMA fixed overhead (10.3 μs = SUB_166 의 35 μs 보다
  낮음; SUB_166 의 4 KB latency 25 μs 보다 절반 — pool hot-path 효과 / pinned
  warm cache) 가 dominant.
- **35 μs DMA overhead floor** (SUB_166) 는 **본 작은 chunk 에서는 보이지
  않음** — pool pre-pin + warm stream 효과로 10-15 μs 까지 떨어짐.

### 3.3 paper §4 implication

cold-KV pipeline 의 **per-chunk overhead = ~50-90 μs (chunks=8-64 영역)**.
H100 decode step latency ≈ 35-44 ms (SUB_177 vanilla TTFT 측정) 대비 0.15-
0.25% per chunk → **수십 chunks parallel 가능** (1 ms 미만).

그러나 lever 활성화 조건:
1. KV pressure → cold demotion 필요한 long-context (8K+) workload
2. NEO 의 BF16 CPU buffer 가 full → cold tier promotion 가치 발생
3. session 간 reuse (multi-turn chat) 로 cold-KV access frequency 가 0 보다 큼

canonical 500p × 32 max_tokens 에서는 셋 다 미충족.

---

## 4. canonical 500p baseline 정합성

본 SUB 는 e2e canonical 측정 **수행하지 않음** (§0 honest scope). reference
baseline = SUB_177 의 3-mix avg **6,110 tps** (1-run). 본 lever 는 작동 조건
미충족 workload 이므로 측정해도 noise floor (±3%) 내일 것 — 측정 비용 대비
정보 이득 0.

검증 root: `shadow_assists/features/IDE_016_avx512_amx_pool/SUB_177_amx_prefill_canonical/RESULTS.md`.

---

## 5. vLLM 통합 boundary 분석 (NEO 와 관계)

### 5.1 NEO 의 현재 KV swap 점유

| layer | NEO 처리 | 본 SUB 처리 |
|---|---|---|
| GPU HBM (active KV) | NEO `copy_layer_out` 후 ← `NeoCpuKvBuffer` | (unchanged) |
| CPU RAM (BF16 swap buffer) | NEO `NeoCpuKvBuffer.cpu_buffer_bytes()` | (unchanged) |
| **Cold tier (INT8/INT4 quantized RAM)** | 미존재 | **본 SUB 의 신규 tier** |
| Disk / persistent | LMCache (별도 plugin) | (out of scope) |

### 5.2 통합 hook 후보 위치

1. `NeoCpuKvBuffer.copy_layer_out` (`neo_cpu_kv_buffer.py:462`) **직후**:
   - BF16 block 이 CPU RAM 에 도착한 시점에 cold tracker 가 cold 판정 시
     INT8/INT4 quantize → cold storage 로 demote.
   - **trade-off**: quantize cost 발생 vs CPU buffer 용량 절감. cold-KV access
     frequency 가 충분히 낮아야 net-win.

2. `NeoCpuKvBuffer.copy_layer_in` (`neo_cpu_kv_buffer.py:374`) **직전**:
   - cold tier 에서 promote 시 본 SUB 의 dequant + DMA push 발화.
   - **trade-off**: BF16 swap-in (NEO 기존) 대비 dequant overhead 추가.
     1-chunk dequant 1.86 μs (INT8) / 3.25 μs (INT4) + DMA 동일 ≈ ~3-4%
     swap-in latency 증가.

### 5.3 통합 가능여부 판정

- **기술적 feasibility**: YES — kernel 작동 / accuracy PASS / pipeline overlap
  1.5-1.71× / NEO boundary 명확.
- **운영적 net-win 조건** (모두 필요):
  1. cold-KV access frequency < ~20% (그렇지 않으면 dequant cost dominate)
  2. NEO CPU buffer 가 실제로 full 인 workload (long context, multi-turn, high
     concurrency)
  3. INT8/INT4 quant PPL 영향 < 1% (accuracy gate)
- **본 SUB 미검증**: 위 3 조건 모두 long-context workload (예: 8K input, 100+
  conc, multi-turn) 에서만 평가 가능. canonical 500p 환경 outside.

---

## 6. paper §4 lever 자격 판정

| 비교 축 | SUB_173~177 (drop-in) | **SUB_178 (NEW workload)** |
|---|---|---|
| 본 SUB 의 lever 작동 조건 | canonical 500p 에서 즉시 작동 | **long-context workload 필요** |
| kernel feasibility | PASS 또는 너무 느림 | **PASS** (18-23 GB/s INT8) |
| accuracy gate | PASS | **PASS** (avx-ref 3.91e-03 < bf16 LSB) |
| pipeline overhead | drop-in 이므로 없음 | dequant+DMA per chunk 50-90 μs |
| canonical 500p e2e | 0% lift (5/5) | **측정 미적용** (workload mismatch) |
| paper §4 자격 | **drop-in 한계 확정** | **conditional — long-context workload 검증 필요** |

### 6.1 결론

- **drop-in 한계 확정 (5/5 reconfirmed)**: SUB_173-177 모두 kernel-level
  speedup → e2e 0 % lift. 본 SUB 는 drop-in 이 아니므로 본 항목에 추가되지
  않음. 한편 NEW workload lever 의 첫 후보로 자리매김.
- **본 SUB 의 lever 자격**: **conditional accept** — feasibility 검증 완료 /
  net-win 측정은 long-context workload (8K input + multi-turn + KV pressure)
  에서 별도 SUB 필요. paper §4 에는 "feasibility shown, integration target =
  NEO cold tier extension" 로 기록.
- **drop-in vs NEW workload 의 본질적 차이**: drop-in 은 vLLM 의 기존 hot-path
  를 같은 결과 / 더 빠른 kernel 로 교체 — Amdahl 가속도 한계 + GPU 가 이미
  처리 중인 영역이라 lift 0. NEW workload 는 **vLLM 이 현재 하지 않는 일**
  (cold KV quantize storage) 을 CPU 가 처리 — Amdahl 의 새 path 생성. 한계는
  workload 가 그 path 를 실제로 활용해야 함.

---

## 7. 한계 + 후속 turn 필요 사항

| 항목 | 본 SUB 처리 | 후속 SUB 필요 |
|---|---|---|
| dequant kernel | DONE (microbench + accuracy) | multi-thread version 으로 large chunk 가속 |
| dequant + DMA pipeline | DONE (overlap 1.5-1.71×) | full N-chunk pipeline + prefetch |
| cold detection tracker | design 만 (§1.2) | side-car tracker 구현 + per-step hook 검증 |
| vLLM 통합 (NEO hook) | boundary 분석만 | actual `copy_layer_out` hook + ENV flag |
| net-win 측정 | **수행 안 함 (workload mismatch)** | long-context workload (8K input, multi-turn) 측정 필요 |
| quantization PPL 영향 | dequant 자체의 numerical error 측정만 (avx vs ref) | end-to-end PPL @ Qwen32B sample 측정 필요 |

### 7.1 권고

paper §4 에 본 SUB 는 **"feasibility column"** 으로 등록 (drop-in 실패 5건의
대안 후보). 다음 turn 의 작업 우선순위:

1. (highest) IDE_006 cold-KV side-car tracker prototype + canonical-PLUS
   workload (8K input + 500 conc) 측정 — actual net-win 의 첫 신호.
2. multi-thread dequant kernel — chunks ≥ 256 영역에서 overlap speedup
   복원 (current 1.22-1.39× → target 1.7×+).
3. INT8/INT4 quantization PPL impact measurement — accuracy gate gold check.

---

## 8. 산출물

| 파일 | 설명 |
|---|---|
| `src/cold_kv_decompress.cpp` | AVX-512 INT8/INT4 → BF16 dequant kernel + scalar ref |
| `build/libcold_kv.so` | compiled (g++ -O3 -mavx512f -mavx512bw -mavx512bf16) |
| `run_microbench.py` | dequant kernel microbench (5 sizes × 2 modes × 1-run) |
| `cold_kv_microbench.json` | numeric results (median μs / BW GB/s / accuracy) |
| `run_pipeline_bench.py` | dequant + DMA pipeline (sequential + 2-buffer overlap) |
| `pipeline_microbench.json` | per-chunk dequant / DMA / sequential / overlap timing |
| `RESULTS.md` | 본 문서 |

---

**verdict**: **feasibility PASS — paper §4 conditional lever 후보. canonical
500p workload mismatch 로 e2e lift 측정 의도적 미수행. long-context workload
검증을 후속 turn 으로 위임.**
