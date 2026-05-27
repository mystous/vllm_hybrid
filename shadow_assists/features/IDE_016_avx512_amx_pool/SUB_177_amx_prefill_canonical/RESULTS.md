# SUB_177 — IDE_016 / TSK_027 AMX medium-context prefill assist canonical 500p e2e

> **parent**: TSK_027 (IDE_016). AMX 의 512-2K context GPU prefill 보조 가능여부 직접 검증.
> **scope**: 2026-05-27 10:10 ~ 10:29 KST 측정. AMX prefill microbench (12 cell) + TTFT profile (512/1024/2048 prefill, 20 req × 4 conc each) + canonical 500p baseline (9 cell).
> **status**: 측정 완료 — AMX prefill assist **drop-in 불가능 확정**, paper §4 TSK_027 lever 자격 부적합 판정.

---

## 0. 두괄식 — AMX prefill assist **drop-in 불가** 확정 ⚠

| 지표 | 측정값 | 평가 |
|---|---:|---|
| AMX 1-thread prefill matmul (Qwen 32B gate_up B=1024) | 0.0479 TFLOPS, **6,058 ms / matmul** | ⚠ |
| PT 1-thread CPU baseline 같은 shape | 0.4456 TFLOPS, 651 ms / matmul | — |
| PT 4-thread CPU baseline | 1.9281 TFLOPS, 150 ms / matmul | — |
| PT 32-thread CPU baseline | 11.6984 TFLOPS, 24.78 ms / matmul | — |
| AMX/PT-1 speedup (B=1024) | 0.107× (10× **slower**) | ⚠ |
| AMX/PT-32 speedup (B=1024) | 0.0041× (244× slower) | ⚠ |
| Qwen 32B 64-layer prefill total estimate (AMX 1-thr) | **888 sec** | ⚠ TTFT budget 의 10⁴× |
| Qwen 32B 64-layer prefill total estimate (PT 32-thr) | **3.60 sec** | TTFT budget 의 ~10² |

**결론 #1**: 본 AMX kernel 의 prefill 보조는 **per-layer 6 초** 수준 — 64 layer Qwen 32B 의 1K context 단일 prefill 만 **888 초**. paper §4 TSK_027 의 −15% TTFT (1K context) target 대비 운영 무의미.

**결론 #2**: AMX vs PT 32-thread = 244× slower → 사용자 명시 100 core max 환경 (실제 PT 가 다중 코어로 분산) 에서는 AMX 1-thread 는 **항상 더 느린 path**. AMX kernel 자체가 PT CPU multi-thread 보다 본질적으로 뒤처지므로 prefill 보조 lever 자격 부적합.

**결론 #3**: 본 SUB 는 **honest scope** — drop-in 측정 강제 안 하고 theoretical analysis + microbench + canonical baseline 정합성 확인까지로 종결. paper §4 TSK_027 의 main contributor 자격 부적합 판정.

---

## 1. AMX prefill kernel microbench (prod Sapphire Rapids 8480+)

### 1.1. 측정 환경

| 항목 | 값 |
|---|---|
| CPU | Intel Xeon 8480+ Sapphire Rapids |
| ISA flags | amx_tile / amx_bf16 / amx_int8 / avx512f-vl-bf16-vnni |
| AMX permission | granted (arch_prctl ARCH_REQ_XCOMP_PERM = OK) |
| AMX kernel | `src/amx_matmul/amx_qwen_draft.cpp` (SUB_175 와 동일 binary) |
| AMX 측정 | OMP/MKL/OPENBLAS/RAYON=1, taskset 0 (single-thread) |
| PT 1-thread 측정 | OMP/MKL=1, taskset 0 |
| PT 4-thread 측정 | OMP/MKL=4, taskset 0-3 |
| PT 32-thread 측정 | OMP/MKL=32, taskset 0-31 |
| iters | 3 (warmup 2) |

### 1.2. AMX kernel (single-thread, taskset 0) — Qwen 32B prefill shapes

| shape | B (context) | K | N | best (ms) | TFLOPS |
|---|---:|---:|---:|---:|---:|
| qwen32b_gate_up | 512 | 5,120 | 27,648 | 3,022.73 | 0.0480 |
| qwen32b_gate_up | 1,024 | 5,120 | 27,648 | 6,058.38 | 0.0479 |
| qwen32b_gate_up | 2,048 | 5,120 | 27,648 | 12,159.89 | 0.0477 |
| qwen32b_down | 512 | 27,648 | 5,120 | 3,183.61 | 0.0455 |
| qwen32b_down | 1,024 | 27,648 | 5,120 | 6,389.10 | 0.0454 |
| qwen32b_down | 2,048 | 27,648 | 5,120 | 12,820.79 | 0.0452 |
| qwen32b_qkv | 512 | 5,120 | 7,680 | 460.83 | 0.0874 |
| qwen32b_qkv | 1,024 | 5,120 | 7,680 | 908.60 | 0.0886 |
| qwen32b_qkv | 2,048 | 5,120 | 7,680 | 1,905.58 | 0.0845 |
| qwen32b_o_proj | 512 | 5,120 | 5,120 | 260.46 | 0.1031 |
| qwen32b_o_proj | 1,024 | 5,120 | 5,120 | 522.95 | 0.1027 |
| qwen32b_o_proj | 2,048 | 5,120 | 5,120 | 1,085.56 | 0.0989 |

→ context 길이에 대해 latency 가 거의 선형 증가 (per-matmul ms ∝ B). throughput 은 0.046-0.103 TFLOPS 로 SUB_175 의 microbench 와 정확히 일치 (재현성 PASS). 단 작은 N (5,120-7,680) shape 가 큰 N (27,648) 보다 약 2× 효율 — tile pack 비용 분산 효과 추정.

### 1.3. PT BF16 CPU baseline (cross-thread sweep)

| shape (B=1024) | PT 1-thr (ms / TFLOPS) | PT 4-thr (ms / TFLOPS) | PT 32-thr (ms / TFLOPS) |
|---|---:|---:|---:|
| qwen32b_gate_up | 650.56 / 0.4456 | 150.36 / 1.9281 | 24.78 / 11.6984 |
| qwen32b_down | 727.89 / 0.3983 | 166.47 / 1.7415 | 24.73 / 11.7217 |
| qwen32b_qkv | 131.15 / 0.6140 | 33.45 / 2.4076 | 4.20 / 19.1597 |
| qwen32b_o_proj | 84.23 / 0.6374 | 20.68 / 2.5964 | 2.60 / 20.6337 |

→ PT 32-thread 가 AMX 대비 ~244× 빠름 (gate_up B=1024). PT 의 oneDNN/MKL backend 가 AMX bf16 tile ops + L2 blocking + register tiling + parallel reduce 를 fully utilize.

### 1.4. Per-layer prefill latency estimate (Qwen 32B = 64 layers)

per-layer ≈ (1 QKV + 1 O_proj + 1 gate_up + 1 down) — attention 본체 (FA) 제외, 순수 dense matmul 만.

| kernel | B (context) | per-layer (ms) | × 64 layer (sec) |
|---|---:|---:|---:|
| **amx 1-thr** | 512 | 6,927.63 | **443.37** |
| **amx 1-thr** | 1,024 | 13,879.04 | **888.26** |
| **amx 1-thr** | 2,048 | 27,971.83 | **1,790.20** |
| pt 1-thr | 512 | 761.22 | 48.72 |
| pt 1-thr | 1,024 | 1,593.83 | 102.01 |
| pt 1-thr | 2,048 | 3,176.48 | 203.29 |
| pt 4-thr | 512 | 171.83 | 11.00 |
| pt 4-thr | 1,024 | 370.96 | 23.74 |
| pt 4-thr | 2,048 | 741.21 | 47.44 |
| **pt 32-thr** | 512 | 25.52 | **1.63** |
| **pt 32-thr** | 1,024 | 56.32 | **3.60** |
| **pt 32-thr** | 2,048 | 122.25 | **7.82** |

**해석**: GPU H100 의 32B 1K context TTFT 는 일반적으로 100-300 ms 범위. AMX 1-thread 의 888 초 / PT 32-thread 의 3.6 초 둘 다 GPU baseline 의 10³-10⁴× 더 느려 prefill 보조 lever 자격 부적합.

---

## 2. TTFT profile (vllm GPU prefill, vanilla port 8001)

### 2.1. 측정 protocol

- Qwen 2.5 32B Instruct TP=4 (port 8001 vanilla, no spec decode)
- input_tokens: 512 / 1024 / 2048 target → tokenizer actual = 481 / 897 / 1729 prompt tokens
- n_requests = 20 per length, concurrency = 4, max_tokens = 32
- stream=True 로 TTFT (time-to-first-token chunk) 측정
- 1-run

### 2.2. 결과 — GPU H100 TP=4 baseline TTFT

| target | actual prompt tok | min (ms) | p50 (ms) | p90 (ms) | p99 (ms) | max (ms) |
|---|---:|---:|---:|---:|---:|---:|
| 512 | 481 | 29.6 | **35.7** | 234.5 | 234.8 | 234.8 |
| 1024 | 897 | 32.6 | **43.6** | 99.7 | 99.9 | 99.9 |
| 2048 | 1,729 | 31.8 | **44.3** | 88.4 | 96.5 | 96.5 |

→ GPU TTFT p50 = 35.7-44.3 ms 범위. p99 가 1K-2K 에서 100 ms 이하, 512 에서 234 ms (4 conc 의 head-of-line bottleneck — 첫 batch 4 동시 시 queue tail). 1K/2K 영역의 작은 p99 분산은 GPU prefill 이 PIECEWISE cudagraph + chunked prefill (`--enable-chunked-prefill` default) 로 안정적 처리됨을 시사.

### 2.3. AMX prefill assist 의 budget 비교

| context | GPU TTFT p50 (ms) | AMX 1-thr per-layer (ms) | AMX 64-layer total (sec) | PT 32-thr 64-layer (sec) |
|---|---:|---:|---:|---:|
| 512 | 35.7 | 6,928 | **443** | **1.63** |
| 1,024 | 43.6 | 13,879 | **888** | **3.60** |
| 2,048 | 44.3 | 27,972 | **1,790** | **7.82** |

→ GPU TTFT (35-44 ms) vs AMX 64-layer (443-1790 sec) = **10⁴~10⁵× gap**. AMX prefill assist 의 본 kernel 로는 prefill 보조가 **시스템 throughput 에 음의 영향만 발생** — CPU 가 prefill 결과를 늦게 내놓아 GPU 가 그것을 기다릴 수 없음. PT 32-thread baseline (1.63-7.82 sec) 도 GPU 의 35-44 ms 대비 100× 느려 동일 결론.

### 2.4. paper §4 TSK_027 의 −15% TTFT 1K context feasibility

paper §4 TSK_027 target = TTFT −15% on 1K context. 본 측정의 GPU baseline 43.6 ms p50 의 −15% = **37.1 ms target**. CPU assist 가 6.5 ms 만 보조해도 달성 — 그러나 본 AMX kernel 의 단일 matmul (B=1024 qkv) = 909 ms 로 budget 의 140× 초과. PT 32-thread (4.20 ms) 가 단일 qkv 만 처리 시 budget 부합 — 그러나 vllm 안 GPU prefill path 에 CPU partial 결과를 inject 하는 인프라 (CPU 결과를 GPU KV 와 merge) 가 **존재하지 않음**. drop-in 불가능.

---

## 3. canonical 500p baseline (control measurement)

본 SUB 의 vllm 측정은 **pure baseline** — AMX kernel 은 vllm 안에서 사용되지 않음. canonical 환경 재현 정합성만 확인.

### 3.1. 측정 protocol

- Qwen 2.5 32B Instruct TP=4×2 (port 8001 vanilla / port 8002 trident suffix-32)
- AGSD router (sub094_router.py) gated, classifier_workers=4
- 500 prompt × 32 conc × 256 max_tokens × 3 mix (balanced/sonnet-heavy/code-heavy)
- 1-run, monitor 0.5s interval
- ENV: `RAYON/OMP/MKL/OPENBLAS=4`, `TOKENIZERS_PARALLELISM=false`

### 3.2. 결과

| mix | scenario | tps | p50 (s) | p99 (s) |
|---|---|---:|---:|---:|
| balanced | vanilla-only | 2,451.21 | 3.22 | 3.33 |
| balanced | trident-only | 3,949.63 | 1.48 | 4.63 |
| balanced | **agsd-gated** | **5,223.52** | **0.76** | **3.28** |
| sonnet-heavy | vanilla-only | 2,610.08 | 3.01 | 3.19 |
| sonnet-heavy | trident-only | 5,984.27 | 0.86 | 3.98 |
| sonnet-heavy | **agsd-gated** | **6,197.81** | **0.67** | **2.92** |
| code-heavy | vanilla-only | 2,544.80 | 3.14 | 3.29 |
| code-heavy | trident-only | 5,995.24 | 0.90 | 4.32 |
| code-heavy | **agsd-gated** | **6,909.43** | **0.64** | **2.85** |

**3-mix avg AGSD**: **6,110 tps** (= (5,224 + 6,198 + 6,909) / 3) vs SUB_174 OFF **6,240 tps** = **−2.08%**, vs SUB_175 OFF **6,256 tps** = **−2.33%** (1-run noise 영역, |Δ| < 3% ✅ 정합성 PASS)

→ canonical 환경 재현 정합성 PASS. 본 SUB 의 측정 환경은 SUB_174/SUB_175 OFF 와 동등. AMX kernel 의 e2e 영향은 없음 (vllm 안에서 사용되지 않음 — pure control measurement).

### 3.3. CPU / GPU util (498 sample × 0.5s, ~249 s total)

| 항목 | SUB_177 | SUB_175 | SUB_174 OFF | 평가 |
|---|---:|---:|---:|---|
| CPU util mean (system %) | 4.53 | 4.47 | 4.50 | 정합 (~4.5%) |
| CPU util p50 | 4.10 | — | — | — |
| CPU util p99 | 10.50 | 9.90 | — | 정합 |
| CPU util max | 36.70 | — | — | benchmark client surge |
| mem mean (GB) | 107.1 | 106.8 | 106.8 | 정합 |
| GPU 0 (vanilla) mean util | 45.5 | 46.7 | 49.4 | 정합 |
| GPU 1-3 (vanilla) mean util | 52.7 / 53.0 / 52.2 | 51.1 / 51.7 / 51.3 | 50.4 / 51.0 / 50.6 | 정합 |
| GPU 4-7 (trident) mean util | 24.5 / 24.3 / 25.0 / 24.1 | 25.4 / 25.2 / 25.2 / 24.7 | 25.8 / 25.0 / 25.7 / 25.4 | 정합 |

→ 모든 util/mem 지표 SUB_174/SUB_175 OFF 와 동등 — 환경 재현 정합성 추가 검증 PASS.

---

## 4. paper §4 TSK_027 lever 의 status

| target | 결과 |
|---|---|
| TSK_027 TTFT −15% on 1K context | **미달** — 본 AMX kernel 보조 시 prefill compute 시간 ↑ 만 발생 (서비스 throughput 음의 영향) |
| AMX e2e drop-in 위치 (vllm prefill 안) | **없음** — vllm 의 prefill 모든 matmul 은 GPU 위 |
| AMX kernel rewrite 후 가능여부 | per-layer 10× upside (L2 blocking + register tiling) + multi-thread 4× upside ≈ ~10 ms/layer 가능. 그래도 GPU prefill (1-3 ms / layer) 대비 4× 느림 — CPU 보조 lever 의 본질적 한계 |

→ **TSK_027 paper §4 main contributor 자격 부적합**. AMX prefill assist 는 본 kernel 로 운영 불가.

---

## 5. 한계 및 후속 turn

### 5.1. 본 SUB 의 한계

| 항목 | 한계 |
|---|---|
| AMX kernel 최적화 | SUB_175 와 동일 — naive tile loop, no L2 blocking / no register tiling / single thread |
| canonical e2e | AMX 가 vllm 내부 사용처 없음 → baseline 정합성 만 |
| TTFT 직접 측정 | concurrency=4 의 head-of-line 영향, 본 SUB scope 내 noise |
| AMX 가 vllm prefill 안 직접 통합 | 불가능 (prefill matmul 은 cublas/cudnn GPU path) |

### 5.2. 후속 turn 제안

1. **AMX prefill 영역 자체 폐기 검토**: paper §4 TSK_027 는 lever 자격 부적합 — IDE_016 의 paper main lever 는 TSK_024 (tokenizer) + TSK_025 (sampling) + TSK_026 (draft head matmul) 로 응축
2. **TSK_036 (CPU draft head) 별도 turn 의 입력**: 본 SUB 의 PT 32-thread 측정 (per-layer 56 ms B=1024) 이 Qwen 0.5B draft head 의 multi-thread budget 입력 (별도 turn)
3. **CPU prefill assist 자체 idea 영구 폐기**: GPU H100 + multi-instance 환경에서 prefill compute-bound 시 CPU 도움 없음 — IDE_002 ↔ IDE_016 TSK_027 mapping 폐기 권장 (별도 turn)

### 5.3. honest verdict

본 SUB 사전 인지대로 **drop-in 불가능 확정**. TSK_027 의 −15% TTFT 1K context target 은 **본 AMX kernel + 본 vllm 환경에서 운영 불가**. AMX hardware feature 자체는 prod 머신에 가용 — 단지 prefill 영역의 보조로는 적합하지 않음 (GPU 의 4×10² 배 throughput gap).

---

## 6. raw data

- `microbench_prefill.py` — AMX prefill kernel + PT baseline microbench source
- `amx_rows.json` / `pt_t1.json` / `pt_t4.json` / `pt_t32.json` — microbench raw rows
- `ttft_profile.py` / `ttft_profile/ttft_vanilla.json` — TTFT measurement source + raw
- `baseline_500p_off/{balanced,sonnet-heavy,code-heavy}/benchmark_*.json` — canonical 9 cell
- `_monitor_off_{cpu,gpu}.csv` — 0.5s system monitor (498 / 3,984 samples)
- `logs/{vanilla,trident,router,monitor,main,ttft_vanilla,amx_microbench,run_full}.log`
- launcher: `run_sub177.sh`
- boot duration: `logs/boot_off_seconds.txt` = 80s

---

## 7. references

- IDE_016 plan: `spec_decoding/plan/README.md` §IDE_016 / §TSK_027
- TSK_027 source: 본 SUB 영역 (별도 신규 코드 없음)
- AMX kernel: SUB_175 의 `src/amx_matmul/{amx_kernels.h, amx_qwen_draft.cpp}` 재사용
- prior measurements: SUB_106 (22 TFLOPS peak), SUB_117 (10.24 TFLOPS available), SUB_174 (canonical 6,240 OFF), SUB_175 (AMX matmul canonical e2e — drop-in 불가)
