# SUB_175 — IDE_016 / TSK_026 AMX BF16 matmul canonical 500p e2e

> **parent**: TSK_026 (IDE_016). AMX BF16 kernel 의 prod (Sapphire Rapids 8480+) 직접 검증.
> **scope**: 2026-05-27 09:39 ~ 10:30 KST 측정. AMX microbench (12 cell) + canonical 500p baseline (9 cell).
> **status**: 완료 — AMX kernel build PASS / 정확도 PASS / **e2e feasibility 한계 확인** ⚠

---

## 0. 두괄식 — AMX kernel **현 implementation 의 e2e 효용 제한** ⚠

| 지표 | 측정값 | 평가 |
|---|---:|---|
| AMX kernel 정확도 (M=16,K=32,N=16, fp32 ref 비교) | max_abs_err **9.5e-7** | ✅ bit-exact 수준 |
| AMX kernel 1-thread throughput (Qwen 7B B=256) | **0.049 TFLOPS** | ⚠ SUB_106 의 22 TFLOPS peak 와 큰 격차 |
| AMX kernel 1-thread throughput (Qwen 32B B=256) | **0.046-0.049 TFLOPS** | ⚠ |
| PT BF16 CPU baseline 1-thread (Qwen 7B B=256) | 0.602 TFLOPS | — |
| PT BF16 CPU baseline 4-thread (Qwen 7B B=256) | 2.126 TFLOPS | — |
| PT BF16 CPU baseline 32-thread (Qwen 32B B=256, gate_up) | 13.79 TFLOPS | — |
| **AMX vs PT 1-thread speedup (Qwen 7B B=256)** | **0.082×** (12× **slower**) | ⚠ ≥3× target 미달 |
| **AMX vs PT 32-thread speedup (Qwen 32B B=256, gate_up)** | **0.0036×** (280× slower) | ⚠ paper §4 target 미달 |

**결론 #1**: 현 `amx_qwen_draft.cpp` 의 naive tile-loop kernel (no L1/L2 blocking, no register tiling, no streaming prefetch, single-thread) 는 PT BF16 multi-thread baseline 보다 한참 느림. SUB_106 의 22 TFLOPS peak 는 multi-core full-machine + oneDNN/libxsmm 외부 라이브러리 기준일 가능성 매우 큼.

**결론 #2**: paper §4 의 TSK_026 ≥3× speedup vs PyTorch CPU matmul target **현 kernel 로는 도달 불가**. AMX 사용 자체는 가능 (정확도 + permission grant + ISA dispatch 모두 PASS), 그러나 본 minimal kernel 의 효율성 부족.

**결론 #3**: vllm 의 거의 모든 matmul 은 GPU 위에 있어 본 AMX kernel 은 vllm e2e 흐름에 직접 통합 불가 — drop-in replacement 시도조차 불가능. IDE_019 TSK_036 (CPU draft head, Qwen 0.5B) 가 실제 사용처 (별도 turn).

---

## 1. AMX microbench (prod Sapphire Rapids 8480+)

### 1.1. 측정 환경

| 항목 | 값 |
|---|---|
| CPU | Intel Xeon 8480+ Sapphire Rapids |
| ISA flags | amx_tile / amx_bf16 / amx_int8 / avx512f-vl-bf16-vnni |
| Python | /workspace/vllm_dev_prj/bin/python (torch 2.11.0+cu128) |
| AMX permission | granted (arch_prctl ARCH_REQ_XCOMP_PERM = OK) |
| AMX kernel 1-thread (taskset 0) | OMP/MKL/OPENBLAS/RAYON=1 |
| PT BF16 baseline | torch.matmul(A_bf16, B_bf16) 1/4/32 thread |

### 1.2. AMX kernel (single-thread, taskset 0)

| shape | B | K | N | best (ms) | TFLOPS |
|---|---:|---:|---:|---:|---:|
| qwen7b_gate_up | 64 | 3,584 | 18,944 | 177.70 | 0.049 |
| qwen7b_gate_up | 128 | 3,584 | 18,944 | 349.20 | 0.050 |
| qwen7b_gate_up | 256 | 3,584 | 18,944 | 710.43 | 0.049 |
| qwen7b_down | 64 | 18,944 | 3,584 | 178.62 | 0.049 |
| qwen7b_down | 128 | 18,944 | 3,584 | 336.63 | 0.052 |
| qwen7b_down | 256 | 18,944 | 3,584 | 696.86 | 0.050 |
| **qwen32b_gate_up** | 64 | 5,120 | 27,648 | 367.44 | 0.049 |
| qwen32b_gate_up | 128 | 5,120 | 27,648 | 743.93 | 0.049 |
| qwen32b_gate_up | 256 | 5,120 | 27,648 | 1,485.97 | 0.049 |
| qwen32b_down | 64 | 27,648 | 5,120 | 389.98 | 0.046 |
| qwen32b_down | 128 | 27,648 | 5,120 | 787.67 | 0.046 |
| qwen32b_down | 256 | 27,648 | 5,120 | 1,577.87 | 0.046 |

→ batch 와 shape 에 거의 무관 **0.046 ~ 0.052 TFLOPS** flat — kernel 의 fixed per-tile overhead 가 dominant. 즉 tile op 의 raw 산술률은 잡혔으나 memory hierarchy 활용이 안 되어 single core 의 native AMX peak (~ 1.5 TFLOPS/core BF16) 의 3% 수준만 활용.

### 1.3. PT BF16 CPU baseline (cross-thread)

| shape (B=256) | PT 1-thr (TFLOPS) | PT 4-thr (TFLOPS) | PT 32-thr (TFLOPS) | AMX 1-thr (TFLOPS) | AMX/PT-1 | AMX/PT-32 |
|---|---:|---:|---:|---:|---:|---:|
| qwen7b_gate_up | 0.602 | 2.126 | 7.313 | 0.049 | 0.082× | 0.007× |
| qwen7b_down | 0.436 | 1.959 | 12.980 | 0.050 | 0.115× | 0.004× |
| qwen32b_gate_up | 0.489 | 2.059 | 13.791 | 0.049 | 0.100× | 0.004× |
| qwen32b_down | 0.409 | 1.124 | 11.106 | 0.046 | 0.112× | 0.004× |

→ 모든 shape 에서 AMX < PT-1thread (12× slower). target ≥3× 대비 **약 40× gap**. paper §4 의 TSK_026 lever 는 본 kernel 로 운영 불가.

### 1.4. 정확도 게이트

| 게이트 | 결과 |
|---|---|
| AMX matmul vs fp32 reference (M=16, K=32, N=16, BF16 input) | max_abs_err **9.5367e-07** |
| permission grant | PASS (arch_prctl ARCH_REQ_XCOMP_PERM = 0) |
| build + load | PASS (`avx512_amx_pool._core` 로딩 OK on prod 머신) |

정확도는 AMX hardware-level dpbf16ps semantics 정확 — kernel 의 alg correctness 는 통과. 단지 micro-architecture 활용 효율이 낮음.

### 1.5. SUB_106 (22 TFLOPS peak) 재현 평가

SUB_106 의 "AMX 22.05 TFLOPS peak Qwen 7B B=256" 측정은 본 kernel 로는 **재현 안 됨** (0.049 TFLOPS 측정). 가능한 격차 원인:

| 가설 | 검증 가능성 |
|---|---|
| (a) SUB_106 의 22 TFLOPS 는 multi-core 56-thread full machine ÷ thread 산정 | 56 × 0.05 = 2.8 TF (그래도 미달) |
| (b) SUB_106 는 oneDNN/libxsmm AMX path 의 측정 (외부 라이브러리, register tiling + L2 blocking) | 가장 plausible — 본 kernel 은 그 best-practice 미적용 |
| (c) SUB_106 측정 환경 의 tile 활용 패턴 (B=256 fully tile-aligned + persistent tile config) | partial — 본 kernel 도 thread_local persistent config 적용 |

본 kernel 의 0.049 TFLOPS / core 는 AMX native peak 1.5 TFLOPS/core 의 ~3% — kernel 의 memory hierarchy (L2 reuse on K-dim, register blocking on N-dim, software prefetch) 보강 시 ≥ 10× upside 가능. 그러나 oneDNN AMX path 대비로는 본 kernel 직접 작성 의미가 약함.

---

## 2. canonical 500p baseline (control measurement)

본 SUB 의 vllm 측정은 **pure baseline** — AMX kernel 은 vllm 안에서 사용되지 않음. canonical 환경 재현 정합성만 확인.

### 2.1. 측정 protocol

- Qwen 2.5 32B Instruct TP=4×2 (port 8001 vanilla / port 8002 trident suffix-32)
- AGSD router (sub094_router.py) gated, classifier_workers=4
- 500 prompt × 32 conc × 256 max_tokens × 3 mix (balanced/sonnet-heavy/code-heavy) × 3 scenario (vanilla/trident/AGSD)
- 1-run, monitor 0.5s interval
- ENV: `RAYON/OMP/MKL/OPENBLAS=4`, `TOKENIZERS_PARALLELISM=false` (pthread EAGAIN 회피)

### 2.2. 결과

| mix | scen | SUB_175 (tps) | SUB_174 OFF (tps) | Δ vs SUB_174 |
|---|---|---:|---:|---:|
| balanced | vanilla | 2,527 | 2,504 | +0.93% |
| balanced | trident | 3,909 | 3,875 | +0.88% |
| balanced | **AGSD** | **5,425** | **5,482** | **−1.04%** |
| sonnet | vanilla | 2,730 | 2,696 | +1.27% |
| sonnet | trident | 5,937 | 5,648 | +5.12% |
| sonnet | **AGSD** | **6,212** | **6,149** | **+1.02%** |
| code | vanilla | 2,606 | 2,590 | +0.62% |
| code | trident | 6,092 | 6,072 | +0.32% |
| code | **AGSD** | **7,131** | **7,089** | **+0.59%** |

**3-mix avg AGSD**: **6,256 tps** (= (5,425 + 6,212 + 7,131) / 3) vs SUB_174 OFF **6,240 tps** = **+0.26%** (1-run noise, |Δ| < 2% ✅ 정합성 PASS)

→ canonical 환경 재현 정합성 PASS. 본 SUB 의 측정 환경은 SUB_174 OFF 와 동등. AMX kernel 의 e2e 영향은 없음 (vllm 안에서 사용되지 않음 — pure control measurement).

### 2.3. CPU / GPU util (488 sample × 0.5s, ~244 s total)

| 항목 | SUB_175 | SUB_174 OFF | Δ |
|---|---:|---:|---:|
| CPU util mean (system %) | 4.47 | 4.50 | −0.03 pp |
| CPU util p99 | 9.90 | — | — |
| mem mean (GB) | 106.8 | 106.8 | ~0 |
| GPU 0-3 (vanilla) mean util | 46.7 / 51.1 / 51.7 / 51.3 | 49.4 / 50.4 / 51.0 / 50.6 | ±2 pp |
| GPU 4-7 (trident) mean util | 25.4 / 25.2 / 25.2 / 24.7 | 25.8 / 25.0 / 25.7 / 25.4 | ~0 |

→ 모든 util/mem 지표 SUB_174 OFF 와 동등 — 환경 재현 정합성 추가 검증 PASS.

---

## 3. IDE_019 TSK_036 (CPU draft head, Qwen 0.5B) latency 점검

paper §4 의 TSK_036 입장에서 본 AMX kernel 이 CPU draft step ≤ 5 ms 충족 가능여부:

### 3.1. Qwen 0.5B MLP shape 측정 (1-thread, taskset 0)

| shape | B | K | N | best (ms) | TFLOPS |
|---|---:|---:|---:|---:|---:|
| qwen0_5b_gate_up | 8 | 896 | 4,864 | 80.76 | 0.001 (M < 16 scalar fallback) |
| qwen0_5b_gate_up | **16** | 896 | 4,864 | **0.87** | 0.160 |
| qwen0_5b_gate_up | 32 | 896 | 4,864 | 1.71 | 0.163 |
| qwen0_5b_down | 16 | 4,864 | 896 | 1.01 | 0.139 |
| qwen0_5b_down | 32 | 4,864 | 896 | 2.00 | 0.139 |

### 3.2. Draft head forward latency 추정

Qwen 0.5B = 24 layer × (gate + up + down) MLP. attention 별도.

| batch | per-layer 3-matmul (ms) | × 24 layer | budget |
|---|---:|---:|---:|
| B=16 | gate(0.87) + up(0.87) + down(1.01) = 2.75 | **66.0 ms** | ≤ 5 ms 미달 13× |
| B=32 | gate(1.71) + up(1.71) + down(2.00) = 5.42 | **130 ms** | 미달 26× |
| B=8 | scalar fallback (M < 16) | — | M=8 path 불가 |

**결론**: 현 AMX kernel 은 IDE_019 TSK_036 의 5 ms / step budget **미달**. budget 달성하려면:

1. AMX kernel rewrite (L2 blocking + register tiling + prefetch) → 10× upside
2. multi-thread parallel (4-16 thread × per-layer batch split) → 4-16× upside
3. 위 두 개 조합 시 ~ 60× upside 가능 — 본 측정 결과 5 ms 가능성 있으나 **본 SUB 의 scope 가 아님** (TSK_036 별도 turn).

또한 M=8 (작은 batch) 에서는 scalar fallback 으로 90 ms — AMX align requirement (M ≥ 16) 때문. draft head 의 small-batch 사용처에선 본 alignment constraint 가 즉시 issue.

### 3.3. TSK_036 implication

- AMX raw 사용 자체는 prod 머신에서 검증됨 (정확도 PASS, permission grant PASS, ISA dispatch PASS)
- 그러나 **본 kernel 의 micro-architecture 활용 효율 (3% of peak) 부족**
- TSK_036 actual implementation 은 oneDNN AMX or libxsmm AMX path 권장 (직접 작성보다 well-tuned 외부 path)
- IDE_019 README 의 ≤5ms / step 목표는 외부 라이브러리 path + multi-thread 의 결합 후에야 도달 가능

---

## 4. 한계 및 후속 turn

### 4.1. 본 SUB 의 한계

| 항목 | 한계 |
|---|---|
| AMX kernel 최적화 | naive tile loop, no L2 blocking / no register tiling / no prefetch / single thread |
| canonical e2e 측정 | AMX 가 vllm 내부 사용처 없음 → AMX OFF baseline 정합성 확인만 가능 |
| SUB_106 22 TFLOPS 재현 | 미재현 — SUB_106 측정 환경 (multi-thread + oneDNN-like path 추정) 과 본 kernel 환경 분리 필요 |
| OMP fork + AMX 충돌 | OMP=4 fork 시 backround thread 가 AMX permission 없어 SIGILL — kernel 호출은 single-thread context 에서만 |

### 4.2. 후속 turn 제안

1. **TSK_036 별도 turn**: oneDNN AMX or libxsmm AMX path 도입 + Qwen 0.5B 실제 forward latency 측정 + draft step ≤5ms budget 검증
2. **AMX kernel rewrite (선택)**: L2 blocking + register tiling + software prefetch → 10× upside 시 0.5 TFLOPS/core 도달 가능. 다만 oneDNN AMX path 대비 maintenance 비용 큼 — pragmatic 하지 않음
3. **multi-thread AMX dispatch**: per-process thread pool 에서 각 thread 가 permission 획득 후 AMX 분할 — backround thread 의 ARCH_REQ_XCOMP_PERM 부재로 인한 SIGILL 회피 패턴 정립 필요

### 4.3. paper §4 본 lever 의 status

| target | 결과 |
|---|---|
| TSK_026 ≥3× speedup vs PyTorch CPU matmul | **미달** (현 kernel 1-thread vs PT 1-thread = 0.082×) |
| AMX e2e drop-in 위치 (vllm 안) | **없음** — vllm 의 matmul 은 GPU 위에 위치 |
| IDE_019 TSK_036 draft head ≤5ms / step (B=16, 24 layer) | **미달** (현 kernel ~66ms / step) |

→ **TSK_026 paper §4 main contributor 자격 부적합**. AMX 라는 hardware feature 자체는 prod 머신에 가용 (정확도 + ISA dispatch + permission 모두 PASS) 하나, **vllm e2e 사용처 부재 + 현 kernel 의 µarch 비활용** 두 조건 동시 만족이라 main lever 로 이동 불가.

---

## 5. raw data

- `/tmp/sub175_amx_microbench.py` — microbench 코드
- `/tmp/sub175_amx_rows.json` — AMX 12 cell raw
- `/tmp/sub175_pt_rows_t{1,4,32}.json` — PT BF16 cross-thread raw
- `/tmp/sub175_pt{1,4,32}_run.log` — PT bench log
- `/tmp/sub175_amx_run.log` — AMX bench log
- `baseline_500p_off/{balanced,sonnet-heavy,code-heavy}/benchmark_*.json` — canonical 3 cell
- `_monitor_off_{cpu,gpu}.csv` — 0.5s monitor
- `logs/{vanilla,trident,router,monitor,main}_off.log`
- launcher: `/tmp/run_sub175_amx_canonical.sh`

---

## 6. references

- IDE_016 plan: `spec_decoding/plan/README.md` §IDE_016
- TSK_026 source: `src/amx_matmul/{amx_kernels.h, amx_qwen_draft.cpp}`
- python binding: `src/python_bindings.cpp` (py_amx_matmul / py_amx_repack_b)
- wrapper: `src/_python/avx512_sampling.py` (amx_matmul / amx_repack_b / amx_is_available)
- prior measurements: SUB_106 (22 TFLOPS peak), SUB_117 (10.24 TFLOPS available), SUB_160 (canonical 6,170 OFF), SUB_174 (canonical 6,240 OFF)
- IDE_019 TSK_036 spec: `features/IDE_019_multi_source_drafter/{README,CLAUDE,task,test}.md`
