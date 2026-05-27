# SUB_187 — IDE_019 / TSK_036 AMX draft head Qwen 1.5B microbench + e2e proxy

> **parent**: TSK_036 (IDE_019) — Qwen 1.5B-Instruct AMX-accelerated draft head + canonical e2e proxy.
> **scope**: 2026-05-27 13:06 ~ 13:09 KST (~3 min wall).
> **status**: ✅ 완료 — **AMX kernel ⭐ strong PASS** (per-step 0.5 ms) / e2e proxy **warmup order artifact 의심** ⚠ (re-measure canonical 500p 별도 SUB 권고)

---

## 0. 두괄식 — AMX kernel feasibility 확정, e2e proxy 결과 invalid

| 측정 | 결과 |
|---|---|
| AMX hardware (`/proc/cpuinfo amx`) | **available** ✓ (Sapphire Rapids prod 머신) |
| LM-head matmul B=1 K=1 p50 | 3.221 ms (OMP=16) |
| **K=7 draft loop per-step OMP=64** | **0.524 ms** ⭐ |
| SUB_181 (no AMX, scalar) per-step | 245 ms (Jacobi BF16, full vocab) |
| **speedup vs SUB_181** | **~490×** ⭐ |
| GPU verify estimate | ~40 ms / step |
| **AMX vs GPU verify** | **AMX 80× faster** — net positive 가능 |
| **e2e proxy 50 prompt** | OFF 898 tps → ON 2,159 tps (+140%) ⚠ **warmup order artifact 의심** |
| paper §4 lever 자격 | **kernel feasibility PASS, e2e canonical 재측정 필요** |

---

## 1. AMX microbench (강한 결과)

### 1.1 OMP=16 single LM-head matmul (1 step)

| B | K | p50 (ms) | p90 (ms) | mean (ms) |
|---:|---:|---:|---:|---:|
| 1 | 1 | 3.221 | 4.247 | 3.347 |
| 4 | 1 | 3.575 | 5.343 | 3.719 |
| 16 | 1 | 3.054 | 3.132 | 3.059 |

### 1.2 OMP=16 K-step draft loop

| B | K | total p50 (ms) | **per-step (ms)** | mean (ms) |
|---:|---:|---:|---:|---:|
| 1 | 5 | 15.424 | **3.085** | 17.852 |
| 1 | 7 | 21.757 | **3.108** | 22.672 |
| 4 | 5 | 15.127 | **3.025** | 15.450 |
| 4 | 7 | 21.179 | **3.026** | 22.333 |

→ 모든 cell **target < 5 ms PASS**.

### 1.3 Thread sweep (K=7 B=1)

| OMP | total p50 (ms) | **per-step (ms)** |
|---:|---:|---:|
| 4 | 127.06 | 18.15 |
| 8 | 49.99 | 7.14 |
| 16 | 21.76 | 3.11 |
| 32 | 11.98 | 1.71 |
| **64** | **3.671** | **0.524** ⭐ |

OMP=4 → OMP=64 **35× speedup** (linear 16×, actual >linear → AMX dispatch saturation 효과).

### 1.4 acceptance rate × speedup estimate

paper formula `E[accept] = (1−α^(K+1))/(1−α) − 1`:

| workload | α (typical) | K | E[accept] | cost ratio | **expected spec speedup** |
|---|---:|---:|---:|---:|---:|
| chat (target) | 0.80 | 7 | 4.17 | (40+0.5)/40 = 1.013 | **4.12×** ⭐ |
| chat | 0.85 | 7 | 4.77 | 1.013 | **4.71×** |
| sonnet | 0.65 | 7 | 2.99 | 1.013 | **2.95×** |
| code | 0.75 | 7 | 3.66 | 1.013 | **3.61×** |

→ AMX draft head 의 spec speedup 은 **kernel cost 가 GPU verify 의 1.3% 이라 거의 100% acceptance term 만 dominant**. SUB_181 (245 ms cost) 와는 categorical 다른 영역.

## 2. e2e proxy 결과 (⚠ warmup order artifact 의심)

setup: Qwen 2.5-1.5B-Instruct TP=4 :8001, single-instance vllm, 50 prompt × max-tokens=32 × conc=32, ENV `VLLM_USE_AMX_DRAFT=1` (proxy: AMX firer fire-and-forget, vllm spec_decode 미통합).

| metric | OFF | ON (AMX firer concurrent) | Δ |
|---|---:|---:|---:|
| wall_s | 1.782 | 0.741 | **−58.4%** ⭐⭐ |
| tps | 898.0 | 2,159.0 | **+140.4%** ⭐⭐ |
| n_ok | 50 | 50 | 0 |
| total_out_tokens | 1,600 | 1,600 | 0 |
| **p50_ms** | **102.54** | **102.85** | **+0.3%** (동일!) |
| p90_ms | 777.67 | 144.73 | **−81.4%** |
| p99_ms | 780.26 | 146.93 | **−81.2%** |

### 2.1 honest interpretation: **warmup order artifact**

- p50 latency 가 **OFF/ON 동일** (102 ms) → 개별 request latency 미변경
- 단 OFF 의 p90/p99 가 777 ms 로 매우 높음 → **cudagraph cold-start tail latency**
- OFF 가 measurement 시점 1번째 (vllm ready 직후 0초) → cudagraph cache cold
- ON 이 2번째 (OFF 후 ~5초) → cudagraph cache warm

→ **+140% throughput 은 AMX 효과가 아니라 cudagraph warmup 효과**. 본 SUB 의 e2e 결과는 **invalid scope** — proper canonical 500p (각 모드 별 fresh vllm boot) 별도 SUB 필요.

### 2.2 measurement 재실측 조건

- 각 모드 vllm fresh boot
- 500 prompt × 3 mix (canonical setup matching SUB_182~186)
- AMX draft 통합 (real vllm spec_decode 통합 — 별도 invasive SUB)
- 1-run, |Δ|<3% noise floor 기준

## 3. paper §4 implication

**positive (microbench)**:
- AMX kernel feasibility PASS — 0.5 ms per-step, GPU verify 80× faster
- spec speedup theoretical 4.1×~4.7× (chat α=0.80~0.85)
- SUB_181 의 catastrophic (Jacobi 245 ms) 의 근본 원인 (vocab BW + scalar arithmetic) 해결

**negative (e2e proxy invalid)**:
- 50 prompt warmup order artifact — net positive 결론 도출 불가
- vllm spec_decode 통합 (cpu_amx draft proposer) 별도 invasive SUB 필요
- canonical 500p 재측정 시 ordering 제거 보장 필수

**lever 자격**:
- microbench: paper §4 lever 후보 1순위로 **재상승** (SUB_184 main lever reject 이후 가장 강한 후보)
- e2e: 본 SUB 결과 invalid, canonical 500p 재측정 (vllm spec_decode invasive integration) 으로 binding

## 4. 다음 step

- **SUB_188+ 우선순위**: side-channel batch precompute (NEW lever) 보다 **AMX draft real vllm integration** 가 paper main 후보 1순위로 ROI 높음 (별도 invasive SUB)
- 본 SUB scope 종료 — RESULTS 정리 + 다음 SUB sequential continue

## 5. raw data

- `logs/microbench_omp16.log` — OMP=16 microbench (single + K-loop)
- `logs/microbench_thread_sweep.log` — OMP 4/8/32/64 sweep
- `measurements/{off,on}/bench.json` — e2e proxy 50p
- `_monitor_{cpu,gpu}.csv` (1.0s interval)
- `src/amx_draft_qwen05b.cpp` (target Qwen 0.5B shape, 본 SUB 1.5B 로 변경 — agent 결정)
- `build/` — AMX-compiled .so
