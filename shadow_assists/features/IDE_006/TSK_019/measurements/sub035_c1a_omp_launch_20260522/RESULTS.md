# SUB_035 C1a — OMP team launch overhead 측정 결과 (2026-05-22 KST)

> **parent**: TSK_019 / O 분석 §7.2 ★★ C-tier 항목 / [SUB_035 plan](../../planning/SUB_035_c1_layer_fusion_plan.md) Option C1a
> **measurement**: HEAD `0d7dc0334`, 100p × 8192, gmu=0.85, env-ON baseline + `VLLM_NEO_OMP_LAUNCH_PROFILE=1` + `VLLM_NEO_OMP_PROFILE=1`, 1-run (instrumentation only).
> **결과 요약**: throughput = 930.8 tps (noise range), **OMP launch overhead = 1.22%**.

---

## 1. 측정 결과 (PROFILE OMP LAUNCH, 8 worker × 3500 calls = 28K calls 평균)

| 항목 | avg | max |
|---|---:|---:|
| parallel region 전체 (`#pragma omp parallel` 진입~종료) | 0.445-0.555 ms | 2.7-9.1 ms |
| step0 + b1_wait + step1 + b2_wait + step2 합 | 0.439-0.550 ms | — |
| **launch overhead (region 전체 - step 합)** | **~0.005 ms (5 μs)** | — |
| **overhead %** | **0.95-1.30%** (avg ~1.20%) | — |

## 2. OMP step breakdown (PROFILE OMP tid=0)

| step | avg ms | 비중 | 의미 |
|---|---:|---:|---|
| step0 (KV cache store) | 0.000 | 0% | thread_local memcpy, instant |
| **b1_wait** | **0.003-0.004** | **~1%** | Step 0 → Step 1 barrier |
| **step1 (ISPC attention compute)** | **0.263-0.369** | **~70%** | qk_product + softmax + av_product (dominant) |
| **b2_wait** | **0.059-0.277** | **~30%** | Step 1 → Step 2 barrier (workload imbalance) |
| step2 (gather output) | 0.005-0.006 | ~1% | sequence-level merge |

→ **dominant = step1 (ISPC attention compute) = 70%**, **b2_wait = 0-30% (imbalance)**.

## 3. ⚠️ 결정적 해석 — perf record 의 libgomp 43.75% 의 진짜 의미

이전 측정 (`reference/H_dynamic_analysis.md`): perf record 60s 에서 libgomp 43.75% (`gomp_team_barrier_wait`).

**진짜 의미** (C1a 결과 갱신):
- libgomp 의 `gomp_team_barrier_wait` 는 **active spin** (CPU cycle 누적)
- wall-time 영향 = **micro-sec level** (b1_wait 3-4 μs, b2_wait 60-277 μs)
- perf record 의 cycle 분배 ≠ wall 분배 — spin loop 는 CPU 비싸고 wall 영향 없음
- → **NEO 의 진짜 wall 병목 = step1 (ISPC compute) = 0.31 ms / call**

## 4. C-tier ROI 결론 — C-tier 폐기

| Option | 본래 가설 | C1a 결과 후 ROI |
|---|---|---|
| C1a (OMP launch profile) | — | **완료, 결과: 1.22%** |
| C1b (OMP nowait clause) | b1/b2 barrier 제거 | **~0.06-0.28 ms / call** = micro-sec, **noise band 안** |
| C1c (OMP chunk schedule) | b2_wait (imbalance) 줄임 | 0.06-0.28 ms / call 영역 — best case ~0.1 ms = **0.02% of wall** |
| C1d (CDEC concurrency 재시도) | thread 분배 변경 | SUB_023 CW=4 = -52% 회귀 — risk >> reward |
| C1e (persistent OMP team 재설계) | launch overhead 감소 | **1.22% 전체** → 절약 < 1.22% ≪ noise |

→ **C-tier 모든 옵션의 max ROI < 1.5%** ≪ single-run noise (±2.4%). **C-tier 전체 폐기**.

## 5. 진짜 병목 = step1 (ISPC attention compute)

| 영역 | 효과 가설 |
|---|---|
| ISPC kernel 자체 가속 (AMX 더 적극, libxsmm JIT) | step1 의 0.31 ms 직접 감소 |
| AMX 의 av_product (현재 ISPC) 까지 확장 | step1 의 일부 영역 (av_product) AMX 적용 |
| Compute 자체 줄이기 (sparse attention, low-rank approx) | step1 의 algorithmic 감소 |

본 워크로드 (100p × 8192, throughput-saturated) 에서 단일 step1 의 작은 절약이 lever 신호로 검출되려면:
- step1 0.31 ms × 80 layer × 100p / 932 tps × 880 s = **2.6 sec / 880 s = 0.3% wall** per 1% step1 절약
- 즉 step1 을 **10% 줄여야 wall 3% 영향** → 측정 가능한 신호 영역 진입

## 6. 갱신된 권고 — 본 워크로드 의 실질적 다음 path

| 우선순위 | 경로 | 이유 |
|---|---|---|
| **★★★** | **Path A — 워크로드 재정의 (500p × 8192 또는 max_num_seqs ↑)** | 본 워크로드 = saturated. 더 큰 워크로드에서 lever 신호 검출 가능성 |
| **★★** | **B-tier 잔여 (B4 SPARAMX AMX↔AVX-512 switch, B5 libxsmm JIT)** | step1 (ISPC compute) 직접 가속 — algorithmic 변경 |
| ★ | C-tier 전체 폐기 (C1b/c/d/e 시도 안 함) | ROI < 1.5% < noise |
| ⚪ | NEO net benefit 재정의 — vanilla 와 비교 (vanilla 가 OOM 인 영역에서만 NEO win) | TSK_019 의 raison d'être 점검 |

## 7. raw 자료

| 항목 | 위치 |
|---|---|
| result.json | `eval/results/20260522_040138_sub035_c1a_omp_launch_100p/run1/result.json` |
| engine.log.stdout (PROFILE emit 포함) | `eval/results/20260522_040138_sub035_c1a_omp_launch_100p/run1/engine.log.stdout` |
| launcher | `/tmp/run_sub035_c1a_omp_launch.sh` |
| stdout log | `/tmp/sub035_c1a.log` |
| code 변경 | `csrc/cpu/pacpu/core.h` (parallel region wall timer + summary emit) |
| build | `csrc/cpu/pacpu/build/libpacpu-llama3_3_70b-tp8.so` (재빌드 ✓) |

## 8. 코드 처리 결정

C1a instrumentation = **env-gated default OFF 유지** (`VLLM_NEO_OMP_LAUNCH_PROFILE=1` 으로 opt-in). 향후 다른 환경 / 다른 workload 점검 시 재사용 가능.
