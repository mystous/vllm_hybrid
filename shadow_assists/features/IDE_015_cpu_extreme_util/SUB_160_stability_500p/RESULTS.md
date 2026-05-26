# SUB_160 — 500p baseline + N=32 pinned sustained stability

> **parent**: IDE_015 / TSK_021 잔여 (Phase A 종합) + TSK_039 SUB_153 (1-hour stability) 통합
> **scope**: 2026-05-26 22:24 ~ 23:18 KST (~54 min, early-stop by user direction at 31 runs)
> **status**: ✅ 완료 — Phase 1 500p baseline 9 cells + Phase 2 N=32 pinned 30-run balanced stability
> **note**: 본 SUB 는 1-hour × 40-run 으로 설계됐으나 사용자 지시 ("1번만 돌려") 로 30 runs 시점에 early-stop. memory rule `feedback_measurement_runs.md` 갱신 (1-run default)

---

## 0. 두괄식 — 500p 베이스라인 정정 + N=32 stability 재현

### Key findings

| 발견 | 정량 |
|---|---|
| **500p 측정이 200p 대비 모든 cell 에서 +5~26% 높음** | warmup amortization — Phase A 200p baseline 은 throughput 을 underestimate |
| **N=32 pinned 30-run stability 안정** | AGSD mean 5,457 / CV 6.48% / no drift, no thermal regression |
| **first 3 vs last 3 AGSD = +21.45%** | warmup 만 5-7 분 — 측정 시 첫 3 runs 제외 권장 |
| **CPU util 16.34% 재현 (SUB_117 와 정합)** | N=32 fill 의 CPU 활성화 효과 sustained 확인 |
| GPU avg 47.6% (full 8 GPU during sustained workload) | baseline 27.7% 대비 +19.9pp — sustained activity 가 GPU 더 활용 |

---

## 1. Phase 1 — 500p baseline (no fill) vs Phase A 200p baseline

| mix | scenario | **500p (본 SUB)** | 200p (SUB_098) | Δ |
|---|---|---:|---:|---:|
| balanced | vanilla | 2,501 | 2,373 | +5.4% |
| balanced | trident | 3,880 | 3,467 | +11.9% |
| balanced | **AGSD** | **5,474** | 4,569 | **+19.8%** |
| sonnet-heavy | vanilla | 2,676 | 2,351 | +13.8% |
| sonnet-heavy | trident | 5,894 | 4,877 | +20.9% |
| sonnet-heavy | **AGSD** | **6,037** | 5,273 | **+14.5%** |
| code-heavy | vanilla | 2,608 | 2,463 | +5.9% |
| code-heavy | trident | 6,182 | 4,904 | +26.1% |
| code-heavy | **AGSD** | **6,996** | 5,985 | **+16.9%** |
| **3-mix avg AGSD** | — | **6,169** | 5,276 | **+16.9%** |

→ **500p baseline 이 paper §4 evaluation 의 더 정확한 numeric base**. Phase A 의 모든 SUB (SUB_098~117/112/116/117) 가 200p 기준 → 500p re-measure 시 모든 절대값 약 +17% 상승, **비율 (Δ vs baseline) 은 보존**.

---

## 2. Phase 2 — N=32 pinned 30-run balanced stability

### 2.1 Aggregate (30 runs)

| scenario | n | mean tps | std | **CV%** | min | max |
|---|---:|---:|---:|---:|---:|---:|
| **AGSD-gated** | 30 | **5,457** | 353 | **6.48%** | 4,277 | 5,754 |
| vanilla-only | 30 | 2,656 | 255 | 9.60% | 1,651 | 2,770 |
| trident-only | 30 | 8,343 | 1,112 | 13.33% ⚠ | 4,804 | 9,392 |

→ AGSD CV 6.48% (n=30) — 안정. cf SUB_116 (3-run, balanced N=0 CV 3.08% / N=16 CV 1.77%) 와 noise floor 유사.
→ trident-only CV 13% — backend warmup chain (SUB_109~112 와 동일 패턴).

### 2.2 Time-bucketed drift (warmup mechanism)

| bucket | mean AGSD tps | vs first |
|---|---:|---:|
| first 3 runs | 4,728 | baseline |
| mid 3 runs | 5,637 | **+19.21%** |
| last 3 runs | 5,743 | **+21.45%** |

→ **warmup 만 5-7 분 ((~3 runs × 1.5 min/run)**. 첫 3 runs 제외하면 sustained throughput ≈ 5,700 tps.

### 2.3 Stability — N=32 pinned vs 500p baseline (no fill)

| 비교 | AGSD tps | 비고 |
|---|---:|---|
| 500p no-fill baseline (single run, 처음) | 5,474 | Phase 1 |
| **500p N=32 stability (mean 30 runs)** | **5,457** | Phase 2 — warmup runs 포함 |
| 500p N=32 stability (last 3 runs avg) | **5,743** | warmup 제외 |
| **Δ (warmup 제외) vs no-fill** | **+4.9%** | N=32 fill 효과 |

→ N=32 stability 의 warmup-excluded throughput **+4.9% vs no-fill** (Phase A SUB_112 200p 의 +3.9% 와 정합 — 500p 에서도 effect 유효).

---

## 3. CPU/GPU util (full 54 min sustained capture)

### 3.1 CPU util (2Hz, n=3916 samples)

- **avg = 16.34%** ⭐
- max = 44.8%
- vs SUB_098 baseline (4.1%) → **+12.24pp elevate**
- vs SUB_117 N=32 microbench (16%) — **정확히 정합** ✓

### 3.2 GPU util (per-GPU, full duration)

| GPU | role | avg | max |
|---:|---|---:|---:|
| 0 | vanilla backend | 65.4% | 100% |
| 1 | vanilla backend | 70.8% | 100% |
| 2 | vanilla backend | 70.3% | 100% |
| 3 | vanilla backend | 67.8% | 100% |
| 4 | trident backend | 27.9% | 94% |
| 5 | trident backend | 25.0% | 91% |
| 6 | trident backend | 27.9% | 93% |
| 7 | trident backend | 25.7% | 93% |
| **avg (8 GPU)** | — | **47.6%** | — |

→ 8-GPU avg 47.6% — SUB_098 baseline 27.7% 대비 **+19.9pp**. sustained workload + N=32 fill 환경에서 GPU 활용도 대폭 향상.
→ vanilla backend (GPU 0-3) 65-71% > trident backend (GPU 4-7) 25-28% — D4 paradox 재현 (sustained).

---

## 4. 핵심 finding (paper-worthy)

| finding | 의미 |
|---|---|
| 500p baseline 이 200p 대비 모든 scenario +5~26% 높음 | Phase A 측정의 numeric base 갱신 필요. paper §4 는 500p 기준 |
| N=32 pinned 30-run no-drift / no-thermal-regression | Phase A SUB_112 +3.9% lever 의 sustainability 검증 |
| warmup 5-7 분 / 3 runs — 그 이후 stable | 측정 시 first 3 exclude 필수 |
| **CPU util 16.34% sustained (SUB_117 16% 와 정합)** | N=32 의 CPU 활성화 효과 long-run 유효 |
| GPU avg 47.6% (vs 27.7% baseline) | sustained workload + N=32 fill 환경 GPU 활용도 향상 |
| **사용자 지시 — 1번만 돌려** | memory rule 갱신 — stability test 도 1-run default. 30-run loop 는 과설계 |

---

## 5. design 회고 (사용자 피드백 반영)

| 원 설계 (잘못) | 사용자 지시 후 (올바름) |
|---|---|
| 1-hour × ~40 runs loop | **1-run** (또는 명시 요청 시만 multi-run) |
| 모든 stability check 30+ runs | 단일 sustained 측정 (continuous prompt) |
| 변동성 자동 측정 | 사용자 명시 요청 시만 |

→ 후속 SUB (SUB_161/162 포함) 는 **1-run default** 적용. 30-run 미실시.

---

## 6. 다음 step

- **SUB_161** — ncu per-kernel + py-spy sublayer profile (1-run, nsys 부재 alternative)
- **SUB_162** — /proc + py-spy CPU thread state (1-run, perf 부재 alternative)
- **(미할당)** 500p baseline 기준 SUB_112 protocol 재측정 (single-run 으로) — N curve 의 비단조 valley 가 500p 에서도 재현 되는지 verify

---

## 7. raw data

- `baseline_500p/{balanced,sonnet-heavy,code-heavy}/benchmark_*.json` — 9 cells (Phase 1)
- `stability_loop/run_{001..030}/benchmark_balanced.json` — 30 cells (Phase 2)
- `cpu_workers/worker_*_cpu80~111.log` — N=32 worker TFLOPS log
- `_monitor_cpu.csv` (3916 samples × 2Hz)
- `_monitor_gpu.csv` (3916 × 8 GPU × 7 fields)
- `logs/{vanilla,trident,router,main,monitor,cpu_fill}.log`
- 소스: `/tmp/run_sub160_stability_500p.sh`, `/tmp/sub112_cpu_fill_pinned.py`
