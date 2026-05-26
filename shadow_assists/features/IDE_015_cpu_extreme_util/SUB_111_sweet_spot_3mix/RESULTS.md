# SUB_111 — Sweet spot 3-mix sweep (qwen32b shape, unpinned)

> **parent**: IDE_015 / TSK_021 후속 — SUB_110 sweet spot 의 mix 의존성 검증
> **scope**: 2026-05-26 15:56 ~ 16:05 KST (~9 min wall)
> **status**: ✅ 완료 — 3 × 3 = 9 cell (N=0/2/4 × balanced/sonnet-heavy/code-heavy)
> **AMX shape**: Qwen 32B MLP (K=5120, N=27648) BF16, batch=128
> **CPU pin**: ❌ 미적용 (sub108_cpu_amx_fill_v2.py — OS scheduler 자율)

---

## 0. 두괄식 — Unpinned 한계 명확화 (SUB_112 pinning 동기)

| N | balanced Δ | sonnet-heavy Δ | code-heavy Δ | **3-mix avg Δ** |
|---:|---:|---:|---:|---:|
| 0 | baseline | baseline | baseline | — |
| **2** | +1.3% | **+3.6%** | −4.7% ⚠ | **+0.07%** ≈ tie |
| 4 | −2.5% | +2.3% | −6.4% ⚠ | **−2.20%** |

→ **Unpinned ceiling 3-mix avg +0.07% (N=2 best)** — 사실상 net-neutral.
→ **code-heavy 일관 음수** (N=2 −4.7%, N=4 −6.4%) — GPU-bound spec 경로에 CPU contention 이 직접 타격.
→ sonnet-heavy 만 net positive (+3.6%) — chat traffic 에 CPU head-room 가장 큼.
→ → → **SUB_112 pinning 결정 결과**: 동일 N=4 가 unpinned −2.2% → pinned +3.5% (Δ +5.7pp swing).

---

## 1. 측정 결과 — 3 × 3 = 9 cell matrix

### 1.1 AGSD-gated tps (★ 핵심)

| N | balanced | sonnet-heavy | code-heavy |
|---:|---:|---:|---:|
| 0 | 4,992.8 | 5,458.9 | 6,201.7 |
| **2** | **5,057.2 (+1.3%)** | **5,655.7 (+3.6%)** | 5,908.5 (−4.7%) |
| 4 | 4,866.7 (−2.5%) | 5,587.1 (+2.3%) | 5,802.7 (−6.4%) |

### 1.2 wall time (s)

| N | balanced | sonnet-heavy | code-heavy |
|---:|---:|---:|---:|
| 0 | 9.94 | 9.30 | 7.93 |
| 2 | 9.81 | 8.96 | 8.44 |
| 4 | 10.18 | 9.06 | 8.47 |

### 1.3 vanilla-only / trident-only (참고)

vanilla-only tps:
| N | balanced | sonnet-heavy | code-heavy |
|---:|---:|---:|---:|
| 0 | 2,410.5 | 2,534.2 | 2,611.6 |
| 2 | 2,399.8 | 2,450.4 | 2,484.7 |
| 4 | 2,460.6 | 2,359.3 | 2,472.8 |

trident-only tps (★ warmup artifact — N progression 누적):
| N | balanced | sonnet-heavy | code-heavy |
|---:|---:|---:|---:|
| 0 | 3,545.5 | 5,423.1 | 5,486.7 |
| 2 | 5,814.2 | 6,599.9 | 7,716.6 |
| 4 | 7,248.3 | 7,356.0 | 7,816.5 |

---

## 2. Util (~9 분 sustained capture)

| 영역 | avg | max |
|---|---:|---:|
| CPU (1Hz, n=949) | **5.0%** | 48.0% |
| GPU 0 (vanilla) | 44.3% | 100% |
| GPU 1 (vanilla) | 45.5% | 100% |
| GPU 2 (vanilla) | 44.6% | 100% |
| GPU 3 (vanilla) | 46.0% | 100% |
| GPU 4 (trident) | 18.7% | 88% |
| GPU 5 (trident) | 18.8% | 82% |
| GPU 6 (trident) | 18.9% | 84% |
| GPU 7 (trident) | 18.9% | 100% |
| **GPU avg (8)** | **32.0%** | — |

→ GPU avg 32.0% — SUB_098 canonical 27.7% 대비 ↑ (3-mix sweep 9 cell 동안 GPU 가 더 일관되게 활용).
→ vanilla backend GPU 44-46% > trident 18-19% — D4 paradox 더 강하게 재현.
→ CPU avg 5.0% — unpinned worker 가 idle gap 거의 못 채움 (baseline 4.1% 와 ±1%).

---

## 3. 핵심 finding (paper-worthy)

| finding | 의미 |
|---|---|
| **Unpinned 3-mix avg ceiling +0.07% (N=2)** | unpinned + qwen32b shape 는 net-neutral 한계 |
| **code-heavy 일관 음수** (N=2 −4.7%, N=4 −6.4%) | spec-bound workload 가 CPU contention 에 가장 취약 |
| sonnet-heavy 만 net positive (+3.6% N=2, +2.3% N=4) | chat traffic 에 CPU head-room 가장 큼 — sustainable target |
| balanced N=2 +1.3% → N=4 −2.5% | sharp degradation window — sweet spot 매우 좁음 |
| CPU avg 5.0% (idle gap 95% 그대로) | unpinned 으로는 CPU util elevate 거의 불가 |
| GPU 0-3 vs GPU 4-7 → 26pp 차이 | D4 paradox sharp (spec-side GPU 더 idle) |

→ **본 SUB 결과로 SUB_112 pinning 결정**. 동일 N=4 에서 unpinned −2.2% (본 SUB) → pinned +3.5% (SUB_112) 의 +5.7pp swing 이 pinning 의 핵심 lever 임을 정량 입증.

---

## 4. SUB_109/110/112 와 비교

| SUB | shape | pin | mix | N range | best Δ |
|---|---|---|---|---|---:|
| SUB_109 | qwen7b | ❌ | balanced 만 | 0/1/2/4/8 | N=2 +3.5% |
| SUB_110 | qwen32b | ❌ | balanced 만 | 0/1/2/4/8/16 | N=2 +2.8% |
| **SUB_111** (본 SUB) | qwen32b | ❌ | **3 mix** | 0/2/4 | **3-mix avg +0.07%** (N=2) |
| **SUB_112** | qwen32b | ✅ (80-111) | 3 mix | 0/4/8/16/32 | **3-mix avg +3.9%** ⭐⭐ (N=32) |

→ SUB_111 (unpinned 3-mix) → SUB_112 (pinned 3-mix) 의 **3-mix avg +0.07% → +3.9%** 가 본 fork 의 핵심 lever 진보.

---

## 5. raw data

- `workers_{0,2,4}/benchmark_{balanced,sonnet-heavy,code-heavy}.json` — 9 cells
- `workers_{N}/cpu_workers/worker_*.log` — per-worker TFLOPS (N=2,4)
- `workers_{N}/cpu_fill.log` — fill summary
- `workers_{N}/bench_{mix}.log` — benchmark stdout
- `_monitor_cpu.csv` (949 samples × 1Hz)
- `_monitor_gpu.csv` (949 × 8 GPU × 7 fields)
- `logs/{vanilla,trident,router,monitor}.log`
- 소스: `/tmp/sub108_cpu_amx_fill_v2.py`

---

## 6. 다음 step

- **SUB_112 ✅ 완료**: pinning (CPU 80-111) 도입 — 3-mix avg +3.9% 달성
- SUB_113: NUMA topology audit — 0-55 vs 80-111 가 실제 NUMA0/1 어디 인지 확인
- SUB_117: per-worker actual CPU util — pinned vs unpinned 의 worker active% 정량
