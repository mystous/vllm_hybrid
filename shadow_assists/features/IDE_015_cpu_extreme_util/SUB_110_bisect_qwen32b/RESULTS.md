# SUB_110 — Worker count bisect (qwen32b shape, unpinned)

> **parent**: IDE_015 / TSK_021 후속 — CPU fill worker count tuning (32B shape)
> **scope**: 2026-05-26 15:48 ~ 15:55 KST (~7 min wall)
> **status**: ✅ 완료 — 6 cell × balanced mix
> **AMX shape**: Qwen 32B MLP (K=5120, N=27648) BF16, batch=128
> **CPU pin**: ❌ 미적용 (sub108_cpu_amx_fill_v2.py — OS scheduler 자율)

---

## 0. 두괄식 — N=2 sweet spot, N≥4 회귀 (SUB_109 와 정합)

| N workers | AGSD balanced tps | Δ vs N=0 |
|---:|---:|---:|
| 0 | 4,891.7 | baseline |
| 1 | 4,850.7 | −0.8% |
| **2** ⭐ | **5,030.9** | **+2.8%** |
| 4 | 4,696.9 | −4.0% |
| 8 | 4,840.4 | −1.0% |
| 16 | 4,665.3 | −4.6% |

→ Unpinned qwen32b shape: **N=2 가 sweet spot (+2.8%)**, N≥4 일관 회귀.
→ qwen32b shape (compute-heavy, K×N = 5120×27648) 는 worker 당 GIL/memory bandwidth 압력 큼 — SUB_109 의 7B shape +3.5% 대비 N=2 절대 gain 도 작음.
→ N=16 까지 시도해도 회귀 패턴 — **unpinned 으로는 qwen32b shape ceiling +2.8%**.

---

## 1. 측정 결과 — 6 × 1 cell

### 1.1 AGSD-gated throughput (★ 핵심)

| N | tps | wall (s) | p50 (s) | p99 (s) | Δ vs N=0 |
|---:|---:|---:|---:|---:|---:|
| 0 | 4,891.7 | 10.19 | 0.75 | 3.12 | — |
| 1 | 4,850.7 | 10.21 | 0.62 | 3.50 | −0.8% |
| **2** | **5,030.9** | 9.88 | 0.64 | 2.97 | **+2.8%** ⭐ |
| 4 | 4,696.9 | 10.65 | 0.67 | 3.19 | −4.0% |
| 8 | 4,840.4 | 10.36 | 0.66 | 3.08 | −1.0% |
| 16 | 4,665.3 | 10.76 | 0.67 | 3.35 | −4.6% |

### 1.2 vanilla-only

| N | tps | Δ vs N=0 |
|---:|---:|---:|
| 0 | 2,480.9 | — |
| 1 | 2,683.1 | +8.2% |
| 2 | 2,429.7 | −2.1% |
| 4 | 2,293.1 | −7.6% |
| 8 | 2,345.4 | −5.5% |
| 16 | 2,231.3 | −10.1% |

→ vanilla path 는 qwen32b shape contention 에 더 민감 — N≥4 부터 명확한 −5% 이상 회귀.

### 1.3 trident-only

| N | tps |
|---:|---:|
| 0 | 3,419.7 |
| 1 | 4,953.6 |
| 2 | 5,907.6 |
| 4 | 5,980.7 |
| 8 | 6,439.8 |
| 16 | 6,506.9 |

→ trident-only N progression 대부분 backend warmup 누적 (SUB_109 와 동일 패턴).

---

## 2. Util (~7 분 sustained capture)

| 영역 | avg | max |
|---|---:|---:|
| CPU (1Hz, n=727) | **4.8%** | 83.6% |
| GPU 0 (vanilla) | 32.9% | 100% |
| GPU 1 (vanilla) | 33.7% | 100% |
| GPU 2 (vanilla) | 35.8% | 100% |
| GPU 3 (vanilla) | 34.5% | 100% |
| GPU 4 (trident) | 15.9% | 92% |
| GPU 5 (trident) | 16.1% | 84% |
| GPU 6 (trident) | 15.6% | 98% |
| GPU 7 (trident) | 16.1% | 92% |
| **GPU avg (8)** | **25.1%** | — |

→ CPU max 83.6% — N=16 cell 에서 unpinned worker 가 OS scheduler 통해 system-wide 침투, 단 avg 는 baseline 수준 (4.8%) 머무름.
→ GPU avg 25.1% — SUB_098 canonical 27.7% 와 비교해 N=16 에서 vanilla GPU 약간 ↓ (서버 contention 영향).

---

## 3. 핵심 finding

| finding | 의미 |
|---|---|
| **qwen32b unpinned ceiling +2.8% (N=2)** | qwen7b 의 +3.5% 대비 0.7pp 낮음 — shape 클수록 contention 커짐 |
| N=1 −0.8%, N=2 +2.8%, N=4 −4.0% | sharp window — unpinned 의 sweet spot 매우 좁음 |
| N=8 −1.0%, N=16 −4.6% | N 늘려도 회복 안 됨 — unpinned 자체 한계 |
| vanilla-only N≥4 일관 −5% 이상 | qwen32b shape worker 가 vllm 코어 침범 |
| CPU avg 4.8% (baseline 4.1%) | unpinned 으로 CPU idle gap 거의 못 채움 (변화 0.7pp) |

---

## 4. SUB_111/112 와 연결

| SUB | 차이 | 결과 |
|---|---|---|
| **SUB_110** (본 SUB) | qwen32b, unpinned, balanced 만 × N=0/1/2/4/8/16 | N=2 +2.8%, N≥4 회귀 |
| SUB_111 | qwen32b, unpinned, **3 mix** × N=0/2/4 | code-heavy 음수 노출 |
| SUB_112 | qwen32b, **pinned (CPU 80-111)** × N=0/4/8/16/32 | N=4/8/32 +3.5~3.9% ⭐ |

→ SUB_110 의 발견 "qwen32b 는 unpinned 으로 N=2 가 한계" 가 SUB_112 의 pinning protocol 동기 부여.

---

## 5. raw data

- `workers_{0,1,2,4,8,16}/benchmark_balanced.json` — 6 cells
- `workers_{N}/cpu_workers/worker_*.log` — per-worker TFLOPS (N=1-16)
- `workers_{N}/cpu_fill.log` — fill summary
- `workers_{N}/bench.log` — benchmark stdout
- `_monitor_cpu.csv` (727 samples × 1Hz)
- `_monitor_gpu.csv` (727 × 8 GPU × 7 fields)
- `logs/{vanilla,trident,router,monitor}.log`
- 소스: `/tmp/sub108_cpu_amx_fill_v2.py`

---

## 6. 다음 step

- SUB_111: 3-mix 확대 — N=2 sweet spot 이 mix 따라 어떻게 변하는지 검증
- SUB_112: pinning + cross-NUMA isolation 으로 ceiling 돌파
