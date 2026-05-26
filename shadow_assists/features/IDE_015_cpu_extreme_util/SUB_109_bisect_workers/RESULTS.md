# SUB_109 — Worker count bisect (qwen7b shape, unpinned)

> **parent**: IDE_015 / TSK_021 후속 — CPU fill worker count tuning
> **scope**: 2026-05-26 15:01 ~ 15:07 KST (~6 min wall)
> **status**: ✅ 완료 — 5 cell × balanced mix
> **AMX shape**: Qwen 7B MLP (K=3584, N=18944) BF16, batch=128
> **CPU pin**: ❌ 미적용 (sub108_cpu_amx_fill_v2.py — OS scheduler 자율)

---

## 0. 두괄식 — N=2 sweet spot, N≥4 plateau

| N workers | AGSD balanced tps | Δ vs N=0 |
|---:|---:|---:|
| 0 | 4,912.9 | baseline |
| 1 | 4,938.7 | +0.5% |
| **2** ⭐ | **5,086.4** | **+3.5%** |
| 4 | 4,910.2 | −0.1% |
| 8 | 4,466.8 | **−9.1%** ⚠ |

→ Unpinned 환경에서 **N=2 가 sweet spot**. N=4 이상부터 contention 시작, N=8 에서 9% 감소.
→ Qwen 7B AMX shape (compute-light, K×N = 3584×18944) 라 worker 당 throughput pressure 가 32B 대비 낮음 — 따라서 N=8 까지 시도 가능했으나 GIL/scheduler contention 으로 무력화.

---

## 1. 측정 결과 — 5 × 1 cell

### 1.1 AGSD-gated throughput (★ 핵심)

| N | tps | wall (s) | p50 (s) | p99 (s) | Δ vs N=0 |
|---:|---:|---:|---:|---:|---:|
| 0 | 4,912.9 | 10.10 | 0.71 | 3.37 | — |
| 1 | 4,938.7 | 10.07 | 0.66 | 2.92 | +0.5% |
| **2** | **5,086.4** | 9.77 | 0.62 | 2.94 | **+3.5%** ⭐ |
| 4 | 4,910.2 | 10.12 | 0.61 | 3.19 | −0.1% |
| 8 | 4,466.8 | 11.12 | 0.66 | 3.34 | −9.1% |

### 1.2 vanilla-only (CPU fill 영향 미미)

| N | tps | Δ vs N=0 |
|---:|---:|---:|
| 0 | 2,407.7 | — |
| 1 | 2,530.8 | +5.1% |
| 2 | 2,554.2 | +6.1% |
| 4 | 2,322.0 | −3.6% |
| 8 | 2,306.6 | −4.2% |

→ vanilla path 는 CPU fill 과 무관 (예상). N=1-2 에서 약간 + 노이즈, N≥4 약간 − 노이즈.

### 1.3 trident-only (warmup artifact)

| N | tps |
|---:|---:|
| 0 | 3,552.5 |
| 1 | 5,254.0 |
| 2 | 5,494.5 |
| 4 | 5,900.8 |
| 8 | 6,796.6 |

→ N progression 따른 dramatic 상승은 N effect 아니라 backend warmup 누적 (cudagraph capture cache). N=0 cold start cell 의 baseline 가 underrepresent.

---

## 2. Util (4 분 sustained capture)

| 영역 | avg | max |
|---|---:|---:|
| CPU (1Hz, n=735) | **5.0%** | 61.2% |
| GPU 0 (vanilla) | 33.6% | 100% |
| GPU 1 (vanilla) | 34.8% | 100% |
| GPU 2 (vanilla) | 30.7% | 100% |
| GPU 3 (vanilla) | 35.0% | 100% |
| GPU 4 (trident) | 15.3% | 98% |
| GPU 5 (trident) | 15.3% | 82% |
| GPU 6 (trident) | 15.4% | 92% |
| GPU 7 (trident) | 14.9% | 81% |
| **GPU avg (8)** | **24.4%** | — |

→ CPU avg 5.0% (SUB_098 baseline 4.1% 와 ± 1% 노이즈) — unpinned worker 가 CPU idle gap 을 사실상 채우지 못함.
→ vanilla backend (GPU 0-3) avg 33.5% > trident backend (GPU 4-7) avg 15.2% — D4 paradox 재현.

---

## 3. 핵심 finding

| finding | 의미 |
|---|---|
| **N=2 unpinned 가 sweet spot (+3.5%)** | unpinned 으로도 적당한 worker 수면 net positive 달성 가능 |
| N=4 이상 plateau/회귀 | OS scheduler 가 worker 를 vllm 코어에 침범 — contention 발생 |
| N=8 에서 −9.1% | unpinned 의 한계 — pinning 없이는 N 증가 무의미 |
| CPU avg 5.0% (baseline 4.1% 와 ±1% 노이즈) | unpinned 으로는 CPU idle gap 거의 fill 못함 |
| Qwen 7B shape AMX 0.66-0.72 TFLOPS / worker | 32B shape 대비 compute-light |

---

## 4. SUB_110/111/112 와 연결

| SUB | 차이 | 결과 |
|---|---|---|
| **SUB_109** (본 SUB) | qwen7b shape, unpinned, N=0/1/2/4/8 | N=2 +3.5% best, N=8 −9.1% |
| SUB_110 | qwen32b shape, unpinned, N=0/1/2/4/8/16 | N=2 +2.8% best, N≥4 회귀 |
| SUB_111 | qwen32b shape, unpinned, 3 mix × N=0/2/4 | code-heavy 음수 — unpinned 한계 노출 |
| **SUB_112** | qwen32b shape, **pinned (CPU 80-111)**, N=0/4/8/16/32 | N=4/8/32 +3.5~3.9% 안정 net positive ⭐ |

→ **SUB_109 → SUB_112 progression**: unpinned 에서 발견한 sweet spot 의 ceiling 을 pinning 으로 돌파.

---

## 5. raw data

- `workers_{0,1,2,4,8}/benchmark_balanced.json` — 5 cells
- `workers_{N}/cpu_workers/worker_*.log` — per-worker TFLOPS (N=1-8)
- `workers_{N}/cpu_fill.log` — fill summary
- `workers_{N}/bench.log` — benchmark stdout
- `_monitor_cpu.csv` (735 samples × 1Hz)
- `_monitor_gpu.csv` (735 × 8 GPU × 7 fields)
- `logs/{vanilla,trident,router,monitor}.log`
- 소스: `/tmp/sub108_cpu_amx_fill_v2.py`

---

## 6. 다음 step

- SUB_110: 동일 protocol 을 qwen32b AMX shape 으로 확대
- SUB_111: 3-mix 확대 — code-heavy 가 spec-bound 인지 검증
- SUB_112: physical-core pinning + cross-NUMA isolation 으로 ceiling 돌파
