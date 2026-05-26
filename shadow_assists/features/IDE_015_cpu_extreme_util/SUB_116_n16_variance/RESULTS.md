# SUB_116 — N=16 outlier variance check (3-run)

> **parent**: IDE_015 — SUB_112 N=16 balanced outlier (−9.9%) 의 variance 검증
> **scope**: 2026-05-26 21:29 ~ 21:36 KST (~7 min wall — 모델 boot 90s + 6 cell)
> **status**: ✅ 완료 — 3-run × (N=0, N=16) × balanced = 6 cell
> **AMX shape**: Qwen 32B MLP (K=5120, N=27648) BF16, batch=128
> **CPU pin**: ✅ 적용 (sub112_cpu_fill_pinned, CPU 80-95 for N=16)

---

## 0. 두괄식 — N=16 은 outlier 가 아니고 **일관된 회귀** ⚠

SUB_112 의 N=16 balanced −9.9% 가 noise 라는 가설은 **기각**.

| Run | N=0 AGSD | N=16 AGSD | Δ |
|---:|---:|---:|---:|
| 1 | 4,882.7 | 4,411.4 | **−9.65%** |
| 2 | 5,152.6 | 4,258.4 | **−17.35%** |
| 3 | 5,153.3 | 4,339.9 | **−15.78%** |
| **mean** | **5,062.9** | **4,336.6** | **−14.35%** ⚠ |

→ **3-run avg Δ = −14.35%** (run-by-run CV: N=0 3.08%, N=16 **1.77%** — variance 극히 낮음).
→ SUB_112 의 single-run 결과 (4,397 ≈ run1 4,411) 가 운 좋게 best end 였음 — 실제 N=16 회귀는 더 큼.

→ **paper-worthy finding**: N 워커 수와 throughput 의 관계는 **비단조 (non-monotonic)** —
- N=4/8/32 → +3.5~3.9% ⭐ (SUB_112)
- **N=16 → −14.35%** ⚠ (본 SUB)

---

## 1. 측정 결과 — 6 cell

### 1.1 AGSD throughput per run

| N | run | tps | wall (s) |
|---:|---:|---:|---:|
| 0 | 1 | 4,882.7 | 10.19 |
| 0 | 2 | 5,152.6 | 9.66 |
| 0 | 3 | 5,153.3 | 9.62 |
| 16 | 1 | 4,411.4 | 11.27 |
| 16 | 2 | 4,258.4 | 11.71 |
| 16 | 3 | 4,339.9 | 11.40 |

### 1.2 Aggregate (3-run mean / std / CV)

| N | scenario | mean | std | **CV%** | min | max |
|---:|---|---:|---:|---:|---:|---:|
| 0 | agsd-gated | 5,062.9 | 156.0 | **3.08%** | 4,882.7 | 5,153.3 |
| 0 | trident-only | 5,648.2 | 1927.6 | 34.13% ⚠ | 3,486.0 | 7,187.0 |
| 0 | vanilla-only | 2,596.6 | 176.8 | 6.81% | 2,392.8 | 2,708.3 |
| 16 | agsd-gated | **4,336.6** | 76.6 | **1.77%** ⭐ | 4,258.4 | 4,411.4 |
| 16 | trident-only | 5,444.3 | 1782.8 | 32.75% ⚠ | 3,889.5 | 7,390.3 |
| 16 | vanilla-only | 2,055.3 | 41.7 | 2.03% | 2,011.1 | 2,093.9 |

→ AGSD CV: N=0 3.08% / N=16 1.77% → **회귀 신호가 noise floor 위로 명확히 우세**.
→ trident-only CV 33% → backend warmup 누적 (SUB_109/110/111/112 과 동일 패턴, run progression 정렬).
→ **N=16 vanilla-only 2055 vs N=0 vanilla-only 2596 → 추가 −20.9%** — 16-worker fill 이 trident NUMA 1 + vanilla 양쪽에 침범.

---

## 2. Util (~7 min sustained capture, 0.2s interval)

| 영역 | avg | max |
|---|---:|---:|
| CPU (5Hz, n=835) | **6.9%** | 75.9% |
| GPU 0 (vanilla) | 42.5% | 100% |
| GPU 1 (vanilla) | 43.4% | 100% |
| GPU 2 (vanilla) | 36.1% | 100% |
| GPU 3 (vanilla) | 43.6% | 100% |
| GPU 4 (trident) | 18.3% | 86% |
| GPU 5 (trident) | 18.4% | 93% |
| GPU 6 (trident) | 17.3% | 86% |
| GPU 7 (trident) | 17.8% | 83% |
| **GPU avg (8)** | **29.7%** | — |

→ CPU avg 6.9% (SUB_098 baseline 4.1%, SUB_112 ~5%) — N=16 도입으로 +2.8pp elevate. 단 throughput 은 회귀.
→ vanilla backend (NUMA 0) GPU 0-3 avg 41.4% > trident backend (NUMA 1) 17.9% — D4 paradox + N=16 fill 의 NUMA 1 contention 으로 trident GPU 더 underutilized.

---

## 3. 핵심 finding (paper-worthy)

| finding | 의미 |
|---|---|
| **N=16 → −14.35% 일관 회귀** (CV 1.77%) | "outlier" 가설 기각. 실제 회귀 |
| **non-monotonic N curve**: N=4/8 (+3.5~3.6%) → N=16 (−14.35%) → N=32 (+3.9%) | "더 많은 worker = 더 좋다" 가설 기각. **N 선택은 비단조 optimization** |
| N=16 vanilla-only −20.9% (3-run avg) | 16-worker fill 이 vllm vanilla path 까지 침범 — NUMA 1 메모리 압력 또는 sched contention |
| N=16 의 CV 1.77% (N=0 CV 3.08% 보다 낮음) | 회귀가 매우 안정적 — 일관된 mechanism |
| 가설 (SUB_113 NUMA audit 기반): N=16 (CPU 80-95) 가 trident backend (GPU 4-7 ↔ NUMA 1) 의 IRQ / kthread / vllm worker 와 정확히 겹침 | 다음 SUB 검증 대상 (`perf trace`) |
| paper 입력 | N tuning curve 의 **valley** 가 존재 — 단순 "physical-core 만 쓰면 됨" 가설 부족, 더 정교한 placement 필요 |

---

## 4. 가설 — N=16 valley 의 mechanism

SUB_113 NUMA audit 결과 + 본 SUB 회귀 패턴 결합:

| 가설 | 근거 | 검증 방법 |
|---|---|---|
| (A) CPU 80-95 가 trident backend 의 IRQ affinity 와 겹침 | nvidia driver IRQ 가 GPU 4-7 ↔ NUMA 1 의 default | `cat /proc/interrupts | grep nvidia` |
| (B) vllm trident 의 worker thread (TP=4) 가 CPU 80-95 사용 | Python multiprocessing 의 default thread placement | `ps -eLo pid,tid,psr` for VLLM::Worker_TP |
| (C) N=32 (80-111) 는 trident worker 가 다른 CPU 로 밀려나서 회복 | "다 점유하면 trident 가 NUMA 0 로 fallback" | 가설 (B) 의 N=32 일 때 cpu0-55 thread 분포 확인 |

→ 본 finding 은 IDE_020 cgroup 설계의 핵심 입력: **단순 `cpuset.cpus=80-111` 만으로는 부족**. trident worker 를 명시적으로 `cpuset.cpus=56-79` 로 격리해야 N=16 valley 회피.

---

## 5. SUB_112 vs SUB_116 N=16 단일 vs 3-run 비교

| 측정 | tps | Δ vs N=0 |
|---|---:|---:|
| SUB_112 (single run, N=0 baseline 4879) | 4,397 | −9.9% |
| SUB_116 run1 (N=0 baseline 4882) | 4,411 | −9.65% |
| SUB_116 run2 (N=0 baseline 5152) | 4,258 | **−17.35%** |
| SUB_116 run3 (N=0 baseline 5153) | 4,340 | **−15.78%** |
| **SUB_116 3-run mean** | **4,337** | **−14.35%** |

→ SUB_112 single-run 은 run1 -equivalent (best end). SUB_116 의 3-run 은 valley 의 진짜 깊이 (−14.35%) 를 노출.

---

## 6. 다음 step

- **SUB_117**: per-worker actual CPU util 정량 (N=8/32 pinned) — 본 SUB 의 회귀 mechanism 검증 입력
- **(미할당)** N=16 mechanism deep-dive: `/proc/interrupts` + `ps -eLo psr` for vllm trident worker → 가설 (A)/(B) 검증
- **SUB_148 (IDE_020/TSK_038)**: NUMA-aware cgroup split (trident → CPU 56-79, fill → CPU 80-111) → N=16 valley 해소 검증

---

## 7. raw data

- `N{0,16}_run{1,2,3}/benchmark_balanced.json` — 6 cells
- `N16_run{1,2,3}/cpu_workers/worker_*.log` — per-worker TFLOPS
- `N16_run{1,2,3}/cpu_fill.log` — fill summary
- `N*_run*/bench.log` — benchmark stdout
- `_monitor_cpu.csv` (835 samples × 5Hz)
- `_monitor_gpu.csv` (835 × 8 GPU × 7 fields)
- `logs/{vanilla,trident,router,main,monitor}.log`
- 소스: `/tmp/run_sub116_n16_variance.sh`, `/tmp/sub112_cpu_fill_pinned.py`
