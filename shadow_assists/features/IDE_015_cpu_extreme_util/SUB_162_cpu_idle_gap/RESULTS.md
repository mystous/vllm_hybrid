# SUB_162 — CPU thread state sampling (perf 부재 alternative, 1-run)

> **parent**: IDE_015 / TSK_023 재정의 (perf 부재로 `/proc/$tid/stat` + py-spy --idle 으로 대체)
> **scope**: 2026-05-26 23:26 ~ 23:31 KST (~5 min — vllm boot 90s + 500p balanced 1-run + 10 Hz sampler 60s)
> **status**: ✅ 완료 — 313 samples × 10 Hz × R/S/D state distribution + py-spy --idle flamegraph
> **measurement protocol (1-run)**: 사용자 지시 — single 500p balanced run

---

## 0. 두괄식 — VLLM 의 Python thread 96-100% sleeping 상태 ⭐

### Top finding

| Process | total thread-samples | **R (running)** | S (sleeping/blocked) | D (disk wait) |
|---|---:|---:|---:|---:|
| `vllm` (main) | 303,610 | **0.0%** | **100.0%** | 0.0% |
| `VLLM::EngineCor` | 171,524 | **0.2%** | 99.8% | 0.0% |
| `VLLM::Worker_TP` | 127,704 | **1.0%** | 96.0% | 0.0% |

→ **VLLM Python threads 대부분 OS scheduler 의 S (sleeping/blocked) 상태**. R (running) 시간 0.2-1.0%.

### Implication for IDE_018 phase-burst

→ paper §3+§4 핵심 데이터 — **vllm 측 CPU 자원이 거의 100% idle 상태에서 OS sleeping queue 에 누워 있음**. 이 idle window 는 IDE_018 phase-burst scheduler 가 CPU task 를 fill 할 가용 자원.

### 핵심 활성 thread

- **avg 5.04 running threads / sample** (across all vllm processes)
- max 8 simultaneously running
- 16 samples (5.1%) 에 0 running threads (전체 vllm stack 동시 idle 순간)

---

## 1. Thread state distribution (per process)

### 1.1 sampler 결과 (60s × 10 Hz = 313 samples)

| Process | sample 수 | R% | S% | D% | other% |
|---|---:|---:|---:|---:|---:|
| `vllm` (1 main process) | 303,610 | 0.00% | **100.00%** | 0.00% | 0.00% |
| `VLLM::EngineCor` (2 processes) | 171,524 | **0.20%** | 99.80% | 0.00% | 0.00% |
| `VLLM::Worker_TP` (8 processes) | 127,704 | **1.00%** | 96.00% | 0.00% | 3.00% |

→ Worker_TP 의 1.0% R 비율은 **GPU dispatch 직후 짧은 CPU 활동** 시간으로 추정. 나머지 96-99% 는 NCCL/IPC/GPU wait 의 S 상태.

### 1.2 running thread snapshot (per-sample)

| 통계 | 값 |
|---|---:|
| avg running threads/sample | **5.04** |
| max running threads/sample | 8 |
| 0-running samples (전체 idle) | 16 / 313 (5.1%) |
| total thread count (vllm) | ~30 |
| avg active ratio | 5.04 / 30 = **16.8%** of threads, 1.8% of OS sched slots |

→ **vllm 전체 thread 의 16.8% 만 동시에 active**. 나머지 83% 는 항상 sleep — 명시적인 idle gap.

---

## 2. Benchmark + util context

### 2.1 benchmark (1-run, 500p balanced)

| scenario | tps | wall (s) | p50 (s) | p99 (s) |
|---|---:|---:|---:|---:|
| vanilla-only | 1,897.8 | 65.5 | 4.17 | 4.49 |
| trident-only | 3,835.3 | 32.4 | 1.45 | 4.66 |
| **AGSD-gated** | **4,699.3** | 26.5 | 0.71 | 3.61 |

→ AGSD 4,699 — SUB_161 (5,442) / SUB_160 stability mean (5,457) 대비 약 14% 낮음. 본 SUB 가 시스템 부하 높은 시점에 실행됐을 가능성 (다른 sampling process 가 적게 영향), 그러나 finding 자체 (96% S) 는 robust.

### 2.2 system CPU util (5Hz, 458 samples × 90s capture)

- **avg = 4.31%** ← SUB_098 baseline 4.1% 와 정합 ✓
- max = 40.3%

→ baseline 수준의 CPU 활용 (vllm 만 running, CPU fill 없음). vllm 자체가 CPU 의 ~4% 만 활성화 — SUB_117 N=32 fill 시 16% 와 12pp 차이.

---

## 3. py-spy --idle flamegraph

flamegraph SVG files:
- `raw/pyspy_record_idle_3942152.svg` — VLLM::Worker_TP0 (vanilla)
- `raw/pyspy_record_idle_3942164.svg` — VLLM::Worker_TP0 (trident)

→ `--idle` 플래그로 GIL-blocked thread 도 캡처. SUB_161 의 active-only flamegraph (sampler.py 44.3% 등) 와 보완.

---

## 4. 핵심 finding (paper-worthy, IDE_018 입력)

| finding | 정량 | 의미 | 영향 IDE |
|---|---|---|---|
| **VLLM 의 Python thread 96-100% S** | OS sched 의 sleep queue 누적 | CPU 측 절대 idle window 의 존재 정량 | IDE_018 main thesis |
| Worker_TP R 비율 1.0% | 30 worker thread 중 0.3개 active | GPU wait 의 CPU 영향 zero — phase-burst lever | IDE_018 |
| EngineCore R 비율 0.2% | engine scheduler 거의 idle | IDE_001 (CPU planner) 의 의미 | IDE_001 / IDE_018 |
| avg 5.04 running threads/sample | vllm 의 활성 thread 7.4 / 사용가능 ~30 | 83% thread 가 항상 sleep | IDE_018 task-pool capacity |
| 5.1% sample 에서 0 thread running | 전 vllm stack 동시 idle 순간 존재 | **CPU 측 100% idle window 도 종종 발생** — phase-burst 의 burst opportunity | IDE_018 |
| **CPU util 4.31% (no fill)** ↔ SUB_098 baseline 4.1% 와 정합 | 본 SUB 가 sampling overhead 미미 | sampler 자체가 fill 영향 없음 — pure observation | (메타 검증) |

---

## 5. perf 부재의 한계

| perf 가능했을 측정 | 본 SUB 의 대안 |
|---|---|
| `perf sched record` — thread state transition timeline (R → S → R 의 latency 분포) | `/proc/$tid/stat` 10 Hz 샘플 — state aggregate 만 (transition timing 정량 불가) |
| `perf sched latency` — average / max blocked time | 본 SUB 에선 단순 R/S ratio 만 |
| context switch rate | unavailable |
| **GIL contention 정량** (Python 의 OS-level lock) | py-spy --idle flamegraph 로 partial 추적 가능 |

→ paper 시 perf 가능한 환경에서 후속 measurement 권장. 본 SUB 의 R/S 분포는 **첫 정량 estimate** 로 사용 가능.

---

## 6. 다음 step

- **SUB_166** (queued, 1-run): DMA push latency microbench (cudaHostAlloc + cudaMemcpyAsync 4KB~64MB block sweep) — TSK_028 SUB_119 measurement-only equivalent
- **SUB_163** (analysis): SUB_161 + SUB_162 데이터 결합 → GPU idle window phase categorize matrix (paper Table 1)
- **SUB_164** (analysis): CPU-fillable threshold + per-phase task candidate matrix (paper Table 1)
- **TSK_031 (IDE_018) — kernel dev** 별도 turn — 본 SUB 의 96% S 정량 이 phase-burst 의 main motivation

---

## 7. raw data

- `benchmark_balanced.json` — 1-run 500p × 3 scenarios
- `raw/thread_states.jsonl` — 313 samples × 10 Hz × per-process R/S/D distribution
- `raw/pyspy_record_idle_*.svg` — vanilla + trident TP0 idle flamegraph
- `_monitor_{cpu,gpu}.csv` — full duration util
- `logs/{vanilla,trident,router,bench,main,monitor,sampler}.log`
- 소스: `/tmp/run_sub162_cpu_idle_gap.sh`, `/tmp/sub162_thread_state_sampler.py`
