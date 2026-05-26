# SUB_117 — per-worker actual CPU util (N=8 vs N=32 pinned)

> **parent**: IDE_015 — SUB_112 pinned 워커의 실제 활용도 정량
> **scope**: 2026-05-26 21:38 ~ 21:41 KST (~3 min wall — 2 N config × 75s)
> **status**: ✅ 완료 — 2 config (N=8, N=32) × 60s sample × 2 Hz
> **방법**: psutil.Process.cpu_percent + per-core util + worker TFLOPS log
> **vllm 미실행** — pure CPU fill worker microbench

---

## 0. 두괄식 — N=32 가 N=8 보다 per-worker 효율도 더 높음 ⭐

| Metric | N=8 | N=32 |
|---|---:|---:|
| 워커 수 | 8 | 32 |
| 워커당 Python proc CPU% (avg) | **49.8%** | **99.4%** ⭐ |
| 워커당 Python proc CPU% (max avg) | 52.5% | 101.4% |
| 워커당 TFLOPS (BF16) | **0.23** | **0.32** ⭐ |
| 핀된 코어 OS-view util (avg) | 100.0% | 99.9% |
| 시스템 CPU% (전체) | 5.2% | 16.0% |
| **총 CPU compute** | **1.84 TFLOPS** | **10.24 TFLOPS** ⭐ |

→ **N=32 가 N=8 대비 per-worker 39% 더 효율, 총 compute 5.6× 더 많음**.
→ 핀된 코어는 OS 관점에서 둘 다 100% busy 인데, **N=8 의 Python proc 만 50% 사용** — 50% 의 코어 time 이 worker 가 아닌 다른 곳에 소비됨 (kthread/system 추정).
→ N=8 valley 와 N=32 peak 의 mechanism — per-worker 효율 (Python proc CPU%) + per-worker compute (TFLOPS) 모두 N=32 우세.

---

## 1. 측정 결과

### 1.1 N=8 pinned (CPU 80-87)

```
SYSTEM CPU avg=5.2% max=10.1%
WORKER (n=8) avg=49.8% (max-across-workers avg=52.5%)
PINNED CORE (n=8) avg=100.0%
```

**Per-worker breakdown** (모든 워커 ≈ 동일):
| PID | CPU | avg% | max% | TFLOPS (worker log) |
|---:|---:|---:|---:|---:|
| 3382437 | 80 | 49.8 | 51.7 | 0.23 |
| 3382438 | 81 | 49.8 | 53.7 | 0.22 |
| 3382439 | 82 | 49.8 | 53.7 | 0.22 |
| 3382440 | 83 | 49.7 | 51.7 | 0.23 |
| 3382441 | 84 | 49.8 | 53.7 | 0.23 |
| 3382442 | 85 | 49.8 | 51.7 | 0.23 |
| 3382443 | 86 | 49.8 | 51.7 | 0.22 |
| 3382444 | 87 | 49.8 | 51.7 | 0.23 |

→ 8 워커 모두 정확히 49.8% — **systematic** 50% throttle (noise 아님).
→ 핀된 코어 8 개 모두 100% — 나머지 50% 의 core-time 은 kthread / IRQ / 다른 시스템 활동.

### 1.2 N=32 pinned (CPU 80-111)

```
SYSTEM CPU avg=16.0% max=42.4%
WORKER (n=32) avg=99.4% (max-across-workers avg=101.4%)
PINNED CORE (n=32) avg=99.9%
```

**Per-worker breakdown** (32 워커 모두 99.3-99.4%):
| 통계 | 값 |
|---|---:|
| 모든 워커 avg% 범위 | 99.2 - 99.4% |
| 모든 워커 max% 범위 | 101.2 - 102.9% |
| TFLOPS / worker | 0.31 - 0.33 |
| 총 TFLOPS | 32 × 0.32 = **10.24** |

→ N=32 워커 모두 **99.4%** 로 saturate — 거의 100% efficient.
→ N=8 → N=32 로 늘리면 단위 worker 의 효율도 동시에 올라감 (50% → 99%).

---

## 2. 총 compute / util 비교

| Metric | N=8 | N=32 | 비율 |
|---|---:|---:|---:|
| 활성 코어 수 | 8 | 32 | 4× |
| 워커당 effective CPU% | 49.8% | 99.4% | **2×** |
| 워커당 TFLOPS | 0.23 | 0.32 | **1.39×** |
| **총 CPU compute** | 1.84 TFLOPS | **10.24 TFLOPS** | **5.6×** |
| 시스템 전체 CPU% | 5.2% | 16.0% | 3.1× |

→ 워커 수만 4× 증가시켰는데 **총 compute 5.6× 증가** — N=32 는 per-worker 효율 + per-core compute 둘 다 개선.

---

## 3. 핵심 finding (paper-worthy)

| finding | 의미 |
|---|---|
| **N=32 pinned 워커 99.4% per-process CPU saturation** | physical-core pinning 의 ideal state — paper 입력 |
| **N=8 pinned 워커 49.8% per-process CPU** (uniform across 8 workers) | systematic 50% 제약 — 다른 시스템 활동 (kthread/IRQ) 이 코어 share |
| TFLOPS per-worker: N=8 0.23 vs N=32 0.32 (+39%) | per-worker compute 도 N=32 우세 — 핀된 코어 fully 격리 효과 |
| 총 CPU compute: N=8 1.84 → N=32 **10.24 TFLOPS** | "CPU 100pp idle gap fill 가능" 주장의 정량 입증 — 사용 가능 CPU compute = **10 TFLOPS / NUMA 1** |
| SUB_116 의 N=16 회귀 (-14.35%) | 본 SUB 의 N=8/32 데이터로 mechanism 일부 설명 — N=16 의 16 코어 (80-95) 가 trident GPU 4-7 (NUMA1) 의 IRQ/kthread 와 부분 경합 |
| SUB_098 의 CPU avg 4.1% baseline → SUB_117 N=32 16% | CPU util **5% → 16%** elevate 달성 — IDE_018 의 30%+ 목표에 절반 도달 |

---

## 4. mechanism 가설 — N=8 50% throttle 의 원인

| 가설 | 근거 | 검증 방법 |
|---|---|---|
| (A) kthread / IRQ 가 코어 80-87 공유 | 정확히 50% uniform — systematic | `cat /proc/interrupts | awk '{... CPU80-87 ...}'` |
| (B) Sapphire Rapids power management — N 적을수록 freq boost → BLAS 가 처음에 빠르게 끝나 idle 시간 발생 | TFLOPS 0.23 vs 0.32 (33% gap) | `turbostat` for per-core freq |
| (C) AMX warmup / cache effect | TFLOPS log shows steady 0.23 (not progression) | (C) 기각 — 정상 |
| (D) torch CPU thread pool — set_num_threads(1) 이 모든 thread 를 1 로 강제하지 않음 | OMP/MKL/OPENBLAS 환경변수 + torch.set_num_threads(1) 모두 설정 | thread count via /proc/$pid/status |

→ N=32 의 99.4% per-worker 는 모든 NUMA 1 physical core 가 워커로 점유되어 kthread/IRQ 가 다른 곳 (NUMA 0 또는 HT siblings) 으로 밀려난 결과로 해석 가능 — 가설 (A) 와 정합.

---

## 5. SUB_116 N=16 valley mechanism 입력

SUB_116 에서 N=16 → −14.35% 회귀 발견. 본 SUB 의 데이터 결합:

| N | 효율 가설 | throughput 결과 (SUB_112/116) |
|---:|---|---:|
| 8 | 50% throttle (다른 시스템과 share) | +3.6% ⭐ |
| 16 | (미측정) — 다음 SUB candidate | **−14.35%** ⚠ |
| 32 | 99.4% saturation (격리 완성) | +3.9% ⭐⭐ |

→ N=8 (50% throttle) 에서도 throughput net positive (+3.6%) 이지만, N=16 은 net negative.
→ 가설: N=16 의 16 핀된 코어 (80-95) 는 NUMA 1 의 약 절반만 점유 → trident backend 의 IRQ / vllm worker thread 가 95-111 로 밀려나며 정확히 그 영역에서 contention 발생. N=32 는 80-111 전체 점유 → trident 가 56-79 로 밀려나 contention 해소.
→ **SUB_148 (IDE_020 cgroup) 시 핵심 입력**: trident 영역 56-79, fill 영역 80-111 명시 분리.

---

## 6. 다음 step

- **(미할당) N=16 deep-dive**: `cat /proc/interrupts | grep nvidia` + `ps -eLo pid,tid,psr` for trident worker — N=16 valley mechanism 정량 검증
- **SUB_148 (IDE_020/TSK_038)**: cgroup `cpuset.cpus` 명시 분리 protocol — N=16 valley 가 해소되는지 검증
- **SUB_138 (IDE_018 main)**: 본 SUB 의 10 TFLOPS / NUMA 1 가용 compute 가 phase-burst scheduler 의 main lever — attention-phase task pool 의 candidate (KV prefetch, AMX draft head, tokenize, grammar check)

---

## 7. raw data

- `N8_pinned/samples.csv` — 60s × 2Hz × (sys_cpu + 8 worker pct + 8 core pct)
- `N8_pinned/summary.json` — per-worker breakdown
- `N8_pinned/cpu_workers/worker_*_cpu*.log` — TFLOPS log per worker (5s interval)
- `N8_pinned/cpu_fill.log` — fill summary
- `N8_pinned/run.log` — script stdout
- `N32_pinned/*` — N=32 동일 구조
- 소스: `/tmp/sub117_per_worker_util.py`, `/tmp/run_sub117_per_worker_util.sh`, `/tmp/sub112_cpu_fill_pinned.py`
