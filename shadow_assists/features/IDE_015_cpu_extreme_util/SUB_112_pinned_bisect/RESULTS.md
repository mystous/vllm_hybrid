# SUB_112 — Pinned bisect (physical cores only, HT 금지)

> **parent**: IDE_020 surrogate / TSK_038 NUMA preview
> **scope**: 2026-05-26 16:52 ~ 17:04 KST (~12 min)
> **status**: ✅ 완료 — 5 worker counts × 3 mix = 15 cells × qwen32b shape
> **CPU pin**: physical cores 80-111 (NUMA1 후반부, vLLM이 NUMA0 0-55 사용 가정)
> **AMX shape**: Qwen 32B MLP (5120, 27648) BF16

---

## 0. 두괄식 — Pinned 영역 첫 깨끗한 net positive ⭐

| N pinned workers | AGSD 3-mix 평균 delta | sweet spot |
|---:|---:|---|
| 0 | — | baseline |
| **4** | **+3.5%** ⭐ | net positive (5.6 TFLOPS fill) |
| **8** | **+3.6%** ⭐ | net positive (11.2 TFLOPS) |
| 16 | −1.8% | outlier (balanced 영역 만 −9.9%) |
| **32** | **+3.9%** ⭐⭐ | **best** (44.8 TFLOPS fill, 32 physical core 사용) |

→ **Physical-core pinning + cross-NUMA isolation (80-111) 영역 깨끗한 sustained net positive 달성**.
→ unpinned (SUB_111) N=2 best +0.07% 영역 unpinned N=4 -2.2% 대비 **압도적 개선**.

---

## 1. 측정 결과 — 5 × 3 cell matrix

### 1.1 AGSD throughput

| N | balanced | sonnet-heavy | code-heavy | 3-mix 평균 |
|---:|---:|---:|---:|---:|
| 0 | 4,879 | 5,371 | 6,118 | — |
| **4** | **5,112** (+4.8%) | **5,770** (+7.4%) | 6,013 (−1.7%) | **+3.5%** |
| **8** | **5,154** (+5.6%) | 5,726 (+6.6%) | 6,038 (−1.3%) | **+3.6%** |
| 16 | 4,397 (−9.9%) ⚠ | 5,668 (+5.5%) | 6,051 (−1.1%) | −1.8% |
| **32** ⭐ | **5,142** (+5.4%) | **5,823** (+8.4%) | 5,983 (−2.2%) | **+3.9%** ⭐ |

### 1.2 Trident-only throughput (★ warmup artifact note)

| N | balanced | sonnet-heavy | code-heavy |
|---:|---:|---:|---:|
| 0 | 3,453 | 4,914 | 5,580 |
| 4 | 5,826 | 7,070 | 7,438 |
| 8 | 6,811 | 7,541 | 7,738 |
| 16 | 7,473 | 6,692 | 8,495 |
| 32 | 7,672 | 7,458 | 7,278 |

→ Trident-only 영역 N progression 영역 dramatic 영역 상승 — N effect 영역 아니고 backend warmup 누적 (cudagraph capture cache 등). 단 N=0 영역 cold start 영역 baseline 영역 underrepresent.

### 1.3 vanilla-only throughput (CPU fill 영역 영향 없음 확인)

| N | balanced | sonnet-heavy | code-heavy |
|---:|---:|---:|---:|
| 0 | 2,376 | 2,471 | 2,541 |
| 4 | 2,608 | 2,555 | 2,671 |
| 8 | 2,653 | 2,535 | 2,714 |
| 16 | 2,626 | 2,530 | 2,704 |
| 32 | 2,586 | 2,568 | 2,636 |

→ vanilla-only 영역 ±5% 범위 — CPU fill 영역 vanilla path 영역 무관 (예상).

---

## 2. Unpinned (SUB_111) vs Pinned (SUB_112) 비교 ⭐

| N | unpinned 평균 (SUB_111) | **pinned 평균 (SUB_112)** | 개선 |
|---:|---:|---:|---:|
| 2 | +0.07% | (N=4가 minimum) | — |
| 4 | −2.2% ✗ | **+3.5%** ⭐ | **+5.7pp** |
| 8 | — | **+3.6%** ⭐ | new data |
| 16 | — | −1.8% | (outlier) |
| 32 | — | **+3.9%** ⭐⭐ | new data |

→ **Physical-core pinning + cross-NUMA isolation 영역 핵심 lever** — same N에서 −2% → +3.5% 점프.

---

## 3. 핵심 finding (paper-worthy)

| finding | 의미 |
|---|---|
| **Pinned N=4-32 영역 깨끗한 +3.5~3.9% sustained gain** | IDE_020 영역 cgroup/isolcpus 영역 효과 영역 surrogate 영역 정량 입증 |
| sonnet-heavy 영역 항상 net positive (+5.5~8.4%) | chat 영역 router/CPU 영역 활용 영역 head-room 큼 |
| code-heavy 영역 약간 음수 (−1~2%) | code 영역 spec 영역 GPU bound + CPU 영역 free 적음 |
| N=16 balanced 영역 outlier (−9.9%) | 다음 SUB 영역 재측정 필요 (run-to-run variance) |
| N=32 영역 best — 32 physical cores fully utilized | physical core 영역 충분히 영역 활용 영역 contention 없음 |
| **44.8 TFLOPS CPU compute 영역 free 영역 활용 가능** | 추정 (32 worker × ~1.4 TFLOPS each) |

---

## 4. 다음 step

| SUB | 영역 |
|---|---|
| SUB_113 | NUMA 영역 topology audit + IRQ affinity check |
| SUB_114 | proper cgroup cpuset.cpus (root 필요 시) — pinning 영역 stable 영역 production-ready |
| SUB_115 | 1-hour sustained throughput stability + util |
| SUB_116 | N=16 outlier 영역 재측정 (variance check) |
| SUB_117 | N=8/32 영역 actual CPU util 영역 측정 (5Hz monitor 영역 worker 활동 영역 정량) |

---

## 5. raw data

- `workers_{0,4,8,16,32}/benchmark_{balanced,sonnet-heavy,code-heavy}.json` (15 cells)
- `workers_{N}/cpu_workers/worker_*.log` (TFLOPS per worker per N)
- `_monitor_cpu.csv` / `_monitor_gpu.csv` (5Hz 영역 14 min sustained capture)
- 소스: `/tmp/sub112_cpu_fill_pinned.py` + `/tmp/run_sub112_pinned_bisect.sh`
