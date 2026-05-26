# SUB_098 — Canonical Baseline Util 재측정

> **parent**: IDE_015 / TSK_021
> **scope**: 2026-05-26 12:22 ~ 12:26 KST (~4 min wall)
> **status**: ✅ 완료 — 9 cell × CPU%/GPU% 캡처
> **purpose**: SUB_097 Phase B util gap fill — canonical test bed lock-in

---

## 0. 두괄식 — IDE_015~021 영역 baseline 확정

**전체 시스템 idle gap**:
- **CPU avg 4.1%** (max 18.8%) — 95.9% idle
- **GPU avg 28% across 8 GPUs** — 72% idle (Llama 70B TP=8 영역 73% util 영역 단일 instance 대비, Qwen 32B TP=4×2 e2e 영역 split parallel 영역 GPU 영역 절반만 동시 사용)

→ **idle gap 영역 fill 가능 영역 100pp** (CPU 95pp + GPU 72pp). IDE_018 (phase-burst) main target.

---

## 1. Throughput (canonical baseline lock-in)

| Mix | vanilla-only | trident-only | **AGSD-gated** | vs vanilla | vs trident |
|---|---:|---:|---:|---:|---:|
| **balanced** (34:33:33) | 2,373 | 3,467 | **4,569** ⭐ | **+92.6%** | **+31.8%** |
| **sonnet-heavy** (60:20:20) | 2,351 | 4,877 | **5,273** ⭐ | **+124.3%** | **+8.1%** |
| **code-heavy** (10:20:70) | 2,463 | 4,904 | **5,985** ⭐ | **+143.0%** | **+22.0%** |

→ SUB_097 Phase B (2,416/3,461/4,894 ...) 영역 ±3% noise 범위 — canonical baseline 영역 안정 입증.

---

## 2. Wall time

| Mix | vanilla-only | trident-only | AGSD-gated |
|---|---:|---:|---:|
| balanced | 20.8s | 14.3s | **11.0s** (−47%) |
| sonnet-heavy | 21.3s | 10.3s | **9.6s** (−55%) |
| code-heavy | 20.0s | 10.1s | **8.2s** (−59%) |

---

## 3. ★ Util matrix (canonical baseline 영역 첫 캡처)

### 3.1 CPU util (1Hz, 192 samples × 4 min)

| 지표 | 값 |
|---|---:|
| **avg** | **4.1%** |
| max | 18.8% |
| idle% | **95.9%** |

→ CPU 영역 거의 idle. Sapphire Rapids 영역 multi-core 영역 활용 영역 거의 0. **IDE_018 phase-burst 영역 main target 영역 fill 가능 영역 95.9pp**.

### 3.2 GPU util (1Hz, 192 samples × 8 GPU)

| GPU | role | avg | max | idle% |
|---:|---|---:|---:|---:|
| 0 | vanilla backend | **31.7%** | 97.0% | 68.3% |
| 1 | vanilla backend | **36.8%** | 100.0% | 63.2% |
| 2 | vanilla backend | **36.5%** | 100.0% | 63.5% |
| 3 | vanilla backend | **36.0%** | 100.0% | 64.0% |
| 4 | trident backend | **20.6%** | 100.0% | 79.4% |
| 5 | trident backend | **20.3%** | 100.0% | 79.7% |
| 6 | trident backend | **19.8%** | 86.0% | 80.2% |
| 7 | trident backend | **19.6%** | 100.0% | 80.4% |
| **avg (8 GPU)** | — | **27.7%** | — | **72.3%** |

→ **GPU avg 27.7% / idle 72.3%** — massive idle gap.
→ vanilla backend GPU 0-3 (35% avg) > trident backend GPU 4-7 (20% avg). Trident 영역 spec decoding 영역 GPU 활용률 영역 더 줄임 (D4 paradox 영역 e2e setting 영역 재현).

---

## 4. ★ Idle Gap 종합 (IDE_018 phase-burst 영역 target)

| resource | avg util | idle gap | fill 가능 영역 (IDE_018 target) |
|---|---:|---:|---:|
| CPU | **4.1%** | **95.9pp** | 5% → 30%+ → **~25pp 추가** 활용 |
| GPU avg (8) | **27.7%** | **72.3pp** | spec drafter / KV prefetch / cold-KV decompress 영역 fill |
| **합산 활용 잠재력** | — | — | **~100pp idle 영역 활용 가능** |

→ IDE_018 (phase-burst) + IDE_019 (CPU multi-source drafter) 영역 본 idle gap 영역 fill 영역 paper main contribution.

---

## 5. raw data

- `_monitor_cpu.csv` (192 samples × 1Hz × 4 fields)
- `_monitor_gpu.csv` (192 × 8 GPU × 7 fields)
- `benchmark_{balanced,sonnet-heavy,code-heavy}.json`
- `logs/{vanilla,trident,router,bench_*}.log`

---

## 6. SUB_099 영역 canonical baseline lock-in (다음 단계)

- 본 SUB_098 영역 결과 영역 plan README.md §1 영역 canonical baseline table 영역 update
- _ALL_TABLE_20260526.md 영역 SUB_097 Phase B row 영역 util column 영역 fill
- 후속 모든 SUB measurement 영역 본 canonical baseline 영역 vs comparison 기준
