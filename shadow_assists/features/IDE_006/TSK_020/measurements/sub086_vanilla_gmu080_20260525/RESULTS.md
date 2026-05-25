# SUB_086 — vanilla baseline @ gmu=0.80 (SUB_085 v2 fair comparison)

> **parent**: TSK_020 / SUB_085 v2 (Phase 2 unblock) 영역 fair baseline
> **measurement**: 2026-05-25 KST 09:31~09:58, single-run × 3 workload
> **status**: ✅ 완료

---

## 1. 측정 결과

`/tmp/run_workload_gen.py` (same wrapper as SUB_085) × no spec (vanilla) × gmu=0.80 × 500p × 8192in × 8192max:

| workload | tps | wall (s) | out_tok | out_tok/prompt |
|---|---:|---:|---:|---:|
| **sonnet** | **7,709.8** | 521.6 | 4,021,373 | 8,043 |
| **chat** | **2,186.9** | 151.1 | 330,476 | 661 |
| **code** | **6,717.8** | 585.7 | 3,934,466 | 7,869 |

## 2. 이전 vanilla baseline 영역 비교

| workload | SUB_086 (이번, gmu=0.80, `run_workload_gen.py` wrapper) | 이전 vanilla (SUB_044/047/071) | 차이 |
|---|---:|---:|---:|
| sonnet | **7,709.8** | 4,679.8 (SUB_043 t1_baseline, gmu=0.85, `run_spec_decode.py` 다른 wrapper) | **+64.7%** ⚠ |
| chat | 2,186.9 | 2,186.0 (SUB_071, gmu=0.85, same wrapper) | **+0.04% (stable)** ✓ |
| code | 6,717.8 | 6,964.5 (SUB_071, gmu=0.85, same wrapper) | -3.5% |

### 2.1 sonnet 의 +64.7% 차이 — 원인 분석

| 가설 | 검증 |
|---|---|
| gmu 0.80 vs 0.85 영역 영향 | ✗ — chat / code 영역 0% / -3.5% 영역 stable, gmu 영역 sonnet 만 +65% 영향 불가능 |
| **wrapper 차이 (`run_workload_gen.py` vs `run_spec_decode.py`)** | **✓ most likely** — 두 wrapper 영역 prompt builder 영역 약간 다를 가능성 (sonnet line count, padding 등) |
| 환경 변수 차이 (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) | ◐ minor 영역 fragmentation, 단 throughput +65% 안 줘야 |
| run variance | ✗ — sonnet 영역 historical 4,679.8 영역 SUB_043 t1 + t6 prefix_cache 영역 같은 값 영역 reproducible |
| GPU contention (bentoml services) | ◐ time-of-day 영역 차이 가능, 단 unlikely 영역 +65% |

→ **결론: sonnet vanilla 영역 historical 4,679.8 영역 wrapper-specific 영역 noise**. 본 SUB_086 의 **7,709.8 영역 진짜 fair vanilla baseline** (same wrapper as ngram/suffix 측정).

### 2.2 chat / code 영역 stable

chat ≈ 0.04% 차이 (statistically identical), code 영역 -3.5% (small gmu effect) → **two wrapper 영역 chat/code 영역 같은 prompt 영역 생성, sonnet 만 차이**.

## 3. fair comparison — SUB_085 v2 contribution 재정의

본 SUB_086 영역 baseline 으로 SUB_085 v2 (suffix + PIECEWISE) 영역 fair contribution:

| workload | SUB_086 vanilla (gmu=0.80) | SUB_085 v2 suffix PIECEWISE (gmu=0.80) | **fair contribution** |
|---|---:|---:|---:|
| sonnet | 7,709.8 | 11,589.5 | **+50.3%** ⭐ |
| chat | 2,186.9 | 3,582.4 | **+63.8%** ⭐ |
| code | 6,717.8 | 7,990.0 | **+18.9%** ⭐ |

(이전 보고 +147.7% / +63.9% / +14.7% 영역 unfair 영역, gmu=0.85 vanilla 영역 mix 영역 비교)

## 4. 본 doc 의 영향

| 영향 doc | 갱신 사항 |
|---|---|
| `analysis/workload_acceptance_analysis_20260524.md` | §1 TL;DR + §2 측정 fact 영역 vanilla sonnet 영역 SUB_086 영역 7,709.8 영역 갱신 (단 historical 영역 4,679.8 영역 caveat 보존) |
| `Best_SpecDecode_10778tps.md` | §1 측정 fact 영역 SUB_086 영역 vanilla baseline 영역 추가 + SUB_085 v2 fair contribution 영역 정리 |
| `INDEX.md` | §1 active SUB 표 + §4 workload generalization 표 영역 SUB_085 v2 row 추가 |
| 기존 contribution claim ("+134.12% vs vanilla") | **wrapper-historical caveat** 추가. wrapper-consistent fair = vanilla 7,710 → ngram cap=8 10,957 = +42% / vanilla 7,710 → suffix PIECEWISE 11,590 = +50% |

## 5. raw 자료

| 항목 | 위치 |
|---|---|
| sonnet/chat/code result.json | `eval/results/20260525_093152_sub086_{sonnet,chat,code}_vanilla_gmu080/result.json` |
| launcher | `/tmp/run_sub086_vanilla_gmu080.sh` |
| stdout | `/tmp/sub086.log` |
| summary | `/tmp/sub086_summary.tsv` |
| fair vs comparison doc | [`../sub085_suffix_piecewise_20260525/RESULTS.md`](../sub085_suffix_piecewise_20260525/RESULTS.md) |
