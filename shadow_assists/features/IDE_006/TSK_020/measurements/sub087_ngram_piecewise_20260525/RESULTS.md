# SUB_087 — ngram cap=8 + cudagraph PIECEWISE + gmu=0.80 fair baseline

> **parent**: TSK_020 / SUB_085 v2 fair sub-comparison
> **measurement**: 2026-05-25 KST 10:15~10:42
> **status**: ✅ 완료 — all-fair comparison (vanilla / ngram / suffix 모두 gmu=0.80 + cudagraph PIECEWISE + same wrapper) 영역 확정

---

## 1. 측정 결과

`/tmp/run_workload_gen.py` × ngram spec=7 cap=8 div_tp=0 × cudagraph PIECEWISE × gmu=0.80 × 500p × 8192in × 8192max:

| workload | tps | wall (s) | out_tok | mean_accept_len (K) | per-pos α |
|---|---:|---:|---:|---:|---:|
| **sonnet** | **10,139.2** | 398.5 | 4,040,787 | 1.66 | 9.5% |
| **chat** | **2,846.4** | 116.8 | 332,521 | 5.98 | 71.2% |
| **code** | **5,326.6** | 727.9 | 3,877,057 | 1.09 | 1.2% |

## 2. ★ all-fair table (vanilla / ngram / suffix 모두 gmu=0.80 + PIECEWISE + same wrapper)

| workload | **vanilla** (SUB_086) | **ngram cap=8** (SUB_087) | **suffix_spec32** (SUB_085 v2) | ngram vs vanilla | suffix vs vanilla | **suffix vs ngram** |
|---|---:|---:|---:|---:|---:|---:|
| sonnet | 7,709.8 | **10,139.2** | **11,589.5** | +31.5% | **+50.3%** ⭐ | +14.3% |
| chat | 2,186.9 | **2,846.4** | **3,582.4** | +30.2% | **+63.8%** ⭐ | +25.9% |
| code | 6,717.8 | **5,326.6** ✗ | **7,990.0** | **−20.7% (회귀)** ✗ | **+18.9%** ⭐ | **+50.0%** ⭐⭐ |

→ **suffix PIECEWISE 영역 3 workload 모두 ngram 영역 fair 능가**:
- sonnet +14.3%
- chat +25.9%
- **code +50.0%** (ngram 회귀 영역 net positive 영역 완전 mitigation)

## 3. ★ 본 fork 영역 최종 fair contribution 정리 (gmu=0.80 + PIECEWISE)

### 3.1 best config per workload

| workload | best | tps | fair contribution (vs vanilla) |
|---|---|---:|---:|
| sonnet | suffix PIECEWISE (SUB_085 v2) | **11,589.5** | **+50.3%** |
| chat | suffix PIECEWISE (SUB_085 v2) | **3,582.4** | **+63.8%** |
| code | suffix PIECEWISE (SUB_085 v2) | **7,990.0** | **+18.9%** |

### 3.2 historical claim 정정

| historical claim | 실제 fair number |
|---|---|
| "SUB_047 sonnet +134.1%" | wrapper-historical (4,679.8 → 10,956.5). fair = **+31.5%** (7,710 → 10,139, SUB_086 영역 SUB_087) |
| "SUB_071 chat +37.5%" | gmu=0.85 영역 fair (gmu 영향 작음). fair = +30.2% (SUB_086 영역 SUB_087) |
| "SUB_071 code −23.2% 회귀" | gmu=0.85. fair = **−20.7%** (SUB_086 영역 SUB_087, ngram cap=8) |

## 4. K / α 비교 (suffix vs ngram, all-fair)

| workload | ngram K | ngram α | suffix K | suffix α | K 비율 | α 비율 |
|---|---:|---:|---:|---:|---:|---:|
| sonnet | 1.66 | 9.5% | 5.11 | 77.0% | **3.08×** | **8.1×** |
| chat | 5.98 | 71.2% | 10.06 | 92.7% | 1.68× | 1.30× |
| **code** | **1.09** | **1.2%** | **4.08** | **40.1%** | **★ 3.74×** | **★ 33×** |

→ suffix 영역 모든 workload 영역 K / α 영역 매우 강함 (특히 code).

## 5. K rank 차이 (vs SUB_075 영역 historical)

| workload | SUB_075 historical (gmu=0.85, FULL_AND_PIECEWISE) | SUB_087 (gmu=0.80, PIECEWISE) |
|---|---:|---:|
| sonnet K | 3.72 / α 38.8% | **1.66 / α 9.5%** ← K ↓ |
| chat K | 6.69 / α 81.2% | **5.98 / α 71.2%** ← 비슷 |
| code K | 1.10 / α 1.4% | **1.09 / α 1.2%** ← 동등 |

→ sonnet K 영역 cudagraph mode 영역 차이 (FULL → PIECEWISE) + gmu 영역 차이 영역 dynamic batch shape 영역 영향 추정. mean_accept_len 영역 interval-based metric 영역 noise 가능. variance 측정 (SUB_089) 영역 후속.

## 6. raw 자료

| 항목 | 위치 |
|---|---|
| sonnet/chat/code result.json | `eval/results/20260525_101534_sub087_{sonnet,chat,code}_ngram_piecewise/result.json` |
| launcher | `/tmp/run_sub087_ngram_piecewise.sh` |
| stdout | `/tmp/sub087.log` |
| summary | `/tmp/sub087_summary.tsv` |
