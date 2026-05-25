# SUB_089 — SUB_085 v2 sonnet canonical 3-run (variance 측정)

> **parent**: TSK_020 / SUB_085 v2 stability
> **measurement**: 2026-05-25 KST 10:53~11:16, 3 runs × sonnet × suffix PIECEWISE + gmu=0.80
> **status**: ✅ 완료 — **canonical avg = 11,687.4 tps, variance 0.20%** (매우 stable)

---

## 1. 3-run 결과

| run | tps | wall (s) | out_tok | K (mean_accept_len) | α (per-pos) |
|---|---:|---:|---:|---:|---:|
| 1 | 11,695.3 | 345.5 | 4,040,828 | 1.65 (interval noise) | 29.5% |
| 2 | 11,694.7 | 345.5 | 4,040,828 | 7.92 | 88.6% |
| 3 | 11,672.1 | 346.2 | 4,040,828 | 7.59 | 87.6% |
| **avg** | **11,687.4** | 345.7 | 4,040,828 | — | — |
| min | 11,672.1 | 345.5 | — | — | — |
| max | 11,695.3 | 346.2 | — | — | — |
| range / avg (variance) | **0.20%** | 0.20% | — | — | — |

→ **throughput variance 0.20% — 매우 stable**. configuration 영역 production-ready.

## 2. K (mean_accept_len) 변동 — last-interval noise

run 1 영역 K=1.65 / run 2 영역 K=7.92 영역 큰 차이 — vLLM `SpecDecoding metrics` 영역 last interval 영역 측정 → batch size 영역 작아질 때 영역 적은 sample 영역 variance.

cumulative average 영역 더 정확 (sum/count) — vLLM v1 metric path 영역 cumulative emit 추가 영역 별도 patch. 단 본 SUB 영역 throughput 영역 main metric (variance 0.20%).

## 3. SUB_086 vanilla (gmu=0.80) 영역 canonical fair contribution

| metric | SUB_086 vanilla (gmu=0.80) | SUB_089 canonical 3-run avg | fair contribution |
|---|---:|---:|---:|
| sonnet tps | 7,709.8 | **11,687.4** | **+51.6%** ⭐ |

→ single-run (SUB_085 v2) 영역 11,589.5 (+50.3%) 영역 매우 일관. canonical 영역 약간 더 높음 (+1.3 pp).

## 4. 본 doc 갱신

- Best doc 영역 sonnet canonical = 11,687.4 tps (variance 0.20%) 영역 정식 등록
- 분석 doc §1 영역 sonnet fair = +51.6% (canonical 3-run avg) 영역 갱신

## 5. raw 자료

| 항목 | 위치 |
|---|---|
| 3 result.json | `eval/results/20260525_105335_sub089_sonnet_suffix_piecewise_run{1,2,3}/result.json` |
| launcher | `/tmp/run_sub089_sonnet_3run.sh` |
| stdout | `/tmp/sub089.log` |
| summary | `/tmp/sub089_summary.tsv` |
