# SUB_032 — A4 단독 3-run avg 결과 (2026-05-21 KST)

> **parent**: TSK_019 / O 분석 §7.2 ★★ 항목 — Stage 1+3 유일 win 으로 보였던 A4 (numactl --localalloc) 의 statistical validation.
> **measurement**: HEAD `0d7dc0334`, 100p × 8192, gmu=0.85, env-ON baseline + `numactl --localalloc --physcpubind=0-111`.

---

## 1. 측정 결과

| run | tps | wall (s) | crash |
|---|---:|---:|:-:|
| run1 | 929.8 | 881.0 | 0 ✓ |
| run2 | 928.4 | 873.9 | 0 ✓ |
| run3 | 932.4 | 870.2 | 0 ✓ |
| **avg** | **930.2** | **875.0** | — |
| std | 1.65 (0.18%) | — | — |

## 2. ⚠️ 결정적 발견 — A4 도 noise 였다

| 비교 | tps | Δ |
|---|---:|---:|
| Stage 1 baseline t00 (1-run) | 932.2 | — |
| **Stage 1 A4 단독 t04 (1-run)** | **941.1** | **+1.0% (1-run artifact)** |
| **SUB_032 A4 3-run avg** | **930.2** | **-0.21% (noise)** |

**Stage 1 의 A4 단독 +1.0% win 은 1-run measurement noise 였습니다.** 3-run avg 로 평균을 내면 baseline 과 통계적으로 구별 불가능 (±0.18% 안).

→ **Stage 1+3 24 measurements 중 유일하게 win 으로 보였던 lever 마저 invalid**.
→ O 분석 §7.1 결론 "유일 win = A4 (+1.0%, numactl --localalloc)" 는 **철회** 필요.

## 3. 갱신된 결론

1. **모든 A-tier lever (A1-A5) 가 noise**: ±2% noise band 안에서 single-run 측정의 통계적 의미 없음.
2. **NEO 의 worker thread NUMA bind 만으로 충분** (현 코드의 `VLLM_NEO_NUMA_BIND=1`): numactl --localalloc 추가가 측정 가능한 이득을 만들지 못함.
3. **다음 단계는 B-tier (kernel-level 변경) 가 유일 path**: A-tier 의 runtime/env tuning 으로는 더 이상 짜낼 수 없음. SUB_033 (B3 FlashDecoding++) / SUB_034 (B1 OmniServe LSE async) 가 실질적 시도.

## 4. 통계 근거

- noise band (3-run std): ±0.18% = ±1.65 tps
- baseline noise (Stage 1+3 24 measurements 의 분포): ±2.0% = ±18.6 tps
- A4 단독의 measured Δ: -0.21% (단일점) — noise band 안에서 random distribution

## 5. raw 자료

| 항목 | 위치 |
|---|---|
| SUMMARY.tsv | `eval/results/20260521_222156_sub032_a4_3run_100p/SUMMARY.tsv` |
| per-run 결과 (3 dirs) | `eval/results/20260521_222156_sub032_a4_3run_100p/run1~3/` |
| launcher | `/tmp/run_sub032_a4_3run.sh` |
| stdout log | `/tmp/sub032_a4_3run.log` |

## 6. 후속 영향

- [O 분석](../../analysis/O_stage1_stage3_root_cause.md) §0 TL;DR / §4 / §7.1 갱신 필요 — "A4 +1.0% win" 을 "A4 도 noise (SUB_032 3-run avg 932 tps)" 로 정정
- [Stage 1 RESULTS](../stage1_a1a2a3a4_matrix_100p_20260521/RESULTS.md) 의 "★ winner = A4" 표시 도 정정
- [Stage 3 RESULTS](../stage3_a5_matrix_100p_20260521/RESULTS.md) 합산 결론 도 갱신
- SUB_033 (B3 FlashDecoding++) 가 진정한 첫 kernel-level lever — 결과에 따라 B/C-tier 전략 결정
