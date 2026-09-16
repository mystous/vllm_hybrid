# IDE_072 RESULT — 수치표·실행 상태·출처만

측정값만 기록한다. 평가·해석·의견은 포함하지 않는다.

상태 **BOUNDED_VALIDATION_COMPLETE** · 실행량 PERF 11/11 REPLAY 3/3 LOAD 1/1 DIAG 0/1 RETRY 0/2 부팅 7/10 GSM 80/80

| cell | attempt | 상태 | boot s | mismatch | per_layer 합 | replay | gsm20 |
|---|---|---|---|---|---|---|---|
| Q0_first | a1 | COMPLETED | 114 | [] | 5952 | None/None healthy=None | GSM20 19/20 correct, 0 unscored, 0 truncated, 0 errors |
| V0_confirm | a1 | COMPLETED | 113 | [] | 5952 | None/None healthy=None | GSM20 20/20 correct, 0 unscored, 0 truncated, 0 errors |
| V0_first | a1 | COMPLETED | 107 | [] | 5952 | 32/0 healthy=True | None |
| V1_confirm | a1 | COMPLETED | 107 | [] | 5952 | None/None healthy=None | GSM20 19/20 correct, 0 unscored, 0 truncated, 0 errors |
| V1_first | a1 | COMPLETED | 114 | [] | 5952 | 32/0 healthy=True | None |
| V1_load | a1 | COMPLETED | 114 | [] | 5952 | None/None healthy=None | None |
| V2_first | a1 | COMPLETED | 108 | [] | 5952 | 32/0 healthy=True | GSM20 20/20 correct, 0 unscored, 0 truncated, 0 errors |

### 성능 통계 (1차·확인·ALT·LOAD 는 cell/workload 별 행) (valid 반복만; sd = 표본표준편차 ddof=1; 표본 1 → sd null)

| cell | workload | C | n_valid | out tok/s 원값 | 평균 | 중앙값 | 최소 | 최대 | sd | mean_of_rep TTFT p95 | mean_of_rep TPOT p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V0_confirm | SHORT_COLD | 64 | 3 | 795.92, 804.02, 799.45 | 799.80 | 799.45 | 795.92 | 804.02 | 4.06 | 2997 | 73.2 |
| V0_confirm | COMPACT_ALT | 64 | 1 | 853.16 | 853.16 | 853.16 | 853.16 | 853.16 | null | 2745 | 69.1 |
| V0_first | SHORT_COLD | 64 | 1 | 795.42 | 795.42 | 795.42 | 795.42 | 795.42 | null | 2995 | 75.9 |
| V1_confirm | SHORT_COLD | 64 | 3 | 827.72, 847.56, 856.82 | 844.03 | 847.56 | 827.72 | 856.82 | 14.87 | 2933 | 67.4 |
| V1_confirm | COMPACT_ALT | 64 | 1 | 887.11 | 887.11 | 887.11 | 887.11 | 887.11 | null | 2681 | 62.5 |
| V1_first | SHORT_COLD | 64 | 1 | 842.44 | 842.44 | 842.44 | 842.44 | 842.44 | null | 3134 | 66.1 |
| V1_load | COMPACT_LOAD | 64 | 1 | 897.75 | 897.75 | 897.75 | 897.75 | 897.75 | null | 2645 | 63.9 |
| V2_first | SHORT_COLD | 64 | 1 | 701.97 | 701.97 | 701.97 | 701.97 | 701.97 | null | 3947 | 80.2 |

### 반복별 원값

| cell | attempt | rep | workload | C | n | 성공/실패 | dur s | in tok | out tok | out tok/s | total tok/s | TTFT p50/p90/p95/p99 | TPOT p50/p90/p95/p99 | ITL p50/p95/p99 | E2EL p50/p95/p99 | pooled TTFT p95/p99 · TPOT p95/p99 | valid | measure_start | flush |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V0_confirm | a1 | SHORT_COLD_C64_rep1 | SHORT_COLD | 64 | 256 | 256/0 | 41.2 | 129237 | 32768 | 795.92 | 3935.0 | 2257/null/3031/3034 | 63.1/null/73.2/75.3 | 62.2/75.8/null | 10270/10763/null | 3031/3034 · 73.2/75.3 | True | 2026-09-16T08:46:20.673713+09:00 | Cache flushed.
Please check backend logs |
| V0_confirm | a1 | SHORT_COLD_C64_rep2 | SHORT_COLD | 64 | 256 | 256/0 | 40.8 | 129237 | 32768 | 804.02 | 3975.1 | 2165/null/3030/3032 | 63.1/null/73.9/74.7 | 61.4/74.1/null | 10187/10442/null | 3030/3032 · 73.9/74.7 | True | 2026-09-16T08:47:21.803642+09:00 | Cache flushed.
Please check backend logs |
| V0_confirm | a1 | SHORT_COLD_C64_rep3 | SHORT_COLD | 64 | 256 | 256/0 | 41.0 | 129237 | 32768 | 799.45 | 3952.5 | 2133/null/2928/2931 | 63.6/null/72.7/74.5 | 63.0/74.6/null | 10211/10456/null | 2928/2931 · 72.7/74.3 | True | 2026-09-16T08:48:22.568979+09:00 | Cache flushed.
Please check backend logs |
| V0_confirm | a1 | COMPACT_ALT_C64_rep1 | COMPACT_ALT | 64 | 256 | 256/0 | 38.4 | 128481 | 32768 | 853.16 | 4198.4 | 2071/null/2745/2761 | 59.8/null/69.1/69.7 | 60.4/72.6/null | 9607/9807/null | 2745/2761 · 69.1/69.7 | True | 2026-09-16T08:49:23.627887+09:00 | Cache flushed.
Please check backend logs |
| V0_first | a1 | SHORT_COLD_C64_rep1 | SHORT_COLD | 64 | 256 | 256/0 | 41.2 | 129237 | 32768 | 795.42 | 3932.6 | 2326/null/2995/2998 | 63.4/null/75.9/75.9 | 63.1/76.1/null | 10337/10639/null | 2995/2998 · 75.9/75.9 | True | 2026-09-16T08:28:37.971753+09:00 | Cache flushed.
Please check backend logs |
| V1_confirm | a1 | SHORT_COLD_C64_rep1 | SHORT_COLD | 64 | 256 | 256/0 | 39.6 | 129237 | 32768 | 827.72 | 4092.3 | 2288/null/2975/2977 | 60.5/null/68.7/71.9 | 58.2/71.8/null | 9893/10064/null | 2975/2977 · 68.7/71.9 | True | 2026-09-16T08:53:58.959807+09:00 | Cache flushed.
Please check backend logs |
| V1_confirm | a1 | SHORT_COLD_C64_rep2 | SHORT_COLD | 64 | 256 | 256/0 | 38.7 | 129237 | 32768 | 847.56 | 4190.3 | 2155/null/2898/2919 | 58.7/null/65.3/70.5 | 56.5/69.3/null | 9632/9759/null | 2898/2919 · 65.2/70.4 | True | 2026-09-16T08:54:58.272677+09:00 | Cache flushed.
Please check backend logs |
| V1_confirm | a1 | SHORT_COLD_C64_rep3 | SHORT_COLD | 64 | 256 | 256/0 | 38.2 | 129237 | 32768 | 856.82 | 4236.1 | 2175/null/2925/2927 | 57.8/null/68.4/69.8 | 56.0/69.4/null | 9626/9903/null | 2925/2927 · 68.4/69.8 | True | 2026-09-16T08:55:56.665454+09:00 | Cache flushed.
Please check backend logs |
| V1_confirm | a1 | COMPACT_ALT_C64_rep1 | COMPACT_ALT | 64 | 256 | 256/0 | 36.9 | 128481 | 32768 | 887.11 | 4365.4 | 2073/null/2681/2702 | 56.1/null/62.5/67.0 | 55.8/65.6/null | 9285/9350/null | 2681/2702 · 62.5/66.6 | True | 2026-09-16T08:56:55.221279+09:00 | Cache flushed.
Please check backend logs |
| V1_first | a1 | SHORT_COLD_C64_rep1 | SHORT_COLD | 64 | 256 | 256/0 | 38.9 | 129237 | 32768 | 842.44 | 4165.0 | 2162/null/3134/3137 | 58.6/null/66.1/71.3 | 56.6/66.5/null | 9725/9853/null | 3133/3137 · 66.1/71.3 | True | 2026-09-16T08:32:50.117026+09:00 | Cache flushed.
Please check backend logs |
| V1_load | a1 | COMPACT_LOAD_C64_rep1 | COMPACT_LOAD | 64 | 1024 | 1024/0 | 146.0 | 514201 | 131072 | 897.75 | 4419.7 | 2104/null/2645/2872 | 55.6/null/63.9/66.3 | 55.5/66.3/null | 9132/9487/null | 2645/2872 · 63.9/66.3 | True | 2026-09-16T09:01:36.835855+09:00 | Cache flushed.
Please check backend logs |
| V2_first | a1 | SHORT_COLD_C64_rep1 | SHORT_COLD | 64 | 256 | 256/0 | 46.7 | 129237 | 32768 | 701.97 | 3470.5 | 3161/null/3947/3950 | 68.5/null/80.2/83.2 | 66.2/78.3/null | 11708/11871/null | 3947/3950 · 80.2/83.2 | True | 2026-09-16T08:36:54.962354+09:00 | Cache flushed.
Please check backend logs |

정확성: quality/paired_question_results.md · 실패: failures/README.md · 상세: FULL_REPORT.md
