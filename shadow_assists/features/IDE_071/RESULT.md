# IDE_071 RESULT — 수치표·실행 상태·출처만

측정값만 기록한다. 평가·해석·의견은 포함하지 않는다.

## compact (cpu_offload_no_02_compact)

### compact 셀 상태

| cell | phase | parent | attempt | exit_status | boot s | 부팅 verdict | config_mismatch | errors | env | server_args (R0 대비 변경) | pin | patch | dir |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B3_recheck | COMPACT | None | a1 | COMPLETED | 108 | HEALTH_OK | [] | 0 | `{"KT_CALLBACK_FREE": "1", "KT_CF_SKIP_EMPTY_IMM": "1"}` | `{"kt-max-deferred-experts-per-token": 8}` | True | None | eval/results/IDE_071_20260916/compact/B3_recheck/a1 |
| S1_b4_pin | COMPACT | B4 | a1 | COMPLETED | 107 | HEALTH_OK | [] | 0 | `{"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1", "KT_AVX_RB": "1", "KT_AVX_PF": "2", "KT_FUSE_QIN": "1", "KT_AMX_MIN_QLEN": "1000000", "KT_AMX_MIN_ROWS": "3", "KT_CALLBACK_FREE": "1", "KT_COLD_DEFER": "1", "KT_COLD_TAU": "0.25"}` | `{"init-expert-location": "/models/kt/ide070/hotmap_mixed_0.25.json", "kt-max-deferred-experts-per-token": 8, "mem-fraction-static": 0.94, "chunked-prefill-size": 4096, "kv-cache-dtype": "fp8_e5m2", "cuda-graph-max-bs": 224, "cuda-graph-bs": [32, 64, 96, 128, 160, 192, 224], "max-total-tokens": 14336` | True | None | eval/results/IDE_071_20260916/compact/S1_b4_pin/a1 |
| S2_b3_v2_nu5952 | COMPACT | B3 | a1 | COMPLETED | 107 | HEALTH_OK | [] | 0 | `{"KT_CALLBACK_FREE": "1", "KT_CF_SKIP_EMPTY_IMM": "1", "KT_GPU_EXPERTS_PER_LAYER": "/models/kt/ide070/layer_budget_5952.json"}` | `{"kt-max-deferred-experts-per-token": 8, "init-expert-location": "/models/kt/ide070/hotmap_v2.json"}` | True | per_layer | eval/results/IDE_071_20260916/compact/S2_b3_v2_nu5952/a1 |
| S2_b3_v2_nu5952__confirm | COMPACT | S2_b3_v2_nu5952 | a1 | COMPLETED | 114 | HEALTH_OK | [] | 0 | `{"KT_CALLBACK_FREE": "1", "KT_CF_SKIP_EMPTY_IMM": "1", "KT_GPU_EXPERTS_PER_LAYER": "/models/kt/ide070/layer_budget_5952.json"}` | `{"kt-max-deferred-experts-per-token": 8, "init-expert-location": "/models/kt/ide070/hotmap_v2.json"}` | True | per_layer | eval/results/IDE_071_20260916/compact/S2_b3_v2_nu5952__confirm/a1 |
| S2_b3_v2_nu5952__load1024 | COMPACT | B3 | a1 | FAILED_RUNTIME | 108 | HEALTH_OK | [] | 3 | `{"KT_CALLBACK_FREE": "1", "KT_CF_SKIP_EMPTY_IMM": "1", "KT_GPU_EXPERTS_PER_LAYER": "/models/kt/ide070/layer_budget_5952.json"}` | `{"kt-max-deferred-experts-per-token": 8, "init-expert-location": "/models/kt/ide070/hotmap_v2.json"}` | True | per_layer | eval/results/IDE_071_20260916/compact/S2_b3_v2_nu5952__load1024/a1 |
| S2_b3_v2_nu5952__load1024 | COMPACT | B3 | a2 | COMPLETED | 107 | HEALTH_OK | [] | 0 | `{"KT_CALLBACK_FREE": "1", "KT_CF_SKIP_EMPTY_IMM": "1", "KT_GPU_EXPERTS_PER_LAYER": "/models/kt/ide070/layer_budget_5952.json"}` | `{"kt-max-deferred-experts-per-token": 8, "init-expert-location": "/models/kt/ide070/hotmap_v2.json"}` | True | per_layer | eval/results/IDE_071_20260916/compact/S2_b3_v2_nu5952__load1024/a2 |
| S3_b4_graph64 | COMPACT | B4 | a1 | COMPLETED | 101 | HEALTH_OK | [] | 0 | `{"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1", "KT_AVX_RB": "1", "KT_AVX_PF": "2", "KT_FUSE_QIN": "1", "KT_AMX_MIN_QLEN": "1000000", "KT_AMX_MIN_ROWS": "3", "KT_CALLBACK_FREE": "1", "KT_COLD_DEFER": "1", "KT_COLD_TAU": "0.25"}` | `{"init-expert-location": "/models/kt/ide070/hotmap_mixed_0.25.json", "kt-max-deferred-experts-per-token": 8, "mem-fraction-static": 0.94, "chunked-prefill-size": 4096, "kv-cache-dtype": "fp8_e5m2", "cuda-graph-max-bs": 64, "cuda-graph-bs": [32, 64], "max-total-tokens": 143360}` | False | None | eval/results/IDE_071_20260916/compact/S3_b4_graph64/a1 |
| S4_b4_kv40960 | COMPACT | B4 | a1 | COMPLETED | 107 | HEALTH_OK | [] | 0 | `{"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1", "KT_AVX_RB": "1", "KT_AVX_PF": "2", "KT_FUSE_QIN": "1", "KT_AMX_MIN_QLEN": "1000000", "KT_AMX_MIN_ROWS": "3", "KT_CALLBACK_FREE": "1", "KT_COLD_DEFER": "1", "KT_COLD_TAU": "0.25"}` | `{"init-expert-location": "/models/kt/ide070/hotmap_mixed_0.25.json", "kt-max-deferred-experts-per-token": 8, "mem-fraction-static": 0.94, "chunked-prefill-size": 4096, "kv-cache-dtype": "fp8_e5m2", "cuda-graph-max-bs": 224, "cuda-graph-bs": [32, 64, 96, 128, 160, 192, 224], "max-total-tokens": 40960` | False | None | eval/results/IDE_071_20260916/compact/S4_b4_kv40960/a1 |
| S5_b4_chunk2048 | COMPACT | B4 | a1 | COMPLETED | 102 | HEALTH_OK | [] | 0 | `{"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1", "KT_AVX_RB": "1", "KT_AVX_PF": "2", "KT_FUSE_QIN": "1", "KT_AMX_MIN_QLEN": "1000000", "KT_AMX_MIN_ROWS": "3", "KT_CALLBACK_FREE": "1", "KT_COLD_DEFER": "1", "KT_COLD_TAU": "0.25"}` | `{"init-expert-location": "/models/kt/ide070/hotmap_mixed_0.25.json", "kt-max-deferred-experts-per-token": 8, "mem-fraction-static": 0.94, "chunked-prefill-size": 2048, "kv-cache-dtype": "fp8_e5m2", "cuda-graph-max-bs": 224, "cuda-graph-bs": [32, 64, 96, 128, 160, 192, 224], "max-total-tokens": 14336` | False | None | eval/results/IDE_071_20260916/compact/S5_b4_chunk2048/a1 |

### compact — 반복 통계 (valid 반복만, sd = 표본표준편차 ddof=1)

| cell | workload | C | n_valid | out tok/s: 값 | 평균 | 중앙값 | 최소 | 최대 | sd | TTFT p95 평균(mean_of_rep_p95) | TPOT p95 평균 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B3_recheck | SHORT_COLD | 64 | 1 | 744.00 | 744.00 | 744.00 | 744.00 | 744.00 | null | 3065 | 78.1 |
| S1_b4_pin | SHORT_COLD | 64 | 1 | 716.86 | 716.86 | 716.86 | 716.86 | 716.86 | null | 4516 | 81.1 |
| S2_b3_v2_nu5952 | SHORT_COLD | 64 | 1 | 795.98 | 795.98 | 795.98 | 795.98 | 795.98 | null | 3046 | 73.5 |
| S2_b3_v2_nu5952__confirm | SHORT_COLD | 64 | 3 | 800.75, 811.33, 805.33 | 805.80 | 805.33 | 800.75 | 811.33 | 5.31 | 3034 | 72.9 |
| S2_b3_v2_nu5952__confirm | COMPACT_ALT | 64 | 1 | 847.99 | 847.99 | 847.99 | 847.99 | 847.99 | null | 2755 | 69.3 |
| S2_b3_v2_nu5952__load1024 | COMPACT_LOAD | 64 | 1 | 842.50 | 842.50 | 842.50 | 842.50 | 842.50 | null | 2645 | 69.2 |
| S3_b4_graph64 | SHORT_COLD | 64 | 1 | 707.60 | 707.60 | 707.60 | 707.60 | 707.60 | null | 4754 | 80.5 |
| S4_b4_kv40960 | SHORT_COLD | 64 | 1 | 703.84 | 703.84 | 703.84 | 703.84 | 703.84 | null | 4669 | 82.0 |
| S5_b4_chunk2048 | SHORT_COLD | 64 | 1 | 620.78 | 620.78 | 620.78 | 620.78 | 620.78 | null | 6239 | 96.0 |

### compact 반복별

| cell | attempt | rep | workload | C | n | 성공/실패 | dur s | in tok | out tok | out tok/s | total tok/s | TTFT p50/p95/p99 ms | TPOT p50/p95/p99 ms | ITL p50/p95 | E2EL p50/p95 | pooled TTFT p95 / TPOT p95 | valid | measure_start (KST) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B3_recheck | a1 | SHORT_COLD_C64_rep1 | SHORT_COLD | 64 | 256 | 256/0 | 44.0 | 129237 | 32768 | 744.00 | 3678.3 | 2344/3065/3067 | 67.4/78.1/78.3 | 68.9/82.6 | 10910/11339 | 3065 / 78.1 | True | 2026-09-16T07:26:57.289264+09:00 |
| S1_b4_pin | a1 | SHORT_COLD_C64_rep1 | SHORT_COLD | 64 | 256 | 256/0 | 45.7 | 129237 | 32768 | 716.86 | 3544.2 | 2899/4516/4551 | 66.6/81.1/84.8 | 57.9/68.8 | 11415/11563 | 4516 / 81.1 | True | 2026-09-16T07:07:46.179554+09:00 |
| S2_b3_v2_nu5952 | a1 | SHORT_COLD_C64_rep1 | SHORT_COLD | 64 | 256 | 256/0 | 41.2 | 129237 | 32768 | 795.98 | 3935.3 | 2260/3046/3049 | 63.0/73.5/74.6 | 62.0/74.2 | 10307/10408 | 3046 / 73.5 | True | 2026-09-16T07:11:34.489765+09:00 |
| S2_b3_v2_nu5952__confirm | a1 | SHORT_COLD_C64_rep1 | SHORT_COLD | 64 | 256 | 256/0 | 40.9 | 129237 | 32768 | 800.75 | 3958.9 | 2281/3133/3136 | 61.7/72.6/73.2 | 60.8/74.8 | 10113/10714 | 3133 / 72.6 | True | 2026-09-16T07:30:48.103988+09:00 |
| S2_b3_v2_nu5952__confirm | a1 | SHORT_COLD_C64_rep2 | SHORT_COLD | 64 | 256 | 256/0 | 40.4 | 129237 | 32768 | 811.33 | 4011.2 | 2138/2991/2993 | 62.2/73.6/73.6 | 61.1/73.9 | 10099/10138 | 2991 / 73.6 | True | 2026-09-16T07:31:49.168943+09:00 |
| S2_b3_v2_nu5952__confirm | a1 | SHORT_COLD_C64_rep3 | SHORT_COLD | 64 | 256 | 256/0 | 40.7 | 129237 | 32768 | 805.33 | 3981.5 | 2344/2978/2981 | 62.4/72.4/72.4 | 61.2/75.3 | 10088/10652 | 2978 / 72.4 | True | 2026-09-16T07:32:49.432404+09:00 |
| S2_b3_v2_nu5952__confirm | a1 | COMPACT_ALT_C64_rep1 | COMPACT_ALT | 64 | 256 | 256/0 | 38.6 | 128481 | 32768 | 847.99 | 4172.9 | 2102/2755/2794 | 59.1/69.3/69.5 | 59.6/72.1 | 9593/9942 | 2755 / 69.3 | True | 2026-09-16T07:33:50.037764+09:00 |
| S2_b3_v2_nu5952__load1024 | a1 | COMPACT_LOAD_C64_rep1 | COMPACT_LOAD | 64 | 1024 | 0/1024 | 1.7 | 0 | 0 | 0.00 | 0.0 | 0/0/0 | 0.0/0.0/0.0 | 0.0/0.0 | 0/0 | 0 / null | False | 2026-09-16T07:43:42.660579+09:00 |
| S2_b3_v2_nu5952__load1024 | a2 | COMPACT_LOAD_C64_rep1 | COMPACT_LOAD | 64 | 1024 | 1024/0 | 155.6 | 514201 | 131072 | 842.50 | 4147.7 | 2090/2645/2918 | 60.6/69.2/71.6 | 60.5/73.4 | 9790/10208 | 2645 / 69.2 | True | 2026-09-16T07:47:44.379411+09:00 |
| S3_b4_graph64 | a1 | SHORT_COLD_C64_rep1 | SHORT_COLD | 64 | 256 | 256/0 | 46.3 | 129237 | 32768 | 707.60 | 3498.3 | 3136/4754/4762 | 67.4/80.5/84.8 | 58.2/69.2 | 11598/11707 | 4754 / 80.5 | True | 2026-09-16T07:15:17.791086+09:00 |
| S4_b4_kv40960 | a1 | SHORT_COLD_C64_rep1 | SHORT_COLD | 64 | 256 | 256/0 | 46.6 | 129237 | 32768 | 703.84 | 3479.8 | 3210/4669/4703 | 66.2/82.0/83.5 | 58.8/75.7 | 11668/11967 | 4669 / 82.0 | True | 2026-09-16T07:19:13.852947+09:00 |
| S5_b4_chunk2048 | a1 | SHORT_COLD_C64_rep1 | SHORT_COLD | 64 | 256 | 256/0 | 52.8 | 129237 | 32768 | 620.78 | 3069.1 | 3658/6239/6545 | 74.7/96.0/97.1 | 59.1/69.4 | 13199/13441 | 6239 / 96.0 | True | 2026-09-16T07:23:03.447717+09:00 |

재사용 대조군: {"B3": {"source_cell": "P2_cf1_skip1_pin1_def8", "attempt": "a1", "runs": [["SHORT_COLD_C64_rep1", 729.6027160730287, 3205.990599526558, 78.80516109876262], ["SHORT_COLD_C64_rep2", 751.0998563996462, 3014.8191818443593, 76.80508520043506], ["SHORT_COLD_C64_rep3", 740.3408904376239, 2945.215696556261, 81.1357430505977]], "mean": 740.3478209700996}, "B4": {"source_cell": "R04_d4_epoch", "attempt": "a1", "runs": [["SHORT_COLD_C64_rep1", 713.2451333059573, 4478.498419979587, 82.02766449975334], ["SHORT_COLD_C64_rep2", 699.675535819436, 4725.242702494143, 80.18592318293535], ["SHORT_COLD_C64_rep3", 702.3221549874545, 4682.442792196525, 80.52045680393692]], "mean": 705.0809413709493}}
실행량: {"bench": 10, "bench_max": 20, "boot": 9, "boot_max": 14, "diag": 0, "load": 1, "retry": 1, "gsm": 0} · 종료 상태 BOUNDED_SEARCH_COMPLETE

## 12단계 계획 실행분 (P1·P2·P3 일부)

### 12단계 — 반복 통계 (valid 반복만, sd = 표본표준편차 ddof=1)

| cell | workload | C | n_valid | out tok/s: 값 | 평균 | 중앙값 | 최소 | 최대 | sd | TTFT p95 평균(mean_of_rep_p95) | TPOT p95 평균 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P2X_cf0_overlap_off | SHORT_COLD | 64 | 3 | 588.41, 588.00, 595.59 | 590.67 | 588.41 | 588.00 | 595.59 | 4.27 | 4185 | 99.8 |
| P2X_cf1_overlap_off | SHORT_COLD | 64 | 3 | 592.77, 597.46, 591.07 | 593.77 | 592.77 | 591.07 | 597.46 | 3.31 | 4146 | 100.0 |
| P2_cf0_skip0_pin0_def0 | SHORT_COLD | 64 | 3 | 578.04, 583.98, 579.00 | 580.34 | 579.00 | 578.04 | 583.98 | 3.19 | 4228 | 98.5 |
| P2_cf0_skip0_pin0_def4 | SHORT_COLD | 64 | 3 | 615.57, 606.56, 615.00 | 612.37 | 615.00 | 606.56 | 615.57 | 5.05 | 4254 | 91.0 |
| P2_cf0_skip0_pin0_def8 | SHORT_COLD | 64 | 3 | 648.31, 639.94, 641.39 | 643.21 | 641.39 | 639.94 | 648.31 | 4.48 | 4100 | 87.2 |
| P2_cf0_skip0_pin1_def0 | SHORT_COLD | 64 | 3 | 587.03, 592.16, 584.02 | 587.74 | 587.03 | 584.02 | 592.16 | 4.11 | 4172 | 96.4 |
| P2_cf0_skip0_pin1_def4 | SHORT_COLD | 64 | 3 | 618.17, 618.99, 615.06 | 617.41 | 618.17 | 615.06 | 618.99 | 2.07 | 4168 | 90.4 |
| P2_cf0_skip0_pin1_def8 | SHORT_COLD | 64 | 3 | 648.36, 654.72, 642.81 | 648.63 | 648.36 | 642.81 | 654.72 | 5.96 | 3976 | 86.5 |
| P2_cf1_skip0_pin0_def0 | SHORT_COLD | 64 | 3 | 591.85, 598.08, 596.39 | 595.44 | 596.39 | 591.85 | 598.08 | 3.22 | 4184 | 96.5 |
| P2_cf1_skip0_pin0_def4 | SHORT_COLD | 64 | 3 | 611.21, 625.90, 620.11 | 619.07 | 620.11 | 611.21 | 625.90 | 7.40 | 4178 | 92.6 |
| P2_cf1_skip0_pin0_def8 | SHORT_COLD | 64 | 3 | 652.84, 656.99, 655.19 | 655.00 | 655.19 | 652.84 | 656.99 | 2.08 | 4012 | 88.3 |
| P2_cf1_skip0_pin1_def0 | SHORT_COLD | 64 | 3 | 603.62, 596.55, 597.39 | 599.18 | 597.39 | 596.55 | 603.62 | 3.86 | 4113 | 96.8 |
| P2_cf1_skip0_pin1_def4 | SHORT_COLD | 64 | 3 | 622.37, 623.96, 618.16 | 621.49 | 622.37 | 618.16 | 623.96 | 3.00 | 4162 | 90.7 |
| P2_cf1_skip0_pin1_def8 | SHORT_COLD | 64 | 3 | 658.93, 657.40, 658.77 | 658.37 | 658.77 | 657.40 | 658.93 | 0.84 | 3973 | 87.3 |
| P2_cf1_skip1_pin0_def0 | SHORT_COLD | 64 | 3 | 614.19, 594.70, 602.34 | 603.74 | 602.34 | 594.70 | 614.19 | 9.82 | 4086 | 96.1 |
| P2_cf1_skip1_pin0_def4 | SHORT_COLD | 64 | 3 | 625.43, 629.98, 624.84 | 626.75 | 625.43 | 624.84 | 629.98 | 2.81 | 4179 | 89.9 |
| P2_cf1_skip1_pin0_def8 | SHORT_COLD | 64 | 3 | 725.77, 744.01, 739.36 | 736.38 | 739.36 | 725.77 | 744.01 | 9.48 | 3058 | 79.3 |
| P2_cf1_skip1_pin1_def0 | SHORT_COLD | 64 | 3 | 598.44, 606.10, 602.67 | 602.40 | 602.67 | 598.44 | 606.10 | 3.84 | 4100 | 95.3 |
| P2_cf1_skip1_pin1_def4 | SHORT_COLD | 64 | 3 | 620.99, 623.43, 626.19 | 623.53 | 623.43 | 620.99 | 626.19 | 2.60 | 4195 | 90.0 |
| P2_cf1_skip1_pin1_def8 | SHORT_COLD | 64 | 3 | 729.60, 751.10, 740.34 | 740.35 | 740.34 | 729.60 | 751.10 | 10.75 | 3055 | 78.9 |
| P3_amx_min_qlen_1000000 | SHORT_COLD | 64 | 3 | 669.74, 680.76, 681.03 | 677.18 | 680.76 | 669.74 | 681.03 | 6.44 | 4111 | 82.4 |
| P3_amx_min_qlen_1024 | SHORT_COLD | 64 | 3 | 743.84, 737.19, 747.33 | 742.79 | 743.84 | 737.19 | 747.33 | 5.15 | 3084 | 78.5 |
| P3_amx_min_qlen_128 | SHORT_COLD | 64 | 3 | 744.20, 741.95, 737.87 | 741.34 | 741.95 | 737.87 | 744.20 | 3.21 | 3036 | 78.9 |
| P3_amx_min_qlen_256 | SHORT_COLD | 64 | 3 | 741.94, 742.37, 738.41 | 740.91 | 741.94 | 738.41 | 742.37 | 2.17 | 3010 | 78.6 |
| P3_amx_min_qlen_32 | SHORT_COLD | 64 | 3 | 674.44, 674.20, 680.80 | 676.48 | 674.44 | 674.20 | 680.80 | 3.75 | 3054 | 87.0 |
| P3_amx_min_qlen_512 | SHORT_COLD | 64 | 3 | 734.80, 730.18, 734.72 | 733.23 | 734.72 | 730.18 | 734.80 | 2.65 | 3092 | 79.6 |
| P3_amx_min_qlen_64 | SHORT_COLD | 64 | 3 | 734.19, 737.36, 756.16 | 742.57 | 737.36 | 734.19 | 756.16 | 11.87 | 2978 | 79.9 |
| P3_amx_min_rows_1 | SHORT_COLD | 64 | 3 | 685.46, 686.34, 678.17 | 683.32 | 685.46 | 678.17 | 686.34 | 4.49 | 3087 | 85.5 |
| P3_amx_min_rows_16 | SHORT_COLD | 64 | 3 | 734.33, 745.21, 733.74 | 737.76 | 734.33 | 733.74 | 745.21 | 6.46 | 3006 | 79.3 |
| P3_amx_min_rows_3 | SHORT_COLD | 64 | 3 | 747.20, 749.98, 741.45 | 746.21 | 747.20 | 741.45 | 749.98 | 4.35 | 2957 | 78.1 |
| P3_amx_min_rows_4 | SHORT_COLD | 64 | 3 | 745.18, 739.86, 733.34 | 739.46 | 739.86 | 733.34 | 745.18 | 5.93 | 3088 | 80.7 |
| P3_amx_min_rows_8 | SHORT_COLD | 64 | 3 | 754.40, 744.84, 761.24 | 753.49 | 754.40 | 744.84 | 761.24 | 8.24 | 3011 | 77.5 |
| P3_avx_pf_0 | SHORT_COLD | 64 | 3 | 742.61, 746.02, 741.65 | 743.43 | 742.61 | 741.65 | 746.02 | 2.30 | 3033 | 78.6 |
| P3_avx_pf_1 | SHORT_COLD | 64 | 3 | 740.26, 732.91, 734.61 | 735.92 | 734.61 | 732.91 | 740.26 | 3.85 | 2972 | 79.0 |
| P3_avx_pf_2 | SHORT_COLD | 64 | 3 | 740.38, 748.10, 738.85 | 742.44 | 740.38 | 738.85 | 748.10 | 4.96 | 3030 | 79.4 |
| P3_avx_pf_4 | SHORT_COLD | 64 | 3 | 732.41, 739.40, 744.76 | 738.86 | 739.40 | 732.41 | 744.76 | 6.19 | 3057 | 79.2 |
| P3_avx_rb_0 | SHORT_COLD | 64 | 3 | 774.08, 792.82, 794.50 | 787.14 | 792.82 | 774.08 | 794.50 | 11.34 | 2952 | 74.1 |
| P3_cpuinfer_104 | SHORT_COLD | 64 | 3 | 737.87, 736.45, 730.57 | 734.96 | 736.45 | 730.57 | 737.87 | 3.87 | 3062 | 80.0 |
| P3_cpuinfer_108 | SHORT_COLD | 64 | 3 | 731.20, 751.67, 743.28 | 742.05 | 743.28 | 731.20 | 751.67 | 10.29 | 3119 | 77.9 |
| P3_cpuinfer_80 | SHORT_COLD | 64 | 3 | 704.70, 695.86, 694.14 | 698.24 | 695.86 | 694.14 | 704.70 | 5.67 | 3119 | 84.0 |
| P3_cpuinfer_88 | SHORT_COLD | 64 | 3 | 729.92, 725.54, 726.99 | 727.48 | 726.99 | 725.54 | 729.92 | 2.23 | 3014 | 79.7 |
| P3_fuse_qin_0 | SHORT_COLD | 64 | 3 | 743.44, 753.05, 759.95 | 752.15 | 753.05 | 743.44 | 759.95 | 8.29 | 3012 | 77.9 |
| P3_fuse_qin_1 | SHORT_COLD | 64 | 3 | 748.68, 731.03, 750.77 | 743.49 | 748.68 | 731.03 | 750.77 | 10.84 | 3012 | 79.1 |
| P3_mkl_num_threads_4 | SHORT_COLD | 64 | 3 | 733.92, 744.10, 745.51 | 741.18 | 744.10 | 733.92 | 745.51 | 6.32 | 3019 | 79.4 |
| P3_omp_num_threads_1 | SHORT_COLD | 64 | 3 | 730.75, 744.56, 733.63 | 736.31 | 733.63 | 730.75 | 744.56 | 7.29 | 3023 | 79.1 |
| P3_omp_num_threads_4 | SHORT_COLD | 64 | 3 | 729.74, 742.52, 747.56 | 739.94 | 742.52 | 729.74 | 747.56 | 9.18 | 3046 | 79.1 |
| P3_openblas_num_threads_1 | SHORT_COLD | 64 | 3 | 743.30, 720.81, 740.14 | 734.75 | 740.14 | 720.81 | 743.30 | 12.18 | 3054 | 79.3 |
| P3_openblas_num_threads_4 | SHORT_COLD | 64 | 3 | 731.41, 740.16, 757.09 | 742.89 | 740.16 | 731.41 | 757.09 | 13.06 | 3019 | 78.9 |
| P3_tokenizer_workers_1 | SHORT_COLD | 64 | 3 | 754.35, 745.85, 739.15 | 746.45 | 745.85 | 739.15 | 754.35 | 7.62 | 2983 | 78.9 |
| P3_tokenizer_workers_2 | SHORT_COLD | 64 | 3 | 724.13, 739.88, 746.69 | 736.90 | 739.88 | 724.13 | 746.69 | 11.57 | 3050 | 78.8 |
| P3_tokenizer_workers_4 | SHORT_COLD | 64 | 3 | 736.87, 744.85, 749.00 | 743.57 | 744.85 | 736.87 | 749.00 | 6.16 | 3031 | 78.2 |
| R00_r0_def4 | LEGACY_070 | 64 | 3 | 661.61, 648.69, 637.05 | 649.12 | 648.69 | 637.05 | 661.61 | 12.28 | 3709 | 85.8 |
| R00_r0_def4 | SHORT_COLD | 64 | 3 | 610.88, 606.05, 619.40 | 612.11 | 610.88 | 606.05 | 619.40 | 6.76 | 4195 | 91.0 |
| R00_r0_def4__anchor1 | SHORT_COLD | 64 | 2 | 604.83, 614.78 | 609.80 | 609.80 | 604.83 | 614.78 | 7.03 | 4288 | 91.3 |
| R01_r0_def0 | SHORT_COLD | 64 | 3 | 577.53, 579.48, 577.15 | 578.05 | 577.53 | 577.15 | 579.48 | 1.25 | 4204 | 99.0 |
| R02_d1_cf | LEGACY_070 | 64 | 3 | 684.07, 654.77, 661.87 | 666.90 | 661.87 | 654.77 | 684.07 | 15.28 | 3688 | 86.4 |
| R04_d4_epoch | LEGACY_070 | 64 | 3 | 817.54, 849.01, 784.17 | 816.90 | 817.54 | 784.17 | 849.01 | 32.42 | 3914 | 69.7 |
| R04_d4_epoch | SHORT_COLD | 64 | 3 | 713.25, 699.68, 702.32 | 705.08 | 702.32 | 699.68 | 713.25 | 7.19 | 4629 | 80.9 |
| R05_v2_u96 | SHORT_COLD | 64 | 3 | 622.90, 627.33, 628.12 | 626.12 | 627.33 | 622.90 | 628.12 | 2.81 | 4263 | 88.9 |
| R06_v2_nu5952 | SHORT_COLD | 64 | 3 | 651.13, 655.23, 660.90 | 655.75 | 655.23 | 651.13 | 660.90 | 4.90 | 4201 | 83.8 |
| R07_gpu_only_tp8_ep8 | SHORT_COLD | 32 | 3 | 1113.64, 1113.68, 1112.36 | 1113.23 | 1113.64 | 1112.36 | 1113.68 | 0.75 | 814 | 24.3 |
| R07_gpu_only_tp8_ep8 | SHORT_COLD | 64 | 3 | 1707.84, 1698.17, 1704.65 | 1703.56 | 1704.65 | 1698.17 | 1707.84 | 4.93 | 1464 | 32.8 |

상세: FULL_REPORT.md §3~§13, manifests/cells.jsonl, reports/request_tables/.
