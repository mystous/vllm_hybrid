# FULL_REPORT — IDE_076 A1·A2·B 구현·검증 (2026-09-18T14:55:48+09:00)

원값 보고서. 예상 speedup·우수성 주장·다음 제안은 없음 (판단은 ANALYSIS_AND_DECISION.md, *_DECISION.md).

## 1. 환경·실효 설정

BASELINE_PROVENANCE.md, SOURCE_MANIFEST.json, PLAN_RESOLVED.md 참조. 바이너리: v4 12926df2… (reference), v5 941a83c4… (A1 런타임 분기), v6 194f3064… (v5 + A2 probe), v7 (A1 템플릿·hoist + 스레드별 카운터; 아래 부팅 표의 .so 열).

## 2. E2E 세션 원값 (OFF = 무계측 성능; vllm bench / vllmw)

| variant | boot | session | .so | A1 | A2 | hotmap | workload | mode | client | done/sent | in/out tok | dur s | tps | TTFT p50/p95/p99 | TPOT p50/p95/p99 | ITL p95 | E2EL p95 | valid | t_start |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A12B | A12B_CONFIRM_131501 | M1_MAIN_OFF | a4add14d | 1 | 1 | hotmap.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 46.5 | 705.32 | 2335/3192/11260 | 66.8/77.4/82.4 | 73.8 | 11335 | False | 13:17:56 |
| A12B | A12B_CONFIRM_131501 | M2_MAIN_OFF | a4add14d | 1 | 1 | hotmap.json | MAIN_SHORT | OFF | vllm | 127/256 | 65372/16256 | 313.7 | 51.83 | 2252/3201/3205 | 65.2/76.0/76.1 | 74.6 | 11194 | False | 13:19:02 |
| A12B | A12B_CONFIRM_135200 | M1_MAIN_OFF | a4add14d | 1 | 1 | hotmap.json | MAIN_SHORT | OFF | vllm | 63/256 | 32409/8064 | 291.0 | 27.71 | 2229/3249/3250 | 69.8/75.6/77.8 | 72.1 | 11108 | False | 13:54:55 |
| A12B | A12B_CONFIRM_142846 | M1_MAIN_OFF | a4add14d | 1 | 1 | hotmap.json | MAIN_SHORT | OFF | vllm | 189/256 | 97303/24192 | 325.3 | 74.37 | 2431/3226/4236 | 67.5/77.3/82.0 | 71.9 | 11252 | False | 14:31:35 |
| A12 | A12_MAIN3_C1_LONG_073716 | C1_OFF | a4add14d | 1 | 1 | hotmap_v2.json | LOW_CONCURRENCY | OFF | vllm | 16/16 | 8234/2048 | 32.4 | 63.27 | 239/312/322 | 13.9/13.9/14.0 | 14.2 | 2079 | True | 07:43:01 |
| A12 | A12_MAIN3_C1_LONG_073716 | LONG_OFF | a4add14d | 1 | 1 | hotmap_v2.json | LONGER_PREFILL | OFF | vllm | 32/32 | 131147/4096 | 24.4 | 167.53 | 2333/3118/3121 | 29.8/41.6/41.7 | 25.2 | 6124 | True | 07:43:53 |
| A12 | A12_MAIN3_C1_LONG_073716 | M1_MAIN_OFF | a4add14d | 1 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 40.0 | 819.29 | 2269/3102/6131 | 53.3/64.4/67.6 | 59.2 | 9495 | True | 07:40:01 |
| A12 | A12_MAIN3_C1_LONG_073716 | M2_MAIN_OFF | a4add14d | 1 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 40.2 | 815.57 | 2123/3066/6048 | 54.0/65.9/70.1 | 61.0 | 9816 | True | 07:41:01 |
| A12 | A12_MAIN3_C1_LONG_073716 | M3_MAIN_OFF | a4add14d | 1 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 39.8 | 822.92 | 2143/3018/6019 | 53.3/63.6/69.0 | 58.8 | 9512 | True | 07:42:02 |
| A12 | A12_MAIN3_C1_LONG_080805 | M1_MAIN_OFF | a4add14d | 1 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 63/256 | 32409/8064 | 300.6 | 26.82 | 2126/3119/3120 | 64.0/69.5/71.7 | 68.2 | 10272 | False | 08:10:49 |
| A12 | A12_MAIN3_C1_LONG_100442 | C1_OFF | a4add14d | 1 | 1 | hotmap_v2.json | LOW_CONCURRENCY | OFF | vllm | 16/16 | 8234/2048 | 32.5 | 63.05 | 251/281/300 | 13.9/14.1/14.6 | 14.2 | 2081 | True | 10:10:32 |
| A12 | A12_MAIN3_C1_LONG_100442 | LONG_OFF | a4add14d | 1 | 1 | hotmap_v2.json | LONGER_PREFILL | OFF | vllm | 32/32 | 131147/4096 | 24.4 | 167.94 | 2318/3099/3103 | 29.8/41.6/41.7 | 25.2 | 6116 | True | 10:11:24 |
| A12 | A12_MAIN3_C1_LONG_100442 | M1_MAIN_OFF | a4add14d | 1 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 42.4 | 772.44 | 2216/3191/10322 | 60.1/69.4/73.4 | 65.2 | 10296 | True | 10:07:27 |
| A12 | A12_MAIN3_C1_LONG_100442 | M2_MAIN_OFF | a4add14d | 1 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 41.4 | 791.01 | 2116/2979/10176 | 59.6/70.7/71.0 | 63.0 | 9767 | True | 10:08:29 |
| A12 | A12_MAIN3_C1_LONG_100442 | M3_MAIN_OFF | a4add14d | 1 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 41.3 | 793.67 | 2127/3001/10043 | 58.9/69.6/71.7 | 64.3 | 9905 | True | 10:09:31 |
| A12 | A12_MAIN3_C1_LONG_103813 | C1_OFF | a4add14d | 1 | 1 | hotmap_v2.json | LOW_CONCURRENCY | OFF | vllm | 16/16 | 8234/2048 | 32.4 | 63.28 | 238/330/330 | 13.9/14.0/14.0 | 14.3 | 2097 | True | 10:44:00 |
| A12 | A12_MAIN3_C1_LONG_103813 | LONG_OFF | a4add14d | 1 | 1 | hotmap_v2.json | LONGER_PREFILL | OFF | vllm | 32/32 | 131147/4096 | 24.5 | 167.32 | 2311/3119/3142 | 29.8/41.6/41.9 | 25.3 | 6159 | True | 10:44:52 |
| A12 | A12_MAIN3_C1_LONG_103813 | M1_MAIN_OFF | a4add14d | 1 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 42.4 | 772.85 | 2226/3250/10457 | 60.6/68.0/71.5 | 64.5 | 10176 | True | 10:40:53 |
| A12 | A12_MAIN3_C1_LONG_103813 | M2_MAIN_OFF | a4add14d | 1 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 42.0 | 781.04 | 2235/3023/10268 | 58.7/69.4/73.2 | 65.0 | 10075 | True | 10:41:56 |
| A12 | A12_MAIN3_C1_LONG_103813 | M3_MAIN_OFF | a4add14d | 1 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 41.4 | 791.43 | 2147/3076/10212 | 59.1/69.6/71.1 | 63.7 | 9812 | True | 10:42:58 |
| A12 | A12_MAIN3_C1_LONG_120750 | C1_OFF | a4add14d | 1 | 1 | hotmap_v2.json | LOW_CONCURRENCY | OFF | vllm | 16/16 | 8234/2048 | 32.2 | 63.57 | 247/260/274 | 13.9/13.9/14.0 | 14.2 | 2024 | True | 12:13:41 |
| A12 | A12_MAIN3_C1_LONG_120750 | LONG_OFF | a4add14d | 1 | 1 | hotmap_v2.json | LONGER_PREFILL | OFF | vllm | 32/32 | 131147/4096 | 25.5 | 160.53 | 2495/3830/3973 | 29.9/41.7/41.9 | 25.4 | 7030 | True | 12:14:33 |
| A12 | A12_MAIN3_C1_LONG_120750 | M1_MAIN_OFF | a4add14d | 1 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 42.5 | 770.84 | 2216/3296/10338 | 60.3/68.6/72.6 | 64.4 | 10241 | True | 12:10:34 |
| A12 | A12_MAIN3_C1_LONG_120750 | M2_MAIN_OFF | a4add14d | 1 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 42.6 | 768.51 | 2168/3019/10239 | 60.2/70.6/75.0 | 69.1 | 10395 | True | 12:11:37 |
| A12 | A12_MAIN3_C1_LONG_120750 | M3_MAIN_OFF | a4add14d | 1 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 41.7 | 786.37 | 2164/3099/10258 | 59.2/66.8/71.4 | 65.1 | 9945 | True | 12:12:40 |
| A12 | A12_MAIN3_C1_LONG_123842 | C1_OFF | a4add14d | 1 | 1 | hotmap_v2.json | LOW_CONCURRENCY | OFF | vllm | 16/16 | 8234/2048 | 32.3 | 63.34 | 247/306/308 | 13.9/13.9/13.9 | 14.2 | 2072 | True | 12:44:34 |
| A12 | A12_MAIN3_C1_LONG_123842 | LONG_OFF | a4add14d | 1 | 1 | hotmap_v2.json | LONGER_PREFILL | OFF | vllm | 32/32 | 131147/4096 | 24.5 | 167.37 | 2332/3114/3173 | 29.7/41.4/42.1 | 25.0 | 6224 | True | 12:45:26 |
| A12 | A12_MAIN3_C1_LONG_123842 | M1_MAIN_OFF | a4add14d | 1 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 42.5 | 770.57 | 2236/3126/10462 | 61.2/71.0/72.3 | 65.9 | 10078 | True | 12:41:27 |
| A12 | A12_MAIN3_C1_LONG_123842 | M2_MAIN_OFF | a4add14d | 1 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 41.6 | 788.09 | 2133/3008/10247 | 60.0/70.8/72.2 | 65.9 | 9899 | True | 12:42:29 |
| A12 | A12_MAIN3_C1_LONG_123842 | M3_MAIN_OFF | a4add14d | 1 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 42.7 | 767.57 | 2174/3003/10413 | 61.6/70.9/74.3 | 68.8 | 10395 | True | 12:43:31 |
| A12e | A12e_MAIN3_C1_LONG_074455 | C1_OFF | a4add14d | 1 | 1 | hotmap_v2.json | LOW_CONCURRENCY | OFF | vllm | 16/16 | 8234/2048 | 32.5 | 63.06 | 242/335/350 | 13.9/14.0/14.0 | 14.3 | 2105 | True | 07:50:44 |
| A12e | A12e_MAIN3_C1_LONG_074455 | LONG_OFF | a4add14d | 1 | 1 | hotmap_v2.json | LONGER_PREFILL | OFF | vllm | 32/32 | 131147/4096 | 24.6 | 166.32 | 2329/3108/3213 | 29.8/41.8/41.9 | 25.0 | 6244 | True | 07:51:36 |
| A12e | A12e_MAIN3_C1_LONG_074455 | M1_MAIN_OFF | a4add14d | 1 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 41.6 | 788.08 | 2209/3209/10134 | 58.6/67.1/72.5 | 64.2 | 10061 | True | 07:47:40 |
| A12e | A12e_MAIN3_C1_LONG_074455 | M2_MAIN_OFF | a4add14d | 1 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 41.3 | 792.84 | 2167/3081/10154 | 59.6/65.6/71.0 | 62.7 | 9817 | True | 07:48:41 |
| A12e | A12e_MAIN3_C1_LONG_074455 | M3_MAIN_OFF | a4add14d | 1 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 41.4 | 792.29 | 2172/3024/10105 | 59.2/66.1/70.2 | 63.5 | 9797 | True | 07:49:42 |
| A12e | A12e_MAIN3_C1_LONG_080026 | C1_OFF | a4add14d | 1 | 1 | hotmap_v2.json | LOW_CONCURRENCY | OFF | vllm | 16/16 | 8234/2048 | 32.4 | 63.12 | 238/327/328 | 13.9/14.0/14.0 | 14.2 | 2091 | True | 08:06:14 |
| A12e | A12e_MAIN3_C1_LONG_080026 | LONG_OFF | a4add14d | 1 | 1 | hotmap_v2.json | LONGER_PREFILL | OFF | vllm | 32/32 | 131147/4096 | 24.9 | 164.19 | 2457/3371/3482 | 29.9/41.7/43.4 | 24.9 | 6455 | True | 08:07:06 |
| A12e | A12e_MAIN3_C1_LONG_080026 | M1_MAIN_OFF | a4add14d | 1 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 40.5 | 808.90 | 2159/3139/9931 | 57.0/64.7/68.8 | 59.8 | 9692 | True | 08:03:12 |
| A12e | A12e_MAIN3_C1_LONG_080026 | M2_MAIN_OFF | a4add14d | 1 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 40.9 | 801.67 | 2154/3077/9995 | 58.1/65.2/70.6 | 61.3 | 9746 | True | 08:04:12 |
| A12e | A12e_MAIN3_C1_LONG_080026 | M3_MAIN_OFF | a4add14d | 1 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 40.9 | 800.65 | 2151/3080/10060 | 57.6/65.1/70.4 | 62.4 | 9660 | True | 08:05:13 |
| A12e | A12e_MAIN3_C1_LONG_101225 | M1_MAIN_OFF | a4add14d | 1 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 40.6 | 806.87 | 2136/3097/10017 | 58.2/64.1/69.5 | 61.5 | 9601 | False | 10:15:10 |
| A12e | A12e_MAIN3_C1_LONG_101225 | M2_MAIN_OFF | a4add14d | 1 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 40.5 | 809.54 | 2126/3060/9853 | 57.5/64.3/68.7 | 60.4 | 9579 | False | 10:16:10 |
| A12e | A12e_MAIN3_C1_LONG_101225 | M3_MAIN_OFF | a4add14d | 1 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 189/256 | 97309/24192 | 313.8 | 77.09 | 2177/3108/3957 | 58.7/66.3/69.6 | 62.7 | 9811 | False | 10:17:11 |
| A12e | A12e_MAIN3_C1_LONG_103040 | C1_OFF | a4add14d | 1 | 1 | hotmap_v2.json | LOW_CONCURRENCY | OFF | vllm | 16/16 | 8234/2048 | 32.2 | 63.54 | 237/283/316 | 13.9/13.9/14.0 | 14.2 | 2047 | True | 10:36:22 |
| A12e | A12e_MAIN3_C1_LONG_103040 | LONG_OFF | a4add14d | 1 | 1 | hotmap_v2.json | LONGER_PREFILL | OFF | vllm | 32/32 | 131147/4096 | 25.0 | 163.93 | 2342/3416/3557 | 29.8/41.5/44.0 | 25.2 | 6665 | True | 10:37:14 |
| A12e | A12e_MAIN3_C1_LONG_103040 | M1_MAIN_OFF | a4add14d | 1 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 40.7 | 804.35 | 2174/3172/9937 | 57.6/64.8/70.1 | 61.2 | 9706 | True | 10:33:21 |
| A12e | A12e_MAIN3_C1_LONG_103040 | M2_MAIN_OFF | a4add14d | 1 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 40.0 | 819.52 | 2233/2969/9806 | 56.4/63.5/67.4 | 60.3 | 9464 | True | 10:34:21 |
| A12e | A12e_MAIN3_C1_LONG_103040 | M3_MAIN_OFF | a4add14d | 1 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 40.8 | 802.99 | 2193/3010/9999 | 58.3/65.0/70.3 | 62.1 | 9636 | True | 10:35:21 |
| A12e | A12e_MAIN3_C1_LONG_121536 | C1_OFF | a4add14d | 1 | 1 | hotmap_v2.json | LOW_CONCURRENCY | OFF | vllm | 16/16 | 8234/2048 | 32.3 | 63.38 | 252/274/278 | 13.9/13.9/13.9 | 14.2 | 2040 | True | 12:21:24 |
| A12e | A12e_MAIN3_C1_LONG_121536 | LONG_OFF | a4add14d | 1 | 1 | hotmap_v2.json | LONGER_PREFILL | OFF | vllm | 32/32 | 131147/4096 | 24.7 | 165.93 | 2340/3112/3161 | 29.9/41.6/42.2 | 25.2 | 6300 | True | 12:22:16 |
| A12e | A12e_MAIN3_C1_LONG_121536 | M1_MAIN_OFF | a4add14d | 1 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 40.8 | 802.37 | 2235/3184/10022 | 58.3/65.7/70.1 | 62.1 | 9641 | True | 12:18:20 |
| A12e | A12e_MAIN3_C1_LONG_121536 | M2_MAIN_OFF | a4add14d | 1 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 41.3 | 792.70 | 2150/2982/10041 | 58.6/68.6/69.8 | 64.7 | 10117 | True | 12:19:21 |
| A12e | A12e_MAIN3_C1_LONG_121536 | M3_MAIN_OFF | a4add14d | 1 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 41.3 | 793.52 | 2086/3009/10168 | 58.8/66.7/70.8 | 63.5 | 9851 | True | 12:20:22 |
| A12e | A12e_MAIN3_C1_LONG_123103 | C1_OFF | a4add14d | 1 | 1 | hotmap_v2.json | LOW_CONCURRENCY | OFF | vllm | 16/16 | 8234/2048 | 32.3 | 63.37 | 246/308/322 | 13.9/13.9/13.9 | 14.2 | 2070 | True | 12:36:48 |
| A12e | A12e_MAIN3_C1_LONG_123103 | LONG_OFF | a4add14d | 1 | 1 | hotmap_v2.json | LONGER_PREFILL | OFF | vllm | 32/32 | 131147/4096 | 24.8 | 165.11 | 2561/3371/3397 | 29.2/41.7/42.3 | 25.1 | 6332 | True | 12:37:40 |
| A12e | A12e_MAIN3_C1_LONG_123103 | M1_MAIN_OFF | a4add14d | 1 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 41.2 | 795.62 | 2170/3215/10186 | 58.8/65.0/69.8 | 61.9 | 9729 | True | 12:33:44 |
| A12e | A12e_MAIN3_C1_LONG_123103 | M2_MAIN_OFF | a4add14d | 1 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 41.4 | 792.03 | 2174/3112/10105 | 57.7/66.5/70.4 | 62.3 | 9951 | True | 12:34:46 |
| A12e | A12e_MAIN3_C1_LONG_123103 | M3_MAIN_OFF | a4add14d | 1 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 40.5 | 808.80 | 2110/3013/9927 | 57.6/64.4/69.2 | 60.4 | 9585 | True | 12:35:47 |
| A1 | A1_MAIN3_C1_LONG_072142 | C1_OFF | a4add14d | 1 | 0 | hotmap_v2.json | LOW_CONCURRENCY | OFF | vllm | 16/16 | 8234/2048 | 32.1 | 63.72 | 235/272/279 | 13.9/14.0/14.0 | 14.3 | 2039 | True | 07:27:38 |
| A1 | A1_MAIN3_C1_LONG_072142 | LONG_OFF | a4add14d | 1 | 0 | hotmap_v2.json | LONGER_PREFILL | OFF | vllm | 32/32 | 131147/4096 | 24.8 | 165.26 | 2408/3206/3283 | 29.7/41.7/42.9 | 25.1 | 6360 | True | 07:28:30 |
| A1 | A1_MAIN3_C1_LONG_072142 | M1_MAIN_OFF | a4add14d | 1 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 43.6 | 751.81 | 2328/3201/10739 | 61.6/72.5/74.6 | 67.2 | 10373 | True | 07:24:26 |
| A1 | A1_MAIN3_C1_LONG_072142 | M2_MAIN_OFF | a4add14d | 1 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 43.6 | 751.91 | 2273/3040/10740 | 62.6/73.5/73.7 | 68.2 | 10479 | True | 07:25:32 |
| A1 | A1_MAIN3_C1_LONG_072142 | M3_MAIN_OFF | a4add14d | 1 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 42.1 | 779.16 | 2204/3182/6274 | 57.6/68.0/73.3 | 67.7 | 10067 | True | 07:26:36 |
| A1 | A1_MAIN3_C1_LONG_082356 | C1_OFF | a4add14d | 1 | 0 | hotmap_v2.json | LOW_CONCURRENCY | OFF | vllm | 16/16 | 8234/2048 | 32.2 | 63.60 | 246/257/261 | 13.9/13.9/14.0 | 14.2 | 2025 | True | 08:29:53 |
| A1 | A1_MAIN3_C1_LONG_082356 | LONG_OFF | a4add14d | 1 | 0 | hotmap_v2.json | LONGER_PREFILL | OFF | vllm | 32/32 | 131147/4096 | 24.9 | 164.30 | 2378/3245/3311 | 30.1/42.2/42.8 | 25.9 | 6349 | True | 08:30:44 |
| A1 | A1_MAIN3_C1_LONG_082356 | M1_MAIN_OFF | a4add14d | 1 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 43.9 | 746.97 | 2342/3278/10853 | 62.6/73.4/74.6 | 68.0 | 10604 | True | 08:26:42 |
| A1 | A1_MAIN3_C1_LONG_082356 | M2_MAIN_OFF | a4add14d | 1 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 43.9 | 747.25 | 2244/3136/10685 | 63.3/72.5/75.3 | 68.5 | 10595 | True | 08:27:45 |
| A1 | A1_MAIN3_C1_LONG_082356 | M3_MAIN_OFF | a4add14d | 1 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 43.8 | 747.38 | 2168/3081/10737 | 62.8/73.7/76.4 | 69.2 | 10490 | True | 08:28:49 |
| A1 | A1_MAIN3_C1_LONG_094916 | C1_OFF | a4add14d | 1 | 0 | hotmap_v2.json | LOW_CONCURRENCY | OFF | vllm | 16/16 | 8234/2048 | 32.3 | 63.34 | 246/302/307 | 13.9/14.0/14.0 | 14.3 | 2069 | True | 09:55:11 |
| A1 | A1_MAIN3_C1_LONG_094916 | LONG_OFF | a4add14d | 1 | 0 | hotmap_v2.json | LONGER_PREFILL | OFF | vllm | 32/32 | 131147/4096 | 25.1 | 163.08 | 2609/3451/3595 | 29.8/41.8/43.7 | 25.4 | 6697 | True | 09:56:03 |
| A1 | A1_MAIN3_C1_LONG_094916 | M1_MAIN_OFF | a4add14d | 1 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.0 | 745.41 | 2239/3117/10926 | 63.0/73.7/77.3 | 68.9 | 10587 | True | 09:52:03 |
| A1 | A1_MAIN3_C1_LONG_094916 | M2_MAIN_OFF | a4add14d | 1 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 41.7 | 785.00 | 2154/3045/6332 | 56.4/67.2/72.3 | 62.8 | 10023 | True | 09:53:07 |
| A1 | A1_MAIN3_C1_LONG_094916 | M3_MAIN_OFF | a4add14d | 1 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 42.2 | 776.14 | 2171/3041/10338 | 60.1/68.3/72.8 | 63.7 | 10042 | True | 09:54:09 |
| A1 | A1_MAIN3_C1_LONG_105333 | C1_OFF | a4add14d | 1 | 0 | hotmap_v2.json | LOW_CONCURRENCY | OFF | vllm | 16/16 | 8234/2048 | 32.4 | 63.19 | 245/328/329 | 13.9/14.0/14.0 | 14.2 | 2091 | True | 10:59:33 |
| A1 | A1_MAIN3_C1_LONG_105333 | LONG_OFF | a4add14d | 1 | 0 | hotmap_v2.json | LONGER_PREFILL | OFF | vllm | 32/32 | 131147/4096 | 24.9 | 164.40 | 2476/3311/3362 | 29.8/42.2/43.3 | 25.2 | 6377 | True | 11:00:25 |
| A1 | A1_MAIN3_C1_LONG_105333 | M1_MAIN_OFF | a4add14d | 1 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.3 | 740.24 | 2322/3145/10878 | 62.6/72.2/75.3 | 68.1 | 10597 | True | 10:56:19 |
| A1 | A1_MAIN3_C1_LONG_105333 | M2_MAIN_OFF | a4add14d | 1 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 43.5 | 753.93 | 2186/2995/10735 | 63.2/74.0/75.6 | 68.0 | 10369 | True | 10:57:23 |
| A1 | A1_MAIN3_C1_LONG_105333 | M3_MAIN_OFF | a4add14d | 1 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.1 | 743.20 | 2246/3039/10882 | 63.6/74.4/75.7 | 68.8 | 10485 | True | 10:58:26 |
| A1 | A1_MAIN3_C1_LONG_114955 | M1_MAIN_OFF | a4add14d | 1 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 63/256 | 32409/8064 | 301.0 | 26.79 | 2142/3200/3201 | 64.7/70.2/72.3 | 65.0 | 10372 | False | 11:52:41 |
| A1 | A1_MAIN3_C1_LONG_232400 | C1_OFF | 194f3064 | 1 | 0 | hotmap_v2.json | LOW_CONCURRENCY | OFF | vllm | 16/16 | 8234/2048 | 32.2 | 63.51 | 243/276/297 | 13.9/14.0/14.0 | 14.3 | 2040 | True | 23:29:54 |
| A1 | A1_MAIN3_C1_LONG_232400 | LONG_OFF | 194f3064 | 1 | 0 | hotmap_v2.json | LONGER_PREFILL | OFF | vllm | 32/32 | 131147/4096 | 24.7 | 165.99 | 2354/3156/3168 | 29.8/41.8/42.3 | 25.5 | 6239 | True | 23:30:46 |
| A1 | A1_MAIN3_C1_LONG_232400 | M1_MAIN_OFF | 194f3064 | 1 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.3 | 739.03 | 2176/3216/10885 | 64.4/72.7/76.9 | 69.5 | 10522 | True | 23:26:42 |
| A1 | A1_MAIN3_C1_LONG_232400 | M2_MAIN_OFF | 194f3064 | 1 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 43.4 | 755.44 | 2219/3084/10695 | 61.8/72.6/73.7 | 67.0 | 10381 | True | 23:27:46 |
| A1 | A1_MAIN3_C1_LONG_232400 | M3_MAIN_OFF | 194f3064 | 1 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.0 | 744.34 | 2184/3072/10801 | 63.6/74.3/77.2 | 69.8 | 10578 | True | 23:28:50 |
| A1ae | A1ae_MAIN3_C1_LONG_070438 | C1_OFF | a4add14d | 1 | 0 | hotmap_v2.json | LOW_CONCURRENCY | OFF | vllm | 16/16 | 8234/2048 | 32.5 | 63.06 | 249/282/299 | 13.9/14.1/14.6 | 14.3 | 2082 | True | 07:10:32 |
| A1ae | A1ae_MAIN3_C1_LONG_070438 | LONG_OFF | a4add14d | 1 | 0 | hotmap_v2.json | LONGER_PREFILL | OFF | vllm | 32/32 | 131147/4096 | 24.7 | 165.62 | 2376/3166/3238 | 29.9/41.8/42.0 | 25.5 | 6244 | True | 07:11:24 |
| A1ae | A1ae_MAIN3_C1_LONG_070438 | M1_MAIN_OFF | a4add14d | 1 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 42.6 | 769.71 | 2305/3237/10473 | 61.0/68.3/73.2 | 64.5 | 10146 | True | 07:07:22 |
| A1ae | A1ae_MAIN3_C1_LONG_070438 | M2_MAIN_OFF | a4add14d | 1 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 42.4 | 772.44 | 2200/3138/10290 | 59.2/69.9/71.8 | 64.9 | 10186 | True | 07:08:25 |
| A1ae | A1ae_MAIN3_C1_LONG_070438 | M3_MAIN_OFF | a4add14d | 1 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 42.7 | 767.94 | 2156/3056/10457 | 60.2/69.6/75.1 | 66.2 | 10317 | True | 07:09:27 |
| A1ae | A1ae_MAIN3_C1_LONG_083930 | M1_MAIN_OFF | a4add14d | 1 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 189/256 | 97303/24192 | 315.6 | 76.65 | 2263/3423/4300 | 62.8/68.9/73.9 | 64.7 | 10326 | False | 08:42:15 |
| A1ae | A1ae_MAIN3_C1_LONG_093058 | M1_MAIN_OFF | a4add14d | 1 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 42.7 | 766.85 | 2260/3312/10430 | 60.3/71.2/71.5 | 65.3 | 10460 | False | 09:33:38 |
| A1ae | A1ae_MAIN3_C1_LONG_093058 | M2_MAIN_OFF | a4add14d | 1 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 40.4 | 811.54 | 2144/3063/6031 | 55.1/65.5/67.7 | 61.4 | 9718 | False | 09:34:41 |
| A1ae | A1ae_MAIN3_C1_LONG_093058 | M3_MAIN_OFF | a4add14d | 1 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 189/256 | 97303/24192 | 321.4 | 75.27 | 2160/3067/3948 | 59.7/66.4/70.7 | 62.9 | 9877 | False | 09:35:41 |
| A1ae | A1ae_MAIN3_C1_LONG_110916 | M1_MAIN_OFF | a4add14d | 1 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 43.0 | 761.83 | 2245/3287/10479 | 61.2/71.5/72.8 | 67.7 | 10638 | False | 11:12:02 |
| A1ae | A1ae_MAIN3_C1_LONG_110916 | M2_MAIN_OFF | a4add14d | 1 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 63/256 | 32405/8064 | 301.6 | 26.74 | 2100/3032/3033 | 62.4/67.7/69.8 | 65.8 | 10036 | False | 11:13:05 |
| A1ae | A1ae_MAIN3_C1_LONG_113424 | C1_OFF | a4add14d | 1 | 0 | hotmap_v2.json | LOW_CONCURRENCY | OFF | vllm | 16/16 | 8234/2048 | 32.2 | 63.51 | 235/270/300 | 13.9/14.1/14.6 | 14.3 | 2082 | True | 11:40:17 |
| A1ae | A1ae_MAIN3_C1_LONG_113424 | LONG_OFF | a4add14d | 1 | 0 | hotmap_v2.json | LONGER_PREFILL | OFF | vllm | 32/32 | 131147/4096 | 24.9 | 164.23 | 2466/3326/3392 | 29.7/41.8/43.1 | 25.0 | 6394 | True | 11:41:09 |
| A1ae | A1ae_MAIN3_C1_LONG_113424 | M1_MAIN_OFF | a4add14d | 1 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 42.4 | 772.11 | 2229/3099/10345 | 60.8/72.2/73.1 | 65.5 | 10099 | True | 11:37:09 |
| A1ae | A1ae_MAIN3_C1_LONG_113424 | M2_MAIN_OFF | a4add14d | 1 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 42.3 | 775.28 | 2123/3055/10396 | 61.0/68.1/73.3 | 65.5 | 10027 | True | 11:38:12 |
| A1ae | A1ae_MAIN3_C1_LONG_113424 | M3_MAIN_OFF | a4add14d | 1 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 42.3 | 773.81 | 2151/3087/10363 | 60.6/67.8/72.6 | 64.0 | 10082 | True | 11:39:14 |
| A1ae | A1ae_MAIN3_C1_LONG_124640 | C1_OFF | a4add14d | 1 | 0 | hotmap_v2.json | LOW_CONCURRENCY | OFF | vllm | 16/16 | 8234/2048 | 32.3 | 63.40 | 236/309/323 | 13.9/14.0/14.0 | 14.3 | 2081 | True | 12:52:35 |
| A1ae | A1ae_MAIN3_C1_LONG_124640 | LONG_OFF | a4add14d | 1 | 0 | hotmap_v2.json | LONGER_PREFILL | OFF | vllm | 32/32 | 131147/4096 | 24.8 | 165.08 | 2354/3148/3154 | 30.4/42.4/42.4 | 26.1 | 6239 | True | 12:53:27 |
| A1ae | A1ae_MAIN3_C1_LONG_124640 | M1_MAIN_OFF | a4add14d | 1 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 43.4 | 754.77 | 2298/3367/10414 | 62.2/73.1/77.4 | 68.6 | 10761 | True | 12:49:26 |
| A1ae | A1ae_MAIN3_C1_LONG_124640 | M2_MAIN_OFF | a4add14d | 1 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 42.7 | 766.86 | 2191/3077/10456 | 61.0/68.9/72.7 | 64.7 | 10161 | True | 12:50:29 |
| A1ae | A1ae_MAIN3_C1_LONG_124640 | M3_MAIN_OFF | a4add14d | 1 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 41.9 | 782.63 | 2143/3030/10251 | 60.6/70.2/72.6 | 64.3 | 9992 | True | 12:51:32 |
| A1ae | A1ae_MAIN3_C1_LONG_234516 | M1_MAIN_OFF | db497d74 | 1 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 42.6 | 769.93 | 2204/3141/10376 | 60.6/68.6/74.1 | 64.0 | 10240 | False | 23:47:58 |
| A1ae | A1ae_MAIN3_C1_LONG_234516 | M2_MAIN_OFF | db497d74 | 1 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 189/256 | 97306/24192 | 318.5 | 75.96 | 2124/3043/3967 | 61.7/69.3/72.6 | 65.1 | 10218 | False | 23:49:01 |
| A1e | A1e_MAIN3_C1_LONG_071226 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 42.5 | 770.14 | 2163/3122/10466 | 62.0/68.3/73.0 | 64.4 | 10106 | False | 07:15:10 |
| A1e | A1e_MAIN3_C1_LONG_071226 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 126/256 | 64867/16128 | 304.4 | 52.99 | 2120/3049/3051 | 62.3/68.5/71.9 | 65.7 | 10146 | False | 07:16:13 |
| A1e | A1e_MAIN3_C1_LONG_083144 | C1_OFF | a4add14d | 0 | 0 | hotmap_v2.json | LOW_CONCURRENCY | OFF | vllm | 16/16 | 8234/2048 | 32.4 | 63.30 | 245/302/304 | 13.9/14.1/14.5 | 14.2 | 2076 | True | 08:37:36 |
| A1e | A1e_MAIN3_C1_LONG_083144 | LONG_OFF | a4add14d | 0 | 0 | hotmap_v2.json | LONGER_PREFILL | OFF | vllm | 32/32 | 131147/4096 | 24.8 | 165.45 | 2368/3168/3169 | 29.9/41.9/42.3 | 25.4 | 6227 | True | 08:38:28 |
| A1e | A1e_MAIN3_C1_LONG_083144 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 42.6 | 769.16 | 2165/3137/10376 | 60.3/69.5/71.8 | 64.3 | 10229 | True | 08:34:28 |
| A1e | A1e_MAIN3_C1_LONG_083144 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 42.8 | 765.41 | 2405/3189/10436 | 60.0/70.9/73.2 | 65.6 | 10274 | True | 08:35:31 |
| A1e | A1e_MAIN3_C1_LONG_083144 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 42.4 | 773.71 | 2143/3048/10367 | 60.9/68.1/72.8 | 64.8 | 10053 | True | 08:36:34 |
| A1e | A1e_MAIN3_C1_LONG_094127 | C1_OFF | a4add14d | 0 | 0 | hotmap_v2.json | LOW_CONCURRENCY | OFF | vllm | 16/16 | 8234/2048 | 32.4 | 63.18 | 242/338/355 | 13.9/14.0/14.0 | 14.2 | 2100 | True | 09:47:22 |
| A1e | A1e_MAIN3_C1_LONG_094127 | LONG_OFF | a4add14d | 0 | 0 | hotmap_v2.json | LONGER_PREFILL | OFF | vllm | 32/32 | 131147/4096 | 25.0 | 163.79 | 2543/3282/3428 | 29.8/41.9/43.1 | 25.2 | 6489 | True | 09:48:14 |
| A1e | A1e_MAIN3_C1_LONG_094127 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 43.3 | 757.15 | 2246/3272/10679 | 62.7/69.7/75.1 | 66.2 | 10394 | True | 09:44:13 |
| A1e | A1e_MAIN3_C1_LONG_094127 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 42.6 | 768.68 | 2171/3094/10416 | 60.8/70.0/73.2 | 64.9 | 10207 | True | 09:45:17 |
| A1e | A1e_MAIN3_C1_LONG_094127 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 42.5 | 770.43 | 2135/3016/10384 | 61.1/71.8/73.3 | 66.4 | 10184 | True | 09:46:19 |
| A1e | A1e_MAIN3_C1_LONG_110127 | C1_OFF | a4add14d | 0 | 0 | hotmap_v2.json | LOW_CONCURRENCY | OFF | vllm | 16/16 | 8234/2048 | 32.2 | 63.52 | 240/289/317 | 13.9/14.0/14.0 | 14.3 | 2056 | True | 11:07:21 |
| A1e | A1e_MAIN3_C1_LONG_110127 | LONG_OFF | a4add14d | 0 | 0 | hotmap_v2.json | LONGER_PREFILL | OFF | vllm | 32/32 | 131147/4096 | 24.8 | 165.18 | 2442/3262/3301 | 29.7/41.8/42.6 | 25.3 | 6287 | True | 11:08:13 |
| A1e | A1e_MAIN3_C1_LONG_110127 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 43.1 | 760.47 | 2301/3230/10479 | 60.1/71.7/72.4 | 66.0 | 10563 | True | 11:04:13 |
| A1e | A1e_MAIN3_C1_LONG_110127 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 42.5 | 771.17 | 2189/3079/10393 | 61.0/68.4/73.6 | 65.0 | 10064 | True | 11:05:17 |
| A1e | A1e_MAIN3_C1_LONG_110127 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 42.4 | 773.13 | 2207/3108/10293 | 59.8/70.5/72.9 | 65.8 | 10264 | True | 11:06:19 |
| A1e | A1e_MAIN3_C1_LONG_114208 | C1_OFF | a4add14d | 0 | 0 | hotmap_v2.json | LOW_CONCURRENCY | OFF | vllm | 16/16 | 8234/2048 | 32.4 | 63.29 | 246/312/325 | 13.9/13.9/14.0 | 14.3 | 2075 | True | 11:48:01 |
| A1e | A1e_MAIN3_C1_LONG_114208 | LONG_OFF | a4add14d | 0 | 0 | hotmap_v2.json | LONGER_PREFILL | OFF | vllm | 32/32 | 131147/4096 | 24.8 | 165.41 | 2369/3158/3225 | 29.9/41.8/42.0 | 25.5 | 6257 | True | 11:48:53 |
| A1e | A1e_MAIN3_C1_LONG_114208 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 42.8 | 765.82 | 2208/3196/10589 | 61.4/68.0/72.6 | 64.4 | 10068 | True | 11:44:53 |
| A1e | A1e_MAIN3_C1_LONG_114208 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 42.5 | 771.25 | 2164/3102/10391 | 60.5/68.3/73.7 | 65.0 | 10122 | True | 11:45:55 |
| A1e | A1e_MAIN3_C1_LONG_114208 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 42.8 | 766.27 | 2126/3021/10498 | 61.5/69.4/73.9 | 65.6 | 10180 | True | 11:46:58 |
| A1e | A1e_MAIN3_C1_LONG_233730 | C1_OFF | db497d74 | 0 | 0 | hotmap_v2.json | LOW_CONCURRENCY | OFF | vllm | 16/16 | 8234/2048 | 32.4 | 63.12 | 250/311/311 | 13.9/14.0/14.0 | 14.3 | 2078 | True | 23:43:23 |
| A1e | A1e_MAIN3_C1_LONG_233730 | LONG_OFF | db497d74 | 0 | 0 | hotmap_v2.json | LONGER_PREFILL | OFF | vllm | 32/32 | 131147/4096 | 24.7 | 165.95 | 2414/3177/3217 | 29.8/41.8/41.9 | 25.1 | 6212 | True | 23:44:15 |
| A1e | A1e_MAIN3_C1_LONG_233730 | M1_MAIN_OFF | db497d74 | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 43.4 | 755.36 | 2301/3261/10616 | 61.6/69.5/73.9 | 65.5 | 10375 | True | 23:40:12 |
| A1e | A1e_MAIN3_C1_LONG_233730 | M2_MAIN_OFF | db497d74 | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 42.7 | 767.27 | 2288/3031/10404 | 59.8/70.5/73.0 | 66.3 | 10302 | True | 23:41:16 |
| A1e | A1e_MAIN3_C1_LONG_233730 | M3_MAIN_OFF | db497d74 | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 43.4 | 754.59 | 2283/3101/10608 | 61.1/71.9/76.1 | 67.3 | 10401 | True | 23:42:19 |
| A2B | A2B_CONFIRM_132439 | C1_OFF | a4add14d | 0 | 1 | hotmap.json | LOW_CONCURRENCY | OFF | vllm | 16/16 | 8234/2048 | 32.5 | 63.01 | 249/267/297 | 14.0/14.3/14.4 | 15.0 | 2100 | True | 13:30:50 |
| A2B | A2B_CONFIRM_132439 | IND1_OFF | a4add14d | 0 | 1 | hotmap.json | CONFIRM | OFF | vllm | 128/128 | 64376/16384 | 23.0 | 713.49 | 2172/3027/8488 | 63.9/71.3/75.0 | 68.3 | 10538 | True | 13:32:28 |
| A2B | A2B_CONFIRM_132439 | IND2_OFF | a4add14d | 0 | 1 | hotmap.json | CONFIRM | OFF | vllm | 128/128 | 64376/16384 | 22.8 | 717.08 | 2176/2994/8421 | 62.8/71.5/74.1 | 68.3 | 10516 | True | 13:33:11 |
| A2B | A2B_CONFIRM_132439 | LONG_OFF | a4add14d | 0 | 1 | hotmap.json | LONGER_PREFILL | OFF | vllm | 32/32 | 131147/4096 | 24.8 | 165.19 | 2378/3171/3176 | 30.1/42.2/42.4 | 27.3 | 6243 | True | 13:31:44 |
| A2B | A2B_CONFIRM_132439 | M1_MAIN_OFF | a4add14d | 0 | 1 | hotmap.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 45.9 | 713.45 | 2299/3285/11186 | 66.1/77.1/77.9 | 70.7 | 11007 | True | 13:27:30 |
| A2B | A2B_CONFIRM_132439 | M2_MAIN_OFF | a4add14d | 0 | 1 | hotmap.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 47.0 | 696.49 | 2202/3147/11624 | 69.4/80.5/81.3 | 75.9 | 11243 | True | 13:28:36 |
| A2B | A2B_CONFIRM_132439 | M3_MAIN_OFF | a4add14d | 0 | 1 | hotmap.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 46.9 | 699.18 | 2374/3172/11641 | 68.8/80.1/81.2 | 73.6 | 11218 | True | 13:29:43 |
| A2B | A2B_CONFIRM_140009 | C1_OFF | a4add14d | 0 | 1 | hotmap.json | LOW_CONCURRENCY | OFF | vllm | 16/16 | 8234/2048 | 32.6 | 62.85 | 242/330/337 | 14.0/14.3/14.4 | 14.9 | 2111 | True | 14:06:22 |
| A2B | A2B_CONFIRM_140009 | IND1_OFF | a4add14d | 0 | 1 | hotmap.json | CONFIRM | OFF | vllm | 128/128 | 64376/16384 | 22.7 | 722.97 | 2139/2979/8399 | 63.4/70.0/73.4 | 67.9 | 10351 | True | 14:07:59 |
| A2B | A2B_CONFIRM_140009 | IND2_OFF | a4add14d | 0 | 1 | hotmap.json | CONFIRM | OFF | vllm | 128/128 | 64376/16384 | 23.1 | 708.17 | 2147/3001/8533 | 64.4/72.3/75.5 | 70.6 | 10639 | True | 14:08:41 |
| A2B | A2B_CONFIRM_140009 | LONG_OFF | a4add14d | 0 | 1 | hotmap.json | LONGER_PREFILL | OFF | vllm | 32/32 | 131147/4096 | 25.0 | 163.79 | 2385/3175/3243 | 30.2/42.2/42.4 | 27.4 | 6324 | True | 14:07:14 |
| A2B | A2B_CONFIRM_140009 | M1_MAIN_OFF | a4add14d | 0 | 1 | hotmap.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 47.0 | 697.20 | 2356/3239/11525 | 68.5/77.0/79.7 | 74.4 | 11298 | True | 14:03:01 |
| A2B | A2B_CONFIRM_140009 | M2_MAIN_OFF | a4add14d | 0 | 1 | hotmap.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 46.7 | 702.28 | 2247/3153/11572 | 67.4/79.3/82.2 | 75.3 | 11268 | True | 14:04:08 |
| A2B | A2B_CONFIRM_140009 | M3_MAIN_OFF | a4add14d | 0 | 1 | hotmap.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 47.1 | 695.24 | 2338/3102/11705 | 68.6/79.1/83.9 | 79.7 | 11494 | True | 14:05:15 |
| A2B | A2B_CONFIRM_143724 | C1_OFF | a4add14d | 0 | 1 | hotmap.json | LOW_CONCURRENCY | OFF | vllm | 16/16 | 8234/2048 | 32.4 | 63.24 | 243/274/308 | 14.0/14.1/14.3 | 14.6 | 2080 | True | 14:43:32 |
| A2B | A2B_CONFIRM_143724 | IND1_OFF | a4add14d | 0 | 1 | hotmap.json | CONFIRM | OFF | vllm | 128/128 | 64376/16384 | 22.2 | 737.84 | 2166/3028/8390 | 61.1/67.8/73.4 | 65.6 | 10069 | True | 14:45:09 |
| A2B | A2B_CONFIRM_143724 | IND2_OFF | a4add14d | 0 | 1 | hotmap.json | CONFIRM | OFF | vllm | 128/128 | 64376/16384 | 21.7 | 755.52 | 2133/2985/8151 | 59.7/65.9/71.4 | 63.7 | 9798 | True | 14:45:52 |
| A2B | A2B_CONFIRM_143724 | LONG_OFF | a4add14d | 0 | 1 | hotmap.json | LONGER_PREFILL | OFF | vllm | 32/32 | 131147/4096 | 24.9 | 164.40 | 2379/3168/3168 | 30.2/42.1/42.9 | 27.9 | 6352 | True | 14:44:24 |
| A2B | A2B_CONFIRM_143724 | M1_MAIN_OFF | a4add14d | 0 | 1 | hotmap.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 45.7 | 717.36 | 2375/3364/11163 | 65.0/75.6/77.9 | 69.6 | 11047 | True | 14:40:18 |
| A2B | A2B_CONFIRM_143724 | M2_MAIN_OFF | a4add14d | 0 | 1 | hotmap.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 42.8 | 764.89 | 2242/3226/6464 | 59.8/72.3/74.2 | 65.6 | 10243 | True | 14:41:24 |
| A2B | A2B_CONFIRM_143724 | M3_MAIN_OFF | a4add14d | 0 | 1 | hotmap.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 45.1 | 727.13 | 2247/3143/11039 | 65.6/74.4/80.0 | 71.7 | 10959 | True | 14:42:27 |
| A2 | A2_MAIN3_C1_LONG_072929 | C1_OFF | a4add14d | 0 | 1 | hotmap_v2.json | LOW_CONCURRENCY | OFF | vllm | 16/16 | 8234/2048 | 32.5 | 63.10 | 247/329/331 | 13.9/13.9/13.9 | 14.3 | 2094 | True | 07:35:21 |
| A2 | A2_MAIN3_C1_LONG_072929 | LONG_OFF | a4add14d | 0 | 1 | hotmap_v2.json | LONGER_PREFILL | OFF | vllm | 32/32 | 131147/4096 | 25.1 | 163.23 | 2348/3437/3549 | 30.1/42.0/44.0 | 25.6 | 6545 | True | 07:36:13 |
| A2 | A2_MAIN3_C1_LONG_072929 | M1_MAIN_OFF | a4add14d | 0 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 42.6 | 768.68 | 2278/3124/10528 | 61.3/71.2/73.2 | 66.0 | 10089 | True | 07:32:14 |
| A2 | A2_MAIN3_C1_LONG_072929 | M2_MAIN_OFF | a4add14d | 0 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 42.4 | 772.86 | 2208/3149/10224 | 59.0/69.4/71.4 | 65.4 | 10238 | True | 07:33:16 |
| A2 | A2_MAIN3_C1_LONG_072929 | M3_MAIN_OFF | a4add14d | 0 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 42.2 | 777.32 | 2135/2988/10331 | 61.3/69.6/73.8 | 65.6 | 10117 | True | 07:34:19 |
| A2 | A2_MAIN3_C1_LONG_081613 | C1_OFF | a4add14d | 0 | 1 | hotmap_v2.json | LOW_CONCURRENCY | OFF | vllm | 16/16 | 8234/2048 | 32.3 | 63.49 | 246/279/319 | 13.9/13.9/13.9 | 14.3 | 2043 | True | 08:22:03 |
| A2 | A2_MAIN3_C1_LONG_081613 | LONG_OFF | a4add14d | 0 | 1 | hotmap_v2.json | LONGER_PREFILL | OFF | vllm | 32/32 | 131147/4096 | 24.5 | 167.05 | 2345/3130/3136 | 29.8/41.6/41.7 | 25.4 | 6154 | True | 08:22:54 |
| A2 | A2_MAIN3_C1_LONG_081613 | M1_MAIN_OFF | a4add14d | 0 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 42.3 | 774.83 | 2161/3207/10370 | 60.7/68.4/72.1 | 64.8 | 10147 | True | 08:18:58 |
| A2 | A2_MAIN3_C1_LONG_081613 | M2_MAIN_OFF | a4add14d | 0 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 42.0 | 780.10 | 2228/2978/10230 | 60.3/71.0/72.7 | 66.1 | 10188 | True | 08:20:00 |
| A2 | A2_MAIN3_C1_LONG_081613 | M3_MAIN_OFF | a4add14d | 0 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 40.4 | 810.63 | 2078/3021/6008 | 54.7/66.5/71.1 | 63.3 | 9854 | True | 08:21:02 |
| A2 | A2_MAIN3_C1_LONG_095702 | C1_OFF | a4add14d | 0 | 1 | hotmap_v2.json | LOW_CONCURRENCY | OFF | vllm | 16/16 | 8234/2048 | 32.2 | 63.64 | 243/267/274 | 13.9/14.0/14.0 | 14.2 | 2033 | True | 10:02:52 |
| A2 | A2_MAIN3_C1_LONG_095702 | LONG_OFF | a4add14d | 0 | 1 | hotmap_v2.json | LONGER_PREFILL | OFF | vllm | 32/32 | 131147/4096 | 24.9 | 164.50 | 2318/3515/3627 | 29.7/41.4/43.7 | 25.1 | 6647 | True | 10:03:43 |
| A2 | A2_MAIN3_C1_LONG_095702 | M1_MAIN_OFF | a4add14d | 0 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 41.9 | 782.21 | 2268/3197/10397 | 59.8/70.6/71.1 | 63.8 | 9957 | True | 09:59:46 |
| A2 | A2_MAIN3_C1_LONG_095702 | M2_MAIN_OFF | a4add14d | 0 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 41.8 | 783.70 | 2169/3115/10187 | 59.5/70.1/71.7 | 64.0 | 10049 | True | 10:00:48 |
| A2 | A2_MAIN3_C1_LONG_095702 | M3_MAIN_OFF | a4add14d | 0 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 41.4 | 792.01 | 2170/3033/10109 | 58.4/69.0/70.0 | 64.0 | 9966 | True | 10:01:50 |
| A2 | A2_MAIN3_C1_LONG_104551 | C1_OFF | a4add14d | 0 | 1 | hotmap_v2.json | LOW_CONCURRENCY | OFF | vllm | 16/16 | 8234/2048 | 32.4 | 63.22 | 251/307/310 | 13.9/14.0/14.0 | 14.3 | 2071 | True | 10:51:43 |
| A2 | A2_MAIN3_C1_LONG_104551 | LONG_OFF | a4add14d | 0 | 1 | hotmap_v2.json | LONGER_PREFILL | OFF | vllm | 32/32 | 131147/4096 | 24.6 | 166.47 | 2342/3123/3126 | 29.9/41.8/42.0 | 25.3 | 6175 | True | 10:52:35 |
| A2 | A2_MAIN3_C1_LONG_104551 | M1_MAIN_OFF | a4add14d | 0 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 42.7 | 766.72 | 2269/3297/10559 | 61.8/67.8/73.0 | 64.9 | 10153 | True | 10:48:35 |
| A2 | A2_MAIN3_C1_LONG_104551 | M2_MAIN_OFF | a4add14d | 0 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 41.7 | 786.63 | 2204/2997/10267 | 59.2/69.7/69.9 | 63.4 | 9842 | True | 10:49:38 |
| A2 | A2_MAIN3_C1_LONG_104551 | M3_MAIN_OFF | a4add14d | 0 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 42.2 | 775.85 | 2157/3030/10310 | 60.6/69.3/71.9 | 65.6 | 10223 | True | 10:50:40 |
| A2 | A2_MAIN3_C1_LONG_115806 | M1_MAIN_OFF | a4add14d | 0 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 42.7 | 766.57 | 2182/3173/10429 | 60.9/69.1/73.1 | 65.9 | 10285 | False | 12:00:52 |
| A2 | A2_MAIN3_C1_LONG_115806 | M2_MAIN_OFF | a4add14d | 0 | 1 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 126/256 | 64866/16128 | 332.0 | 48.58 | 2103/3021/3023 | 60.7/67.0/72.4 | 63.4 | 9947 | False | 12:01:55 |
| B | B_CONFIRM_130403 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 49.3 | 664.28 | 2465/3330/12222 | 72.4/83.9/84.1 | 77.4 | 11746 | False | 13:06:58 |
| B | B_CONFIRM_130403 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 48.8 | 671.26 | 2409/3149/12174 | 72.7/85.0/85.5 | 78.7 | 11896 | False | 13:08:08 |
| B | B_CONFIRM_130403 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap.json | MAIN_SHORT | OFF | vllm | 189/256 | 97309/24192 | 319.8 | 75.65 | 2430/3216/4320 | 72.3/83.5/83.5 | 78.0 | 11715 | False | 13:09:18 |
| B | B_CONFIRM_134337 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap.json | MAIN_SHORT | OFF | vllm | 126/256 | 64855/16128 | 304.6 | 52.96 | 2472/3204/3207 | 72.1/83.3/83.3 | 76.5 | 11704 | False | 13:46:31 |
| B | B_CONFIRM_141906 | C1_OFF | a4add14d | 0 | 0 | hotmap.json | LOW_CONCURRENCY | OFF | vllm | 16/16 | 8234/2048 | 32.5 | 63.01 | 251/289/330 | 13.9/14.2/14.8 | 14.4 | 2114 | True | 14:25:24 |
| B | B_CONFIRM_141906 | IND1_OFF | a4add14d | 0 | 0 | hotmap.json | CONFIRM | OFF | vllm | 128/128 | 64376/16384 | 23.3 | 703.34 | 2298/2987/8777 | 65.6/74.9/77.0 | 70.3 | 10660 | True | 14:27:02 |
| B | B_CONFIRM_141906 | IND2_OFF | a4add14d | 0 | 0 | hotmap.json | CONFIRM | OFF | vllm | 128/128 | 64376/16384 | 22.8 | 717.54 | 2236/3040/8384 | 61.9/73.0/73.0 | 70.1 | 10566 | True | 14:27:46 |
| B | B_CONFIRM_141906 | LONG_OFF | a4add14d | 0 | 0 | hotmap.json | LONGER_PREFILL | OFF | vllm | 32/32 | 131147/4096 | 25.3 | 162.03 | 2411/3208/3210 | 30.3/42.9/43.6 | 28.6 | 6422 | True | 14:26:17 |
| B | B_CONFIRM_141906 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 49.1 | 667.04 | 2418/3206/12169 | 72.9/83.1/84.9 | 77.6 | 11738 | True | 14:22:03 |
| B | B_CONFIRM_141906 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 45.7 | 717.68 | 2336/3262/6697 | 66.1/77.1/82.3 | 73.4 | 11247 | True | 14:23:12 |
| B | B_CONFIRM_141906 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 46.4 | 705.90 | 2329/3142/11263 | 67.3/78.6/78.7 | 74.9 | 11410 | True | 14:24:18 |
| B_DEC1_SEL | B_DEC1_SEL_SELECT_085223 | SEL1_OFF | a4add14d | 0 | 0 | hotmap.json | SELECT | OFF | vllm | 128/128 | 64617/16384 | 32.3 | 507.54 | 4488/6947/12988 | 82.0/97.9/109.8 | 74.5 | 15161 | True | 08:55:12 |
| B_DEC1_SEL | B_DEC1_SEL_SELECT_085223 | SEL2_OFF | a4add14d | 0 | 0 | hotmap.json | SELECT | OFF | vllm | 128/128 | 64617/16384 | 32.3 | 507.99 | 4673/6787/12944 | 80.4/98.1/107.9 | 75.0 | 15097 | True | 08:56:04 |
| B_DEC1_SEL | B_DEC1_SEL_SELECT_090615 | SEL1_OFF | a4add14d | 0 | 0 | hotmap.json | SELECT | OFF | vllm | 128/128 | 64617/16384 | 32.7 | 500.80 | 5146/6936/13231 | 79.1/107.0/107.0 | 75.0 | 15245 | True | 09:08:59 |
| B_DEC1_SEL | B_DEC1_SEL_SELECT_090615 | SEL2_OFF | a4add14d | 0 | 0 | hotmap.json | SELECT | OFF | vllm | 128/128 | 64617/16384 | 32.8 | 499.95 | 4958/6820/13147 | 80.4/107.9/107.9 | 78.5 | 15378 | True | 09:09:52 |
| B_H0_SEL | B_H0_SEL_SELECT_084754 | SEL1_OFF | a4add14d | 0 | 0 | hotmap_v2.json | SELECT | OFF | vllm | 128/128 | 64617/16384 | 25.8 | 634.20 | 2149/3041/9618 | 76.2/81.5/86.9 | 81.4 | 11846 | True | 08:50:38 |
| B_H0_SEL | B_H0_SEL_SELECT_084754 | SEL2_OFF | a4add14d | 0 | 0 | hotmap_v2.json | SELECT | OFF | vllm | 128/128 | 64617/16384 | 25.0 | 656.52 | 2275/2978/9201 | 71.7/79.6/79.6 | 81.4 | 11543 | True | 08:51:24 |
| B_H0_SEL | B_H0_SEL_SELECT_090142 | SEL1_OFF | a4add14d | 0 | 0 | hotmap_v2.json | SELECT | OFF | vllm | 128/128 | 64617/16384 | 25.7 | 638.08 | 2266/3055/9585 | 74.2/84.7/84.7 | 82.1 | 11752 | True | 09:04:27 |
| B_H0_SEL | B_H0_SEL_SELECT_090142 | SEL2_OFF | a4add14d | 0 | 0 | hotmap_v2.json | SELECT | OFF | vllm | 128/128 | 64617/16384 | 25.1 | 652.18 | 2279/2907/9454 | 73.0/83.8/83.8 | 79.7 | 11581 | True | 09:05:13 |
| B_MIX1_SEL | B_MIX1_SEL_SELECT_085714 | SEL1_OFF | a4add14d | 0 | 0 | hotmap.json | SELECT | OFF | vllm | 128/128 | 64617/16384 | 24.2 | 677.54 | 2224/3094/9014 | 68.3/75.2/79.7 | 73.0 | 11067 | True | 08:59:59 |
| B_MIX1_SEL | B_MIX1_SEL_SELECT_085714 | SEL2_OFF | a4add14d | 0 | 0 | hotmap.json | SELECT | OFF | vllm | 128/128 | 64617/16384 | 24.0 | 683.87 | 2234/3199/8916 | 67.1/78.3/78.3 | 74.2 | 11025 | True | 09:00:43 |
| B_MIX1_SEL | B_MIX1_SEL_SELECT_091102 | SEL1_OFF | a4add14d | 0 | 0 | hotmap.json | SELECT | OFF | vllm | 128/128 | 64617/16384 | 24.0 | 683.53 | 2434/3090/8852 | 65.0/76.1/76.1 | 71.8 | 11019 | True | 09:13:42 |
| B_MIX1_SEL | B_MIX1_SEL_SELECT_091102 | SEL2_OFF | a4add14d | 0 | 0 | hotmap.json | SELECT | OFF | vllm | 128/128 | 64617/16384 | 23.9 | 684.44 | 2190/3033/8921 | 67.5/74.4/78.9 | 74.0 | 10930 | True | 09:14:26 |
| CALIB_R0 | CALIB_R0_CALIB_000347 | CAL1_CORR | db497d74 | 0 | 0 | hotmap_v2.json | CALIB | CORR | vllmw | 128/128 | 64584/16384 | 26.5 | 618.95 | 2495/3456/9840 | 73.7/84.3/84.3 | 81.8 | 12155 | True | 00:06:41 |
| CALIB_R0 | CALIB_R0_CALIB_000347 | CAL2_CORR | db497d74 | 0 | 0 | hotmap_v2.json | CALIB_B | CORR | vllmw | 128/128 | 64588/16384 | 26.3 | 622.44 | 2422/3102/9801 | 74.1/81.3/84.1 | 83.4 | 11931 | True | 00:09:10 |
| CFFIX_v4 | CFFIX_v4_MAIN3_010227 | M1_MAIN_OFF | 12926df2 | unset | unset | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.5 | 736.22 | 2329/3231/11015 | 64.2/74.5/75.2 | 68.5 | 10769 | True | 01:05:09 |
| CFFIX_v4 | CFFIX_v4_MAIN3_010227 | M2_MAIN_OFF | 12926df2 | unset | unset | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.1 | 743.86 | 2162/3035/10828 | 64.2/74.9/76.7 | 68.9 | 10491 | True | 01:06:13 |
| CFFIX_v4 | CFFIX_v4_MAIN3_010227 | M3_MAIN_OFF | 12926df2 | unset | unset | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 42.5 | 770.84 | 2119/3066/6395 | 58.4/69.2/74.1 | 66.3 | 10168 | True | 01:07:18 |
| CFFIX_v4 | CFFIX_v4_MAIN3_010834 | M1_MAIN_OFF | 12926df2 | unset | unset | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.5 | 737.09 | 2318/3219/10988 | 64.5/75.4/75.4 | 68.8 | 10630 | False | 01:11:17 |
| CFFIX_v4 | CFFIX_v4_MAIN3_010834 | M2_MAIN_OFF | 12926df2 | unset | unset | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 63/256 | 32409/8064 | 292.5 | 27.57 | 2172/3105/3106 | 64.6/70.0/72.1 | 67.7 | 10391 | False | 01:12:22 |
| CTRL_noprobe | CTRL_noprobe_MAIN3_062915 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 63/256 | 32410/8064 | 296.2 | 27.22 | 2138/3118/3119 | 64.2/69.7/71.8 | 65.9 | 10307 | False | 06:32:00 |
| DIAG_legacycb | DIAG_legacycb_MAIN3_015832 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 51.4 | 637.24 | 2865/4330/12588 | 72.1/82.4/87.5 | 73.0 | 12242 | True | 02:01:17 |
| DIAG_legacycb | DIAG_legacycb_MAIN3_015832 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 51.1 | 640.94 | 2755/4092/12554 | 71.5/80.9/87.2 | 73.6 | 12048 | True | 02:02:29 |
| DIAG_legacycb | DIAG_legacycb_MAIN3_015832 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 51.4 | 637.75 | 2746/4055/12728 | 72.6/81.2/87.9 | 73.5 | 12102 | True | 02:03:40 |
| DIAG_legacycb | DIAG_legacycb_MAIN3_020506 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 50.7 | 646.64 | 2793/4141/12407 | 70.5/80.3/86.1 | 70.6 | 11941 | True | 02:07:50 |
| DIAG_legacycb | DIAG_legacycb_MAIN3_020506 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 50.3 | 651.80 | 2780/4104/12291 | 70.9/79.5/85.9 | 70.4 | 11856 | True | 02:09:01 |
| DIAG_legacycb | DIAG_legacycb_MAIN3_020506 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 49.5 | 662.42 | 2898/4202/7763 | 64.1/78.4/81.0 | 70.4 | 11702 | True | 02:10:11 |
| DIAG_legacycb | DIAG_legacycb_MAIN3_021138 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 51.5 | 636.05 | 2909/4280/12619 | 71.6/81.3/87.4 | 72.8 | 12198 | True | 02:14:23 |
| DIAG_legacycb | DIAG_legacycb_MAIN3_021138 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 52.0 | 629.89 | 2874/4211/12801 | 72.2/82.3/88.0 | 72.6 | 12341 | True | 02:15:35 |
| DIAG_legacycb | DIAG_legacycb_MAIN3_021138 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 51.2 | 640.45 | 2829/4174/12490 | 70.9/81.5/87.1 | 72.5 | 12106 | True | 02:16:46 |
| DIAG_nooverlap | DIAG_nooverlap_MAIN3_021811 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 46.9 | 699.18 | 1752/3327/11053 | 72.0/80.1/83.7 | 72.6 | 11188 | True | 02:20:53 |
| DIAG_nooverlap | DIAG_nooverlap_MAIN3_021811 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 46.2 | 709.93 | 1743/3111/10897 | 70.0/78.0/82.6 | 71.9 | 10958 | True | 02:22:00 |
| DIAG_nooverlap | DIAG_nooverlap_MAIN3_021811 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 46.5 | 704.34 | 1810/3147/11007 | 70.9/79.1/83.1 | 73.1 | 11099 | True | 02:23:06 |
| DIAG_nooverlap | DIAG_nooverlap_MAIN3_022427 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 46.7 | 702.12 | 1737/3281/11087 | 72.0/78.4/83.8 | 71.8 | 10989 | True | 02:27:10 |
| DIAG_nooverlap | DIAG_nooverlap_MAIN3_022427 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 46.3 | 708.38 | 1811/3157/10865 | 70.5/80.0/82.2 | 72.4 | 11132 | True | 02:28:17 |
| DIAG_nooverlap | DIAG_nooverlap_MAIN3_022427 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 46.4 | 705.61 | 1848/3201/11029 | 71.0/79.4/83.0 | 71.4 | 11095 | True | 02:29:23 |
| DIAG_nooverlap | DIAG_nooverlap_MAIN3_023043 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 46.2 | 709.44 | 1836/3397/10955 | 70.8/77.5/82.6 | 70.5 | 10991 | True | 02:33:29 |
| DIAG_nooverlap | DIAG_nooverlap_MAIN3_023043 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 46.8 | 699.62 | 1773/3137/11019 | 71.1/80.0/82.8 | 72.8 | 11186 | True | 02:34:35 |
| DIAG_nooverlap | DIAG_nooverlap_MAIN3_023043 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 46.6 | 703.39 | 1735/3081/11016 | 70.7/79.2/83.2 | 71.2 | 11102 | True | 02:35:41 |
| DIAG_noskipimm | DIAG_noskipimm_MAIN3_023702 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 50.3 | 651.86 | 3123/4105/12589 | 69.8/84.7/85.5 | 72.4 | 11990 | True | 02:39:47 |
| DIAG_noskipimm | DIAG_noskipimm_MAIN3_023702 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 49.8 | 657.36 | 2981/3992/12321 | 68.6/83.3/85.4 | 73.4 | 11967 | True | 02:40:57 |
| DIAG_noskipimm | DIAG_noskipimm_MAIN3_023702 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 50.5 | 648.39 | 3017/4041/12388 | 70.4/88.4/88.4 | 78.3 | 12664 | True | 02:42:07 |
| DIAG_noskipimm | DIAG_noskipimm_MAIN3_024335 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 50.4 | 650.55 | 3176/4282/12643 | 69.3/84.0/84.8 | 72.7 | 12132 | True | 02:46:18 |
| DIAG_noskipimm | DIAG_noskipimm_MAIN3_024335 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 50.0 | 654.82 | 3033/4068/12582 | 69.1/84.2/85.2 | 72.5 | 11961 | True | 02:47:29 |
| DIAG_noskipimm | DIAG_noskipimm_MAIN3_024335 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 49.5 | 662.41 | 2857/4125/12205 | 68.8/83.6/85.4 | 71.2 | 11939 | True | 02:48:39 |
| DIAG_noskipimm | DIAG_noskipimm_MAIN3_025005 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 50.2 | 652.31 | 3073/4103/12570 | 69.7/84.4/84.7 | 74.0 | 12082 | True | 02:52:50 |
| DIAG_noskipimm | DIAG_noskipimm_MAIN3_025005 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 50.8 | 644.41 | 2991/3988/12625 | 70.8/85.5/86.4 | 76.0 | 12340 | True | 02:54:01 |
| DIAG_noskipimm | DIAG_noskipimm_MAIN3_025005 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 50.3 | 651.27 | 2980/4100/12526 | 70.1/84.9/85.2 | 73.8 | 12143 | True | 02:55:12 |
| DIAG_v4_legacycb | DIAG_v4_legacycb_MAIN3_005212 | M1_MAIN_OFF | 12926df2 | unset | unset | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 49.8 | 657.58 | 2820/4225/7762 | 64.3/79.1/81.8 | 69.0 | 11870 | True | 00:54:58 |
| DIAG_v4_legacycb | DIAG_v4_legacycb_MAIN3_005212 | M2_MAIN_OFF | 12926df2 | unset | unset | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 50.0 | 655.00 | 2833/4172/12179 | 68.5/80.1/84.2 | 69.6 | 12036 | True | 00:56:07 |
| DIAG_v4_legacycb | DIAG_v4_legacycb_MAIN3_005212 | M3_MAIN_OFF | 12926df2 | unset | unset | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 49.7 | 659.83 | 2874/4214/12236 | 69.0/77.8/84.4 | 69.0 | 11750 | True | 00:57:17 |
| ENVCHK2_v4 | ENVCHK2_v4_MAIN3_003935 | M1_MAIN_OFF | 12926df2 | unset | unset | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.4 | 737.27 | 2271/3246/10850 | 64.2/71.9/76.5 | 69.0 | 10685 | False | 00:44:13 |
| ENVCHK2_v4 | ENVCHK2_v4_MAIN3_003935 | M2_MAIN_OFF | 12926df2 | unset | unset | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 126/256 | 64859/16128 | 309.1 | 52.18 | 2318/3133/3135 | 63.9/74.9/74.9 | 66.4 | 10450 | False | 00:45:18 |
| ENVCHK_v4 | ENVCHK_v4_MAIN3_C1_LONG_002200 | M1_MAIN_OFF | 12926df2 | unset | unset | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 45.1 | 727.33 | 2276/3437/11058 | 65.1/76.1/76.6 | 71.2 | 10904 | False | 00:24:44 |
| ENVCHK_v4 | ENVCHK_v4_MAIN3_C1_LONG_002200 | M2_MAIN_OFF | 12926df2 | unset | unset | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 43.6 | 750.71 | 2175/3110/10679 | 62.0/72.7/75.3 | 68.7 | 10492 | False | 00:25:48 |
| ENVCHK_v4 | ENVCHK_v4_MAIN3_C1_LONG_002200 | M3_MAIN_OFF | 12926df2 | unset | unset | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 189/256 | 97303/24192 | 313.2 | 77.25 | 2215/3093/4055 | 65.7/75.6/78.5 | 72.6 | 10815 | False | 00:26:52 |
| PROBE_nan | PROBE_nan_MAIN3_025640 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 0/256 | 0/0 | 0.5 | 0.00 | 0/0/0 | 0.0/0.0/0.0 | 0.0 | 0 | False | 03:04:08 |
| PROBE_nan | PROBE_nan_MAIN3_030606 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 0/256 | 0/0 | 0.4 | 0.00 | 0/0/0 | 0.0/0.0/0.0 | 0.0 | 0 | False | 03:13:28 |
| PROBE_nan | PROBE_nan_MAIN3_031453 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 45.3 | 723.09 | 2524/3370/11092 | 63.7/74.4/80.0 | 74.3 | 11179 | True | 03:17:38 |
| PROBE_nan | PROBE_nan_MAIN3_031453 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 43.8 | 748.22 | 2395/3137/10871 | 62.3/73.3/74.6 | 67.9 | 10425 | True | 03:18:43 |
| PROBE_nan | PROBE_nan_MAIN3_031453 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 43.8 | 748.37 | 2420/3128/10860 | 61.5/72.4/74.7 | 68.1 | 10432 | True | 03:19:46 |
| PROBE_nan | PROBE_nan_MAIN3_032108 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 45.0 | 728.03 | 2441/3252/11015 | 62.6/72.6/75.7 | 69.8 | 10901 | True | 03:23:53 |
| PROBE_nan | PROBE_nan_MAIN3_032108 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.1 | 743.01 | 2389/3087/10855 | 62.6/72.3/74.5 | 68.5 | 10545 | True | 03:25:00 |
| PROBE_nan | PROBE_nan_MAIN3_032108 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.1 | 743.56 | 2462/3127/11003 | 62.2/73.2/73.7 | 68.3 | 10516 | True | 03:26:04 |
| PROBE_nan | PROBE_nan_MAIN3_032728 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 43.5 | 753.85 | 2480/3411/6630 | 58.0/68.9/74.2 | 66.0 | 10516 | True | 03:30:11 |
| PROBE_nan | PROBE_nan_MAIN3_032728 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 43.0 | 761.18 | 2405/3111/10694 | 60.8/72.2/73.5 | 65.8 | 10269 | True | 03:31:14 |
| PROBE_nan | PROBE_nan_MAIN3_032728 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 43.2 | 757.99 | 2365/3132/10751 | 61.2/67.5/72.9 | 66.4 | 10223 | True | 03:32:17 |
| PROBE_nan | PROBE_nan_MAIN3_033341 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.6 | 735.06 | 2411/3444/10938 | 62.6/72.8/75.1 | 68.8 | 10879 | True | 03:36:26 |
| PROBE_nan | PROBE_nan_MAIN3_033341 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.6 | 735.28 | 2390/3125/10978 | 63.0/73.9/76.3 | 69.5 | 10652 | True | 03:37:31 |
| PROBE_nan | PROBE_nan_MAIN3_033341 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.0 | 744.34 | 2356/3160/10870 | 62.1/73.0/75.1 | 67.1 | 10495 | True | 03:38:35 |
| PROBE_nan | PROBE_nan_MAIN3_034047 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 0/256 | 0/0 | 0.4 | 0.00 | 0/0/0 | 0.0/0.0/0.0 | 0.0 | 0 | False | 03:48:11 |
| PROBE_nan | PROBE_nan_MAIN3_034943 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.3 | 740.37 | 2316/3215/10760 | 62.4/73.2/77.0 | 69.0 | 10697 | True | 03:52:35 |
| PROBE_nan | PROBE_nan_MAIN3_034943 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 42.2 | 777.07 | 2202/3182/6324 | 56.8/68.9/73.5 | 64.7 | 10234 | True | 03:53:39 |
| PROBE_nan | PROBE_nan_MAIN3_034943 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 43.0 | 762.57 | 2255/3209/10605 | 60.5/71.6/74.1 | 67.0 | 10420 | True | 03:54:41 |
| PROBE_nan | PROBE_nan_MAIN3_035604 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.6 | 734.48 | 2312/3473/10942 | 63.2/72.8/76.0 | 67.8 | 10862 | False | 03:58:48 |
| PROBE_nan | PROBE_nan_MAIN3_035604 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 189/256 | 97310/24192 | 321.6 | 75.22 | 2352/3121/4085 | 64.4/75.5/75.7 | 68.7 | 10581 | False | 03:59:52 |
| PROBE_nan | PROBE_nan_MAIN3_040906 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 45.1 | 726.41 | 2537/3456/11065 | 62.1/72.4/76.5 | 70.1 | 10964 | False | 04:11:51 |
| PROBE_nan | PROBE_nan_MAIN3_040906 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 126/256 | 64859/16128 | 303.5 | 53.14 | 2497/3215/3217 | 63.2/73.1/74.4 | 67.6 | 10525 | False | 04:12:57 |
| PROBE_nan | PROBE_nan_MAIN3_042900 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 45.0 | 727.46 | 2577/3524/11225 | 61.9/73.0/74.7 | 68.3 | 10826 | True | 04:31:46 |
| PROBE_nan | PROBE_nan_MAIN3_042900 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.2 | 740.63 | 2526/3232/11079 | 61.7/72.9/73.1 | 67.3 | 10512 | True | 04:32:51 |
| PROBE_nan | PROBE_nan_MAIN3_042900 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.7 | 732.50 | 2532/3229/11193 | 63.7/73.2/75.0 | 68.2 | 10684 | True | 04:33:56 |
| PROBE_nan | PROBE_nan_MAIN3_043521 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.8 | 731.81 | 2470/3375/11172 | 63.4/69.5/75.0 | 68.0 | 10578 | True | 04:38:05 |
| PROBE_nan | PROBE_nan_MAIN3_043521 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.9 | 729.23 | 2565/3257/11262 | 63.3/73.5/74.8 | 68.5 | 10686 | True | 04:39:10 |
| PROBE_nan | PROBE_nan_MAIN3_043521 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 45.2 | 724.21 | 2492/3223/11267 | 63.4/75.1/77.6 | 71.8 | 10962 | True | 04:40:15 |
| PROBE_nan | PROBE_nan_MAIN3_044141 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.6 | 735.52 | 2519/3419/11028 | 61.2/71.0/74.7 | 66.9 | 10678 | True | 04:44:26 |
| PROBE_nan | PROBE_nan_MAIN3_044141 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.5 | 736.04 | 2596/3174/11153 | 62.7/74.0/74.2 | 68.7 | 10602 | True | 04:45:32 |
| PROBE_nan | PROBE_nan_MAIN3_044141 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 45.0 | 727.71 | 2548/3215/11231 | 62.4/73.6/74.4 | 69.6 | 10834 | True | 04:46:37 |
| PROBE_nan | PROBE_nan_MAIN3_044802 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.3 | 739.49 | 2549/3466/11094 | 60.8/71.8/73.0 | 66.3 | 10688 | False | 04:50:47 |
| PROBE_nan | PROBE_nan_MAIN3_044802 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 63/256 | 32412/8064 | 288.9 | 27.91 | 2376/3168/3169 | 63.7/69.2/71.3 | 67.8 | 10477 | False | 04:51:51 |
| PROBE_nan | PROBE_nan_MAIN3_050025 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 47.7 | 687.05 | 3116/3990/12045 | 63.6/76.2/77.9 | 70.0 | 11371 | True | 05:03:17 |
| PROBE_nan | PROBE_nan_MAIN3_050025 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.8 | 731.14 | 3144/3810/7124 | 55.3/72.1/72.4 | 64.5 | 10763 | True | 05:04:24 |
| PROBE_nan | PROBE_nan_MAIN3_050025 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 45.8 | 715.40 | 3118/3770/11661 | 60.4/73.5/74.3 | 65.3 | 10849 | True | 05:05:30 |
| PROBE_nan | PROBE_nan_MAIN3_050653 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 47.9 | 684.02 | 3219/4271/12153 | 63.5/76.6/76.6 | 69.3 | 11512 | True | 05:09:45 |
| PROBE_nan | PROBE_nan_MAIN3_050653 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 47.0 | 697.61 | 3126/3775/11987 | 63.0/76.0/76.5 | 67.6 | 11174 | True | 05:10:53 |
| PROBE_nan | PROBE_nan_MAIN3_050653 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 46.8 | 700.75 | 3092/3790/11812 | 62.3/75.1/77.7 | 67.6 | 11254 | True | 05:12:01 |
| PROBE_nan | PROBE_nan_MAIN3_051321 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 47.4 | 691.23 | 3109/4012/11885 | 61.6/74.7/77.3 | 68.8 | 11513 | True | 05:16:07 |
| PROBE_nan | PROBE_nan_MAIN3_051321 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 47.2 | 694.89 | 3069/3802/11986 | 62.4/76.6/77.9 | 71.7 | 11337 | True | 05:17:14 |
| PROBE_nan | PROBE_nan_MAIN3_051321 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 47.2 | 694.64 | 3087/3712/11919 | 64.2/76.8/77.3 | 69.7 | 11331 | True | 05:18:21 |
| PROBE_nan | PROBE_nan_MAIN3_051942 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 47.2 | 693.88 | 3187/3956/11913 | 61.5/74.4/76.6 | 68.8 | 11448 | True | 05:22:28 |
| PROBE_nan | PROBE_nan_MAIN3_051942 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 46.9 | 699.18 | 3091/3787/11944 | 62.4/74.2/77.1 | 68.7 | 11247 | True | 05:23:36 |
| PROBE_nan | PROBE_nan_MAIN3_051942 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 46.9 | 698.22 | 3224/3796/11882 | 62.3/71.7/75.7 | 67.9 | 11207 | True | 05:24:43 |
| PROBE_nan | PROBE_nan_MAIN3_052604 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 45.2 | 724.69 | 2430/3385/11127 | 64.1/75.9/78.9 | 70.9 | 11052 | True | 05:28:53 |
| PROBE_nan | PROBE_nan_MAIN3_052604 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 42.5 | 770.94 | 2297/3235/6493 | 57.2/68.3/73.8 | 64.8 | 10172 | True | 05:29:58 |
| PROBE_nan | PROBE_nan_MAIN3_052604 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 43.9 | 746.06 | 2273/3220/10689 | 62.2/73.2/74.7 | 68.8 | 10784 | True | 05:31:01 |
| PROBE_nan | PROBE_nan_MAIN3_053225 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 45.1 | 726.42 | 2443/3361/11250 | 64.2/73.9/79.6 | 70.5 | 11116 | True | 05:35:09 |
| PROBE_nan | PROBE_nan_MAIN3_053225 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.1 | 742.97 | 2365/3273/10863 | 63.5/73.0/75.7 | 67.2 | 10496 | True | 05:36:14 |
| PROBE_nan | PROBE_nan_MAIN3_053225 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.2 | 741.15 | 2353/3253/10798 | 62.1/72.9/74.8 | 68.6 | 10912 | True | 05:37:19 |
| PROBE_nan | PROBE_nan_MAIN3_053857 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.8 | 731.29 | 2427/3403/11042 | 63.7/74.9/75.2 | 67.8 | 10665 | True | 05:41:42 |
| PROBE_nan | PROBE_nan_MAIN3_053857 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.1 | 742.61 | 2356/3181/10838 | 62.5/73.2/76.9 | 68.2 | 10620 | True | 05:42:47 |
| PROBE_nan | PROBE_nan_MAIN3_053857 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 45.5 | 719.87 | 2408/3261/11228 | 65.3/74.9/79.3 | 72.5 | 10991 | True | 05:43:51 |
| PROBE_nan | PROBE_nan_MAIN3_054516 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.9 | 729.42 | 2440/3444/11105 | 63.9/75.2/76.7 | 69.1 | 10696 | True | 05:48:01 |
| PROBE_nan | PROBE_nan_MAIN3_054516 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.6 | 734.60 | 2333/3252/10932 | 62.7/74.0/75.6 | 69.2 | 10841 | True | 05:49:05 |
| PROBE_nan | PROBE_nan_MAIN3_054516 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.9 | 730.11 | 2350/3229/11184 | 64.5/74.8/77.2 | 68.9 | 10686 | True | 05:50:10 |
| PROBE_nan | PROBE_nan_MAIN3_055129 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 42.5 | 770.25 | 2330/3294/6606 | 56.8/68.6/74.3 | 64.0 | 10340 | True | 05:54:15 |
| PROBE_nan | PROBE_nan_MAIN3_055129 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.0 | 744.27 | 2357/3229/10759 | 62.7/72.8/76.3 | 68.1 | 10793 | True | 05:55:18 |
| PROBE_nan | PROBE_nan_MAIN3_055129 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 42.7 | 767.67 | 2291/3180/10336 | 59.3/69.3/72.5 | 64.3 | 10283 | True | 05:56:22 |
| PROBE_nan | PROBE_nan_MAIN3_055739 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.4 | 737.37 | 2499/3314/10910 | 62.2/73.3/76.0 | 68.5 | 10701 | True | 06:00:22 |
| PROBE_nan | PROBE_nan_MAIN3_055739 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.4 | 737.56 | 2265/3143/10986 | 64.6/70.8/76.1 | 68.0 | 10499 | True | 06:01:26 |
| PROBE_nan | PROBE_nan_MAIN3_055739 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.7 | 733.49 | 2278/3213/10947 | 63.3/74.5/76.1 | 69.4 | 10860 | True | 06:02:31 |
| PROBE_nan | PROBE_nan_MAIN3_060356 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 45.2 | 724.48 | 2443/3300/11175 | 64.6/74.6/77.7 | 69.4 | 10771 | True | 06:06:41 |
| PROBE_nan | PROBE_nan_MAIN3_060356 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.3 | 739.75 | 2359/3190/10930 | 63.5/74.9/75.0 | 67.9 | 10532 | True | 06:07:46 |
| PROBE_nan | PROBE_nan_MAIN3_060356 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.5 | 735.75 | 2316/3264/10957 | 63.6/74.9/77.3 | 68.8 | 10637 | True | 06:08:50 |
| PROBE_nan | PROBE_nan_MAIN3_061008 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.8 | 731.13 | 2432/3442/10995 | 62.8/74.2/76.6 | 68.6 | 10817 | True | 06:12:53 |
| PROBE_nan | PROBE_nan_MAIN3_061008 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.6 | 735.32 | 2337/3204/11114 | 63.9/73.8/78.0 | 70.6 | 10741 | True | 06:13:58 |
| PROBE_nan | PROBE_nan_MAIN3_061008 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 45.1 | 727.13 | 2333/3237/11084 | 64.4/75.7/77.3 | 68.5 | 10690 | True | 06:15:03 |
| PROBE_nan | PROBE_nan_MAIN3_061628 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 45.7 | 717.30 | 2475/3454/11416 | 64.6/74.4/80.0 | 71.7 | 11056 | True | 06:19:12 |
| PROBE_nan | PROBE_nan_MAIN3_061628 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.2 | 741.49 | 2358/3202/10932 | 63.3/74.8/76.2 | 67.9 | 10590 | True | 06:20:18 |
| PROBE_nan | PROBE_nan_MAIN3_061628 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.4 | 737.78 | 2293/3200/10903 | 62.7/74.0/76.4 | 68.9 | 10610 | True | 06:21:23 |
| PROBE_nan | PROBE_nan_MAIN3_062248 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 45.3 | 723.82 | 2484/3345/11154 | 63.5/74.7/77.0 | 70.6 | 10933 | True | 06:25:39 |
| PROBE_nan | PROBE_nan_MAIN3_062248 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.5 | 736.49 | 2287/3173/10952 | 63.7/74.6/75.3 | 68.1 | 10736 | True | 06:26:44 |
| PROBE_nan | PROBE_nan_MAIN3_062248 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.3 | 739.52 | 2315/3199/10905 | 64.1/75.3/76.4 | 69.0 | 10588 | True | 06:27:49 |
| R0C | R0C_CONFIRM_125429 | C1_OFF | a4add14d | 0 | 0 | hotmap_v2.json | LOW_CONCURRENCY | OFF | vllm | 16/16 | 8234/2048 | 32.6 | 62.92 | 251/309/315 | 13.9/14.1/14.6 | 14.2 | 2096 | True | 13:00:36 |
| R0C | R0C_CONFIRM_125429 | IND1_OFF | a4add14d | 0 | 0 | hotmap_v2.json | CONFIRM | OFF | vllm | 128/128 | 64376/16384 | 25.2 | 650.61 | 2090/2926/9338 | 74.5/81.0/81.0 | 81.5 | 11565 | True | 13:02:15 |
| R0C | R0C_CONFIRM_125429 | IND2_OFF | a4add14d | 0 | 0 | hotmap_v2.json | CONFIRM | OFF | vllm | 128/128 | 64376/16384 | 25.1 | 652.07 | 2128/2934/9351 | 73.4/79.0/84.3 | 80.7 | 11469 | True | 13:03:00 |
| R0C | R0C_CONFIRM_125429 | LONG_OFF | a4add14d | 0 | 0 | hotmap_v2.json | LONGER_PREFILL | OFF | vllm | 32/32 | 131147/4096 | 24.7 | 166.09 | 2371/3165/3188 | 29.9/41.8/42.1 | 25.1 | 6204 | True | 13:01:30 |
| R0C | R0C_CONFIRM_125429 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.1 | 743.67 | 2193/3251/10729 | 63.1/73.5/75.2 | 67.7 | 10784 | True | 12:57:23 |
| R0C | R0C_CONFIRM_125429 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 43.9 | 745.86 | 2180/3102/10836 | 64.1/74.9/75.7 | 68.6 | 10380 | True | 12:58:27 |
| R0C | R0C_CONFIRM_125429 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 43.6 | 751.04 | 2159/3034/10739 | 63.8/70.5/75.6 | 68.0 | 10428 | True | 12:59:32 |
| R0C | R0C_CONFIRM_133411 | C1_OFF | a4add14d | 0 | 0 | hotmap_v2.json | LOW_CONCURRENCY | OFF | vllm | 16/16 | 8234/2048 | 32.4 | 63.15 | 237/338/365 | 13.9/14.0/14.0 | 14.3 | 2108 | True | 13:40:17 |
| R0C | R0C_CONFIRM_133411 | IND1_OFF | a4add14d | 0 | 0 | hotmap_v2.json | CONFIRM | OFF | vllm | 128/128 | 64376/16384 | 24.4 | 672.75 | 2203/2906/9055 | 69.5/80.2/80.2 | 77.3 | 11086 | True | 13:41:54 |
| R0C | R0C_CONFIRM_133411 | IND2_OFF | a4add14d | 0 | 0 | hotmap_v2.json | CONFIRM | OFF | vllm | 128/128 | 64376/16384 | 24.2 | 677.83 | 2078/2918/9035 | 69.9/78.1/78.1 | 79.4 | 11030 | True | 13:42:38 |
| R0C | R0C_CONFIRM_133411 | LONG_OFF | a4add14d | 0 | 0 | hotmap_v2.json | LONGER_PREFILL | OFF | vllm | 32/32 | 131147/4096 | 25.0 | 163.80 | 2482/3310/3340 | 29.8/42.5/43.1 | 25.8 | 6336 | True | 13:41:09 |
| R0C | R0C_CONFIRM_133411 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.9 | 730.07 | 2257/3262/11067 | 64.5/75.2/76.5 | 70.0 | 10799 | True | 13:37:06 |
| R0C | R0C_CONFIRM_133411 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.1 | 742.33 | 2234/3047/10825 | 64.8/73.0/77.0 | 70.2 | 10548 | True | 13:38:11 |
| R0C | R0C_CONFIRM_133411 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 41.8 | 783.64 | 2159/3095/6240 | 56.7/67.4/72.8 | 64.0 | 10007 | True | 13:39:15 |
| R0C | R0C_CONFIRM_140939 | C1_OFF | a4add14d | 0 | 0 | hotmap_v2.json | LOW_CONCURRENCY | OFF | vllm | 16/16 | 8234/2048 | 32.3 | 63.50 | 244/273/317 | 13.9/14.0/14.0 | 14.2 | 2045 | True | 14:15:46 |
| R0C | R0C_CONFIRM_140939 | IND1_OFF | a4add14d | 0 | 0 | hotmap_v2.json | CONFIRM | OFF | vllm | 128/128 | 64376/16384 | 24.7 | 663.55 | 2242/2896/9224 | 70.9/81.4/81.4 | 78.6 | 11256 | True | 14:17:23 |
| R0C | R0C_CONFIRM_140939 | IND2_OFF | a4add14d | 0 | 0 | hotmap_v2.json | CONFIRM | OFF | vllm | 128/128 | 64376/16384 | 24.8 | 661.88 | 2219/2926/9219 | 71.3/81.9/81.9 | 81.1 | 11298 | True | 14:18:08 |
| R0C | R0C_CONFIRM_140939 | LONG_OFF | a4add14d | 0 | 0 | hotmap_v2.json | LONGER_PREFILL | OFF | vllm | 32/32 | 131147/4096 | 25.4 | 161.45 | 2415/3753/3901 | 29.8/41.8/44.3 | 25.4 | 6949 | True | 14:16:37 |
| R0C | R0C_CONFIRM_140939 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.3 | 739.04 | 2204/3194/10822 | 64.9/74.3/77.5 | 69.1 | 10633 | True | 14:12:33 |
| R0C | R0C_CONFIRM_140939 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.0 | 744.28 | 2175/3008/10819 | 65.2/73.9/76.8 | 70.5 | 10504 | True | 14:13:38 |
| R0C | R0C_CONFIRM_140939 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.1 | 743.26 | 2161/3005/10927 | 63.7/74.5/75.4 | 70.4 | 10574 | True | 14:14:42 |
| R0 | R0_MAIN3_C1_LONG_001157 | M1_MAIN_OFF | db497d74 | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 189/256 | 97303/24192 | 321.2 | 75.32 | 2317/3170/4155 | 64.8/74.2/76.4 | 70.1 | 10671 | False | 00:14:40 |
| R0 | R0_MAIN3_C1_LONG_012815 | C1_OFF | a4add14d | 0 | 0 | hotmap_v2.json | LOW_CONCURRENCY | OFF | vllm | 16/16 | 8234/2048 | 32.3 | 63.50 | 233/294/322 | 13.9/13.9/14.0 | 14.3 | 2061 | True | 01:34:11 |
| R0 | R0_MAIN3_C1_LONG_012815 | LONG_OFF | a4add14d | 0 | 0 | hotmap_v2.json | LONGER_PREFILL | OFF | vllm | 32/32 | 131147/4096 | 24.6 | 166.50 | 2347/3152/3171 | 29.8/41.8/42.0 | 25.3 | 6179 | True | 01:35:03 |
| R0 | R0_MAIN3_C1_LONG_012815 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.8 | 732.06 | 2274/3210/11124 | 64.8/72.6/77.7 | 69.2 | 10780 | True | 01:30:58 |
| R0 | R0_MAIN3_C1_LONG_012815 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.1 | 743.40 | 2186/3032/10788 | 63.6/74.3/74.8 | 68.6 | 10670 | True | 01:32:03 |
| R0 | R0_MAIN3_C1_LONG_012815 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 43.8 | 748.37 | 2176/3116/10769 | 64.3/71.4/75.5 | 67.9 | 10413 | True | 01:33:07 |
| R0 | R0_MAIN3_C1_LONG_013601 | C1_OFF | a4add14d | 0 | 0 | hotmap_v2.json | LOW_CONCURRENCY | OFF | vllm | 16/16 | 8234/2048 | 32.4 | 63.11 | 243/310/319 | 13.9/14.0/14.0 | 14.3 | 2076 | True | 01:41:56 |
| R0 | R0_MAIN3_C1_LONG_013601 | LONG_OFF | a4add14d | 0 | 0 | hotmap_v2.json | LONGER_PREFILL | OFF | vllm | 32/32 | 131147/4096 | 25.1 | 163.12 | 2359/3580/3727 | 29.8/41.5/44.2 | 25.2 | 6791 | True | 01:42:48 |
| R0 | R0_MAIN3_C1_LONG_013601 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.6 | 735.35 | 2248/3225/10844 | 63.1/73.2/77.0 | 70.5 | 10840 | True | 01:38:44 |
| R0 | R0_MAIN3_C1_LONG_013601 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 43.8 | 748.79 | 2163/3063/10689 | 64.0/74.7/75.5 | 68.1 | 10526 | True | 01:39:49 |
| R0 | R0_MAIN3_C1_LONG_013601 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 43.5 | 752.88 | 2175/3085/10659 | 63.4/74.1/74.3 | 69.5 | 10397 | True | 01:40:53 |
| R0 | R0_MAIN3_C1_LONG_014350 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.4 | 737.78 | 2252/3242/10989 | 64.9/75.7/76.4 | 69.0 | 10607 | False | 01:46:33 |
| R0 | R0_MAIN3_C1_LONG_014350 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 126/256 | 64861/16128 | 303.5 | 53.14 | 2214/3000/3003 | 63.5/74.3/74.3 | 68.7 | 10290 | False | 01:47:38 |
| R0 | R0_MAIN3_C1_LONG_065651 | C1_OFF | a4add14d | 0 | 0 | hotmap_v2.json | LOW_CONCURRENCY | OFF | vllm | 16/16 | 8234/2048 | 32.3 | 63.49 | 249/272/275 | 13.9/13.9/13.9 | 14.2 | 2038 | True | 07:02:47 |
| R0 | R0_MAIN3_C1_LONG_065651 | LONG_OFF | a4add14d | 0 | 0 | hotmap_v2.json | LONGER_PREFILL | OFF | vllm | 32/32 | 131147/4096 | 24.6 | 166.62 | 2348/3139/3143 | 29.8/41.7/41.9 | 25.6 | 6184 | True | 07:03:39 |
| R0 | R0_MAIN3_C1_LONG_065651 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.7 | 732.52 | 2327/3200/10835 | 63.6/74.9/75.9 | 69.5 | 11000 | True | 06:59:34 |
| R0 | R0_MAIN3_C1_LONG_065651 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 43.8 | 747.78 | 2156/3051/10674 | 63.2/72.8/75.0 | 69.2 | 10683 | True | 07:00:39 |
| R0 | R0_MAIN3_C1_LONG_065651 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.1 | 743.81 | 2161/3058/10584 | 62.8/73.0/78.4 | 70.5 | 10683 | True | 07:01:43 |
| R0 | R0_MAIN3_C1_LONG_075238 | C1_OFF | a4add14d | 0 | 0 | hotmap_v2.json | LOW_CONCURRENCY | OFF | vllm | 16/16 | 8234/2048 | 32.3 | 63.49 | 246/285/303 | 13.9/14.0/14.0 | 14.3 | 2050 | True | 07:58:36 |
| R0 | R0_MAIN3_C1_LONG_075238 | LONG_OFF | a4add14d | 0 | 0 | hotmap_v2.json | LONGER_PREFILL | OFF | vllm | 32/32 | 131147/4096 | 25.0 | 163.60 | 2394/3338/3483 | 29.9/41.8/44.1 | 25.4 | 6550 | True | 07:59:27 |
| R0 | R0_MAIN3_C1_LONG_075238 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.8 | 731.51 | 2280/3182/10907 | 64.4/76.2/76.2 | 72.2 | 11117 | True | 07:55:23 |
| R0 | R0_MAIN3_C1_LONG_075238 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.0 | 744.81 | 2185/3048/10861 | 64.7/74.3/75.7 | 69.2 | 10424 | True | 07:56:28 |
| R0 | R0_MAIN3_C1_LONG_075238 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 43.4 | 754.55 | 2161/3072/10706 | 63.2/69.8/75.2 | 67.1 | 10318 | True | 07:57:32 |
| R0 | R0_MAIN3_C1_LONG_092311 | C1_OFF | a4add14d | 0 | 0 | hotmap_v2.json | LOW_CONCURRENCY | OFF | vllm | 16/16 | 8234/2048 | 32.3 | 63.39 | 249/288/318 | 13.9/14.0/14.0 | 14.3 | 2063 | True | 09:29:07 |
| R0 | R0_MAIN3_C1_LONG_092311 | LONG_OFF | a4add14d | 0 | 0 | hotmap_v2.json | LONGER_PREFILL | OFF | vllm | 32/32 | 131147/4096 | 25.4 | 161.37 | 2351/3637/3751 | 30.1/42.3/44.8 | 26.5 | 6781 | True | 09:29:59 |
| R0 | R0_MAIN3_C1_LONG_092311 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 43.8 | 748.04 | 2146/3097/10789 | 63.5/73.5/75.5 | 67.2 | 10439 | True | 09:25:57 |
| R0 | R0_MAIN3_C1_LONG_092311 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 43.5 | 753.88 | 2218/2994/10766 | 63.1/73.7/74.1 | 67.8 | 10330 | True | 09:27:01 |
| R0 | R0_MAIN3_C1_LONG_092311 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 43.5 | 753.63 | 2135/3060/10706 | 63.5/69.6/74.7 | 67.1 | 10276 | True | 09:28:04 |
| R0 | R0_MAIN3_C1_LONG_102248 | C1_OFF | a4add14d | 0 | 0 | hotmap_v2.json | LOW_CONCURRENCY | OFF | vllm | 16/16 | 8234/2048 | 32.5 | 63.09 | 252/337/338 | 13.9/13.9/14.0 | 14.2 | 2101 | True | 10:28:46 |
| R0 | R0_MAIN3_C1_LONG_102248 | LONG_OFF | a4add14d | 0 | 0 | hotmap_v2.json | LONGER_PREFILL | OFF | vllm | 32/32 | 131147/4096 | 25.3 | 161.99 | 2377/3501/3615 | 30.0/42.6/44.6 | 25.6 | 6628 | True | 10:29:38 |
| R0 | R0_MAIN3_C1_LONG_102248 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.3 | 740.07 | 2301/3158/10782 | 62.8/73.2/78.5 | 69.8 | 10785 | True | 10:25:34 |
| R0 | R0_MAIN3_C1_LONG_102248 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 43.8 | 748.88 | 2256/3146/10795 | 63.7/74.5/75.2 | 68.7 | 10419 | True | 10:26:38 |
| R0 | R0_MAIN3_C1_LONG_102248 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 45.0 | 728.68 | 2219/3078/11192 | 65.1/77.5/78.8 | 73.0 | 10913 | True | 10:27:42 |
| R0 | R0_MAIN3_C1_LONG_112641 | C1_OFF | a4add14d | 0 | 0 | hotmap_v2.json | LOW_CONCURRENCY | OFF | vllm | 16/16 | 8234/2048 | 32.4 | 63.27 | 236/328/329 | 13.9/14.0/14.0 | 14.2 | 2099 | True | 11:32:33 |
| R0 | R0_MAIN3_C1_LONG_112641 | LONG_OFF | a4add14d | 0 | 0 | hotmap_v2.json | LONGER_PREFILL | OFF | vllm | 32/32 | 131147/4096 | 24.6 | 166.69 | 2350/3136/3151 | 30.0/41.8/41.9 | 25.3 | 6198 | True | 11:33:25 |
| R0 | R0_MAIN3_C1_LONG_112641 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.2 | 741.82 | 2279/3136/10954 | 63.3/73.5/74.5 | 70.2 | 10795 | True | 11:29:20 |
| R0 | R0_MAIN3_C1_LONG_112641 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 43.8 | 748.02 | 2195/3029/10757 | 64.1/74.7/75.7 | 69.7 | 10531 | True | 11:30:25 |
| R0 | R0_MAIN3_C1_LONG_112641 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.1 | 743.05 | 2165/3030/10860 | 64.7/75.4/76.1 | 69.6 | 10467 | True | 11:31:29 |
| R0 | R0_MAIN3_C1_LONG_122315 | C1_OFF | a4add14d | 0 | 0 | hotmap_v2.json | LOW_CONCURRENCY | OFF | vllm | 16/16 | 8234/2048 | 32.5 | 62.96 | 248/330/335 | 13.9/14.0/14.0 | 14.3 | 2100 | True | 12:29:13 |
| R0 | R0_MAIN3_C1_LONG_122315 | LONG_OFF | a4add14d | 0 | 0 | hotmap_v2.json | LONGER_PREFILL | OFF | vllm | 32/32 | 131147/4096 | 24.7 | 165.59 | 2379/3170/3247 | 29.8/41.7/42.0 | 25.4 | 6253 | True | 12:30:05 |
| R0 | R0_MAIN3_C1_LONG_122315 | M1_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.5 | 736.33 | 2229/3265/10901 | 64.2/73.7/77.1 | 71.7 | 10741 | True | 12:26:00 |
| R0 | R0_MAIN3_C1_LONG_122315 | M2_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.0 | 744.83 | 2206/3083/10837 | 64.3/71.0/75.6 | 67.9 | 10416 | True | 12:27:05 |
| R0 | R0_MAIN3_C1_LONG_122315 | M3_MAIN_OFF | a4add14d | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 43.6 | 751.68 | 2180/3047/10579 | 62.3/72.4/74.8 | 67.8 | 10502 | True | 12:28:09 |
| R0 | R0_MAIN3_C1_LONG_231609 | C1_OFF | 194f3064 | 0 | 0 | hotmap_v2.json | LOW_CONCURRENCY | OFF | vllm | 16/16 | 8234/2048 | 32.5 | 63.08 | 246/305/308 | 13.9/14.0/14.0 | 14.3 | 2077 | True | 23:22:06 |
| R0 | R0_MAIN3_C1_LONG_231609 | LONG_OFF | 194f3064 | 0 | 0 | hotmap_v2.json | LONGER_PREFILL | OFF | vllm | 32/32 | 131147/4096 | 24.8 | 165.41 | 2372/3177/3183 | 29.9/41.9/42.4 | 26.2 | 6252 | True | 23:22:58 |
| R0 | R0_MAIN3_C1_LONG_231609 | M1_MAIN_OFF | 194f3064 | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.9 | 730.01 | 2321/3259/11145 | 64.7/75.6/77.1 | 71.7 | 10728 | True | 23:18:51 |
| R0 | R0_MAIN3_C1_LONG_231609 | M2_MAIN_OFF | 194f3064 | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.5 | 736.81 | 2217/3075/10900 | 65.2/73.8/77.6 | 70.5 | 10684 | True | 23:19:56 |
| R0 | R0_MAIN3_C1_LONG_231609 | M3_MAIN_OFF | 194f3064 | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.3 | 739.11 | 2169/3057/10772 | 63.9/73.4/77.2 | 71.5 | 10706 | True | 23:21:00 |
| R0 | R0_MAIN3_C1_LONG_235443 | M1_MAIN_OFF | db497d74 | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 42.1 | 778.40 | 2281/3285/6387 | 57.1/67.2/70.9 | 63.3 | 10032 | False | 23:57:31 |
| R0 | R0_MAIN3_C1_LONG_235443 | M2_MAIN_OFF | db497d74 | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 63/256 | 32407/8064 | 290.2 | 27.79 | 2134/3066/3067 | 65.9/71.3/73.4 | 68.8 | 10521 | False | 23:58:33 |
| R0_REF | R0_REF_MAIN3_C1_LONG_064858 | C1_OFF | f437b6f1 | 0 | 0 | hotmap_v2.json | LOW_CONCURRENCY | OFF | vllm | 16/16 | 8234/2048 | 32.5 | 62.93 | 259/316/342 | 13.9/14.0/14.0 | 14.2 | 2081 | True | 06:54:56 |
| R0_REF | R0_REF_MAIN3_C1_LONG_064858 | LONG_OFF | f437b6f1 | 0 | 0 | hotmap_v2.json | LONGER_PREFILL | OFF | vllm | 32/32 | 131147/4096 | 24.6 | 166.31 | 2367/3151/3154 | 29.8/41.8/41.9 | 25.2 | 6191 | True | 06:55:48 |
| R0_REF | R0_REF_MAIN3_C1_LONG_064858 | M1_MAIN_OFF | f437b6f1 | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.6 | 735.10 | 2280/3157/10925 | 64.7/73.7/76.1 | 70.0 | 10674 | True | 06:51:44 |
| R0_REF | R0_REF_MAIN3_C1_LONG_064858 | M2_MAIN_OFF | f437b6f1 | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.2 | 741.82 | 2173/3017/10840 | 64.2/74.6/76.7 | 70.9 | 10511 | True | 06:52:48 |
| R0_REF | R0_REF_MAIN3_C1_LONG_064858 | M3_MAIN_OFF | f437b6f1 | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 43.8 | 748.48 | 2256/3038/10781 | 64.1/74.8/75.2 | 69.6 | 10414 | True | 06:53:53 |
| R0_REF | R0_REF_MAIN3_C1_LONG_091528 | C1_OFF | f437b6f1 | 0 | 0 | hotmap_v2.json | LOW_CONCURRENCY | OFF | vllm | 16/16 | 8234/2048 | 32.2 | 63.66 | 241/271/297 | 13.9/13.9/14.0 | 14.2 | 2040 | True | 09:21:19 |
| R0_REF | R0_REF_MAIN3_C1_LONG_091528 | LONG_OFF | f437b6f1 | 0 | 0 | hotmap_v2.json | LONGER_PREFILL | OFF | vllm | 32/32 | 131147/4096 | 24.8 | 165.32 | 2359/3177/3203 | 30.1/42.1/42.3 | 25.4 | 6248 | True | 09:22:13 |
| R0_REF | R0_REF_MAIN3_C1_LONG_091528 | M1_MAIN_OFF | f437b6f1 | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 43.1 | 759.55 | 2286/3363/6522 | 57.9/68.5/73.5 | 65.3 | 10175 | True | 09:18:10 |
| R0_REF | R0_REF_MAIN3_C1_LONG_091528 | M2_MAIN_OFF | f437b6f1 | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 42.4 | 772.49 | 2188/3136/10278 | 60.7/71.0/74.6 | 65.2 | 10214 | True | 09:19:14 |
| R0_REF | R0_REF_MAIN3_C1_LONG_091528 | M3_MAIN_OFF | f437b6f1 | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 42.8 | 764.99 | 2218/3038/10540 | 62.3/73.2/74.3 | 65.4 | 10283 | True | 09:20:16 |
| R0_REF | R0_REF_MAIN3_C1_LONG_111830 | M1_MAIN_OFF | f437b6f1 | 0 | 0 | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 63/256 | 32409/8064 | 302.2 | 26.68 | 2188/3195/3195 | 65.7/71.2/73.4 | 67.7 | 10544 | False | 11:21:16 |
| R0_REF | R0_REF_R0_REF_224948 | C1_OFF | 12926df2 | unset | unset | hotmap_v2.json | LOW_CONCURRENCY | OFF | vllm | 16/16 | 8234/2048 | 38.5 | 53.23 | 237/330/330 | 13.9/26.3/53.9 | 14.4 | 3622 | True | 22:55:43 |
| R0_REF | R0_REF_R0_REF_224948 | LONG_OFF | 12926df2 | unset | unset | hotmap_v2.json | LONGER_PREFILL | OFF | vllm | 32/32 | 131147/4096 | 25.1 | 163.01 | 2366/3443/3589 | 29.9/41.8/44.4 | 25.2 | 6619 | True | 22:56:41 |
| R0_REF | R0_REF_R0_REF_224948 | M1_MAIN_OFF | 12926df2 | unset | unset | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.0 | 745.40 | 2223/3175/10733 | 62.3/73.0/75.1 | 68.5 | 10706 | True | 22:52:31 |
| R0_REF | R0_REF_R0_REF_224948 | M2_MAIN_OFF | 12926df2 | unset | unset | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 43.9 | 745.99 | 2186/3007/10807 | 63.9/74.4/75.4 | 68.7 | 10470 | True | 22:53:35 |
| R0_REF | R0_REF_R0_REF_224948 | M3_MAIN_OFF | 12926df2 | unset | unset | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 43.8 | 747.53 | 2189/2991/10697 | 63.4/74.1/74.9 | 69.1 | 10640 | True | 22:54:39 |
| ROLLBACK_v4 | ROLLBACK_v4_MAIN3_144659 | M1_MAIN_OFF | 12926df2 | unset | unset | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.2 | 741.75 | 2287/3175/10868 | 63.2/73.9/76.0 | 69.3 | 10581 | True | 14:49:39 |
| ROLLBACK_v4 | ROLLBACK_v4_MAIN3_144659 | M2_MAIN_OFF | 12926df2 | unset | unset | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 43.5 | 752.76 | 2297/3243/10581 | 62.0/72.8/74.8 | 67.0 | 10594 | True | 14:50:43 |
| ROLLBACK_v4 | ROLLBACK_v4_MAIN3_144659 | M3_MAIN_OFF | 12926df2 | unset | unset | hotmap_v2.json | MAIN_SHORT | OFF | vllm | 256/256 | 131786/32768 | 44.0 | 744.93 | 2217/2996/10761 | 63.9/74.5/76.6 | 69.6 | 10606 | True | 14:51:47 |

## 3. variant × workload 통계 (유효 세션만; sd ddof=1; pooled = Σtok/Σdur)

| variant | workload | mode | n | boots | mean | median | sd | min | max | pooled | TTFT p95 mean | TPOT p95 mean | invalid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 [post_fix] | LONGER_PREFILL | OFF | 4 | 4 | 164.26 | 164.35 | 0.90 | 163.1 | 165.3 | 164.26 | 3303 | 42.0 | 0 |
| A1 [post_fix] | LOW_CONCURRENCY | OFF | 4 | 4 | 63.46 | 63.47 | 0.24 | 63.2 | 63.7 | 63.46 | 290 | 14.0 | 0 |
| A1 [post_fix] | MAIN_SHORT | OFF | 12 | 4 | 755.70 | 749.60 | 15.31 | 740.2 | 785.0 | 755.42 | 3108 | 72.0 | 1 |
| A12 [post_fix] | LONGER_PREFILL | OFF | 5 | 5 | 166.14 | 167.37 | 3.14 | 160.5 | 167.9 | 166.09 | 3256 | 41.6 | 0 |
| A12 [post_fix] | LOW_CONCURRENCY | OFF | 5 | 5 | 63.30 | 63.28 | 0.18 | 63.1 | 63.6 | 63.30 | 298 | 14.0 | 0 |
| A12 [post_fix] | MAIN_SHORT | OFF | 15 | 5 | 787.48 | 786.37 | 18.76 | 767.6 | 822.9 | 787.07 | 3084 | 68.6 | 1 |
| A12e [post_fix] | LONGER_PREFILL | OFF | 5 | 5 | 165.10 | 165.11 | 1.05 | 163.9 | 166.3 | 165.09 | 3276 | 41.7 | 0 |
| A12e [post_fix] | LOW_CONCURRENCY | OFF | 5 | 5 | 63.30 | 63.37 | 0.20 | 63.1 | 63.5 | 63.30 | 305 | 13.9 | 0 |
| A12e [post_fix] | MAIN_SHORT | OFF | 15 | 5 | 799.76 | 800.65 | 8.47 | 788.1 | 819.5 | 799.67 | 3085 | 65.6 | 3 |
| A1ae [post_fix] | LONGER_PREFILL | OFF | 3 | 3 | 164.97 | 165.08 | 0.70 | 164.2 | 165.6 | 164.97 | 3213 | 42.0 | 0 |
| A1ae [post_fix] | LOW_CONCURRENCY | OFF | 3 | 3 | 63.32 | 63.40 | 0.23 | 63.1 | 63.5 | 63.32 | 287 | 14.1 | 0 |
| A1ae [post_fix] | MAIN_SHORT | OFF | 9 | 3 | 770.62 | 772.11 | 7.54 | 754.8 | 782.6 | 770.55 | 3127 | 69.8 | 6 |
| A1e [post_fix] | LONGER_PREFILL | OFF | 4 | 4 | 164.96 | 165.29 | 0.78 | 163.8 | 165.4 | 164.95 | 3217 | 41.9 | 0 |
| A1e [post_fix] | LOW_CONCURRENCY | OFF | 4 | 4 | 63.32 | 63.30 | 0.14 | 63.2 | 63.5 | 63.32 | 310 | 14.0 | 0 |
| A1e [post_fix] | MAIN_SHORT | OFF | 12 | 4 | 767.72 | 768.92 | 5.00 | 757.2 | 773.7 | 767.69 | 3124 | 69.7 | 2 |
| A2 [post_fix] | LONGER_PREFILL | OFF | 4 | 4 | 165.31 | 165.49 | 1.76 | 163.2 | 167.0 | 165.30 | 3301 | 41.7 | 0 |
| A2 [post_fix] | LOW_CONCURRENCY | OFF | 4 | 4 | 63.36 | 63.36 | 0.25 | 63.1 | 63.6 | 63.36 | 295 | 13.9 | 0 |
| A2 [post_fix] | MAIN_SHORT | OFF | 12 | 4 | 780.96 | 778.71 | 11.83 | 766.7 | 810.6 | 780.80 | 3095 | 69.4 | 2 |
| A2B [post_fix] | LONGER_PREFILL | OFF | 3 | 3 | 164.46 | 164.40 | 0.70 | 163.8 | 165.2 | 164.46 | 3171 | 42.2 | 0 |
| A2B [post_fix] | LOW_CONCURRENCY | OFF | 3 | 3 | 63.03 | 63.01 | 0.20 | 62.8 | 63.2 | 63.03 | 290 | 14.2 | 0 |
| A2B [post_fix] | MAIN_SHORT | OFF | 9 | 3 | 712.58 | 702.28 | 22.51 | 695.2 | 764.9 | 711.98 | 3203 | 77.3 | 0 |
| A2B [post_fix] | CONFIRM | OFF | 6 | 3 | 725.85 | 720.03 | 17.75 | 708.2 | 755.5 | 725.49 | 3002 | 69.8 | 0 |
| B [post_fix] | LONGER_PREFILL | OFF | 1 | 1 | 162.03 | 162.03 | null | 162.0 | 162.0 | 162.03 | 3208 | 42.9 | 0 |
| B [post_fix] | LOW_CONCURRENCY | OFF | 1 | 1 | 63.01 | 63.01 | null | 63.0 | 63.0 | 63.01 | 289 | 14.2 | 0 |
| B [post_fix] | MAIN_SHORT | OFF | 3 | 1 | 696.87 | 705.90 | 26.50 | 667.0 | 717.7 | 696.19 | 3203 | 79.6 | 4 |
| B [post_fix] | CONFIRM | OFF | 2 | 1 | 710.44 | 710.44 | 10.04 | 703.3 | 717.5 | 710.37 | 3013 | 74.0 | 0 |
| B_DEC1_SEL [post_fix] | SELECT | OFF | 4 | 2 | 504.07 | 504.17 | 4.28 | 500.0 | 508.0 | 504.04 | 6872 | 102.7 | 0 |
| B_H0_SEL [post_fix] | SELECT | OFF | 4 | 2 | 645.25 | 645.13 | 10.78 | 634.2 | 656.5 | 645.11 | 2995 | 82.4 | 0 |
| B_MIX1_SEL [post_fix] | SELECT | OFF | 4 | 2 | 682.35 | 683.70 | 3.23 | 677.5 | 684.4 | 682.33 | 3104 | 76.0 | 0 |
| DIAG_legacycb [post_fix] | MAIN_SHORT | OFF | 9 | 3 | 642.58 | 640.45 | 9.73 | 629.9 | 662.4 | 642.45 | 4176 | 80.9 | 0 |
| DIAG_nooverlap [post_fix] | MAIN_SHORT | OFF | 9 | 3 | 704.67 | 704.34 | 4.01 | 699.2 | 709.9 | 704.65 | 3204 | 79.1 | 0 |
| DIAG_noskipimm [post_fix] | MAIN_SHORT | OFF | 9 | 3 | 652.60 | 651.86 | 5.19 | 644.4 | 662.4 | 652.56 | 4089 | 84.8 | 0 |
| PROBE_nan [post_fix] | MAIN_SHORT | OFF | 66 | 22 | 731.75 | 734.83 | 20.40 | 684.0 | 777.1 | 731.19 | 3368 | 73.4 | 9 |
| R0 [post_fix] | LONGER_PREFILL | OFF | 8 | 8 | 164.43 | 164.59 | 2.18 | 161.4 | 166.7 | 164.41 | 3332 | 41.9 | 0 |
| R0 [post_fix] | LOW_CONCURRENCY | OFF | 8 | 8 | 63.29 | 63.33 | 0.21 | 63.0 | 63.5 | 63.29 | 306 | 14.0 | 0 |
| R0 [post_fix] | MAIN_SHORT | OFF | 24 | 8 | 743.95 | 744.82 | 7.73 | 728.7 | 754.6 | 743.87 | 3103 | 73.5 | 2 |
| R0C [post_fix] | LONGER_PREFILL | OFF | 3 | 3 | 163.78 | 163.80 | 2.32 | 161.4 | 166.1 | 163.76 | 3409 | 42.1 | 0 |
| R0C [post_fix] | LOW_CONCURRENCY | OFF | 3 | 3 | 63.19 | 63.15 | 0.30 | 62.9 | 63.5 | 63.19 | 307 | 14.0 | 0 |
| R0C [post_fix] | MAIN_SHORT | OFF | 9 | 3 | 747.02 | 743.67 | 14.85 | 730.1 | 783.6 | 746.77 | 3111 | 73.0 | 0 |
| R0C [post_fix] | CONFIRM | OFF | 6 | 3 | 663.12 | 662.72 | 10.86 | 650.6 | 677.8 | 662.97 | 2917 | 80.2 | 0 |
| R0_REF [post_fix] | LONGER_PREFILL | OFF | 2 | 2 | 165.82 | 165.82 | 0.70 | 165.3 | 166.3 | 165.82 | 3164 | 41.9 | 0 |
| R0_REF [post_fix] | LOW_CONCURRENCY | OFF | 2 | 2 | 63.29 | 63.29 | 0.52 | 62.9 | 63.7 | 63.29 | 294 | 13.9 | 0 |
| R0_REF [post_fix] | MAIN_SHORT | OFF | 6 | 2 | 753.74 | 754.01 | 14.35 | 735.1 | 772.5 | 753.51 | 3125 | 72.6 | 1 |
| A1 [pre_fix] | LONGER_PREFILL | OFF | 1 | 1 | 165.99 | 165.99 | null | 166.0 | 166.0 | 165.99 | 3156 | 41.8 | 0 |
| A1 [pre_fix] | LOW_CONCURRENCY | OFF | 1 | 1 | 63.51 | 63.51 | null | 63.5 | 63.5 | 63.51 | 276 | 14.0 | 0 |
| A1 [pre_fix] | MAIN_SHORT | OFF | 3 | 1 | 746.27 | 744.34 | 8.38 | 739.0 | 755.4 | 746.21 | 3124 | 73.2 | 0 |
| A1e [pre_fix] | LONGER_PREFILL | OFF | 1 | 1 | 165.95 | 165.95 | null | 165.9 | 165.9 | 165.95 | 3177 | 41.8 | 0 |
| A1e [pre_fix] | LOW_CONCURRENCY | OFF | 1 | 1 | 63.12 | 63.12 | null | 63.1 | 63.1 | 63.12 | 311 | 14.0 | 0 |
| A1e [pre_fix] | MAIN_SHORT | OFF | 3 | 1 | 759.07 | 755.36 | 7.11 | 754.6 | 767.3 | 759.03 | 3131 | 70.6 | 0 |
| CALIB_R0 [pre_fix] | CALIB | CORR | 1 | 1 | 618.95 | 618.95 | null | 618.9 | 618.9 | 618.95 | 3456 | 84.3 | 0 |
| CALIB_R0 [pre_fix] | CALIB_B | CORR | 1 | 1 | 622.44 | 622.44 | null | 622.4 | 622.4 | 622.44 | 3102 | 81.3 | 0 |
| DIAG_v4_legacycb [pre_fix] | MAIN_SHORT | OFF | 3 | 1 | 657.47 | 657.58 | 2.41 | 655.0 | 659.8 | 657.46 | 4203 | 79.0 | 0 |
| R0 [pre_fix] | LONGER_PREFILL | OFF | 1 | 1 | 165.41 | 165.41 | null | 165.4 | 165.4 | 165.41 | 3177 | 41.9 | 0 |
| R0 [pre_fix] | LOW_CONCURRENCY | OFF | 1 | 1 | 63.08 | 63.08 | null | 63.1 | 63.1 | 63.08 | 305 | 14.0 | 0 |
| R0 [pre_fix] | MAIN_SHORT | OFF | 3 | 1 | 735.31 | 736.81 | 4.73 | 730.0 | 739.1 | 735.29 | 3130 | 74.3 | 3 |
| R0_REF [pre_fix] | LONGER_PREFILL | OFF | 1 | 1 | 163.01 | 163.01 | null | 163.0 | 163.0 | 163.01 | 3443 | 41.8 | 0 |
| R0_REF [pre_fix] | LOW_CONCURRENCY | OFF | 1 | 1 | 53.23 | 53.23 | null | 53.2 | 53.2 | 53.23 | 330 | 26.3 | 0 |
| R0_REF [pre_fix] | MAIN_SHORT | OFF | 3 | 1 | 746.31 | 745.99 | 1.10 | 745.4 | 747.5 | 746.31 | 3058 | 73.8 | 0 |
| ROLLBACK_v4 [pre_fix] | MAIN_SHORT | OFF | 3 | 1 | 746.48 | 744.93 | 5.66 | 741.7 | 752.8 | 746.45 | 3138 | 73.7 | 0 |
| CFFIX_v4 [sync_only] | MAIN_SHORT | OFF | 3 | 1 | 750.31 | 743.86 | 18.19 | 736.2 | 770.8 | 750.01 | 3111 | 72.9 | 2 |

R0 대비 이득 (MAIN OFF; 총 개선 = 평균비 − 1, 짝 = 부팅 순서 짝; 퍼센트 합산 없음):

```
{
 "post_fix|A1": {
  "population": "post_fix",
  "pair_blocks": [
   "p1",
   "p2",
   "p3",
   "p4"
  ],
  "total_gain_vs_R0_mean": 0.015798023385537263,
  "paired_block_gains": [
   0.026424280446206705,
   0.004809548217203918,
   0.022609294459967844,
   0.0089040104658058
  ],
  "all_pairs_positive": true,
  "n_pairs": 4,
  "ttft_p95_ratio": 1.0018342707573014,
  "tpot_p95_ratio": 0.9789496557658912
 },
 "post_fix|A12": {
  "population": "post_fix",
  "pair_blocks": [
   "p1",
   "p3",
   "p4",
   "p5",
   "p6"
  ],
  "total_gain_vs_R0_mean": 0.05851140932056076,
  "paired_block_gains": [
   0.10506294856096887,
   0.045027926276181685,
   0.057578518308039506,
   0.04156827128514795,
   0.04182607618275602
  ],
  "all_pairs_positive": true,
  "n_pairs": 5,
  "ttft_p95_ratio": 0.9939255362356788,
  "tpot_p95_ratio": 0.9333900100632703
 },
 "post_fix|A12e": {
  "population": "post_fix",
  "pair_blocks": [
   "p1",
   "p2",
   "p4",
   "p5",
   "p6"
  ],
  "total_gain_vs_R0_mean": 0.07501721431497788,
  "paired_block_gains": [
   0.06703676157471472,
   0.08084622666979602,
   0.09434821278182781,
   0.06973327535223661,
   0.07327722559663674
  ],
  "all_pairs_positive": true,
  "n_pairs": 5,
  "ttft_p95_ratio": 0.9942944721925842,
  "tpot_p95_ratio": 0.8925506239274253
 },
 "post_fix|A1ae": {
  "population": "post_fix",
  "pair_blocks": [
   "p1",
   "p5",
   "p6"
  ],
  "total_gain_vs_R0_mean": 0.035848810937614495,
  "paired_block_gains": [
   0.038657335471576904,
   0.03955161266331175,
   0.03198708458742261
  ],
  "all_pairs_positive": true,
  "n_pairs": 3,
  "ttft_p95_ratio": 1.007959456090621,
  "tpot_p95_ratio": 0.9493424633868712
 },
 "post_fix|A1e": {
  "population": "post_fix",
  "pair_blocks": [
   "p2",
   "p3",
   "p4",
   "p5"
  ],
  "total_gain_vs_R0_mean": 0.03195417930385136,
  "paired_block_gains": [
   0.03469926387382527,
   0.018046682914119483,
   0.03929279513480366,
   0.031547338163594896
  ],
  "all_pairs_positive": true,
  "n_pairs": 4,
  "ttft_p95_ratio": 1.0070009222441294,
  "tpot_p95_ratio": 0.9480131800435435
 },
 "post_fix|A2": {
  "population": "post_fix",
  "pair_blocks": [
   "p1",
   "p2",
   "p3",
   "p4"
  ],
  "total_gain_vs_R0_mean": 0.04975209240882594,
  "paired_block_gains": [
   0.04259662816122356,
   0.06037507815708887,
   0.04538374599132844,
   0.0503099383755643
  ],
  "all_pairs_positive": true,
  "n_pairs": 4,
  "ttft_p95_ratio": 0.9973781717540279,
  "tpot_p95_ratio": 0.9437560516820525
 },
 "post_fix|A2B": {
  "population": "post_fix",
  "pair_blocks": null,
  "total_gain_vs_R0_mean": -0.04216323777161479,
  "paired_block_gains": [
   -0.05157821507876992,
   -0.06361634911293446,
   -0.00661918531281902
  ],
  "all_pairs_positive": false,
  "n_pairs": 3,
  "ttft_p95_ratio": 1.0324740513111428,
  "tpot_p95_ratio": 1.0510073766841876
 },
 "post_fix|B": {
  "population": "post_fix",
  "pair_blocks": null,
  "total_gain_vs_R0_mean": -0.06328087995086751,
  "paired_block_gains": [
   -0.0599036756762924
  ],
  "all_pairs_positive": false,
  "n_pairs": 1,
  "ttft_p95_ratio": 1.0323320469168258,
  "tpot_p95_ratio": 1.0828136728355522
 },
 "post_fix|DIAG_legacycb": {
  "population": "post_fix",
  "pair_blocks": null,
  "total_gain_vs_R0_mean": -0.13626137851634101,
  "paired_block_gains": [
   -0.13845171203077056,
   -0.12344879942868836,
   -0.14285293044127922
  ],
  "all_pairs_positive": false,
  "n_pairs": 3,
  "ttft_p95_ratio": 1.3460770861602622,
  "tpot_p95_ratio": 1.1002067651541563
 },
 "post_fix|DIAG_nooverlap": {
  "population": "post_fix",
  "pair_blocks": null,
  "total_gain_vs_R0_mean": -0.05280214872879274,
  "paired_block_gains": [
   -0.049637314931818444,
   -0.05405119004143,
   -0.050208239093876506
  ],
  "all_pairs_positive": false,
  "n_pairs": 3,
  "ttft_p95_ratio": 1.0327806944996705,
  "tpot_p95_ratio": 1.0757385634747239
 },
 "post_fix|DIAG_noskipimm": {
  "population": "post_fix",
  "pair_blocks": null,
  "total_gain_vs_R0_mean": -0.12279170182959465,
  "paired_block_gains": [
   -0.11971007193849603,
   -0.12035716308157385,
   -0.1241522269809252
  ],
  "all_pairs_positive": false,
  "n_pairs": 3,
  "ttft_p95_ratio": 1.317967365284665,
  "tpot_p95_ratio": 1.1533170814872347
 },
 "post_fix|PROBE_nan": {
  "population": "post_fix",
  "pair_blocks": null,
  "total_gain_vs_R0_mean": -0.016390665898224133,
  "paired_block_gains": [
   -0.001868645055904783,
   -0.010021973856032207,
   0.021988401405307334,
   -0.007260697800823168,
   0.010838217495080515,
   -0.007684671778860119,
   -0.02133812065996965,
   -0.015033087294011649
  ],
  "all_pairs_positive": false,
  "n_pairs": 8,
  "ttft_p95_ratio": 1.0854868930799197,
  "tpot_p95_ratio": 0.9983120533402524
 },
 "post_fix|R0C": {
  "population": "post_fix",
  "pair_blocks": null,
  "total_gain_vs_R0_mean": 0.004132175649382619,
  "paired_block_gains": [
   0.0075314207105894315,
   0.008499733914091978,
   0.0011131045474879198
  ],
  "all_pairs_positive": true,
  "n_pairs": 3,
  "ttft_p95_ratio": 1.0025951652424832,
  "tpot_p95_ratio": 0.9933528238199952
 },
 "post_fix|R0_REF": {
  "population": "post_fix",
  "pair_blocks": [
   "p1",
   "p3"
  ],
  "total_gain_vs_R0_mean": 0.013160415901601885,
  "paired_block_gains": [
   0.0005808411338770192,
   0.01838729900395819
  ],
  "all_pairs_positive": true,
  "n_pairs": 2,
  "ttft_p95_ratio": 1.0071702418296533,
  "tpot_p95_ratio": 0.9879161300133134
 },
 "pre_fix|A1": {
  "population": "pre_fix",
  "pair_blocks": null,
  "total_gain_vs_R0_mean": 0.014907892634564268,
  "paired_block_gains": [
   0.014907892634564268
  ],
  "all_pairs_positive": true,
  "n_pairs": 1,
  "ttft_p95_ratio": 0.9980234669212973,
  "tpot_p95_ratio": 0.9855422083640607
 },
 "pre_fix|A1e": {
  "population": "pre_fix",
  "pair_blocks": null,
  "total_gain_vs_R0_mean": 0.03231627531303194,
  "paired_block_gains": [
   0.03231627531303194
  ],
  "all_pairs_positive": true,
  "n_pairs": 1,
  "ttft_p95_ratio": 1.0003283312364344,
  "tpot_p95_ratio": 0.951043221467536
 },
 "pre_fix|DIAG_v4_legacycb": {
  "population": "pre_fix",
  "pair_blocks": null,
  "total_gain_vs_R0_mean": -0.10585909288582585,
  "paired_block_gains": [
   -0.10585909288582585
  ],
  "all_pairs_positive": false,
  "n_pairs": 1,
  "ttft_p95_ratio": 1.3427968332373397,
  "tpot_p95_ratio": 1.063532883933962
 },
 "pre_fix|R0_REF": {
  "population": "pre_fix",
  "pair_blocks": null,
  "total_gain_vs_R0_mean": 0.014955055285566399,
  "paired_block_gains": [
   0.014955055285566399
  ],
  "all_pairs_positive": true,
  "n_pairs": 1,
  "ttft_p95_ratio": 0.9768737248401421,
  "tpot_p95_ratio": 0.9941690315963178
 },
 "pre_fix|ROLLBACK_v4": {
  "population": "pre_fix",
  "pair_blocks": null,
  "total_gain_vs_R0_mean": 0.01519140991510981,
  "paired_block_gains": [
   0.01519140991510981
  ],
  "all_pairs_positive": true,
  "n_pairs": 1,
  "ttft_p95_ratio": 1.0024274110615714,
  "tpot_p95_ratio": 0.992838702173633
 }
}
```

## 4. expert replay 원값 (컨테이너 CPU 전용, 난수 가중치, p50/p10/p90 µs; replay/<variant>/expert_results.csv)

| variant | scenario | n | n_unique | rows/expert | p50 | p10 | p90 | min |
|---|---|---|---|---|---|---|---|---|
| a1off | hot_r1 | 40 | 160 | 1 | 9950.6 | 9897.3 | 10109.8 | 9843.5 |
| a1off | hot_r1_e18 | 40 | 16 | 1 | 837.6 | 803.5 | 963.9 | 780.2 |
| a1off | hot_r2 | 40 | 160 | 2 | 11190.0 | 11092.0 | 11289.0 | 11060.1 |
| a1off | hot_r2_e18 | 40 | 16 | 2 | 1050.6 | 1017.0 | 1082.8 | 1009.4 |
| a1off | hot_r3 | 40 | 160 | 3 | 12865.0 | 12508.4 | 13099.3 | 12474.1 |
| a1off | seq_mixed | 40 | 159 | p50=3,max=8 | 12330.7 | 12083.0 | 12806.9 | 11899.2 |
| a1on | hot_r1 | 40 | 160 | 1 | 9967.2 | 9915.2 | 10266.6 | 9842.9 |
| a1on | hot_r1_e18 | 40 | 16 | 1 | 837.1 | 807.2 | 968.2 | 787.1 |
| a1on | hot_r2 | 40 | 160 | 2 | 11020.7 | 10942.7 | 11337.8 | 10916.6 |
| a1on | hot_r2_e18 | 40 | 16 | 2 | 1029.5 | 1019.3 | 1046.7 | 1006.0 |
| a1on | hot_r3 | 40 | 160 | 3 | 12784.3 | 12364.4 | 12973.9 | 12315.6 |
| a1on | seq_mixed | 40 | 159 | p50=3,max=8 | 12128.3 | 11862.9 | 12454.5 | 11674.7 |
| a1proof | hot_r1_e18 | 5 | 16 | 1 | 1497.9 | 1487.4 | 1538.7 | 1487.4 |
| nscale | hot_r1_n128 | 40 | 128 | 1 | 7838.9 | 7779.2 | 8076.3 | 7725.9 |
| nscale | hot_r1_n16 | 40 | 16 | 1 | 771.3 | 742.4 | 809.0 | 735.4 |
| nscale | hot_r1_n32 | 40 | 32 | 1 | 1925.2 | 1885.3 | 1977.3 | 1869.5 |
| nscale | hot_r1_n64 | 40 | 64 | 1 | 4068.5 | 4034.6 | 4110.6 | 4011.6 |
| nscale | hot_r1_n8 | 40 | 8 | 1 | 325.0 | 320.7 | 341.3 | 318.7 |
| nscale | hot_r2_n128 | 40 | 128 | 2 | 8578.4 | 8375.7 | 8811.2 | 8316.0 |
| nscale | hot_r2_n16 | 40 | 16 | 2 | 934.2 | 909.6 | 967.9 | 900.1 |
| nscale | hot_r2_n32 | 40 | 32 | 2 | 2204.5 | 2180.9 | 2233.3 | 2169.1 |
| nscale | hot_r2_n64 | 40 | 64 | 2 | 4063.4 | 4020.4 | 4178.6 | 3998.3 |
| nscale | hot_r2_n8 | 40 | 8 | 2 | 445.6 | 432.4 | 480.6 | 428.4 |
| pf1 | hot_r1 | 40 | 160 | 1 | 9490.5 | 9406.9 | 9603.7 | 9382.8 |
| pf1 | hot_r1_e18 | 40 | 16 | 1 | 728.3 | 697.0 | 839.7 | 678.1 |
| pf1 | hot_r2 | 40 | 160 | 2 | 9974.8 | 9898.7 | 10081.0 | 9870.2 |
| pf1 | hot_r2_e18 | 40 | 16 | 2 | 857.7 | 839.3 | 916.6 | 820.5 |
| pf2 | hot_r1 | 40 | 160 | 1 | 9436.5 | 9368.5 | 9581.4 | 9290.1 |
| pf2 | hot_r1_e18 | 40 | 16 | 1 | 760.1 | 713.6 | 923.9 | 705.2 |
| pf2 | hot_r2 | 40 | 160 | 2 | 10005.5 | 9903.4 | 10083.8 | 9882.6 |
| pf2 | hot_r2_e18 | 40 | 16 | 2 | 855.7 | 840.2 | 890.1 | 826.0 |
| pf4 | hot_r1 | 40 | 160 | 1 | 9550.3 | 9448.4 | 9736.5 | 9431.3 |
| pf4 | hot_r1_e18 | 40 | 16 | 1 | 730.5 | 700.4 | 859.0 | 687.5 |
| pf4 | hot_r2 | 40 | 160 | 2 | 10058.5 | 10025.0 | 10156.7 | 9954.5 |
| pf4 | hot_r2_e18 | 40 | 16 | 2 | 870.4 | 852.9 | 893.7 | 840.2 |
| rb_off | hot_r1 | 40 | 160 | 1 | 10780.4 | 10715.7 | 10892.0 | 10682.8 |
| rb_off | hot_r1_e18 | 40 | 16 | 1 | 1058.5 | 997.3 | 1145.6 | 983.5 |
| rb_off | hot_r2 | 40 | 160 | 2 | 11802.8 | 11710.2 | 12012.4 | 11668.2 |
| rb_off | hot_r2_e18 | 40 | 16 | 2 | 1215.6 | 1195.8 | 1263.4 | 1188.3 |
| rb_off | hot_r3 | 40 | 160 | 3 | 13602.6 | 13413.9 | 14010.4 | 13280.7 |
| rb_off | seq_mixed | 40 | 159 | p50=3,max=8 | 13852.7 | 13707.9 | 14150.4 | 13412.7 |
| rb_on | hot_r1 | 40 | 160 | 1 | 9890.4 | 9770.5 | 10036.7 | 9726.8 |
| rb_on | hot_r1_e18 | 40 | 16 | 1 | 829.6 | 804.0 | 986.8 | 785.7 |
| rb_on | hot_r2 | 40 | 160 | 2 | 10585.6 | 10490.5 | 11202.8 | 10448.2 |
| rb_on | hot_r2_e18 | 40 | 16 | 2 | 955.7 | 941.5 | 976.5 | 927.9 |
| rb_on | hot_r3 | 40 | 160 | 3 | 12116.4 | 11809.1 | 12663.6 | 11786.1 |
| rb_on | seq_mixed | 40 | 159 | p50=3,max=8 | 11918.2 | 11622.1 | 12049.9 | 11554.7 |

## 5. feature proof

```
{
 "a1_requested": "1",
 "a1_effective": true,
 "a2_effective": false,
 "proof_on": true,
 "a1_policy": "a1-v1: GFNI lo-nibble unpack (0x0000000001020408) in avx_rb_rows<RB,SPLIT>, all RB; integer accumulation order unchanged",
 "a1_eligible_calls": 112640,
 "a1_r1_calls": 112640,
 "a1_r2_calls": 0,
 "a1_r4_calls": 0,
 "a1_r8_calls": 0,
 "a1_fallback_not_enabled": 0,
 "a2_eligible_jobs": 0,
 "a2_bundled_jobs": 0,
 "a2_fallback_jobs": 0,
 "scenario": "hot_r1_e18 x (5 warm + 5 iters), 16 experts R=1",
 "binary_sha256": "941a83c43187789427aa74335b60233b6f453b99ae97bbf2586e425fc2b6219a (v5: v4 + A1-a + kt_opt)",
 "note": "eligible_calls = avx_rb_rows 호출 수 (RB 블록 단위); R=1 만 존재하는 시나리오라 r1_calls==eligible. 정수 결과는 a1on==a1off==v4 bitwise 동일 (6 시나리오)"
}
```

### a1_flag_test.log

```
validate True
{'a1_requested': 'unset', 'a1_effective': False, 'a2_effective': False, 'proof_on': False, 'a1_policy': 'a1-v1: GFNI lo-nibble unpack (0x0000000001020408) in avx_rb_rows<RB,SPLIT>, all RB; integer accumulation order unchanged', 'a1_eligible_calls': 0, 'a1_r1_calls': 0, 'a1_r2_calls': 0, 'a1_r4_calls': 0, 'a1_r8_calls': 0, 'a1_fallback_not_enabled': 0, 'a2_eligible_jobs': 0, 'a2_bundled_jobs': 0, 'a2_fallback_jobs': 0}
Traceback (most recent call last):
  File "<string>", line 1, in <module>
RuntimeError: kt_opt: invalid value for KT_OPT_A1_ENABLE = 'true' (only 0 or 1 allowed)
time="2026-09-17T23:08:36+09:00" level=fatal msg="exec failed with exit code 1"
invalid-value exit=1

```

### a1_fp8_regression.log

```
== test_fp8_perchannel_moe.py
Mean relative L1 diff: 0.5829%
PASS: Mean error 0.5829% < 15.0% threshold
== test_fp8_moe.py
Mean relative L1 diff: 0.5829%
PASS: Mean error 0.5829% < 15.0% threshold

```

### v6_fp8_regression.log

```
== test_fp8_perchannel_moe.py
Mean relative L1 diff: 0.5829%
PASS: Mean error 0.5829% < 15.0% threshold
== test_fp8_moe.py
Mean relative L1 diff: 0.5829%
PASS: Mean error 0.5829% < 15.0% threshold

```

### v7_fp8_regression.log

```
== test_fp8_perchannel_moe.py
Mean relative L1 diff: 0.5829%
PASS: Mean error 0.5829% < 15.0% threshold
== test_fp8_moe.py
Mean relative L1 diff: 0.5829%
PASS: Mean error 0.5829% < 15.0% threshold

```

### replay4_a2probe.log (A2 pool probe)

```
PROBE {'a1_requested': '0', 'a1_effective': False, 'a2_effective': False, 'proof_on': True, 'a1_policy': 'a1-v1: GFNI lo-nibble unpack (0x0000000001020408) in avx_rb_rows<RB,SPLIT>, all RB; integer accumulation order unchanged', 'a1_eligible_calls': 3793856, 'a1_r1_calls': 0, 'a1_r2_calls': 0, 'a1_r4_calls': 0, 'a1_r8_calls': 0, 'a1_fallback_not_enabled': 3793856, 'pool_n_dispatch': 721, 'pool_sum_tasks': 697900, 'pool_sum_wall_ns': 1835316430, 'pool_sum_parent_ns': 1779653969, 'pool_sum_tail_ns': 67957629, 'pool_max_tail_ns': 20430377, 'pool_sum_worker_span_ns': 90322966, 'a2_eligible_jobs': 0, 'a2_bundled_jobs': 0, 'a2_fallback_jobs': 0}

```

### replay5_a2probe2_a1off.log (A2 pool probe)

```
PROBE {'a1_requested': '0', 'a1_effective': False, 'a2_effective': False, 'proof_on': True, 'a1_policy': 'a1-v1: GFNI lo-nibble unpack (0x0000000001020408) in avx_rb_rows<RB,SPLIT>, all RB; integer accumulation order unchanged', 'a1_eligible_calls': 3793856, 'a1_r1_calls': 0, 'a1_r2_calls': 0, 'a1_r4_calls': 0, 'a1_r8_calls': 0, 'a1_fallback_not_enabled': 3793856, 'pool_n_dispatch': 721, 'pool_sum_tasks': 697900, 'pool_sum_wall_ns': 1555155111, 'pool_sum_parent_ns': 1501889656, 'pool_sum_tail_ns': 56924289, 'pool_max_tail_ns': 18196968, 'pool_sum_worker_span_ns': 80210898, 'a2_eligible_jobs': 0, 'a2_bundled_jobs': 0, 'a2_fallback_jobs': 0}

```

### replay5_a2probe2_a1on.log (A2 pool probe)

```
PROBE {'a1_requested': '1', 'a1_effective': True, 'a2_effective': False, 'proof_on': True, 'a1_policy': 'a1-v1: GFNI lo-nibble unpack (0x0000000001020408) in avx_rb_rows<RB,SPLIT>, all RB; integer accumulation order unchanged', 'a1_eligible_calls': 3793856, 'a1_r1_calls': 1403776, 'a1_r2_calls': 1507264, 'a1_r4_calls': 856768, 'a1_r8_calls': 26048, 'a1_fallback_not_enabled': 0, 'pool_n_dispatch': 721, 'pool_sum_tasks': 697900, 'pool_sum_wall_ns': 1571385522, 'pool_sum_parent_ns': 1515460402, 'pool_sum_tail_ns': 68584022, 'pool_max_tail_ns': 18777982, 'pool_sum_worker_span_ns': 94016352, 'a2_eligible_jobs': 0, 'a2_bundled_jobs': 0, 'a2_fallback_jobs': 0}

```

## 6. 실행 원장·오류

```
 CTRL_noprobe c1 exit 0
2026-09-18T06:37:20+0900 ctrl boot CTRL_noprobe_MAIN3_062915 died — probe masks (hazard still present)
2026-09-18T06:37:20+0900 ctrl done
2026-09-18T06:42:00+0900 post3 start (died boots = failures, excluded)
0c14adef5967b853 ae763d078fe30137 df0480b0c67f17ee 90d2e73c2ac36ccc 319aaf8fb18688e1 
cpuinfer.h already 4ad59b792de3c501
ext_bindings patched 943dd90996979f72
experts_base patched 3b0efe2ced0d9552
cpuinfer.h 4ad59b792de3c501 cf-rearm
ext_bindings.cpp 943dd90996979f72 cf-rearm
experts_base.py 3b0efe2ced0d9552 cf-rearm
BUILD_EXIT=0
f437b6f1690d8687a8b6a4d1f0185839af81e8ee0e13d81b0e927264b39ea1a9  /sgl-workspace/ide076_backup/kt_kernel_ext.so.v4f
7
2026-09-18T06:45:06+0900 v4f built (v8f sources restored)
a4add14d4708940d91a2bbe61313b351fbc1a340d7a2e0257e5dd718010d69c1  /usr/local/lib/python3.12/dist-packages/kt_kernel/kt_kernel_ext.cpython-312-x86_64-linux-gnu.so
2026-09-18T06:46:08+0900 replay6 a2off exit 0
2026-09-18T06:47:04+0900 replay6 a2on exit 0
2026-09-18T06:48:01+0900 replay6 a12on exit 0
2026-09-18T06:48:58+0900 a2 probe3 exit 0
2026-09-18T06:48:58+0900 v8f replay done
f437b6f1690d8687a8b6a4d1f0185839af81e8ee0e13d81b0e927264b39ea1a9  /usr/local/lib/python3.12/dist-packages/kt_kernel/kt_kernel_ext.cpython-312-x86_64-linux-gnu.so
2026-09-18T06:56:51+0900 R0_REF p1 exit 0
a4add14d4708940d91a2bbe61313b351fbc1a340d7a2e0257e5dd718010d69c1  /usr/local/lib/python3.12/dist-packages/kt_kernel/kt_kernel_ext.cpython-312-x86_64-linux-gnu.so
2026-09-18T07:04:37+0900 R0 p1 exit 0
2026-09-18T07:12:26+0900 A1ae p1 exit 0
2026-09-18T07:21:41+0900 A1e p1 exit 0
2026-09-18T07:29:28+0900 A1 p1 exit 0
2026-09-18T07:37:15+0900 A2 p1 exit 0
2026-09-18T07:44:54+0900 A12 p1 exit 0
2026-09-18T07:52:38+0900 A12e p1 exit 0
2026-09-18T07:52:38+0900 post block p1 done
2026-09-18T08:00:26+0900 R0 p2 exit 0
2026-09-18T08:08:04+0900 A12e p2 exit 0
2026-09-18T08:16:13+0900 A12 p2 exit 0
2026-09-18T08:23:55+0900 A2 p2 exit 0
2026-09-18T08:31:43+0900 A1 p2 exit 0
2026-09-18T08:39:29+0900 A1e p2 exit 0
2026-09-18T08:47:54+0900 A1ae p2 exit 0
2026-09-18T08:47:54+0900 post block p2 done
2026-09-18T08:52:23+0900 B_H0_SEL s1 exit 0
2026-09-18T08:57:13+0900 B_DEC1_SEL s1 exit 0
2026-09-18T09:01:41+0900 B_MIX1_SEL s1 exit 0
2026-09-18T09:06:15+0900 B_H0_SEL s2 exit 0
2026-09-18T09:11:02+0900 B_DEC1_SEL s2 exit 0
2026-09-18T09:15:27+0900 B_MIX1_SEL s2 exit 0
2026-09-18T09:15:27+0900 bsel done
f437b6f1690d8687a8b6a4d1f0185839af81e8ee0e13d81b0e927264b39ea1a9  /usr/local/lib/python3.12/dist-packages/kt_kernel/kt_kernel_ext.cpython-312-x86_64-linux-gnu.so
2026-09-18T09:23:11+0900 R0_REF p3 exit 0
a4add14d4708940d91a2bbe61313b351fbc1a340d7a2e0257e5dd718010d69c1  /usr/local/lib/python3.12/dist-packages/kt_kernel/kt_kernel_ext.cpython-312-x86_64-linux-gnu.so
2026-09-18T09:30:58+0900 R0 p3 exit 0
2026-09-18T09:41:27+0900 A1ae p3 exit 0
2026-09-18T09:49:15+0900 A1e p3 exit 0
2026-09-18T09:57:02+0900 A1 p3 exit 0
2026-09-18T10:04:42+0900 A2 p3 exit 0
2026-09-18T10:12:25+0900 A12 p3 exit 0
2026-09-18T10:22:48+0900 A12e p3 exit 0
2026-09-18T10:22:48+0900 post block p3 done
2026-09-18T10:30:40+0900 R0 p4 exit 0
2026-09-18T10:38:12+0900 A12e p4 exit 0
2026-09-18T10:45:50+0900 A12 p4 exit 0
2026-09-18T10:53:33+0900 A2 p4 exit 0
2026-09-18T11:01:27+0900 A1 p4 exit 0
2026-09-18T11:09:15+0900 A1e p4 exit 0
2026-09-18T11:18:30+0900 A1ae p4 exit 0
2026-09-18T11:18:30+0900 post block p4 done
f437b6f1690d8687a8b6a4d1f0185839af81e8ee0e13d81b0e927264b39ea1a9  /usr/local/lib/python3.12/dist-packages/kt_kernel/kt_kernel_ext.cpython-312-x86_64-linux-gnu.so
2026-09-18T11:26:41+0900 R0_REF p5 exit 0
a4add14d4708940d91a2bbe61313b351fbc1a340d7a2e0257e5dd718010d69c1  /usr/local/lib/python3.12/dist-packages/kt_kernel/kt_kernel_ext.cpython-312-x86_64-linux-gnu.so
2026-09-18T11:34:23+0900 R0 p5 exit 0
2026-09-18T11:42:08+0900 A1ae p5 exit 0
2026-09-18T11:49:55+0900 A1e p5 exit 0
2026-09-18T11:58:05+0900 A1 p5 exit 0
2026-09-18T12:07:50+0900 A2 p5 exit 0
2026-09-18T12:15:35+0900 A12 p5 exit 0
2026-09-18T12:23:14+0900 A12e p5 exit 0
2026-09-18T12:23:14+0900 post block p5 done
2026-09-18T12:31:03+0900 R0 p6 exit 0
2026-09-18T12:38:41+0900 A12e p6 exit 0
2026-09-18T12:41:23+0900 post3/post4 chains stopped by operator (user time constraint): p6 reduced to A1ae only; CONFIRMATION reduced to R0C/B/A12B/A2B x3
2026-09-18T12:46:39+0900 post5 start
2026-09-18T12:54:28+0900 A1ae p6 exit 0
2026-09-18T12:54:28+0900 post block p6 done (A1ae only)
2026-09-18T12:54:28+0900 post3 chain done
2026-09-18T12:54:28+0900 post4 start (B CONFIRMATION with HB_MIX1; reduced: R0C/B/A12B/A2B)
2026-09-18T13:04:02+0900 R0C c1 exit 0
2026-09-18T13:15:01+0900 B c1 exit 0
2026-09-18T13:24:39+0900 A12B c1 exit 0
2026-09-18T13:34:11+0900 A2B c1 exit 0
2026-09-18T13:34:11+0900 post4 block c1 done
2026-09-18T13:43:36+0900 R0C c2 exit 0
2026-09-18T13:51:59+0900 B c2 exit 0
2026-09-18T14:00:09+0900 A12B c2 exit 0
2026-09-18T14:09:38+0900 A2B c2 exit 0
2026-09-18T14:09:38+0900 post4 block c2 done
2026-09-18T14:19:06+0900 R0C c3 exit 0
2026-09-18T14:28:45+0900 B c3 exit 0
2026-09-18T14:37:24+0900 A12B c3 exit 0
2026-09-18T14:46:51+0900 A2B c3 exit 0
2026-09-18T14:46:51+0900 post4 block c3 done
2026-09-18T14:46:51+0900 post4 chain done
2026-09-18T14:46:58+0900 rollback test start
12926df2c30b724694171108635637e8cda14064ae9a643c03b114d6c9d6b926  /usr/local/lib/python3.12/dist-packages/kt_kernel/kt_kernel_ext.cpython-312-x86_64-linux-gnu.so
fc44bf94770903a662675199d8d37b93cb10c4621d2e383350c21228b842a09f  /usr/local/lib/python3.12/dist-packages/kt_kernel/experts_base.py
2026-09-18T14:53:04+0900 ROLLBACK_v4 r1 exit 0
a4add14d4708940d91a2bbe61313b351fbc1a340d7a2e0257e5dd718010d69c1  /usr/local/lib/python3.12/dist-packages/kt_kernel/kt_kernel_ext.cpython-312-x86_64-linux-gnu.so
f8a6f0569d97978cfa4ca3d543985d62ab5e373472edc71d0e0fabadec7e3983  /usr/local/lib/python3.12/dist-packages/kt_kernel/experts_base.py
2026-09-18T14:53:04+0900 rollback test done (adopted stack restored)

```

원장 event 수 1142: {"boot_end": 115, "boot_start": 116, "calibration_dump": 2, "numprobe": 12, "server_died": 28, "session_end": 408, "smoke_end": 115, "variant_end": 115, "variant_start": 116, "warmup_end": 115}

## 7. 파일

- 문서: A1_BRANCH_PROOF.json, A1_DECISION.md, A1_GAP_ANALYSIS.md, A2_APPLICABILITY.md, A2_DECISION.md, ACCEPTANCE_CRITERIA.yaml, ANALYSIS_AND_DECISION.md, BASELINE_PROVENANCE.md, B_COST_MODEL.md, B_DECISION.md, CALIBRATION_MANIFEST.json, CLAUDE.md, CODE_MAP.md, COMPLETION_STATUS.md, EXECUTION_POLICY_TESTS.md, FULL_REPORT.md, INTEGRATION_DECISION.md, NUMERICAL_CONTRACT.md, PLAN.md, PLAN_RESOLVED.md, PROGRESS.md, README.md, ROLLBACK.md, SOURCE_MANIFEST.json, TOOL_INTERFACES.md, WORK_LOG.md, e2e_stats.json, execution_policy_test_results.json, task.md, test.md
- 원자료: eval/results/IDE_076_20260917/ (qwen/OPT4/<boot>/<session>/, replay/, calibration/, placements/, variants/, state/)
- 도구: eval/ide076/ (TOOL_INTERFACES.md)
