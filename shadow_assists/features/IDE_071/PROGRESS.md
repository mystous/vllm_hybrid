# IDE_071 PROGRESS (30분 보고 누적)

- 2026-09-16 00:20 KST 지시서 수령 → ID 등록, 브랜치 feat/cpu-offload-ide071, P0 환경 동결 (evidence/), 옵션 등록부, 하네스 작성.
- 00:50 셀 실행기 smoke (T00: hot96 def4, SHORT_COLD C16 32요청 1회) — 부팅 110 s, effective config 불일치 0, 비-kt 스레드 1,197개 재배치, 350.9 tok/s.
- 00:52 캠페인 시작: P1 (R00~R07) → P2 (20셀). 보고 루프 report_progress.py 기동.

# 중간 실행 보고 — 2026-09-16T00:50:40.114703+09:00

- campaign_id: IDE_071_20260916
- elapsed_seconds: 0
- utc: 2026-09-15T15:50:40.114703+00:00
- current_phase / cell_id / attempt / rep_id: None / None / None / None
- phase_state: IDLE
- registered / completed / failed / blocked / pending: 0 / 0 / 0 / 0 / 0
- last_completed_cells: 없음
- latest_measurements:

- latest_error: 없음
- retries_and_recovery: 0 셀 재시도
- HBM: 0, 0 MiB;1, 0 MiB;2, 0 MiB;3, 0 MiB;4, 0 MiB;5, 0 MiB;6, 0 MiB;7, 0 MiB;
- RAM: used 75 GB / free 354 GB; disk /data: 21T avail
- latest_persisted_artifacts: 없음
- next_registered_cells: 없음
- report_delivery_state: saved (세션 cron 이 전달 후 delivered 로 갱신)

# 중간 실행 보고 — 2026-09-16T01:20:42.008031+09:00

- campaign_id: IDE_071_20260916
- elapsed_seconds: 1801
- utc: 2026-09-15T16:20:42.008031+00:00
- current_phase / cell_id / attempt / rep_id: P1 / R00_r0_def4 / a1 / LEGACY_070_C64_rep1
- phase_state: MEASURING
- registered / completed / failed / blocked / pending: 4 / 4 / 0 / 0 / 0
- last_completed_cells: R02_d1_cf=COMPLETED, R06_v2_nu5952=COMPLETED, R07_gpu_only_tp8_ep8=COMPLETED, R04_d4_epoch=COMPLETED
- latest_measurements:
  - R04_d4_epoch/a1 LEGACY_070_C64_rep1: out_tps=817.5374145665195 ttft50=2376.5790119650774 tpot50=58.195536043201606 ok=256/256 valid=True
  - R04_d4_epoch/a1 LEGACY_070_C64_rep2: out_tps=849.0054602334179 ttft50=2530.4530595312826 tpot50=54.99730060632363 ok=256/256 valid=True
  - R04_d4_epoch/a1 LEGACY_070_C64_rep3: out_tps=784.1690904263121 ttft50=2962.3660430079326 tpot50=58.62306290504149 ok=256/256 valid=True
  - R04_d4_epoch/a1 SHORT_COLD_C64_rep1: out_tps=713.2451333059573 ttft50=2912.242538994178 tpot50=66.68152244061261 ok=256/256 valid=True
  - R04_d4_epoch/a1 SHORT_COLD_C64_rep2: out_tps=699.675535819436 ttft50=3168.5475604608655 tpot50=66.94472481553466 ok=256/256 valid=True
  - R04_d4_epoch/a1 SHORT_COLD_C64_rep3: out_tps=702.3221549874545 ttft50=3136.447592987679 tpot50=66.90294347615458 ok=256/256 valid=True
- latest_error: 없음
- retries_and_recovery: 0 셀 재시도
- HBM: 0, 80334 MiB;1, 80214 MiB;2, 80214 MiB;3, 79734 MiB;4, 0 MiB;5, 0 MiB;6, 0 MiB;7, 0 MiB;
- RAM: used 312 GB / free 46 GB; disk /data: 21T avail
- latest_persisted_artifacts: eval/results/IDE_071_20260916/R04_d4_epoch/a1/SHORT_COLD_C64_rep1/metrics.json, eval/results/IDE_071_20260916/R04_d4_epoch/a1/SHORT_COLD_C64_rep2/metrics.json, eval/results/IDE_071_20260916/R04_d4_epoch/a1/SHORT_COLD_C64_rep3/metrics.json
- next_registered_cells: R00_r0_def4, R01_r0_def0, R03_d3_cf_skip_def8_pin, R05_v2_u96
- report_delivery_state: saved (세션 cron 이 전달 후 delivered 로 갱신)

# 중간 실행 보고 — 2026-09-16T01:50:43.985795+09:00

- campaign_id: IDE_071_20260916
- elapsed_seconds: 3603
- utc: 2026-09-15T16:50:43.985795+00:00
- current_phase / cell_id / attempt / rep_id: P1 / R05_v2_u96 / a1 / SHORT_COLD_C64_rep2
- phase_state: MEASURING
- registered / completed / failed / blocked / pending: 8 / 7 / 1 / 0 / 0
- last_completed_cells: R06_v2_nu5952=COMPLETED, R07_gpu_only_tp8_ep8=COMPLETED, R04_d4_epoch=COMPLETED, R00_r0_def4=COMPLETED, R03_d3_cf_skip_def8_pin=FAILED_RUNTIME, R01_r0_def0=COMPLETED
- latest_measurements:
  - R00_r0_def4__anchor1/a1 SHORT_COLD_C64_rep1: out_tps=604.8295506350951 ttft50=3097.4090229719877 tpot50=81.99799684285163 ok=256/256 valid=True
  - R00_r0_def4__anchor1/a1 SHORT_COLD_C64_rep2: out_tps=614.7776524797105 ttft50=3050.7574634975754 tpot50=80.50672242584764 ok=256/256 valid=True
  - R01_r0_def0/a1 SHORT_COLD_C64_rep1: out_tps=577.5345343538698 ttft50=2924.462423485238 tpot50=88.01548584688851 ok=256/256 valid=True
  - R01_r0_def0/a1 SHORT_COLD_C64_rep2: out_tps=579.4805603964882 ttft50=2900.135300005786 tpot50=86.51300646432522 ok=256/256 valid=True
  - R01_r0_def0/a1 SHORT_COLD_C64_rep3: out_tps=577.1480735606349 ttft50=2853.0228104791604 tpot50=89.25859818480704 ok=256/256 valid=True
  - R05_v2_u96/a1 SHORT_COLD_C64_rep1: out_tps=622.9032246720977 ttft50=3113.6132700485177 tpot50=77.6970060707224 ok=256/256 valid=True
- latest_error: 2026-09-16T01:34:58.948093+09:00 R03_d3_cf_skip_def8_pin stage=MEASURING server not healthy after rep log=eval/results/IDE_071_20260916/R03_d3_cf_skip_def8_pin/a1/errors.jsonl
- retries_and_recovery: 0 셀 재시도
- HBM: 0, 77540 MiB;1, 77422 MiB;2, 77422 MiB;3, 76942 MiB;4, 0 MiB;5, 0 MiB;6, 0 MiB;7, 0 MiB;
- RAM: used 311 GB / free 56 GB; disk /data: 21T avail
- latest_persisted_artifacts: eval/results/IDE_071_20260916/R01_r0_def0/a1/SHORT_COLD_C64_rep2/metrics.json, eval/results/IDE_071_20260916/R01_r0_def0/a1/SHORT_COLD_C64_rep3/metrics.json, eval/results/IDE_071_20260916/R05_v2_u96/a1/SHORT_COLD_C64_rep1/metrics.json
- next_registered_cells: R05_v2_u96
- report_delivery_state: saved (세션 cron 이 전달 후 delivered 로 갱신)

# 중간 실행 보고 — 2026-09-16T02:20:45.943807+09:00

- campaign_id: IDE_071_20260916
- elapsed_seconds: 5405
- utc: 2026-09-15T17:20:45.943807+00:00
- current_phase / cell_id / attempt / rep_id: P2 / P2_cf1_skip0_pin0_def4 / a1 / None
- phase_state: WARMUP
- registered / completed / failed / blocked / pending: 13 / 12 / 1 / 0 / 0
- last_completed_cells: R01_r0_def0=COMPLETED, R05_v2_u96=COMPLETED, P2_cf1_skip1_pin0_def4=COMPLETED, P2_cf1_skip0_pin1_def4=COMPLETED, P2_cf1_skip0_pin0_def0=COMPLETED, P2_cf1_skip1_pin0_def8=COMPLETED
- latest_measurements:
  - P2_cf1_skip0_pin0_def0/a1 SHORT_COLD_C64_rep1: out_tps=591.8522163494574 ttft50=3287.796239485033 tpot50=82.51468385074757 ok=256/256 valid=True
  - P2_cf1_skip0_pin0_def0/a1 SHORT_COLD_C64_rep2: out_tps=598.0776421895763 ttft50=3335.9578929957934 tpot50=81.080828870346 ok=256/256 valid=True
  - P2_cf1_skip0_pin0_def0/a1 SHORT_COLD_C64_rep3: out_tps=596.392154212511 ttft50=3163.5328929987736 tpot50=85.50490360655712 ok=256/256 valid=True
  - P2_cf1_skip1_pin0_def8/a1 SHORT_COLD_C64_rep1: out_tps=725.772565366425 ttft50=2354.1355105116963 tpot50=69.97863501979508 ok=256/256 valid=True
  - P2_cf1_skip1_pin0_def8/a1 SHORT_COLD_C64_rep2: out_tps=744.0104447702321 ttft50=2309.653021511622 tpot50=68.99510151578303 ok=256/256 valid=True
  - P2_cf1_skip1_pin0_def8/a1 SHORT_COLD_C64_rep3: out_tps=739.3568143645108 ttft50=2327.2160229389556 tpot50=69.99538705489913 ok=256/256 valid=True
- latest_error: 2026-09-16T01:34:58.948093+09:00 R03_d3_cf_skip_def8_pin stage=MEASURING server not healthy after rep log=eval/results/IDE_071_20260916/R03_d3_cf_skip_def8_pin/a1/errors.jsonl
- retries_and_recovery: 0 셀 재시도
- HBM: 0, 79574 MiB;1, 79454 MiB;2, 79454 MiB;3, 78974 MiB;4, 0 MiB;5, 0 MiB;6, 0 MiB;7, 0 MiB;
- RAM: used 311 GB / free 56 GB; disk /data: 21T avail
- latest_persisted_artifacts: eval/results/IDE_071_20260916/P2_cf1_skip1_pin0_def8/a1/SHORT_COLD_C64_rep1/metrics.json, eval/results/IDE_071_20260916/P2_cf1_skip1_pin0_def8/a1/SHORT_COLD_C64_rep2/metrics.json, eval/results/IDE_071_20260916/P2_cf1_skip1_pin0_def8/a1/SHORT_COLD_C64_rep3/metrics.json
- next_registered_cells: P2_cf0_skip0_pin0_def0, P2_cf0_skip0_pin0_def4, P2_cf0_skip0_pin0_def8, P2_cf0_skip0_pin1_def0, P2_cf0_skip0_pin1_def4
- report_delivery_state: saved (세션 cron 이 전달 후 delivered 로 갱신)

# 중간 실행 보고 — 2026-09-16T02:50:47.925316+09:00

- campaign_id: IDE_071_20260916
- elapsed_seconds: 7207
- utc: 2026-09-15T17:50:47.925316+00:00
- current_phase / cell_id / attempt / rep_id: P2 / P2_cf1_skip1_pin1_def0 / a1 / None
- phase_state: BOOTING
- registered / completed / failed / blocked / pending: 18 / 17 / 1 / 0 / 0
- last_completed_cells: P2_cf1_skip1_pin0_def8=COMPLETED, P2_cf1_skip0_pin0_def4=COMPLETED, P2_cf1_skip0_pin0_def8=COMPLETED, P2_cf1_skip1_pin0_def0=COMPLETED, P2_cf0_skip0_pin0_def4=COMPLETED, P2_cf0_skip0_pin0_def8=COMPLETED
- latest_measurements:
  - P2_cf0_skip0_pin0_def4/a1 SHORT_COLD_C64_rep1: out_tps=615.5695696249272 ttft50=3110.989469510969 tpot50=79.48821171656192 ok=256/256 valid=True
  - P2_cf0_skip0_pin0_def4/a1 SHORT_COLD_C64_rep2: out_tps=606.5557424666554 ttft50=3042.126410466153 tpot50=81.80471858620189 ok=256/256 valid=True
  - P2_cf0_skip0_pin0_def4/a1 SHORT_COLD_C64_rep3: out_tps=614.9950364825752 ttft50=3031.906169548165 tpot50=80.80773199193445 ok=256/256 valid=True
  - P2_cf0_skip0_pin0_def8/a1 SHORT_COLD_C64_rep1: out_tps=648.3149693882558 ttft50=2879.1128550074063 tpot50=75.83391080302457 ok=256/256 valid=True
  - P2_cf0_skip0_pin0_def8/a1 SHORT_COLD_C64_rep2: out_tps=639.9367797461395 ttft50=2809.5286855241284 tpot50=77.11891998036391 ok=256/256 valid=True
  - P2_cf0_skip0_pin0_def8/a1 SHORT_COLD_C64_rep3: out_tps=641.387942411505 ttft50=2927.788804518059 tpot50=77.63689365393783 ok=256/256 valid=True
- latest_error: 2026-09-16T01:34:58.948093+09:00 R03_d3_cf_skip_def8_pin stage=MEASURING server not healthy after rep log=eval/results/IDE_071_20260916/R03_d3_cf_skip_def8_pin/a1/errors.jsonl
- retries_and_recovery: 0 셀 재시도
- HBM: 0, 1812 MiB;1, 1860 MiB;2, 1860 MiB;3, 1620 MiB;4, 0 MiB;5, 0 MiB;6, 0 MiB;7, 0 MiB;
- RAM: used 82 GB / free 285 GB; disk /data: 21T avail
- latest_persisted_artifacts: eval/results/IDE_071_20260916/P2_cf0_skip0_pin0_def8/a1/SHORT_COLD_C64_rep1/metrics.json, eval/results/IDE_071_20260916/P2_cf0_skip0_pin0_def8/a1/SHORT_COLD_C64_rep2/metrics.json, eval/results/IDE_071_20260916/P2_cf0_skip0_pin0_def8/a1/SHORT_COLD_C64_rep3/metrics.json
- next_registered_cells: P2_cf0_skip0_pin0_def0, P2_cf0_skip0_pin1_def0, P2_cf0_skip0_pin1_def4, P2_cf0_skip0_pin1_def8, P2_cf1_skip0_pin1_def0
- report_delivery_state: saved (세션 cron 이 전달 후 delivered 로 갱신)

# 중간 실행 보고 — 2026-09-16T03:20:49.880760+09:00

- campaign_id: IDE_071_20260916
- elapsed_seconds: 9009
- utc: 2026-09-15T18:20:49.880760+00:00
- current_phase / cell_id / attempt / rep_id: P2 / P2X_cf0_overlap_off / a1 / SHORT_COLD_C64_rep3
- phase_state: MEASURING
- registered / completed / failed / blocked / pending: 22 / 21 / 1 / 0 / 0
- last_completed_cells: P2_cf0_skip0_pin0_def4=COMPLETED, P2_cf0_skip0_pin0_def8=COMPLETED, P2_cf1_skip1_pin1_def0=COMPLETED, P2_cf1_skip1_pin1_def8=COMPLETED, P2_cf1_skip0_pin1_def0=COMPLETED, P2_cf0_skip0_pin1_def4=COMPLETED
- latest_measurements:
  - P2_cf1_skip0_pin1_def0/a1 SHORT_COLD_C64_rep3: out_tps=597.38939507432 ttft50=2785.3660699911416 tpot50=85.15119627961434 ok=256/256 valid=True
  - P2_cf0_skip0_pin1_def4/a1 SHORT_COLD_C64_rep1: out_tps=618.1652932735304 ttft50=3050.735149998218 tpot50=77.28496386200248 ok=256/256 valid=True
  - P2_cf0_skip0_pin1_def4/a1 SHORT_COLD_C64_rep2: out_tps=618.992767960836 ttft50=3028.8943744963035 tpot50=78.42666446093807 ok=256/256 valid=True
  - P2_cf0_skip0_pin1_def4/a1 SHORT_COLD_C64_rep3: out_tps=615.0613037122972 ttft50=3002.224460069556 tpot50=79.85368490587612 ok=256/256 valid=True
  - P2X_cf0_overlap_off/a1 SHORT_COLD_C64_rep1: out_tps=588.4114699032338 ttft50=2466.508748009801 tpot50=88.7559928384052 ok=256/256 valid=True
  - P2X_cf0_overlap_off/a1 SHORT_COLD_C64_rep2: out_tps=587.9963713506604 ttft50=2410.360082925763 tpot50=90.1494725785167 ok=256/256 valid=True
- latest_error: 2026-09-16T01:34:58.948093+09:00 R03_d3_cf_skip_def8_pin stage=MEASURING server not healthy after rep log=eval/results/IDE_071_20260916/R03_d3_cf_skip_def8_pin/a1/errors.jsonl
- retries_and_recovery: 0 셀 재시도
- HBM: 0, 79700 MiB;1, 79598 MiB;2, 79598 MiB;3, 79118 MiB;4, 0 MiB;5, 0 MiB;6, 0 MiB;7, 0 MiB;
- RAM: used 311 GB / free 55 GB; disk /data: 21T avail
- latest_persisted_artifacts: eval/results/IDE_071_20260916/P2_cf0_skip0_pin1_def4/a1/SHORT_COLD_C64_rep3/metrics.json, eval/results/IDE_071_20260916/P2X_cf0_overlap_off/a1/SHORT_COLD_C64_rep1/metrics.json, eval/results/IDE_071_20260916/P2X_cf0_overlap_off/a1/SHORT_COLD_C64_rep2/metrics.json
- next_registered_cells: P2_cf0_skip0_pin0_def0, P2_cf0_skip0_pin1_def0, P2_cf0_skip0_pin1_def8, P2_cf1_skip0_pin1_def8, P2_cf1_skip1_pin1_def4
- report_delivery_state: saved (세션 cron 이 전달 후 delivered 로 갱신)

# 중간 실행 보고 — 2026-09-16T03:50:51.849111+09:00

- campaign_id: IDE_071_20260916
- elapsed_seconds: 10811
- utc: 2026-09-15T18:50:51.849111+00:00
- current_phase / cell_id / attempt / rep_id: P2 / P2_cf1_skip1_pin1_def4 / a1 / SHORT_COLD_C64_rep1
- phase_state: MEASURING
- registered / completed / failed / blocked / pending: 27 / 26 / 1 / 0 / 0
- last_completed_cells: P2_cf0_skip0_pin1_def4=COMPLETED, P2X_cf0_overlap_off=COMPLETED, P2_cf0_skip0_pin0_def0=COMPLETED, P2X_cf1_overlap_off=COMPLETED, P2_cf1_skip0_pin1_def8=COMPLETED, P2_cf0_skip0_pin1_def8=COMPLETED
- latest_measurements:
  - P2_cf1_skip0_pin1_def8/a1 SHORT_COLD_C64_rep1: out_tps=658.9344615730749 ttft50=3229.0029324940406 tpot50=73.18272171624548 ok=256/256 valid=True
  - P2_cf1_skip0_pin1_def8/a1 SHORT_COLD_C64_rep2: out_tps=657.3993741482325 ttft50=2958.5348804830573 tpot50=75.48726241766131 ok=256/256 valid=True
  - P2_cf1_skip0_pin1_def8/a1 SHORT_COLD_C64_rep3: out_tps=658.7736947457938 ttft50=2813.6195485130884 tpot50=76.93962819276626 ok=256/256 valid=True
  - P2_cf0_skip0_pin1_def8/a1 SHORT_COLD_C64_rep1: out_tps=648.3601773406892 ttft50=2792.7839159965515 tpot50=75.80196558301681 ok=256/256 valid=True
  - P2_cf0_skip0_pin1_def8/a1 SHORT_COLD_C64_rep2: out_tps=654.7211950895158 ttft50=2773.4001660137437 tpot50=75.73649994469812 ok=256/256 valid=True
  - P2_cf0_skip0_pin1_def8/a1 SHORT_COLD_C64_rep3: out_tps=642.809196086129 ttft50=2766.155236517079 tpot50=77.58029671266719 ok=256/256 valid=True
- latest_error: 2026-09-16T01:34:58.948093+09:00 R03_d3_cf_skip_def8_pin stage=MEASURING server not healthy after rep log=eval/results/IDE_071_20260916/R03_d3_cf_skip_def8_pin/a1/errors.jsonl
- retries_and_recovery: 0 셀 재시도
- HBM: 0, 79618 MiB;1, 79598 MiB;2, 79598 MiB;3, 79118 MiB;4, 0 MiB;5, 0 MiB;6, 0 MiB;7, 0 MiB;
- RAM: used 312 GB / free 54 GB; disk /data: 21T avail
- latest_persisted_artifacts: eval/results/IDE_071_20260916/P2_cf0_skip0_pin1_def8/a1/SHORT_COLD_C64_rep1/metrics.json, eval/results/IDE_071_20260916/P2_cf0_skip0_pin1_def8/a1/SHORT_COLD_C64_rep2/metrics.json, eval/results/IDE_071_20260916/P2_cf0_skip0_pin1_def8/a1/SHORT_COLD_C64_rep3/metrics.json
- next_registered_cells: P2_cf0_skip0_pin1_def0, P2_cf1_skip1_pin1_def4
- report_delivery_state: saved (세션 cron 이 전달 후 delivered 로 갱신)

# 중간 실행 보고 — 2026-09-16T04:20:53.829612+09:00

- campaign_id: IDE_071_20260916
- elapsed_seconds: 12613
- utc: 2026-09-15T19:20:53.829612+00:00
- current_phase / cell_id / attempt / rep_id: P3 / P3_amx_min_qlen_128 / a1 / SHORT_COLD_C64_rep1
- phase_state: MEASURING
- registered / completed / failed / blocked / pending: 32 / 31 / 1 / 0 / 0
- last_completed_cells: P2_cf0_skip0_pin1_def8=COMPLETED, P2_cf1_skip1_pin1_def4=COMPLETED, P2_cf0_skip0_pin1_def0=COMPLETED, P3_amx_min_qlen_64=COMPLETED, P3_amx_min_rows_3=COMPLETED, P3_amx_min_qlen_1024=COMPLETED
- latest_measurements:
  - P3_amx_min_rows_3/a1 SHORT_COLD_C64_rep1: out_tps=747.2008124791545 ttft50=2380.494203942362 tpot50=67.38402438579186 ok=256/256 valid=True
  - P3_amx_min_rows_3/a1 SHORT_COLD_C64_rep2: out_tps=749.975620441854 ttft50=2268.250548047945 tpot50=69.819935078652 ok=256/256 valid=True
  - P3_amx_min_rows_3/a1 SHORT_COLD_C64_rep3: out_tps=741.4538494927192 ttft50=2298.4775604563765 tpot50=69.78700779457247 ok=256/256 valid=True
  - P3_amx_min_qlen_1024/a1 SHORT_COLD_C64_rep1: out_tps=743.8401964298007 ttft50=2329.855428019073 tpot50=70.08550252754416 ok=256/256 valid=True
  - P3_amx_min_qlen_1024/a1 SHORT_COLD_C64_rep2: out_tps=737.1919547846808 ttft50=2337.2848604922183 tpot50=67.85287606707722 ok=256/256 valid=True
  - P3_amx_min_qlen_1024/a1 SHORT_COLD_C64_rep3: out_tps=747.3286590435293 ttft50=2231.2875709612854 tpot50=68.53719215746602 ok=256/256 valid=True
- latest_error: 2026-09-16T01:34:58.948093+09:00 R03_d3_cf_skip_def8_pin stage=MEASURING server not healthy after rep log=eval/results/IDE_071_20260916/R03_d3_cf_skip_def8_pin/a1/errors.jsonl
- retries_and_recovery: 0 셀 재시도
- HBM: 0, 79618 MiB;1, 79598 MiB;2, 79598 MiB;3, 79118 MiB;4, 0 MiB;5, 0 MiB;6, 0 MiB;7, 0 MiB;
- RAM: used 312 GB / free 54 GB; disk /data: 21T avail
- latest_persisted_artifacts: eval/results/IDE_071_20260916/P3_amx_min_qlen_1024/a1/SHORT_COLD_C64_rep1/metrics.json, eval/results/IDE_071_20260916/P3_amx_min_qlen_1024/a1/SHORT_COLD_C64_rep2/metrics.json, eval/results/IDE_071_20260916/P3_amx_min_qlen_1024/a1/SHORT_COLD_C64_rep3/metrics.json
- next_registered_cells: P3_avx_rb_0, P3_avx_rb_1, P3_avx_pf_0, P3_avx_pf_1, P3_avx_pf_2
- report_delivery_state: saved (세션 cron 이 전달 후 delivered 로 갱신)

# 중간 실행 보고 — 2026-09-16T04:50:55.602909+09:00

- campaign_id: IDE_071_20260916
- elapsed_seconds: 14415
- utc: 2026-09-15T19:50:55.602909+00:00
- current_phase / cell_id / attempt / rep_id: P3 / P3_amx_min_rows_4 / a1 / SHORT_COLD_C64_rep1
- phase_state: MEASURING
- registered / completed / failed / blocked / pending: 37 / 36 / 1 / 0 / 0
- last_completed_cells: P3_amx_min_qlen_1024=COMPLETED, P3_amx_min_qlen_128=COMPLETED, P3_amx_min_qlen_512=COMPLETED, P3_avx_pf_4=COMPLETED, P3_amx_min_qlen_256=COMPLETED, P3_avx_pf_2=COMPLETED
- latest_measurements:
  - P3_amx_min_qlen_256/a1 SHORT_COLD_C64_rep1: out_tps=741.9391642777978 ttft50=2201.119018427562 tpot50=71.15068823680295 ok=256/256 valid=True
  - P3_amx_min_qlen_256/a1 SHORT_COLD_C64_rep2: out_tps=742.3722127931829 ttft50=2267.0664480538107 tpot50=68.16782081851692 ok=256/256 valid=True
  - P3_amx_min_qlen_256/a1 SHORT_COLD_C64_rep3: out_tps=738.4105479831912 ttft50=2200.434987025801 tpot50=69.0234410471107 ok=256/256 valid=True
  - P3_avx_pf_2/a1 SHORT_COLD_C64_rep1: out_tps=740.3776539453046 ttft50=2267.733371059876 tpot50=69.04069411405177 ok=256/256 valid=True
  - P3_avx_pf_2/a1 SHORT_COLD_C64_rep2: out_tps=748.0955435668423 ttft50=2300.605299533345 tpot50=68.23667926405827 ok=256/256 valid=True
  - P3_avx_pf_2/a1 SHORT_COLD_C64_rep3: out_tps=738.8473618331108 ttft50=2266.6200934909284 tpot50=70.51131222488755 ok=256/256 valid=True
- latest_error: 2026-09-16T01:34:58.948093+09:00 R03_d3_cf_skip_def8_pin stage=MEASURING server not healthy after rep log=eval/results/IDE_071_20260916/R03_d3_cf_skip_def8_pin/a1/errors.jsonl
- retries_and_recovery: 0 셀 재시도
- HBM: 0, 79620 MiB;1, 79598 MiB;2, 79598 MiB;3, 79118 MiB;4, 0 MiB;5, 0 MiB;6, 0 MiB;7, 0 MiB;
- RAM: used 312 GB / free 54 GB; disk /data: 21T avail
- latest_persisted_artifacts: eval/results/IDE_071_20260916/P3_avx_pf_2/a1/SHORT_COLD_C64_rep1/metrics.json, eval/results/IDE_071_20260916/P3_avx_pf_2/a1/SHORT_COLD_C64_rep2/metrics.json, eval/results/IDE_071_20260916/P3_avx_pf_2/a1/SHORT_COLD_C64_rep3/metrics.json
- next_registered_cells: P3_avx_rb_0, P3_avx_rb_1, P3_avx_pf_0, P3_avx_pf_1, P3_fuse_qin_0
- report_delivery_state: saved (세션 cron 이 전달 후 delivered 로 갱신)

# 중간 실행 보고 — 2026-09-16T05:20:57.543610+09:00

- campaign_id: IDE_071_20260916
- elapsed_seconds: 16217
- utc: 2026-09-15T20:20:57.543610+00:00
- current_phase / cell_id / attempt / rep_id: P3 / P3_amx_min_qlen_1000000 / a1 / SHORT_COLD_C64_rep2
- phase_state: MEASURING
- registered / completed / failed / blocked / pending: 42 / 41 / 1 / 0 / 0
- last_completed_cells: P3_avx_pf_2=COMPLETED, P3_amx_min_rows_4=COMPLETED, P3_avx_pf_0=COMPLETED, P3_omp_num_threads_1=COMPLETED, P3_fuse_qin_0=COMPLETED, P3_cpuinfer_88=COMPLETED
- latest_measurements:
  - P3_fuse_qin_0/a1 SHORT_COLD_C64_rep2: out_tps=753.0487865698855 ttft50=2239.5815544878133 tpot50=67.96238774811394 ok=256/256 valid=True
  - P3_fuse_qin_0/a1 SHORT_COLD_C64_rep3: out_tps=759.951207243469 ttft50=2319.2358720116317 tpot50=67.03709837051079 ok=256/256 valid=True
  - P3_cpuinfer_88/a1 SHORT_COLD_C64_rep1: out_tps=729.9167562914907 ttft50=2126.874511013739 tpot50=70.73483503548296 ok=256/256 valid=True
  - P3_cpuinfer_88/a1 SHORT_COLD_C64_rep2: out_tps=725.5438750597626 ttft50=2078.1175525626168 tpot50=72.93596718135825 ok=256/256 valid=True
  - P3_cpuinfer_88/a1 SHORT_COLD_C64_rep3: out_tps=726.9892427257722 ttft50=2160.609190526884 tpot50=72.60493517717654 ok=256/256 valid=True
  - P3_amx_min_qlen_1000000/a1 SHORT_COLD_C64_rep1: out_tps=669.7379343631043 ttft50=3203.437753021717 tpot50=71.15905213043256 ok=256/256 valid=True
- latest_error: 2026-09-16T01:34:58.948093+09:00 R03_d3_cf_skip_def8_pin stage=MEASURING server not healthy after rep log=eval/results/IDE_071_20260916/R03_d3_cf_skip_def8_pin/a1/errors.jsonl
- retries_and_recovery: 0 셀 재시도
- HBM: 0, 77358 MiB;1, 77422 MiB;2, 77422 MiB;3, 76942 MiB;4, 0 MiB;5, 0 MiB;6, 0 MiB;7, 0 MiB;
- RAM: used 311 GB / free 54 GB; disk /data: 21T avail
- latest_persisted_artifacts: eval/results/IDE_071_20260916/P3_cpuinfer_88/a1/SHORT_COLD_C64_rep2/metrics.json, eval/results/IDE_071_20260916/P3_cpuinfer_88/a1/SHORT_COLD_C64_rep3/metrics.json, eval/results/IDE_071_20260916/P3_amx_min_qlen_1000000/a1/SHORT_COLD_C64_rep1/metrics.json
- next_registered_cells: P3_avx_rb_0, P3_avx_rb_1, P3_avx_pf_1, P3_fuse_qin_1, P3_amx_min_rows_1
- report_delivery_state: saved (세션 cron 이 전달 후 delivered 로 갱신)

# 중간 실행 보고 — 2026-09-16T05:50:59.494059+09:00

- campaign_id: IDE_071_20260916
- elapsed_seconds: 18019
- utc: 2026-09-15T20:50:59.494059+00:00
- current_phase / cell_id / attempt / rep_id: P3 / P3_amx_min_rows_16 / a1 / SHORT_COLD_C64_rep2
- phase_state: MEASURING
- registered / completed / failed / blocked / pending: 47 / 46 / 1 / 0 / 0
- last_completed_cells: P3_cpuinfer_88=COMPLETED, P3_amx_min_qlen_1000000=COMPLETED, P3_amx_min_rows_8=COMPLETED, P3_amx_min_qlen_32=COMPLETED, P3_cpuinfer_108=COMPLETED, P3_openblas_num_threads_1=COMPLETED
- latest_measurements:
  - P3_cpuinfer_108/a1 SHORT_COLD_C64_rep2: out_tps=751.6747079594855 ttft50=2209.856874542311 tpot50=67.8193310984249 ok=256/256 valid=True
  - P3_cpuinfer_108/a1 SHORT_COLD_C64_rep3: out_tps=743.2764457407209 ttft50=2250.8107795147225 tpot50=68.51662013415746 ok=256/256 valid=True
  - P3_openblas_num_threads_1/a1 SHORT_COLD_C64_rep1: out_tps=743.3037158468746 ttft50=2458.5490815225057 tpot50=67.66679435812567 ok=256/256 valid=True
  - P3_openblas_num_threads_1/a1 SHORT_COLD_C64_rep2: out_tps=720.8122069524522 ttft50=2343.861931061838 tpot50=71.86287031490649 ok=256/256 valid=True
  - P3_openblas_num_threads_1/a1 SHORT_COLD_C64_rep3: out_tps=740.1396212920115 ttft50=2380.975320993457 tpot50=68.92441629476141 ok=256/256 valid=True
  - P3_amx_min_rows_16/a1 SHORT_COLD_C64_rep1: out_tps=734.3330515227744 ttft50=2374.390361015685 tpot50=69.95085058651323 ok=256/256 valid=True
- latest_error: 2026-09-16T01:34:58.948093+09:00 R03_d3_cf_skip_def8_pin stage=MEASURING server not healthy after rep log=eval/results/IDE_071_20260916/R03_d3_cf_skip_def8_pin/a1/errors.jsonl
- retries_and_recovery: 0 셀 재시도
- HBM: 0, 77428 MiB;1, 77422 MiB;2, 77422 MiB;3, 76942 MiB;4, 0 MiB;5, 0 MiB;6, 0 MiB;7, 0 MiB;
- RAM: used 312 GB / free 54 GB; disk /data: 21T avail
- latest_persisted_artifacts: eval/results/IDE_071_20260916/P3_openblas_num_threads_1/a1/SHORT_COLD_C64_rep2/metrics.json, eval/results/IDE_071_20260916/P3_openblas_num_threads_1/a1/SHORT_COLD_C64_rep3/metrics.json, eval/results/IDE_071_20260916/P3_amx_min_rows_16/a1/SHORT_COLD_C64_rep1/metrics.json
- next_registered_cells: P3_avx_rb_0, P3_avx_rb_1, P3_avx_pf_1, P3_fuse_qin_1, P3_amx_min_rows_1
- report_delivery_state: saved (세션 cron 이 전달 후 delivered 로 갱신)

# 중간 실행 보고 — 2026-09-16T06:21:01.360731+09:00

- campaign_id: IDE_071_20260916
- elapsed_seconds: 19821
- utc: 2026-09-15T21:21:01.360731+00:00
- current_phase / cell_id / attempt / rep_id: P3 / P3_avx_pf_1 / a1 / SHORT_COLD_C64_rep2
- phase_state: MEASURING
- registered / completed / failed / blocked / pending: 52 / 51 / 1 / 0 / 0
- last_completed_cells: P3_openblas_num_threads_1=COMPLETED, P3_amx_min_rows_16=COMPLETED, P3_mkl_num_threads_4=COMPLETED, P3_omp_num_threads_4=COMPLETED, P3_openblas_num_threads_4=COMPLETED, P3_cpuinfer_104=COMPLETED
- latest_measurements:
  - P3_openblas_num_threads_4/a1 SHORT_COLD_C64_rep2: out_tps=740.1635762537027 ttft50=2283.1679029623047 tpot50=69.04449890547352 ok=256/256 valid=True
  - P3_openblas_num_threads_4/a1 SHORT_COLD_C64_rep3: out_tps=757.0885430860491 ttft50=2074.1411205381155 tpot50=67.51586075580687 ok=256/256 valid=True
  - P3_cpuinfer_104/a1 SHORT_COLD_C64_rep1: out_tps=737.8652971502402 ttft50=2326.540629088413 tpot50=68.59335334656217 ok=256/256 valid=True
  - P3_cpuinfer_104/a1 SHORT_COLD_C64_rep2: out_tps=736.4482213772181 ttft50=2252.35253298888 tpot50=69.87232004712138 ok=256/256 valid=True
  - P3_cpuinfer_104/a1 SHORT_COLD_C64_rep3: out_tps=730.5664681807848 ttft50=2328.0179450521246 tpot50=70.75785181859112 ok=256/256 valid=True
  - P3_avx_pf_1/a1 SHORT_COLD_C64_rep1: out_tps=740.2586648404599 ttft50=2383.9503474882804 tpot50=68.82998768073381 ok=256/256 valid=True
- latest_error: 2026-09-16T01:34:58.948093+09:00 R03_d3_cf_skip_def8_pin stage=MEASURING server not healthy after rep log=eval/results/IDE_071_20260916/R03_d3_cf_skip_def8_pin/a1/errors.jsonl
- retries_and_recovery: 0 셀 재시도
- HBM: 0, 79662 MiB;1, 79600 MiB;2, 79600 MiB;3, 79120 MiB;4, 0 MiB;5, 0 MiB;6, 0 MiB;7, 0 MiB;
- RAM: used 312 GB / free 53 GB; disk /data: 21T avail
- latest_persisted_artifacts: eval/results/IDE_071_20260916/P3_cpuinfer_104/a1/SHORT_COLD_C64_rep2/metrics.json, eval/results/IDE_071_20260916/P3_cpuinfer_104/a1/SHORT_COLD_C64_rep3/metrics.json, eval/results/IDE_071_20260916/P3_avx_pf_1/a1/SHORT_COLD_C64_rep1/metrics.json
- next_registered_cells: P3_avx_rb_0, P3_avx_rb_1, P3_avx_pf_1, P3_fuse_qin_1, P3_amx_min_rows_1
- report_delivery_state: saved (세션 cron 이 전달 후 delivered 로 갱신)

# 중간 실행 보고 — 2026-09-16T06:51:03.375331+09:00

- campaign_id: IDE_071_20260916
- elapsed_seconds: 21623
- utc: 2026-09-15T21:51:03.375331+00:00
- current_phase / cell_id / attempt / rep_id: P3 / P3_tokenizer_workers_2 / a1 / SHORT_COLD_C64_rep2
- phase_state: MEASURING
- registered / completed / failed / blocked / pending: 57 / 56 / 1 / 0 / 0
- last_completed_cells: P3_cpuinfer_104=COMPLETED, P3_avx_pf_1=COMPLETED, P3_avx_rb_0=COMPLETED, P3_tokenizer_workers_1=COMPLETED, P3_cpuinfer_80=COMPLETED, P3_tokenizer_workers_4=COMPLETED
- latest_measurements:
  - P3_cpuinfer_80/a1 SHORT_COLD_C64_rep2: out_tps=695.8611342869164 ttft50=2436.041168461088 tpot50=74.2067312954661 ok=256/256 valid=True
  - P3_cpuinfer_80/a1 SHORT_COLD_C64_rep3: out_tps=694.1420530964383 ttft50=2158.8150180177763 tpot50=75.96348099617697 ok=256/256 valid=True
  - P3_tokenizer_workers_4/a1 SHORT_COLD_C64_rep1: out_tps=736.8723688067071 ttft50=2159.860439482145 tpot50=70.81978356302531 ok=256/256 valid=True
  - P3_tokenizer_workers_4/a1 SHORT_COLD_C64_rep2: out_tps=744.8481414203271 ttft50=2126.15744053619 tpot50=70.247798637894 ok=256/256 valid=True
  - P3_tokenizer_workers_4/a1 SHORT_COLD_C64_rep3: out_tps=748.9997766796371 ttft50=2065.1473319740035 tpot50=68.15823276782056 ok=256/256 valid=True
  - P3_tokenizer_workers_2/a1 SHORT_COLD_C64_rep1: out_tps=724.1250266741264 ttft50=2188.710345479194 tpot50=71.85112153167111 ok=256/256 valid=True
- latest_error: 2026-09-16T01:34:58.948093+09:00 R03_d3_cf_skip_def8_pin stage=MEASURING server not healthy after rep log=eval/results/IDE_071_20260916/R03_d3_cf_skip_def8_pin/a1/errors.jsonl
- retries_and_recovery: 0 셀 재시도
- HBM: 0, 79668 MiB;1, 79598 MiB;2, 79598 MiB;3, 79118 MiB;4, 0 MiB;5, 0 MiB;6, 0 MiB;7, 0 MiB;
- RAM: used 316 GB / free 50 GB; disk /data: 21T avail
- latest_persisted_artifacts: eval/results/IDE_071_20260916/P3_tokenizer_workers_4/a1/SHORT_COLD_C64_rep2/metrics.json, eval/results/IDE_071_20260916/P3_tokenizer_workers_4/a1/SHORT_COLD_C64_rep3/metrics.json, eval/results/IDE_071_20260916/P3_tokenizer_workers_2/a1/SHORT_COLD_C64_rep1/metrics.json
- next_registered_cells: P3_avx_rb_1, P3_fuse_qin_1, P3_amx_min_rows_1, P3_amx_min_rows_2, P3_cpuinfer_112
- report_delivery_state: saved (세션 cron 이 전달 후 delivered 로 갱신)


## compact (cpu_offload_no_02_compact)
- compact_status: RUNNING · current: S1_b4_pin/a1
- 일반 벤치 소비 1/20 · 진단 0 · 연속부하 0 · 재시도 0 · GSM 0
- 부팅 소비 1/14 · 셀 상태: {}
- 재사용 대조군: B3=P2_cf1_skip1_pin1_def8 mean 740.3, B4=R04_d4_epoch mean 705.1
- 직전 실측: 
- targets: None
# 중간 실행 보고 — 2026-09-16T07:05:11.900287+09:00

- campaign_id: IDE_071_20260916
- elapsed_seconds: 0
- utc: 2026-09-15T22:05:11.900287+00:00
- current_phase / cell_id / attempt / rep_id: COMPACT / None / None / None
- phase_state: IDLE
- registered / completed / failed / blocked / pending: 60 / 59 / 1 / 0 / 0
- last_completed_cells: P3_tokenizer_workers_1=COMPLETED, P3_cpuinfer_80=COMPLETED, P3_tokenizer_workers_4=COMPLETED, P3_tokenizer_workers_2=COMPLETED, P3_fuse_qin_1=COMPLETED, P3_amx_min_rows_1=COMPLETED
- latest_measurements:
  - P3_fuse_qin_1/a1 SHORT_COLD_C64_rep1: out_tps=748.6789525425781 ttft50=2371.4729950297624 tpot50=68.24946181489317 ok=256/256 valid=True
  - P3_fuse_qin_1/a1 SHORT_COLD_C64_rep2: out_tps=731.0309896347675 ttft50=2290.8179209916852 tpot50=70.85396505078288 ok=256/256 valid=True
  - P3_fuse_qin_1/a1 SHORT_COLD_C64_rep3: out_tps=750.7683739950322 ttft50=2365.683015435934 tpot50=67.46101434256397 ok=256/256 valid=True
  - P3_amx_min_rows_1/a1 SHORT_COLD_C64_rep1: out_tps=685.4564112404156 ttft50=2381.0364624951035 tpot50=75.66496822845528 ok=256/256 valid=True
  - P3_amx_min_rows_1/a1 SHORT_COLD_C64_rep2: out_tps=686.3419134073296 ttft50=2334.9402880412526 tpot50=75.86369623235741 ok=256/256 valid=True
  - P3_amx_min_rows_1/a1 SHORT_COLD_C64_rep3: out_tps=678.1663207014772 ttft50=2213.4035764611326 tpot50=78.2853307208875 ok=256/256 valid=True
- latest_error: 2026-09-16T01:34:58.948093+09:00 R03_d3_cf_skip_def8_pin stage=MEASURING server not healthy after rep log=eval/results/IDE_071_20260916/R03_d3_cf_skip_def8_pin/a1/errors.jsonl
- retries_and_recovery: 0 셀 재시도
- HBM: 0, 0 MiB;1, 0 MiB;2, 0 MiB;3, 0 MiB;4, 0 MiB;5, 0 MiB;6, 0 MiB;7, 0 MiB;
- RAM: used 75 GB / free 290 GB; disk /data: 21T avail
- latest_persisted_artifacts: eval/results/IDE_071_20260916/P3_amx_min_rows_1/a1/SHORT_COLD_C64_rep1/metrics.json, eval/results/IDE_071_20260916/P3_amx_min_rows_1/a1/SHORT_COLD_C64_rep2/metrics.json, eval/results/IDE_071_20260916/P3_amx_min_rows_1/a1/SHORT_COLD_C64_rep3/metrics.json
- next_registered_cells: 없음
- report_delivery_state: saved (세션 cron 이 전달 후 delivered 로 갱신)
- 07:15 compact 지시서 수령: 12단계 대기열 중지(실행 중 셀 P3_amx_min_rows_1 은 완료 후), compact 실행 시작. 대조군 B3/B4 재사용.


## compact (cpu_offload_no_02_compact)
- compact_status: RUNNING · current: S2_b3_v2_nu5952__confirm/a1
- 일반 벤치 소비 10/20 · 진단 0 · 연속부하 0 · 재시도 0 · GSM 0
- 부팅 소비 7/14 · 셀 상태: {'S1_b4_pin': 'COMPLETED', 'S2_b3_v2_nu5952': 'COMPLETED', 'S3_b4_graph64': 'COMPLETED', 'S4_b4_kv40960': 'COMPLETED', 'S5_b4_chunk2048': 'COMPLETED', 'B3_recheck': 'COMPLETED'}
- 재사용 대조군: B3=P2_cf1_skip1_pin1_def8 mean 740.3, B4=R04_d4_epoch mean 705.1
- 직전 실측: S2_b3_v2_nu5952__confirm SHORT_COLD_C64_rep1: out_tps=800.7496281186085 ttft95=3133.148448483553 tpot95=72.61442669688387 ok=256/256 valid=True | S2_b3_v2_nu5952__confirm SHORT_COLD_C64_rep2: out_tps=811.3348364662361 ttft95=2990.651207510382 tpot95=73.55375951967261 ok=256/256 valid=True | S2_b3_v2_nu5952__confirm SHORT_COLD_C64_rep3: out_tps=805.3264716537267 ttft95=2977.9231769789476 tpot95=72.40991345120916 ok=256/256 valid=True | S2_b3_v2_nu5952__confirm COMPACT_ALT_C64_rep1: out_tps=847.9914319553452 ttft95=2754.7082164965104 tpot95=69.2580917400517 ok=256/256 valid=True
- targets: {'B3': 'S2_b3_v2_nu5952', 'B4': None}
# 중간 실행 보고 — 2026-09-16T07:35:13.665699+09:00

- campaign_id: IDE_071_20260916
- elapsed_seconds: 1801
- utc: 2026-09-15T22:35:13.665699+00:00
- current_phase / cell_id / attempt / rep_id: COMPACT / None / None / None
- phase_state: IDLE
- registered / completed / failed / blocked / pending: 60 / 59 / 1 / 0 / 0
- last_completed_cells: P3_tokenizer_workers_1=COMPLETED, P3_cpuinfer_80=COMPLETED, P3_tokenizer_workers_4=COMPLETED, P3_tokenizer_workers_2=COMPLETED, P3_fuse_qin_1=COMPLETED, P3_amx_min_rows_1=COMPLETED
- latest_measurements:
  - P3_fuse_qin_1/a1 SHORT_COLD_C64_rep1: out_tps=748.6789525425781 ttft50=2371.4729950297624 tpot50=68.24946181489317 ok=256/256 valid=True
  - P3_fuse_qin_1/a1 SHORT_COLD_C64_rep2: out_tps=731.0309896347675 ttft50=2290.8179209916852 tpot50=70.85396505078288 ok=256/256 valid=True
  - P3_fuse_qin_1/a1 SHORT_COLD_C64_rep3: out_tps=750.7683739950322 ttft50=2365.683015435934 tpot50=67.46101434256397 ok=256/256 valid=True
  - P3_amx_min_rows_1/a1 SHORT_COLD_C64_rep1: out_tps=685.4564112404156 ttft50=2381.0364624951035 tpot50=75.66496822845528 ok=256/256 valid=True
  - P3_amx_min_rows_1/a1 SHORT_COLD_C64_rep2: out_tps=686.3419134073296 ttft50=2334.9402880412526 tpot50=75.86369623235741 ok=256/256 valid=True
  - P3_amx_min_rows_1/a1 SHORT_COLD_C64_rep3: out_tps=678.1663207014772 ttft50=2213.4035764611326 tpot50=78.2853307208875 ok=256/256 valid=True
- latest_error: 2026-09-16T01:34:58.948093+09:00 R03_d3_cf_skip_def8_pin stage=MEASURING server not healthy after rep log=eval/results/IDE_071_20260916/R03_d3_cf_skip_def8_pin/a1/errors.jsonl
- retries_and_recovery: 0 셀 재시도
- HBM: 0, 79862 MiB;1, 79720 MiB;2, 79720 MiB;3, 79240 MiB;4, 0 MiB;5, 0 MiB;6, 0 MiB;7, 0 MiB;
- RAM: used 311 GB / free 54 GB; disk /data: 21T avail
- latest_persisted_artifacts: eval/results/IDE_071_20260916/P3_amx_min_rows_1/a1/SHORT_COLD_C64_rep1/metrics.json, eval/results/IDE_071_20260916/P3_amx_min_rows_1/a1/SHORT_COLD_C64_rep2/metrics.json, eval/results/IDE_071_20260916/P3_amx_min_rows_1/a1/SHORT_COLD_C64_rep3/metrics.json
- next_registered_cells: 없음
- report_delivery_state: saved (세션 cron 이 전달 후 delivered 로 갱신)


## compact (cpu_offload_no_02_compact)
- compact_status: BOUNDED_SEARCH_COMPLETE · current: None/None
- 일반 벤치 소비 10/20 · 진단 0 · 연속부하 1 · 재시도 1 · GSM 0
- 부팅 소비 9/14 · 셀 상태: {'S1_b4_pin': 'COMPLETED', 'S2_b3_v2_nu5952': 'COMPLETED', 'S3_b4_graph64': 'COMPLETED', 'S4_b4_kv40960': 'COMPLETED', 'S5_b4_chunk2048': 'COMPLETED', 'B3_recheck': 'COMPLETED', 'S2_b3_v2_nu5952__confirm': 'COMPLETED', 'S2_b3_v2_nu5952__load1024': 'COMPLETED'}
- 재사용 대조군: B3=P2_cf1_skip1_pin1_def8 mean 740.3, B4=R04_d4_epoch mean 705.1
- 직전 실측: S2_b3_v2_nu5952__confirm SHORT_COLD_C64_rep3: out_tps=805.3264716537267 ttft95=2977.9231769789476 tpot95=72.40991345120916 ok=256/256 valid=True | S2_b3_v2_nu5952__confirm COMPACT_ALT_C64_rep1: out_tps=847.9914319553452 ttft95=2754.7082164965104 tpot95=69.2580917400517 ok=256/256 valid=True | S2_b3_v2_nu5952__load1024 COMPACT_LOAD_C64_rep1: out_tps=0.0 ttft95=0.0 tpot95=0.0 ok=0/1024 valid=False | S2_b3_v2_nu5952__load1024 COMPACT_LOAD_C64_rep1: out_tps=842.5046478882293 ttft95=2645.400928601157 tpot95=69.19744888235317 ok=1024/1024 valid=True
- targets: {'B3': 'S2_b3_v2_nu5952', 'B4': None}
# 중간 실행 보고 — 2026-09-16T07:53:14.762216+09:00

- campaign_id: IDE_071_20260916
- elapsed_seconds: 2882
- utc: 2026-09-15T22:53:14.762216+00:00
- current_phase / cell_id / attempt / rep_id: COMPACT / None / None / None
- phase_state: IDLE
- registered / completed / failed / blocked / pending: 60 / 59 / 1 / 0 / 0
- last_completed_cells: P3_tokenizer_workers_1=COMPLETED, P3_cpuinfer_80=COMPLETED, P3_tokenizer_workers_4=COMPLETED, P3_tokenizer_workers_2=COMPLETED, P3_fuse_qin_1=COMPLETED, P3_amx_min_rows_1=COMPLETED
- latest_measurements:
  - P3_fuse_qin_1/a1 SHORT_COLD_C64_rep1: out_tps=748.6789525425781 ttft50=2371.4729950297624 tpot50=68.24946181489317 ok=256/256 valid=True
  - P3_fuse_qin_1/a1 SHORT_COLD_C64_rep2: out_tps=731.0309896347675 ttft50=2290.8179209916852 tpot50=70.85396505078288 ok=256/256 valid=True
  - P3_fuse_qin_1/a1 SHORT_COLD_C64_rep3: out_tps=750.7683739950322 ttft50=2365.683015435934 tpot50=67.46101434256397 ok=256/256 valid=True
  - P3_amx_min_rows_1/a1 SHORT_COLD_C64_rep1: out_tps=685.4564112404156 ttft50=2381.0364624951035 tpot50=75.66496822845528 ok=256/256 valid=True
  - P3_amx_min_rows_1/a1 SHORT_COLD_C64_rep2: out_tps=686.3419134073296 ttft50=2334.9402880412526 tpot50=75.86369623235741 ok=256/256 valid=True
  - P3_amx_min_rows_1/a1 SHORT_COLD_C64_rep3: out_tps=678.1663207014772 ttft50=2213.4035764611326 tpot50=78.2853307208875 ok=256/256 valid=True
- latest_error: 2026-09-16T01:34:58.948093+09:00 R03_d3_cf_skip_def8_pin stage=MEASURING server not healthy after rep log=eval/results/IDE_071_20260916/R03_d3_cf_skip_def8_pin/a1/errors.jsonl
- retries_and_recovery: 0 셀 재시도
- HBM: 0, 0 MiB;1, 0 MiB;2, 0 MiB;3, 0 MiB;4, 0 MiB;5, 0 MiB;6, 0 MiB;7, 0 MiB;
- RAM: used 75 GB / free 305 GB; disk /data: 21T avail
- latest_persisted_artifacts: eval/results/IDE_071_20260916/P3_amx_min_rows_1/a1/SHORT_COLD_C64_rep1/metrics.json, eval/results/IDE_071_20260916/P3_amx_min_rows_1/a1/SHORT_COLD_C64_rep2/metrics.json, eval/results/IDE_071_20260916/P3_amx_min_rows_1/a1/SHORT_COLD_C64_rep3/metrics.json
- next_registered_cells: 없음
- report_delivery_state: saved (세션 cron 이 전달 후 delivered 로 갱신)

# 종료 보고 — 2026-09-16 08:05 KST (compact)
- compact 상태: BOUNDED_SEARCH_COMPLETE (07:05 시작 → 07:58 연속 부하 재시도 완료)
- 일반 벤치 소비 10/20 (1차 5 + 부모 B3 재확인 1 + S2 확인 3 + 별도 입력 1) · 연속 부하 1 (시도 2, 재시도 1/2) · 진단 0 · GSM40 1/2 · 부팅 9/14
- 완료 8 셀 시도 (S1~S5, B3_recheck, S2 확인, S2 연속 부하 a2) / 실패 1 시도 (연속 부하 a1: 워밍업 중 서버 NaN assert 종료, 1,024 요청 전부 실패) / NOT_TRIGGERED 2 (S6, B4 계열 확인) / 미실행 1 (선택적 진단)
- 확인 대상 S2_b3_v2_nu5952__confirm: SHORT_COLD C64 800.75 / 811.33 / 805.33, COMPACT_ALT 847.99, GSM40 38/40, 연속 부하 1,024 요청 842.50 tok/s (155.6 s)
- 문서: FULL_REPORT.md, RESULT.md, COMPLETION_STATUS.md, VALIDATION.md (15/15 통과), manifests/cells.jsonl, artifact_manifest.jsonl (3,652 파일), reports/request_tables/ (198 반복)
- 12단계 계획 실행분 (P1 8, P2 20, P3 20, 앵커 재측정) 은 FULL_REPORT §3.2·§5.2 에 보존, 잔여 OUT_OF_SCOPE
