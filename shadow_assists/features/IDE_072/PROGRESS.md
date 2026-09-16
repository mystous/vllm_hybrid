# IDE_072 PROGRESS (30분 보고 누적)

- 2026-09-16 08:20 KST 지시서 `cpu_offload_no_03_compact_verification` 수령 → IDE_072 등록, S2 원본 대조 (PLAN_RESOLVED.md), 하네스 확장 (REPLAY·GSM20_PAIRED·품질 전용 셀·ServerArgs/capture 증거), 실행기 `run_verification.py`, 보고 루프 `report_verification.py`.

# 중간 실행 보고 — 2026-09-16T08:25:56.778127+09:00 (IDE_072)

- 경과: 0 s · 현재 cell·stage·run_id: None / IDLE / None.None
- PERF None/None · REPLAY None/None · LOAD None/None · DIAG None/None · RETRY None/None
- 부하 세션 None/None · 부팅 None/None · 워밍업 0 · smoke 0 · GSM 문항 None/None
- 셀 상태: {}
- 직전 원측정값: 없음
- 오류: 없음
- 다음 등록 항목: ['V0_first', 'V1_first', 'V2_first', 'Q0_first']
- 저장된 최신 MD: 없음
- 전달: saved (세션 cron 이 사용자 전달 후 delivered 기록)

# 중간 실행 보고 — 2026-09-16T08:55:58.331134+09:00 (IDE_072)

- 경과: 1801 s · 현재 cell·stage·run_id: V1_confirm / MEASURING / V1_confirm.a1
- PERF 7/11 · REPLAY 3/3 · LOAD 0/1 · DIAG 0/1 · RETRY 0/2
- 부하 세션 10/18 · 부팅 5/10 · 워밍업 0 · smoke 6 · GSM 문항 60/80
- 셀 상태: {'V0_first': 'COMPLETED', 'V1_first': 'COMPLETED', 'V2_first': 'COMPLETED', 'Q0_first': 'FAILED_REQUESTS', 'V0_confirm': 'COMPLETED'}
- 직전 원측정값: V0_confirm SHORT_COLD_C64_rep3: out_tps=799.4491209450437 ok=256/256 valid=True | V0_confirm COMPACT_ALT_C64_rep1: out_tps=853.1634492712064 ok=256/256 valid=True | V1_confirm SHORT_COLD_C64_rep1: out_tps=827.7227098285144 ok=256/256 valid=True | V1_confirm SHORT_COLD_C64_rep2: out_tps=847.561277173985 ok=256/256 valid=True
- 오류: 없음
- 다음 등록 항목: ['V1_confirm', 'LOAD', 'D0(조건부)']
- 저장된 최신 MD: /home/mystous/projects/vllm_hybrid/shadow_assists/features/IDE_072/progress/20260916T082556+0900.md
- 전달: saved (세션 cron 이 사용자 전달 후 delivered 기록)

# 중간 실행 보고 — 2026-09-16T09:05:58.899289+09:00 (IDE_072)

- 경과: 2402 s · 현재 cell·stage·run_id: None / IDLE / None.None
- PERF 11/11 · REPLAY 3/3 · LOAD 1/1 · DIAG 0/1 · RETRY 0/2
- 부하 세션 15/18 · 부팅 7/10 · 워밍업 0 · smoke 7 · GSM 문항 80/80
- 셀 상태: {'V0_first': 'COMPLETED', 'V1_first': 'COMPLETED', 'V2_first': 'COMPLETED', 'Q0_first': 'COMPLETED', 'V0_confirm': 'COMPLETED', 'V1_confirm': 'COMPLETED', 'V1_load': 'COMPLETED'}
- 직전 원측정값: V1_confirm SHORT_COLD_C64_rep2: out_tps=847.561277173985 ok=256/256 valid=True | V1_confirm SHORT_COLD_C64_rep3: out_tps=856.8201954717177 ok=256/256 valid=True | V1_confirm COMPACT_ALT_C64_rep1: out_tps=887.1100536401574 ok=256/256 valid=True | V1_load COMPACT_LOAD_C64_rep1: out_tps=897.7511545397236 ok=1024/1024 valid=True
- 오류: 없음
- 다음 등록 항목: ['LOAD', 'D0(조건부)']
- 저장된 최신 MD: /home/mystous/projects/vllm_hybrid/shadow_assists/features/IDE_072/progress/20260916T085558+0900.md
- 전달: saved (세션 cron 이 사용자 전달 후 delivered 기록)

# 종료 보고 — 2026-09-16 09:08 KST (IDE_072)
- 상태: BOUNDED_VALIDATION_COMPLETE (08:29 시작 → 09:03 V1_load 완료). 오류 재현: V0·V1·V2 모두 NOT_REPRODUCED_WITHIN_BUDGET → D0 NOT_TRIGGERED.
- 실행량 (원장 재계산): PERF 11/11 · REPLAY 3/3 · LOAD 1/1 · DIAG 0/1 · RETRY 0/2 · 부하 세션 15/18 · 부팅 7/10 · 워밍업 7 · smoke 7 · GSM 80/80
- 완료 8 셀 (V0_first, V1_first, V2_first, Q0_first[상태 정정], V0_confirm, V1_confirm, V1_load) · 실패 0 · 차단 0 · 조건미충족 1 (D0)
- 원측정값 SHORT_COLD C64: V0_first 795.42; V1_first 842.44; V2_first 701.97; V0_confirm 795.92/804.02/799.45, ALT 853.16; V1_confirm 827.72/847.56/856.82, ALT 887.11; V1_load 1,024 요청 897.75. GSM20: V2 20/20, Q0 19/20, V0 20/20, V1 19/20.
- Q0_first 상태 정정 (FAILED_REQUESTS → COMPLETED, 품질 전용 셀 판정 오류; 원값 보존, branch_trace/원장 기록).
- 08:39 cron 미발화 → 08:53 수동 전달; 08:54 정기 전달. 파일 보고 08:25/08:55/09:05.
