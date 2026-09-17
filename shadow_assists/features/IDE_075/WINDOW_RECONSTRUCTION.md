# WINDOW_RECONSTRUCTION — IDE_075 (IDE_074 세션 재구성)

창 정의는 스크립트 docstring. 단위 ns 정수. `ESTIMATED_WINDOW` 는 같은 기준 증거가 없는 역산.

- B_r1 (OFF, B1_131834): client_process 57.763 s, envelope NOT_COLLECTED (OFF 모드: 서버측 이벤트 없음): lead None s / tail None s, phases None, drain 1.3 s
- B_r2 (OFF, B1_131834): client_process 57.448 s, envelope NOT_COLLECTED (OFF 모드: 서버측 이벤트 없음): lead None s / tail None s, phases None, drain 1.3 s
- B_r3 (OFF, B1_131834): client_process 56.322 s, envelope NOT_COLLECTED (OFF 모드: 서버측 이벤트 없음): lead None s / tail None s, phases None, drain 1.3 s
- P0_r1 (OFF, B1_131834): client_process 36.009 s, envelope NOT_COLLECTED (OFF 모드: 서버측 이벤트 없음): lead None s / tail None s, phases None, drain 1.3 s
- P0_r2 (OFF, B1_131834): client_process 34.626 s, envelope NOT_COLLECTED (OFF 모드: 서버측 이벤트 없음): lead None s / tail None s, phases None, drain 1.3 s
- P0_r3 (OFF, B1_131834): client_process 35.929 s, envelope NOT_COLLECTED (OFF 모드: 서버측 이벤트 없음): lead None s / tail None s, phases None, drain 1.3 s
- P1_r1 (P1, B2_132633): client_process 36.948 s, envelope trace step annotation: lead 13.109 s / tail 0.449 s, phases {"step[EXTEND bs=1 toks=51": 2, "step[EXTEND bs=16 toks=8": 1, "step[EXTEND bs=17 toks=8": 5, "step[EXTEND bs=15 toks=7": 1, "step[DECODE bs=63]": 256, "step[EXTEND bs=14 toks=7": 1, "step[EXTEND bs=2 toks=10": 1, "step[DECODE bs=2]": 128}, drain 1.3 s
- P1_r2 (P1, B2_132633): client_process 36.401 s, envelope trace step annotation: lead 12.805 s / tail 0.407 s, phases {"step[EXTEND bs=1 toks=51": 1, "step[EXTEND bs=16 toks=8": 1, "step[EXTEND bs=17 toks=8": 5, "step[EXTEND bs=15 toks=7": 1, "step[DECODE bs=63]": 256, "step[EXTEND bs=3 toks=15": 1, "step[EXTEND bs=12 toks=6": 1, "step[EXTEND bs=2 toks=10": 1, "step[DECODE bs=2]": 128}, drain 1.3 s
- P1_r3 (P1, B2_132633): client_process 37.078 s, envelope trace step annotation: lead 12.991 s / tail 0.398 s, phases {"step[EXTEND bs=1 toks=51": 2, "step[EXTEND bs=16 toks=8": 1, "step[EXTEND bs=17 toks=8": 5, "step[EXTEND bs=15 toks=7": 1, "step[DECODE bs=63]": 256, "step[EXTEND bs=14 toks=7": 1, "step[EXTEND bs=2 toks=10": 1, "step[DECODE bs=2]": 128}, drain 1.3 s
- P2_r1 (P2, B3_133528): client_process 36.205 s, envelope kt_evt (첫 go ~ 마지막 done/def_end): lead 12.649 s / tail 0.401 s, phases None, drain 1.3 s
- P2_r2 (P2, B3_133528): client_process 36.121 s, envelope kt_evt (첫 go ~ 마지막 done/def_end): lead 13.024 s / tail 0.396 s, phases None, drain 1.3 s
- P2_r3 (P2, B3_133528): client_process 38.063 s, envelope kt_evt (첫 go ~ 마지막 done/def_end): lead 14.417 s / tail 0.451 s, phases None, drain 1.3 s
- F1_C1_OFF (OFF, F1_OFF_134055): client_process 45.716 s, envelope NOT_COLLECTED (OFF 모드: 서버측 이벤트 없음): lead None s / tail None s, phases None, drain 1.3 s
- F1_LONG_OFF (OFF, F1_OFF_134055): client_process 38.371 s, envelope NOT_COLLECTED (OFF 모드: 서버측 이벤트 없음): lead None s / tail None s, phases None, drain 1.3 s
- F1_C1_P1 (P1, F1_ON_134513): client_process 48.715 s, envelope trace step annotation: lead 12.927 s / tail 0.361 s, phases {"step[EXTEND bs=1 toks=51": 11, "step[DECODE bs=1]": 2048, "step[EXTEND bs=1 toks=50": 5}, drain 1.3 s
- F1_LONG_P1 (P1, F1_ON_134513): client_process 39.308 s, envelope trace step annotation: lead 13.196 s / tail 0.417 s, phases {"step[EXTEND bs=1 toks=40": 7, "step[EXTEND bs=2 toks=81": 1, "step[EXTEND bs=3 toks=81": 11, "step[EXTEND bs=2 toks=41": 1, "step[DECODE bs=8]": 512}, drain 1.2 s
- S1_P1_PROBE128 (P1, SMOKE_131024): client_process 37.463 s, envelope trace step annotation: lead 13.091 s / tail 0.425 s, phases {"step[EXTEND bs=1 toks=51": 1, "step[EXTEND bs=16 toks=8": 1, "step[EXTEND bs=17 toks=8": 5, "step[EXTEND bs=15 toks=7": 1, "step[DECODE bs=63]": 256, "step[EXTEND bs=4 toks=20": 1, "step[EXTEND bs=11 toks=5": 1, "step[EXTEND bs=2 toks=10": 1, "step[DECODE bs=2]": 128}, drain 1.3 s

## PCM 정정 (System|Read / Write, MB/s)

| session | window | n_full/boundary/outside | full coverage | mean_full | mean_overlap_est | legacy(collector 전체 평균) |
|---|---|---|---|---|---|---|
| P2_r1 | forward_envelope_evt | System|Read 23/2/13 | 0.993 | 236992.9 | 235500.7 | 144783.7 |
| P2_r1 | forward_envelope_evt | System|Write 23/2/13 | 0.993 | 26535.2 | 26376.3 | 16728.6 |
| P2_r1 | client_process_window(legacy) | System|Read 35/2/1 | 0.967 | 156799.9 | 151750.3 | 144783.7 |
| P2_r1 | client_process_window(legacy) | System|Write 35/2/1 | 0.967 | 17987.4 | 17453.4 | 16728.6 |
| P2_r2 | forward_envelope_evt | System|Read 21/2/15 | 0.925 | 250972.7 | 237233.1 | 143391.1 |
| P2_r2 | forward_envelope_evt | System|Write 21/2/15 | 0.925 | 26994.0 | 26366.3 | 16680.8 |
| P2_r2 | client_process_window(legacy) | System|Read 35/2/1 | 0.969 | 155476.4 | 150753.4 | 143391.1 |
| P2_r2 | client_process_window(legacy) | System|Write 35/2/1 | 0.969 | 17970.6 | 17468.6 | 16680.8 |
| P2_r3 | forward_envelope_evt | System|Read 22/2/16 | 0.948 | 247825.6 | 237167.2 | 139422.6 |
| P2_r3 | forward_envelope_evt | System|Write 22/2/16 | 0.948 | 27151.1 | 26161.5 | 16056.1 |
| P2_r3 | client_process_window(legacy) | System|Read 37/2/1 | 0.972 | 150525.3 | 146421.7 | 139422.6 |
| P2_r3 | client_process_window(legacy) | System|Write 37/2/1 | 0.972 | 17217.6 | 16793.8 | 16056.1 |

legacy 값은 `legacy_collector_window_mean` (bench 호출 창, 초 단위 파싱 실패 → 전 표본 포함). perf stat 은 세션 전체 누적값만 있어 부하 창 delta 복원 불가 (NOT_RECOVERABLE).
