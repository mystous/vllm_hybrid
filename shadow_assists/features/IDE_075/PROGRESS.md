# PROGRESS — IDE_075

## 09-17 15:25 30분 보고 (착수 14:45)
- A00: 잔여 예산 세션 7·부팅 5·품질 76 확정. IDE_075 등록, 브랜치 feat/cpu-moe-bottleneck-followup-20260917.
- A01: tsparse 단위시험 PASS; 17 세션 창 재구성(WINDOW_RECONSTRUCTION.md); PCM 정정: forward_envelope 창 read 237.0/251.0/247.8 GB/s (legacy 144.8/143.4/139.4). perf NOT_RECOVERABLE.
- A02: v2 지표·cohort·민감도(0/1/2/5 µs)·항등식 residual 을 IDE_074 P1 6세션에 적용. bs=63 cold_present_nonempty 15,597~15,601 행: cold_pub_delta p50 307.8/312.5/334.0 µs, late 비율 0.898/0.897/0.909 (thr5: 0.894/0.893/0.906), pre_h2d gap 안 GPU idle = gap 전체(다른 op 0), residual 0.
- A03: 기록기 v2 (stamp task 제거) 구현·재빌드(.so SHA a4e9045a…), 합성 시험 PASS (링 overflow 검출, 순서, SPSC stale, clock 27 ns/call).
- A11: OFF_OPEN → CORE(S2 CORE, S3/S4 CORR, S5 RESOURCE) → OFF_CLOSE 체인 실행 중 (3 부팅, 6 세션).

## 09-17 15:45 A11 완료
- 세션 6/7: S1_OFF_A 700.9, S6_OFF_B 706.9, S2_CORE 708.2, S3_CORR 699.1, S4_CORR 701.5, S5_RESOURCE 781.6 tok/s. 부팅 시도 5/5 (OFF_OPEN·CORE_152703 은 하네스 결함으로 세션 없이 소진). 예비 S7 없음.
- 분석 체인(v2 지표·FIFO 분해·자원·검증·렌더) 실행 중.

## 09-17 15:59 종료 보고
- 세션 6/7 (예비 없음), 부팅 시도 5/5 (2회 하네스 결함 소진). 분석·검증·보고서 생성 완료. READINESS: 후보 A·B READY_WITH_LIMITED_SCOPE. 게시 진행.
