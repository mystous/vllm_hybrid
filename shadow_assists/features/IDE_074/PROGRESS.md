# PROGRESS — IDE_074

## 09-17 13:18 M0~M5 완료, M6 착수
- M0 환경·원자료 고정 (SOURCE_INDEX.md, evidence/). M1 분석기 §4.3 8건 PASS, IDE_073 D2 재집계 (profiles/corrected_metrics.csv).
- M3/M4: kt-kernel CPU-side 이벤트 프로브 (kt_evt_patch.py, .so 재빌드 SHA adf49e48…), GPU-side 는 트레이스 패턴 분절.
- M5 SMOKE(P1, 부팅 1·세션 1): execution/gpu_activity/window/cpu_events/correlation_clock 전부 PASS. 23,808 층-replay 매칭, clock 순서 위반 0.
- M6 체인 B1(OFF: B×3+P0×3) → B2(P1×3) → B3(P2×3) 백그라운드 실행 시작.

## 09-17 13:45 30분 보고 (사용자 요청으로 정기 보고 cron 5b706318 등록, 13/43분)
- M6 완료: B1 OFF (B 743.0/747.5/768.9, P0 720.8/780.7/732.6 tok/s), B2 P1 (696.4/705.6/690.8), B3 P2 (706.5/720.7/705.2). 서버 사망 0.
- M5/M7: P1_r1·r2 검증 5항목 PASS, bs=63 lateness p50 302.3 / 307.0 µs, 노출 대기 315.3 / 319.8, CPU 계산 898.3 / 902.3, 제출→시작 650.5 / 655.3, 예산 1247.5 / 1251.8. r3 분석 중.
- 조건부 F1_OFF 완료 (C1 63.3, LONG 164.9 tok/s), F1_ON 부팅 중 → GLM 게이트 예정. 예산: 세션 15/24, 부팅 5/12.

## 09-17 14:03 조건부 완료
- F1_ON: C1 P1 57.6 tok/s (OFF 63.3), LONG P1 159.3 (OFF 164.9). GLM 게이트: 4문항 전부 finish_reason=length·절단·비정상 텍스트 → BLOCKED_NORMAL_OUTPUT, G OFF/ON 미실행. 예산: 세션 17/24, 부팅 7/12, 품질 4/80.
- P1 4세션 재검증(정렬 지터 5 µs 허용) 전부 PASS. F1 세션 분석 중 → 보고서 재생성·게시 예정.

## 09-17 14:08 종료 보고
- 상태 COMPLETED_WITH_FAILURES (GLM 게이트 BLOCKED_NORMAL_OUTPUT) · 게시 PUBLISHED (data 87dee8e55 / receipt d95accc16) · 파일 5개 + 아카이브 2분할 전달 · 정기 보고 cron 삭제. 예산: 세션 17/24, 부팅 7/12, 품질 4/80.
