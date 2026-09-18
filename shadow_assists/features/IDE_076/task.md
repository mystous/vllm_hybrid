# IDE_076 task — PLAN.md 부록 C 작업 단위
| ID | 작업 | 상태 |
|---|---|---|
| C00 | 출처·소스·실효 환경 확인 → BASELINE_PROVENANCE, SOURCE_MANIFEST, CODE_MAP | 완료 (22:55) |
| C01 | 횟수 정책·안전 검사 정리 → EXECUTION_POLICY_TESTS | 완료 (22:49) |
| C02 | 기준 OFF·수치·입력 계약 | 완료 (12:55): post_fix 기준선 R0 6 부팅(MAIN 744.6 평균), R0_REF(v4f) 2 부팅 +1.3 % → 새 바이너리 OFF 경로에 설명되지 않은 변화 없음; 사고는 원인 미해결·실패 처리 |
| C03 | 플래그·feature proof (kt_opt.h, KT_OPT_A1/A2_ENABLE 0/1, kt_opt_status; 잘못된 값 → 부팅 전 실패 확인) | 완료 (23:08); v7 스레드별 카운터 예정 |
| C04 | expert replay 비교기 (expert_comparator.py; 난수 가중치·실제 차원·pool 96/2) — 실제 체크포인트 job replay 는 CORR 계측으로 보완 | 진행 |
| C05 | A1 잔여 비용 확인 → A1_GAP_ANALYSIS (asm 계수·R 의존성·대역 한계) | 완료 (23:08) |
| C06 | A1-a GFNI 언팩 구현 → bitwise 동일·replay −0~2 %; A1-e(PF) 후보 추가 | 진행 (E2E 판정 대기) |
| C07 | A1 서비스 검증 → A1_DECISION | 완료 (12:55): A1 단독 +1.6 % 불채택, A1e +3.1 % 채택, A1ae +3.7 % 채택 (유효 블록 4/4/3) |
| C08 | A2 적용성 → v7 probe 장벽 tail 3.7~4.4 % → 구현 결정 (A2_APPLICABILITY) | 완료 (00:20) |
| C09 | A2 묶음 executor (`forward_prefill_a2_`, kt_opt_a2_patch.py, 소스 적용·문법 통과) → v8 빌드는 post 체인 | 진행 |
| C10 | A2/A12 검증 → A2_DECISION | 완료 (12:55): A2 +5.0 % (4 블록), A12 +5.9 % (5), A12e +7.7 % (5) 채택; bitwise 동일·proof 142/142 |
| C11 | 비용표: v0 불일치 확인 → CALIB 수집 완료(00:03) → 디코드/prefill 커버리지 불일치 발견 → calib_build v1 분석 실행 중 (01:06~) | 진행 |
| C12 | hotmap_tools 완료; 후보 HB_DEC1(920 swaps)·HB_MIX1(582 swaps) 생성·검증 (00:30) | 완료 |
| C13 | B 정상성·선별·확인 → B_DECISION | 완료 (14:50): SELECTION HB_MIX1 +5.7 % → CONFIRMATION 고정 입력 −6 %(완주 4 부팅 전부 음수)·독립 입력 +6~13 %·수치 기준 미달 → **불채택** |
| C14 | 고정-map 개별·통합 비교 → INTEGRATION_DECISION | 완료 (14:50): 최종 A12e(+7.7 %, H0); A2B −6.2 %, A12B 3 부팅 사망 |
| C15 | 독립 확인·회귀 | 완료 (14:50): C1/LONG/TTFT/TPOT 회귀 전 변형 한도 내; 독립 입력(seed 20261021) CONFIRMATION 수집; 지속 실행은 사고(NaN) 로 부팅당 30~40 % 사망 — 실패 처리·별도 과제 |
| C16 | 원복 시험·보고·전달 | 원복 시험 완료(smoke 4/4·MAIN 동일, 채택 스택 복원); FULL_REPORT·ARTIFACT_INDEX·SHA256SUMS 렌더; 게시는 사용자 승인 대기 |
