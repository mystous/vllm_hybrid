# COMPLETION_STATUS — IDE_076 A1·A2·B (2026-09-18 14:58 KST 최종)

## 1. 질문별 상태
| 질문 | 상태 | 근거 |
|---|---|---|
| A1: RB 내부 R=1/2 잔여 비용을 줄이면 이득이 있는가 | **판정 완료** | GFNI 언팩 단독 +1.6 %(4 블록) 불채택; PF=1 +3.1 %(4) 채택; GFNI+PF +3.7 %(3) 채택. bitwise 동일. A1_DECISION |
| A2: 같은 Cold job 내부 descriptor 묶음이 이득이 있는가 | **판정 완료** | 적용성 확인(장벽 tail 3.7~4.4 %) → 구현 → bitwise 동일, proof 142/142 → E2E +5.0 %(4 블록) 채택; A12 +5.9 %(5), A12e +7.7 %(5). A2_DECISION |
| B: 생산자 비용 기반 정적 Hot 배치가 이득이 있는가 | **판정 완료(불채택)** | calibration v1 → HB_DEC1/HB_MIX1 → SELECTION HB_MIX1 +5.7 % → CONFIRMATION 고정 입력 −6 %(완주 4 부팅 전부 음수), 독립 입력 +6~13 %, 수치 기준 미달. 배치가 calibration 분포에 과적합. B_DECISION |
| 통합 조건 | **판정 완료** | A12e (v8f + A1 + A2 + PF, H0) +7.7 %, 회귀 한도 내. INTEGRATION_DECISION |
| 정상성 사고 | **원인 미해결(실패 처리)** | callback-free ∧ overlap ∧ 빈 imm 생략에서 GPU fused_moe 출력 NaN; IDE_076 이전부터 존재; 사망 부팅 제외(사용자 지시 06:40). ANALYSIS §5 |
| 원복 시험 | **완료** | v4 원본 .so + 원본 래퍼 + 플래그 unset + H0: smoke 4/4 = R0_REF, MAIN 746.5 = R0_REF 746.3; 채택 스택 복원 확인. ROLLBACK §4 |
| 게시 | 대기 | 사용자 승인 후 커밋·푸시·영수증 |

## 2. 실행 집계 (수정 후 모집단)
- 부팅 총 61 (E2E 43 + SELECTION 6 + CONFIRMATION 12), 사망 17 (E2E 12, CONFIRMATION 5) → 유효 44. pre_fix 8 부팅(사망 5)은 방향 참고용.
- 유효 블록: R0 6, R0_REF 2, A1 4, A1e 4, A1ae 3, A2 4, A12 5, A12e 5; B 1, A2B 3, A12B 0.

## 3. 범위 밖 작업(기록)
- 09-18 01:00~06:40 사고 원인 추적(진단 부팅·프로브 체인·수정 시도 2 회)은 지시서 §17.2 범위를 넘어선 것으로 사용자 지시로 중단. 유지된 변경: rearm 세대 대기(v8f·v4f 공통, 무해). 제거: eager layer-0 sync. 프로브 코드는 원복.

## 4. 인계 최소 정보 (§17.3)
- 소스: kt-kernel upstream 6d460cc + 로컬 11 파일 + IDE_076 패치(kt_opt.h, avx_kernel_rb 템플릿/GFNI, worker_pool probe, moe_base A2 executor, cpuinfer.h/ext_bindings CF-rearm). 바이너리 v8f `a4add14d…`(채택), 기준 v4f `f437b6f1…`, 원본 v4 `12926df2…`.
- 채택 구성: `KT_OPT_A1_ENABLE=1 KT_OPT_A2_ENABLE=1 KT_AVX_PF=1`, hotmap_v2 + layer_budget_5952(H0), 기타 기존 env(KT_CALLBACK_FREE=1, KT_CF_SKIP_EMPTY_IMM=1, KT_AVX_RB=0).
- 끄는 것: B(HB_MIX1) 미적용; A1 단독은 켜지 않음.
- 검증 입력: MAIN_SHORT(seed 20260916 manifest), C1, LONG, SELECT(20261011), CONFIRM IND(20261021), calibration(20261001/2); replay 장면 6 종; 수치 프로브 sonnet 8 텍스트.
- 실제 명령: 각 부팅 디렉터리의 `launch_cmd.sh` / 세션의 `bench_cmd.sh`.
