# IDE_076 test — PLAN.md 부록 B.1 최소 자동화 시험 + 단계별 통과 증거
- 설정: A1/A2 미지정·0·1·잘못된 문자열 → 기본 OFF, 명시 파싱, invalid 실패 (부팅 전).
- 기준 경로: 신규 빌드 두 기능 OFF ↔ 원 v4 바이너리: 수치 계약·OFF MAIN 동등.
- 타입: FP8/BF16 native 경로 실제 실행 (kt FP8 단독 테스트 2종 PASS 유지).
- A1: R=1/2/3, padding/tail, 잘못된 layout → 지원 분기·fallback·입력 오류 구분. tile comparator 로 기존 RB 와 출력 비교.
- A2: output tile 중복·누락·조기 stage 진행 → validator 실패.
- B: 중복 expert, layer 범위 오류, 합계≠5952, KV 축소 → 실패.
- 수명: 미생산·중복·stale generation 검출 (기존 v2 validate 계승).
- 실행 정책: 합성 원장 횟수 초과에서도 유효 작업 계속.
- 통과 증거 파일: 각 단계의 *_BRANCH_PROOF.json, *_results.csv, validation_results.json, DECISION.md.
