# IDE_026 — test.md

## TST 대응: 신규 TST 는 T2 진입 시 발급 (현 단계는 harness self-test 수준)

## G0. harness 정합 (T0)

- `rdt_ctl.py` self-test: CLOS 생성→schemata 기록→재판독 일치, TID 등록→tasks 재판독 포함 확인
- mon_data 판독: 동일 그룹 2회 판독 delta ≥ 0 (mbm 누적 단조성)
- 도메인 2개 (socket 0/1) 모두 키 존재

## G1. 간섭 재현 게이트 (T1)

| 판정 | 조건 |
|---|---|
| 간섭 실재 | B0 에서 victim p99 가 aggressor 무부하 대비 ≥ +10% 악화 (간섭이 안 보이면 R1 노출 — aggressor 강도 재설계 1회 한도) |
| L2 GO | B2 에서 victim p99 회복 ≥ 80% AND aggressor 처리량 ≥ B0 의 70% |
| L2 kill | 회복 < 50% (CAT 가 이 워크로드 조합에 무력) |

- 3-run CV < 5%, mon_data 로 격리 실효 확인 (harvest llc_occupancy 가 way 비율에 수렴)

## G2. vLLM attribution 게이트 (T2, GPU 가용 후)

- 스레드 분류 누락 0 (boot 후 `/proc/<pid>/task` 전수 vs 등록 TID diff = ∅)
- C-8a 재측정: CAT 격리 ON 에서 delta ≥ −0.1% (무해 수준) 이면 "간섭 원인" 입증
- canonical tps 의 측정 분산 (CV) 이 SUB_212 와 동급 (< 2%) 유지

## G3. 출력 등가 (C3)

- RDT 는 수치 경로 무접촉 — **형식 확인만**: 격리 ON/OFF greedy 8 prompt × 32 tok
  token 일치 (BF16 비결정성 범위 내) + per-token logprob max abs diff 기록 (informational)

## G4. Objective 검증 (C4)

- T3 frontier 측정 시 mpstat 224-thread util 병기 — harvest 100% 코어 채움 셀에서
  전체 CPU util ≥ 90% AND serving tps 저하 ≤ 사전 선언 상한

## G5. 사전 예측 (pre-commit, 측정 전 고정)

| 항목 | 예측 |
|---|---|
| T1 B0 victim p99 악화 | +15~60% (LLC 300MB 라 약할 수 있음 — 하한 +10% 게이트) |
| T1 B2 회복 | 80~95% |
| T2 C-8a CAT ON 재측정 | −0.35% → −0.1~0% |
| T3 100%-harvest 셀 | serving 저하 ≤ 3%, CPU util ≥ 90% |
