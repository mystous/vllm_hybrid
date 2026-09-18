# NUMERICAL_CONTRACT — IDE_076 (PLAN.md §5.3)

## 비교 방법
- comparator: `expert_comparator.py` 가 같은 seed 의 난수 가중치·입력·routing 으로 변형별 `outputs.pt` 를 만들고, `torch.equal` (bitwise) 과 max|Δ| 를 계산 (`/tmp/ide076_replay_summary.py`, 컨테이너). 실제 체크포인트 대신 난수 가중치를 쓰는 이유: 커널 산술 계약(정수 누산·스케일) 검증에는 값 분포가 아니라 경로가 중요하고, 실제 가중치 검증은 서버 smoke(greedy 4 요청 텍스트)·품질 문항으로 보완.
- 기준 경로 결정성: RB ON/OFF, v4 vs v5(A1=0) 모두 6 시나리오 bitwise 동일 → 기존 정상 경로는 결정적이고 연산 순서가 같다 (§5.3 "bitwise 비교도 수행" 조건 충족).

## A1 계약 (int4_same_path)
- A1-a(GFNI 언팩): `(x & 0x0F) << 4` 를 `vgf2p8affineqb(x, 0x0000000001020408)` 로 대체 — 256 바이트 전수 검산 동일, dpbusd 입력 bit 동일 → int32 누산·apply_scale·bf16 변환 전부 동일. 결과: a1on == a1off == v4 bitwise (6 시나리오). atol=rtol=0.
- A1-e(프리페치 거리 KT_AVX_PF): 산술 무관(캐시 힌트) → bitwise 동일 요구; E2E smoke 텍스트로 확인.
- 허용치를 결과에 맞춰 늘리지 않는다. 순서를 바꾸는 변경(예: to_mat 융합) 은 별도 계약 등록 전 실행 금지.

## 공통 native 경로 (unchanged_fp8_bf16_paths)
- kt-kernel `examples/test_fp8_perchannel_moe.py`, `test_fp8_moe.py`: v4·v5·v6 모두 상대 L1 0.5829 % (동일값). GLM 4문항 게이트는 공통 wrapper 변경 시에만 재실행.

## B (placement_int4_to_fp8)
- bitwise 요구 없음. 기준: 동일 입력 greedy 텍스트·logits 차이(가능 시)·quality20 정답(≥ H0 − 1)·smoke finish_reason 동일. H0 자체의 실행 간 변동을 먼저 측정 (R0_REF smoke 와 후속 R0 부팅 smoke 텍스트 비교).

## 수명 (lifecycle)
- IDE_075 v3 기록기 + `validate_v2.py` (lifecycle_valid, task_count_valid: 미생산·중복·stale 0). CORR 세션에서만 확인.
