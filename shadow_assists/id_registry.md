# ID Registry

CLAUDE.md Ground RULE 의 ID Rule 에 따라, 본 저장소에서 사용되는 영문 3글자 prefix 별 **넘버링 규칙** 과 **ID 할당 현황** 을 본 파일에서 단일 출처로 관리한다.

> 본 파일은 ID 의 "할당·상태" 만 다룬다. ID 간 파생 관계(Tree) 는 `shadow_assists/README.md` Part VII Trace Tree, ID 의 본문 정의는 각 feature 디렉토리의 `README.md` 가 담당한다.

## 공통 넘버링 규칙

1. 번호는 **prefix 별로 독립된 카운터** 를 가진다. `IDE_###` 의 번호 공간과 `PLN_###` 의 번호 공간은 분리된다.
2. 번호는 **1 부터 시작하고 1 씩 증가** 한다 (예: `IDE_001`, `IDE_002`, …).
3. 번호는 **3 자리 zero-padding** (`001` ~ `999`).
4. 한 번 부여된 번호는 **재사용하지 않는다**. 기각·재정의된 ID 도 번호는 그대로 보존하고 상태(status)만 갱신한다.
5. 새 ID 부여 시 본 파일의 해당 prefix 섹션에서 "**다음 부여 번호**" 를 가져온 뒤, 같은 섹션의 표에 새 항목을 **즉시** 추가하고 "다음 부여 번호" 를 +1 한다. 본문에서의 사용은 그 이후.
6. 상태(status) 값은 다음 중 하나로 통일한다: `활성` / `대기` / `재정의` / `기각` / `완료`.

## 상태(status) 정의

| 상태 | 의미 |
|---|---|
| `활성` | 현재 진입 조건 충족 또는 작업 중 |
| `대기` | 정의되어 있으나 진입 조건 미충족, 측정·판정 대기 |
| `재정의` | 동일 ID 로 정의 내용이 1 회 이상 교체됨. 본문에 재정의 사유 명시 필요 |
| `기각` | 진입 조건 미달 또는 중복 사유로 폐기. 번호는 재사용 안 함 |
| `완료` | 작업 완료 후 더 이상 변경 예정 없음 |

---

## Prefix: `IDE` — Idea

구현 후보 단계. profile / 측정 결과로 진입·기각 판정 후에야 다음 단계 prefix(`PLN` 등) 로 파생된다.

**다음 부여 번호**: `IDE_009`

| ID | 상태 | 제목 | 비고 |
|---|---|---|---|
| `IDE_001` | 대기 | CPU-assisted 동적 배치 planner | Phase 0 profile 진입 조건 대기 |
| `IDE_002` | 대기 | CPU prefill-assist (medium context) | 512~2K GPU compute-bound 확인 대기 |
| `IDE_003` | 대기 | CPU Background Compiler | quantization 정합성 확인 대기 |
| `IDE_004` | 대기 | GPU-idle-phase CPU Burst | sublayer phase 감지 가능성 확인 대기 |
| `IDE_005` | 대기 | CPU drafter + GPU verifier | 다중화 request 큐 가정 검증 대기 |
| `IDE_006` | 재정의 | Cold-KV CPU Partial Attention | 1 차 정의(Cold KV staging) 업스트림 중복으로 기각 후 재정의 (commit `8f50eeb2a4`) |
| `IDE_007` | 대기 | CPU Speculative Logits Rerank | 정확도 영향 측정 대기 |
| `IDE_008` | 대기 | Constrained Decoding 전담 CPU Worker | constrained workload 시점 대기 |

---

## Prefix: `PLN` — Plan

IDE 의 진입·정확도·throughput 가정을 풀기 위한 PoC / microbench 플랜. PLN 결과에 따라 `FEA_###` 진입 또는 IDE 기각.

**다음 부여 번호**: `PLN_002`

| ID | 상태 | 제목 | 비고 |
|---|---|---|---|
| `PLN_001` | 활성 (Phase 1 dev) | Cold-KV CPU Partial Attention PoC 플랜 | 부모 `IDE_006`. Phase 1 (dev simulation — ≥8K prompt 합성, RTX 3090 + 12900KF) 진행 중. Phase 2 (prod — 운영 (a) 충족 후 Xeon SPR + H100×8) 는 사용자 직접 진행 |

---

## Prefix: `TSK` — Task

FEA 구현을 위한 단계별 작업 단위. CLAUDE.md Method 의 feature 디렉토리 내 `task.md` 항목과 매핑된다.

**다음 부여 번호**: `TSK_004`

| ID | 상태 | 제목 | 비고 |
|---|---|---|---|
| `TSK_001` | 완료 | LSE-반환 CPU partial-attention kernel — dev 부분 | 부모 `PLN_001`. 4.0 KVViewAdapter / 4.1 Python reference / 4.2c portable C++ / 4.3 wrapper. 검증 게이트 = `TST_001` (A·B(i)·C — dev 87 + prod 87 pytest 통과, 2026-04-26). prod SIMD (AVX-512/AMX) 는 `TSK_003` 으로 이관 |
| `TSK_002` | 대기 | scheduler / attention metadata 의 hot/cold partition 통합 | 부모 `PLN_001`. `TSK_001` 후속. 검증 게이트 = `TST_003` (e2e 통합 정확도) |
| `TSK_003` | 대기 (Phase 2 prod) | prod SIMD kernels — AVX-512 + AMX | 부모 `PLN_001`. 선행 `TSK_001` (portable + algorithm reference). §4.2a AVX-512 + §4.2b AMX C++ kernel 작성 + 빌드 통합 + wrapper enable + §4.4 prod microbench. 검증 게이트 = `TST_004` (portable vs AVX-512 / portable vs AMX cross-check). 사용자 직접 prod 머신 (Xeon SPR+) 진행 |

---

## Prefix: `TST` — Test

PLN/FEA 의 검증 단위. 정확도·throughput·통합성을 각각의 TST 로 분리해 측정한다. CLAUDE.md Method 의 feature 디렉토리 내 `test.md` 및 test 코드와 매핑된다.

**다음 부여 번호**: `TST_005`

| ID | 상태 | 제목 | 비고 |
|---|---|---|---|
| `TST_001` | 완료 | TSK_001 dev kernel 정확도 검증 | 부모 `PLN_001`. 검증 대상 **`TSK_001` 단독**. 단계 A (KVViewAdapter round-trip) · B(i) (Python ref vs portable) · C (wrapper dispatch). dev 87 (12900KF + RTX 3090, 2026-04-25) + prod 87 (Xeon Platinum 8480+ x2 + H100 x8, 2026-04-26) 통과 — eval/results/20260426_050608_..._prod_smoke. prod SIMD cross-check (B(ii)/B(iii)) 는 `TST_004` 로 이관 |
| `TST_002` | 대기 | Cold-KV CPU Partial Attention throughput / overlap profile | 부모 `PLN_001`. 모든 TSK (001/002/003) 의 perf 통합 검증 — kernel throughput + overlap profile. IDE_006 §9 (b)(g) 충족 게이트 |
| `TST_003` | 대기 | Cold-KV CPU Partial Attention e2e 통합 정확도 검증 | 부모 `PLN_001`. 검증 대상 `TSK_002` 의 vLLM forward 통합 (kernel ISA 무관). **D-i** generated token divergence + **D-ii** logprob/PPL diff. IDE_006 §9 (c) 의 e2e 측 게이트 |
| `TST_004` | 대기 (Phase 2 prod) | TSK_003 prod SIMD cross-check | 부모 `PLN_001`. 검증 대상 **`TSK_003` 단독**. 단계 B(ii) (portable vs AVX-512) · B(iii) (portable vs AMX). IDE_006 §9 (c) 의 prod ISA 측 게이트 |

---

## Prefix: `FEA` — Feature

PLN/TST 통과 후 본 코드 베이스에 들어가는 단위 기능. CLAUDE.md Method 의 feature 디렉토리 구조를 따른다.

**다음 부여 번호**: `FEA_001`

(등록된 ID 없음)

---

## 향후 추가될 prefix

새 prefix 를 도입할 때는 (a) 본 파일에 섹션 신설 + 카운터 초기화, (b) `shadow_assists/README.md` Part VII Legend 에 한 줄 추가, (c) CLAUDE.md ID Rule 변경이 필요하면 그쪽도 갱신.
