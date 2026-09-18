# ROLLBACK — IDE_076 변경분 원복 절차 (초안 2026-09-18 01:14 KST; 원복 시험 결과는 §4 에 추가)

대상 컨테이너 sgl-kt. 모든 단계는 서버 정지 상태에서 수행하고, 원복 후 `sha256sum` 으로 확인한다.

## 1. 층별 원복 대상
| 층 | 변경 | 원복 방법 | 확인값 |
|---|---|---|---|
| kt-kernel 바이너리 | v6/v7/v8 (A1 GFNI 템플릿·카운터·A2 executor·probe) | `cp /sgl-workspace/ide076_backup/kt_kernel_ext.so.v4 …/.kt_new.so && mv …/.kt_new.so <SO>` | SO sha256 `12926df2c30b7246…` |
| kt-kernel 소스 | kt_opt.h 추가, moe_base.hpp(A2), worker_pool.{h,cpp}(probe), avx_kernel_rb 템플릿/GFNI | `python3 /tmp/ide076_kt_opt_a2_patch.py revert` 등 각 패치 스크립트 `revert`, 또는 `/sgl-workspace/ide076_backup/*.pre_*` 로 복사 | SOURCE_MANIFEST.json 의 v4 시점 SHA |
| 파이썬 래퍼 | experts_base.py `IDE_076 CF-fix` (eager 스텝 layer 0 sync) | `python3 /tmp/ide076_cf_transition_sync_patch.py revert` (백업 `experts_base.py.pre_cfsync`) — **주의: 원복 시 callback-free 경로의 NaN 사망 사고(WORK_LOG 00:08~00:59) 가 재발한다. 사고 회피가 필요하면 `KT_CALLBACK_FREE` 를 해제(기존 callback 경로, MAIN −12 %)** | 파일 sha 원본값(BASELINE_PROVENANCE) |
| 런타임 플래그 | `KT_OPT_A1_ENABLE`, `KT_OPT_A2_ENABLE`, `KT_OPT_PROOF`, `KT_AVX_PF`, `KT_CF_NO_STEP_SYNC` | 모두 unset (기본 0/off) | `kt_opt_status()` 의 a1/a2/proof = 0 |
| Hot 배치 | HB_DEC1 / HB_MIX1 (`/models/kt/ide076/placements/`) | 부팅 인자 `--init-expert-location` = H0 hotmap_v2, `KT_GPU_EXPERTS_PER_LAYER` = layer_budget_5952 | hot_sets_digest `68f74531…` |
| 호스트 도구 | eval/ide076/*, eval/ide075 분석기 수정(IDE_KT_HOST 등) | git 에서 IDE_076 커밋 revert | — |

## 2. 순서
1. 서버 정지(자기 소유 프로세스만). 2. 바이너리 v4 복원. 3. 래퍼 원복 여부 결정(§1 주의). 4. 플래그 unset·H0 배치로 부팅. 5. smoke(greedy 4) 텍스트가 R0_REF smoke 와 동일한지 확인. 6. MAIN 1 세션으로 R0_REF 범위(±3 %) 확인.

## 3. 부분 채택 시 (INTEGRATION_DECISION 에 따라)
- A1 만 채택: v8 바이너리 유지 + `KT_OPT_A2_ENABLE=0`. A1 OFF 경로는 v4 와 asm 동일(A1_GAP_ANALYSIS §9) 이므로 플래그만으로 원복 가능.
- B 만 채택: 바이너리 v4 + 배치 파일만 교체.

## 4. 원복 시험 결과 (2026-09-18 14:47~14:53)
- 절차: 서버 정지 → `.so` 를 원본 v4(`12926df2…`) 로, 래퍼를 원본(`experts_base.py.pre_cfsync`, `fc44bf94…`) 로, `KT_OPT_A1_ENABLE/KT_OPT_A2_ENABLE/KT_AVX_PF` unset, H0 배치 → MAIN3 부팅(ROLLBACK_v4_MAIN3_144659).
- 결과: smoke greedy 4 문항 = pre_fix R0_REF(22:49) 와 **4/4 동일**; MAIN 741.7 / 752.8 / 744.9 (평균 746.5) = R0_REF 746.3 (+0.0 %); NaN 0.
- 시험 후 채택 스택 복원: `.so` v8f `a4add14d…`, 래퍼 `f8a6f056…`(CF-rearm 링 + kt-opt 상태 로그, 프로브 없음) — sha 확인.
- 참고: 채택 구성 A12e 의 smoke 4 문항도 R0_REF 와 4/4 동일(CPU 경로 bitwise 동일과 정합).
