# SUB_080 — Phase 1: workload-aware gating production 적용

> **parent**: TSK_020 (성능 향상 plan Phase 1)
> **status**: 활성 (2026-05-24 신설)
> **effort**: 4-8 시간 (router script + 3 mix scenario 측정)
> **based on**: IDE_012 (classifier accuracy 1.000) + SUB_047 (cap=8 best)

## 1. 목표

mixed traffic 영역 본 fork best 영역 즉시 적용 — code workload 회귀 차단 + sonnet/chat workload 영역 SUB_047 best 활용.

## 2. 진행 절차

1. router wrapper (`/tmp/run_sub080_router.py`) 신설 — workload classifier (IDE_012) 영역 prompt 분류 → spec config 선택:
   - code → `speculative_config=None` (vanilla)
   - sonnet / chat → `speculative_config={"method":"ngram", "num_speculative_tokens":7, ...}` + cap=8 env
2. mix scenario generator — 3 traffic mix (sonnet-heavy 60/20/10, balanced 33/33/33, code-heavy 10/20/70) prompt 생성.
3. launcher (`/tmp/run_sub080_mix.sh`) — 3 mix × {항상 spec ON, workload-aware gating} = 6 cell.
4. 측정 + 결과 doc + 분석 doc §6 갱신.

## 3. cell matrix

| mix | scenario | sonnet : chat : code | spec mode |
|---|---|---|---|
| M1 | sonnet-heavy | 60 : 20 : 10 | always-on / gating |
| M2 | balanced | 33 : 33 : 33 | always-on / gating |
| M3 | code-heavy | 10 : 20 : 70 | always-on / gating |

## 4. 예상 결과 (가설)

| mix | always-on tps (mixed avg) | gating tps | gating 추가 향상 |
|---|---:|---:|---:|
| M1 | sonnet 70%×10956 + chat 20%×3007 + code 10%×5347 = ~8967 | sonnet 70%×10956 + chat 20%×3007 + code 10%×6964(vanilla) = ~9128 | +1.8% |
| M2 | (10956+3007+5347)/3 = 6437 | (10956+3007+6964)/3 = 6976 | +8.4% |
| M3 | sonnet 10%×10956 + chat 20%×3007 + code 70%×5347 = ~5440 | sonnet 10%×10956 + chat 20%×3007 + code 70%×6964 = ~6571 | +20.8% |

## 5. 산출물

- `/tmp/run_sub080_router.py`, `/tmp/run_sub080_mix.sh`
- `measurements/sub080_gating_prod_<TS>/RESULTS.md`
- 분석 doc §6 갱신
