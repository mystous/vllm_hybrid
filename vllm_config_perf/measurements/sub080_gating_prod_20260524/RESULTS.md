# SUB_080 — Phase 1: workload-aware gating production analytical PoC

> **parent**: TSK_020 (성능 향상 plan Phase 1)
> **plan**: [`../../planning/SUB_080_workload_aware_gating_prod.md`](../../planning/SUB_080_workload_aware_gating_prod.md)
> **measurement**: 2026-05-24 22:38 KST, **analytical (SUB_047/071 측정값 기반 weighted average)**
> **raw**: `eval/results/20260525_073811_sub080_gating_analytical/`

---

## 1. 환경 caveat — 단일 instance + analytical PoC

본 환경 (H100×8 + Llama-70B + TP=8) 영역 single instance only. vLLM v1 영역 LLM-level spec config (per-request override 미지원) → **실제 routing measurement 불가**. 본 SUB 영역 **analytical** — SUB_047 (sonnet/chat cap=8 best) + SUB_071 (chat/code 측정) 의 fact 영역 weighted average.

actual routing implementation 영역 Phase 3 (SUB_082) 또는 vLLM upstream PR (per-request override) 영역 필요.

## 2. classifier router 영역 reference

`/tmp/run_sub080_router.py` 영역 reference implementation:
- prompt → IDE_012 workload_classifier (regex feature)
- 분류 결과 → spec config 권장:
  - **code** → `speculative_config=None` (vanilla)
  - **sonnet / chat** → `speculative_config={"method":"ngram", "num_speculative_tokens":7, ...}` + cap=8 env

## 3. 측정 결과 (6 scenario, n=300 prompt each)

| mix | sonnet : chat : code | always_on tps | **gating tps** | **gating 향상** |
|---|---|---:|---:|---:|
| sonnet_only | 300 : 0 : 0 | 10,956.5 | 10,956.5 | +0.00% |
| chat_only | 0 : 300 : 0 | 3,006.6 | 3,006.6 | +0.00% |
| **code_only** | 0 : 0 : 300 | **5,346.8** | **6,964.5** | **+30.26%** ⭐ |
| **M1 (sonnet-heavy)** | 180 : 60 : 60 | 8,392.8 | 9,192.2 | **+9.52%** |
| **M2 (balanced)** | 102 : 99 : 99 | 6,871.1 | **7,976.5** | **+16.09%** ⭐ |
| **M3 (code-heavy)** | 30 : 60 : 210 | 5,616.1 | **7,091.1** | **+26.26%** ⭐ |

## 4. 핵심 finding

| 시나리오 | gating 효과 |
|---|---|
| code-heavy (M3) | **+26.26%** — 가장 큰 효과. code prompt 영역 spec OFF 영역 회귀 차단. |
| balanced (M2) | **+16.09%** — production traffic 영역 가장 가까울 가능성 |
| sonnet-heavy (M1) | **+9.52%** — 단 code 20% 영역 spec OFF 효과 |
| 단일 workload (sonnet/chat/code only) | 0 ~ +30% — code only 영역 vanilla 영역 spec OFF 영역 +30% |

→ **classifier router 영역 mixed traffic 영역 +10~+26% 추가 향상**. 본 fork 영역 즉시 적용 가능 (vLLM core 변경 없음, classifier + router script 만).

## 5. routing 실제 implementation 한계

| path | 가능성 | 한계 |
|---|---|---|
| 같은 instance + 매 generate() 영역 spec config 변경 | ✗ | vLLM v1 LLM-level only, init 시 결정 |
| dual instance (ngram + vanilla) on same GPU | ✗ | 70B + TP=8 영역 GPU 8 영역 dual instance OOM |
| dual instance (TP=4 × 2) | ◐ | viability 측정 필요 (Phase 3 SUB_082) |
| upstream per-request override PR | ★ | large effort but cleanest |

→ 본 fork 영역 즉시 dual instance routing 불가능. 본 SUB 영역 **analytical PoC + router reference + 후속 Phase 3 가이드**.

## 6. 본 doc 갱신

- `analysis/workload_acceptance_analysis_20260524.md` §6.0 다음 영역 SUB_080 analytical PoC 결과 추가
- `INDEX.md` §1 active SUB 표 — SUB_080 entry

## 7. 후속

- **Phase 2 (SUB_081)**: suffix cuda graph 호환 patch 영역 code workload 영역 spec OFF 대신 suffix decoding 사용 가능 시 gating 영역 더 큰 향상 (+30% → +50-70% 추정)
- **Phase 3 (SUB_082)**: dual instance routing viability 확인
- **Phase 4 (SUB_083)**: sonnet/chat 영역 ngram K 추가 향상 (top-M tree verify)

## 8. raw 자료

| 항목 | 위치 |
|---|---|
| 6 mix scenario JSON | `eval/results/20260525_073811_sub080_gating_analytical/{M1,M2,M3,sonnet_only,chat_only,code_only}.json` |
| router reference | `/tmp/run_sub080_router.py` |
| launcher | `/tmp/run_sub080_mix.sh` |
| classifier (IDE_012/SUB_076) | `/tmp/workload_classifier.py` |
