# SUB_076 — I004 workload-aware gating PoC (classifier first-pass)

> **parent**: TSK_020 / idea I004
> **status**: 활성 (2026-05-24 신설)
> **effort**: 1-2 시간 (classifier + 분류 accuracy 측정만, routing/serving 통합은 후속)
> **idea**: [`../idea/IDE_012_workload_aware_gating_poc.md`](../idea/IDE_012_workload_aware_gating_poc.md)

## 1. scope 제한

본 SUB 는 idea I004 의 전체 PoC (classifier + routing + 3-mix 측정, 1-2일) 중 **1차 first-pass = classifier only**.

이유:
- 본 환경 GPU 8 (H100×8) + Llama-3.3-70B TP=8 = single instance only. 두 instance (spec ON / OFF) 동시 운영 불가.
- vLLM v1 의 per-request spec config override 미지원 (LLM-level only).
- 단일 instance 의 dynamic spec switch (예: workload classifier 결과 따라 매 request 별 spec ON/OFF) 는 vLLM core 변경 필요 — large effort.

→ 1차 PoC = **prompt classifier accuracy 만 측정**. 실제 routing/serving 통합은 후속 SUB candidate.

## 2. 진행 절차

1. `tools/workload_classifier.py` 작성 (regex feature 추출 + rule-based 분류).
2. SUB_044/047/071 의 prompt set 으로 분류 accuracy 측정:
   - sonnet 500 prompt × 3 (SUB_044 t1/t3/SUB_047 canonical) — 모두 sonnet 으로 분류되어야
   - chat 500 prompt (SUB_071) — chat 으로 분류
   - code 500 prompt (SUB_071) — code 으로 분류
3. confusion matrix + precision/recall 계산.
4. 결과 doc + 분석 doc §6 갱신.

## 3. classifier features

idea I004 §2.1 표 그대로:

| feature | extractor | threshold |
|---|---|---|
| ` def ` / ` class ` / `def __init__` count | regex | ≥ 1 = code |
| triple-backtick \`\`\` count | regex | ≥ 2 = code |
| `import ` / `from ` count | regex | ≥ 2 = code |
| `<\|system\|>` / `<\|user\|>` / `<\|assistant\|>` tag | regex | ≥ 1 = chat |
| top-20 영문 단어 빈도 비율 | tokenize + count | ≥ 0.30 = sonnet-like |
| unique token / total token ratio | tokenize | ≤ 0.40 = low-repeat → spec 효과 ↑ |

rule 1차:
- code 지표 (3 중 ≥ 2 hit) → label = "code" → 권장 spec_method = "off" 또는 "suffix" (I002 결과 보고 결정)
- chat 지표 hit → label = "chat" → spec_method = "ngram"
- 그 외 → label = "sonnet-like" → spec_method = "ngram"

## 4. 측정 (CPU only, GPU 불필요)

본 SUB 는 GPU 측정 없음 — prompt text 의 regex/tokenize 만. classifier accuracy 가 main metric. 다른 GPU 측정 SUB 와 병행 가능.

## 5. 산출물

- `tools/workload_classifier.py` (또는 `shadow_assists/.../tools/`)
- `measurements/sub076_classifier_<TS>/RESULTS.md` (confusion matrix + accuracy)
- 분석 doc §6.1 갱신 (실측 accuracy)
- idea I004 §7 결과 (1차 PoC 부분만)
