# SUB_096 — Large-model validation (no MoE)

> **parent**: TSK_020 / SUB_093 Phase 1 형식 (single-instance per cell)
> **scope**: 2026-05-26 00:10 ~ 03:02 KST (~3 hours)
> **status**: ✅ 부분 완료 — Qwen 72B + Phi-3 14B 영역 36 cell 완료 / Llama 3.1-405B FP8 영역 측정 불가

---

## 0. 두괄식

| Model | Trident core 6 workload net positive? | 비고 |
|---|---|---|
| **Qwen 2.5-72B** (TP=8) | **5/6** (code 영역 −5% 회귀) | Llama 70B 와 다른 패턴 (Llama 영역 6/6 positive) |
| **Phi-3-medium 14B** (TP=1) | **6/6 모두 net positive** | 새 mid-size class boundary 확인 |
| Llama 3.1-405B FP8 (TP=8) | ❌ **측정 불가** | tokenizer 파일 캐시 미완 + fbgemm_fp8 quantization deprecated 영역 별도 작업 필요 |

---

## 1. Qwen 2.5-72B (TP=8, 500p × 8192in × 8192max) — Llama 70B 동일 scale

### 1.1 6 workload × 3 config matrix

| Workload | vanilla | ngram | **Trident core** | Trident vs vanilla |
|---|---:|---:|---:|---:|
| sonnet | 6,456 | 7,968 (+23%) | **9,959** | **+54%** ⭐ |
| chat | 2,560 | 3,462 (+35%) | **3,770** | **+47%** ⭐ |
| **code** | **6,227** | 5,766 (−7%) | 5,941 | **−5%** ✗ |
| mix-sh (M1) | 5,832 | 6,586 (+13%) | **8,795** | **+51%** ⭐ |
| mix-bal (M2) | 5,702 | 6,692 (+17%) | **8,228** | **+44%** ⭐ |
| mix-ch (M3) | 6,231 | 6,952 (+12%) | **8,491** | **+36%** ⭐ |

### 1.2 vs Llama 70B (SUB_093 Phase 1) 비교

| Workload | Llama 70B Trident gain | Qwen 72B Trident gain | diff |
|---|---:|---:|---:|
| sonnet | +52.1% | +54% | +1.9pp |
| chat | +68.9% | +47% | −21.9pp |
| **code** | **+18.8%** ⭐ | **−5%** ✗ | **−23.8pp** |
| mix-sh | +62.8% | +51% | −11.8pp |
| mix-bal | +57.2% | +44% | −13.2pp |
| mix-ch | +45.6% | +36% | −9.6pp |

→ **Qwen 72B 영역 Llama 70B 보다 Trident core gain 영역 일관적으로 작음**. 특히 code 영역 회귀 (−5%). chat 영역 +21pp 적음.
→ 가설: Qwen 72B 영역 vanilla tps 영역 Llama 보다 낮음 (6,456 vs 7,679 sonnet) — model architecture / tokenizer 영역 차이 + suffix decoding 영역 acceptance rate 영역 model-dependent.

### 1.3 wall 종합

| config | wall sum (s) |
|---|---:|
| vanilla | 2,700 (~45 min) |
| ngram | 2,690 (~45 min) |
| **Trident core** | **2,196** (~37 min, −19% vs vanilla) |

→ Trident core wall 영역 19% 단축 (Llama 70B 영역 31% 단축 대비 작음).

---

## 2. Phi-3-medium 14B (TP=1, 500p × 2048in × 1024max)

### 2.1 6 workload × 3 config matrix

| Workload | vanilla | ngram | **Trident core** | Trident vs vanilla |
|---|---:|---:|---:|---:|
| sonnet | 3,438 | 4,881 (+42%) | **6,523** | **+90%** ⭐ |
| chat | 3,138 | 3,242 (+3%) | **4,173** | **+33%** ⭐ |
| code | 3,340 | 5,140 (+54%) | **5,085** | **+52%** ⭐ |
| mix-sh (M1) | 3,434 | 4,744 (+38%) | **6,563** | **+91%** ⭐ |
| mix-bal (M2) | 3,374 | 4,762 (+41%) | **7,312** | **+117%** ⭐⭐ |
| mix-ch (M3) | 3,342 | 5,246 (+57%) | **6,390** | **+91%** ⭐ |

### 2.2 ★ Phi-3 14B 영역 특징

- **모든 6 workload 영역 Trident core net positive** ⭐
- **code 영역 회귀 없음** (+52%, Qwen 72B 영역 −5%, Llama 70B 영역 +19% 와 비교)
- ngram 도 모든 workload positive (+3% ~ +57%)
- vanilla baseline 영역 일정 (3,138~3,438) — TP=1 + 14B 모델 영역 GPU 영역 fully bound, workload 영역 throughput 영향 적음
- **mix-bal 영역 +117%** (가장 큰 향상) — workload diversity 영역 spec decoding 효과 증폭

### 2.3 R/K boundary 위치

| 모델 size | Trident gain 패턴 |
|---|---|
| Qwen 0.5B / 1.5B | universal regression (R≫K) — SUB_093 Phase 2 |
| Qwen 7B | mixed (5/6 positive, chat 영역 boundary) — SUB_093 Phase 2 / SUB_094 |
| **Phi-3 14B** | **6/6 net positive** ⭐ — 새 boundary point |
| Qwen 32B | 6/6 positive (SUB_095) |
| Llama 70B | 6/6 positive (SUB_093 Phase 1) |
| **Qwen 72B** | **5/6 positive (code −5%)** — 모델-specific 변동 |
| Llama 3.1-405B FP8 | 측정 불가 |

→ **net positive transition boundary 영역 ≥14B** (Phi-3 14B 부터 6/6 positive). 7B 영역 boundary 근접 / 14B 영역 안전 net positive.

---

## 3. Llama 3.1-405B FP8 영역 측정 불가 이유

| 시도 | 에러 |
|---|---|
| 1st (vanilla) | `pydantic_core._pydantic_core.ValidationError: ... 'fbgemm_fp8' is deprecated` — fbgemm_fp8 quantization vLLM 1.6 영역 deprecated |
| 2nd (with allow_deprecated_quantization=True) | `TypeError: not a string` (SentencePiece tokenizer Load) — Llama 3.1 영역 TikToken 기반인데 SentencePiece fallback 시도 |
| 원인 | HF cache 영역 405B FP8 영역 tokenizer 파일 (tokenizer.json / tokenizer.model / tokenizer_config.json) 미캐시 — config.json + generation_config.json 만 존재. weights 영역 캐시 영역 확인 불가 |

→ 405B 측정 영역 별도 작업 필요: (1) tokenizer 파일 다운로드 OR (2) FP8 영역 다른 quantization variant 활용 (예: bnb-int4 / awq).

---

## 4. 종합 contribution

| 영역 | 값 |
|---|---|
| **Qwen 72B Trident core 평균 vs vanilla** | **+38% (5/6 positive)** ⭐ |
| **Phi-3 14B Trident core 평균 vs vanilla** | **+79% (6/6 positive)** ⭐ |
| 14B class 영역 새 boundary 확인 | Phi-3 14B 영역 net positive transition 영역 확정 (R/K boundary < 14B) |
| 동일 size 다른 vendor 비교 | Llama 70B (6/6 +19% ~ +69%) vs Qwen 72B (5/6, code 회귀) — model architecture 영역 spec acceptance 영향 |
| 본 fork vLLM core 변경 | **여전히 14 줄** (SUB_096 영역 wrapper env-tunable 영역 추가 1 줄 — VLLM_ALLOW_DEPRECATED_QUANT) |

---

## 5. raw data

- `qwen72b/{vanilla,ngram,agsd}/{sonnet,chat,code,mix-sh,mix-bal,mix-ch}/result.json` — 18 cell
- `phi3_14b/{...}/...` — 18 cell
- `qwen72b/{cfg}/_monitor_{cpu,gpu}.csv` / `phi3_14b/...` — util time-series

소스:
- `/tmp/run_sub096_orchestrator.sh` — 3 모델 sequential orchestrator
- `/tmp/run_sub096_llama405b_retry.sh` — 405B FP8 retry (실패)
- `/tmp/run_workload_gen_v3.py` — wrapper (VLLM_ALLOW_DEPRECATED_QUANT env 영역 추가)
