# SUB_071 — chat/code workload large-scale validation (2026-05-24 KST)

> **parent**: TSK_020 / workload generalization (post-plateau verification)
> **plan**: [`../../planning/SUB_071_workload_large_chatcode.md`](../../planning/SUB_071_workload_large_chatcode.md)
> **base config**: HEAD `0757c4e0f`, Llama-3.3-70B, TP=8, gmu=0.85, fp8 KV, **500p × 8192in × 8192max**
> **wrapper**: `/tmp/run_workload_gen.py` (Rec 1 v2 와 동일 builder)
> **launcher**: `/tmp/run_sub071_workload_large.sh`

---

## 1. 결과 (4 cell, 1-run each)

| workload | config | output_tps | wall (s) | init (s) | in_tok | out_tok | crash |
|---|---|---:|---:|---:|---:|---:|---:|
| chat | vanilla | 2,186.0 | 151.06 | 85.13 | 883,269 | 330,212 | 0 |
| chat | spec7+cap8 | 3,006.6 | 113.11 | 86.54 | 883,269 | 340,078 | 0 |
| code | vanilla | 6,964.5 | 562.10 | 88.13 | 1,304,836 | 3,914,740 | 0 |
| code | spec7+cap8 | 5,346.8 | 726.23 | 91.89 | 1,304,836 | 3,883,055 | 0 |

## 2. speedup (spec / vanilla)

| workload | vanilla | spec7+cap8 | speedup | Δ% |
|---|---:|---:|---:|---:|
| **chat** | 2,186.0 | 3,006.6 | **1.376×** | **+37.5%** |
| **code** | 6,964.5 | 5,346.8 | **0.768×** | **−23.2% (회귀)** |

## 3. vs SUB_047 sonnet baseline 비교 (workload generalization)

본 SUB_071 은 SUB_047 의 sonnet 500p × 8192 best (10,956.5 tps, +134.1%) 가 다른 workload 에서도 generalize 되는지 검증.

| workload | scale | vanilla | spec7+cap8 | speedup | source |
|---|---|---:|---:|---:|---|
| sonnet | **500p × 8192** | 4,679.8 | **10,956.5** (3-run avg) | **+134.1%** ⭐ | SUB_047 canonical 3-run |
| **chat** | **500p × 8192** | **2,186.0** | **3,006.6** | **+37.5%** | **SUB_071 (본 측정)** |
| **code** | **500p × 8192** | **6,964.5** | **5,346.8** | **−23.2%** | **SUB_071 (본 측정)** |
| sonnet | 200p × 4096 | 8,395.2 | 9,370.1 | +12% | Rec 1 v2 |
| chat | 200p × 4096 | 2,113.6 | 2,577.1 | +22% | Rec 1 v2 |
| code | 200p × 4096 | 7,889.1 | 5,505.6 | −30% | Rec 1 v2 |

→ **workload 별 speedup 순위**: sonnet (+134%) ≫ chat (+37.5%) > 0 > code (−23.2%).

## 4. scale 별 chat/code 비교 (medium ↔ large)

| workload | scale | speedup | 변화 |
|---|---|---:|---|
| chat | 200p × 4096 | +22% | — |
| chat | 500p × 8192 | **+37.5%** | medium → large 에서 **+15.5pp 개선** (긴 prompt 의 ngram pool 효과 추정) |
| code | 200p × 4096 | −30% | — |
| code | 500p × 8192 | **−23.2%** | medium → large 에서 **+6.8pp 개선** (여전히 회귀 — 부호 안 뒤집힘) |

## 5. 해석

### 5.1 chat (+37.5%)

- prompt 는 `<|system|>…<|user|>{sonnet excerpt}\n{question}<|assistant|>` 형식. sonnet excerpt 가 prompt 안에 포함되어 ngram acceptance 가 sonnet workload 의 일부 효과를 가져옴.
- 단 system/user/assistant template token 과 question 부분은 ngram 패턴 매칭이 어려워 sonnet (+134%) 만큼은 못 따라옴.
- **output token 개수가 매우 적음** (330k / 500 = ~660 tok/prompt) — chat 응답이 EOS 에 조기 도달. wall 자체도 짧음 (151s).
- 500p × 8192 (대형) 가 200p × 4096 (medium) 보다 +15.5pp 더 좋음 → batch 가 크고 prompt 가 길수록 spec 의 amortization 효과 ↑.

### 5.2 code (−23.2% 회귀 재확인)

- code prompt = Python function stub (def name(args): docstring) + comment padding. **응답은 거의 8192 max_tokens 까지 끝없이 생성** (out_tok 3.9M / 500 = ~7,830 tok/prompt = 95.6% of max).
- ngram acceptance 매우 낮음 — code generation 의 token sequence 는 매 시점 새 변수명·함수명·로직을 만들어내므로 prompt 안의 n-gram 과 매칭되는 다음-토큰 시퀀스가 거의 없음.
- spec 실패 시 단계당 비용:
  - 1 forward 가 1+7=8 token (vanilla 의 8 배 FLOPs)
  - rejection sampler 가 0 token accept (또는 1 token) → step 당 1 token 만 진척
  - 결과: 같은 token 을 생성하는 데 더 많은 forward + 더 많은 KV ops + 더 많은 ngram lookup 비용
- spec output tokens (3,883k) ≈ vanilla output tokens (3,914k) — 거의 동일 token 수, wall 만 +29% 증가 (562 → 726s) → spec overhead 가 그대로 nett wall 에 누적.

### 5.3 핵심 fact

> **SUB_047 best 의 +134% 는 sonnet (반복 어휘) 특화 효과**. chat 에서는 +37.5% 로 약화, code 에서는 **−23.2% 의 negative net effect**.

→ production 적용 시 workload-aware gating (code workload 검출 시 spec OFF) 가 필수.

## 6. 판정 (plan §5 기준)

| workload | 결과 | 분류 | 다음 lever |
|---|---:|---|---|
| chat | +37.5% | **partial benefit** (+10~50% 영역) | spec 활성 set 에 chat 포함, 단 expected speedup 1.3~1.4× |
| code | −23.2% | **regression** (< −10%) | **production 권장: code workload 검출 시 spec OFF** |

## 7. raw 자료

| 항목 | 위치 |
|---|---|
| chat vanilla | `eval/results/20260524_183239_sub071_chat_vanilla/result.json` |
| chat spec7+cap8 | `eval/results/20260524_183239_sub071_chat_spec7_cap8/result.json` |
| code vanilla | `eval/results/20260524_183239_sub071_code_vanilla/result.json` |
| code spec7+cap8 | `eval/results/20260524_183239_sub071_code_spec7_cap8/result.json` |
| summary tsv | `/tmp/sub071_summary.tsv` |
| launcher log | `/tmp/sub071.log` |
| wrapper | `/tmp/run_workload_gen.py` (sonnet / chat / code 3-builder) |
| launcher | `/tmp/run_sub071_workload_large.sh` |

## 8. 보완 측정 후보 (미실행, 본 1-run 측정으로 충분 판정)

- chat 3-run variance 확인 → ngram lookup 의 system template 의존 noise 측정 (effort 30 min)
- chat workload 의 ngram acceptance rate 직접 측정 (vllm log 분석)
- workload-aware gating 의 detector (prompt 안의 ` def ` / ```` ``` ```` / `class ` token ratio) PoC

본 SUB 의 핵심 fact (chat positive / code regression) 는 1-run 으로도 결론 가능 — speedup 차이가 noise (≤1% / ±100 tps) 보다 훨씬 큼.
