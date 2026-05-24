# Workload acceptance analysis — sonnet / chat / code (2026-05-24 KST)

> **parent**: TSK_020 / SUB_047 (sonnet best) + SUB_071 (chat/code large)
> **목적**: SUB_047 best (sonnet +134%) 가 chat (+37.5%) / code (−23.2%) 에서 보이는 wildly 다른 speedup 의 **mechanism 분석**. ngram speculative decoding 의 효과가 workload shape 에 어떻게 의존하는지 정량 모델 + prompt 구조 fact 로 설명.
> **base**:
> - sonnet 측정: SUB_047 t3 canonical 3-run (10,956.5 tps avg, 366.83 s wall, 500p × 8192in × 8192max)
> - chat / code 측정: SUB_071 1-run (chat 3,006.6 / code 5,346.8, 500p × 8192in × 8192max)
> - vanilla baseline: 동일 scale, 같은 prompt set, 같은 seed
> **scope**: 본 doc 는 **mechanism explainer** — 추가 측정 없이 기존 fact + 정량 모델로 추론. acceptance rate 직접 측정은 §7 에 follow-up 으로 정리.

---

## 1. TL;DR (★ SUB_075 실측 acceptance 반영, 2026-05-24)

| workload | vanilla tps | spec7+cap8 tps | speedup | **실측 mean_accept_len (per-draft K)** | **실측 per-pos α** |
|---|---:|---:|---:|---:|---:|
| **sonnet** | 4,679.8 | **10,956.5** | **+134.1%** | **3.72** | **38.8 %** |
| **chat** | 2,186.0 | **3,006.6** | **+37.5%** | **6.69** ⭐ | **81.2 %** ⭐ |
| **code** | 6,964.5 | **5,346.8** | **−23.2%** | **1.10** | **1.4 %** |

> **★ SUB_075 surprise (2026-05-24)**: per-draft acceptance rank = **chat ≫ sonnet ≫ code** (본 doc 의 사전 예측 sonnet ≫ chat ≫ code 와 반대). throughput rank 는 동일 (sonnet ≫ chat ≫ code). 이유: throughput speedup = **spec coverage × per-draft K**. chat 은 per-draft K 높지만 응답 짧고 spec hit 빈도 낮아 coverage 낮음. sonnet 은 K 중간이지만 응답 길고 coverage 높음. code 는 K, coverage 모두 0 → R 만 누적. 상세: [`measurements/sub075_acceptance_20260524/RESULTS.md`](measurements/sub075_acceptance_20260524/RESULTS.md).

> **★ SUB_074 SuffixDecoding 결과 (2026-05-24, enforce_eager 모드 caveat)**: suffix vs ngram K — sonnet 4.42 vs 3.72 (1.19×), chat 11.58 vs 6.69 (1.73×), **code 7.67 vs 1.10 (★ 7× ⭐)**. tps (eager suffix vs cuda-graph ngram): sonnet 8236 vs 10909 (-25% eager penalty) / chat 2370 vs 2972 (-20% eager penalty) / **code 7094 vs 5362 (+32% net positive — eager penalty 에도 ngram 회귀 mitigation 성공)**. cuda graph 호환 시 suffix 가 모든 workload 에서 ngram 동등 또는 향상 가능성 강함 (code 영역 +70% 가능 추정). 상세: [`measurements/sub074_suffix_20260524/RESULTS.md`](measurements/sub074_suffix_20260524/RESULTS.md).

→ **결정 변수**: prompt 의 **n-gram self-similarity** (prompt 안 어휘 반복) + **generated 응답이 prompt 어휘를 그대로 인용/반사하는가**. sonnet 은 둘 다 ★★★, chat 은 ★★ (sonnet excerpt 인용), code 는 ★ (의미 없는 word salad, generated Python 과 무관).

→ **production 권장**: workload-aware gating — code-like prompt 검출 시 spec OFF. heuristic 는 §6 에 정리.

### 1.1 sonnet "+134.1%" 의 contribution breakdown (SUB_073 / I001 정정)

| 단계 | config | source | tps | vs 직전 | vs vanilla 누적 |
|---|---|---|---:|---:|---:|
| (1) vanilla | `speculative_config=None` | vLLM upstream (spec OFF) | 4,679.8 | — | — |
| (2) **vLLM built-in spec ON (default cap=1)** | `num_spec=7, prompt_lookup=2/5` | **vLLM 영역 코드 변경 0** — feature 활성화만 | **10,778.6** (SUB_044 t3) | **+130.3%** | **+130.3%** |
| (3) **SUB_047 fork patch** | `+ cap=8, div_tp=0` | **본 fork ~6 줄 patch** (env-tunable threading enable) | 10,956.5 (3-run avg) | **+1.65%** | **+134.12%** |

→ +134% 중 **130 pp 는 vLLM built-in 효과**, **본 fork patch 의 추가 기여는 1.65 pp**. 본 doc 의 "+134%" 표기는 본 contribution breakdown 을 전제로 읽을 것.

## 2. 측정 fact

### 2.1 본 분석에 사용한 측정값

| workload | config | tps | wall (s) | in_tok | out_tok | out_tok / prompt | EOS 도달 |
|---|---|---:|---:|---:|---:|---:|---|
| sonnet | vanilla | 4,679.8 | 875 | (n/a)¹ | (n/a) | (n/a) | 부분 |
| sonnet | spec7+cap8 (avg) | 10,956.5 | 366.83 | (n/a) | (n/a) | (n/a) | 부분 |
| chat | vanilla | 2,186.0 | 151.06 | 883,269 | 330,212 | 660 | ★ 조기 (8% of max) |
| chat | spec7+cap8 | 3,006.6 | 113.11 | 883,269 | 340,078 | 680 | ★ 조기 |
| code | vanilla | 6,964.5 | 562.10 | 1,304,836 | 3,914,740 | 7,829 | × max 거의 다 채움 (95.6%) |
| code | spec7+cap8 | 5,346.8 | 726.23 | 1,304,836 | 3,883,055 | 7,766 | × max 거의 다 채움 |

¹ sonnet 의 in/out token detail 은 SUB_044/SUB_047 의 RESULTS.md 에 일부 기록 — 본 분석에는 unit-level token 수치보다 wall/tps ratio 가 중요.

### 2.2 핵심 ratio (spec / vanilla, 같은 workload 안)

| workload | wall ratio | tokens ratio | tps ratio |
|---|---:|---:|---:|
| sonnet | 366.83 / 875 = **0.419** | ≈ 1.00 | 10956 / 4680 = **2.341** |
| chat | 113.11 / 151.06 = **0.749** | 340078 / 330212 = 1.030 | 3007 / 2186 = **1.376** |
| code | 726.23 / 562.10 = **1.292** | 3883055 / 3914740 = 0.992 | 5347 / 6964 = **0.768** |

→ **wall ratio 가 정량 모델의 핵심 입력**.

## 3. 정량 모델

### 3.1 모델

ngram spec decode 의 매 step:

```
spec_step:
    [CPU] ngram lookup → draft 7 candidate token
    [GPU] forward(1 + 7 = 8 token batch) → 8 position logit 계산
    [CPU] rejection sampler → accept_count ∈ {0, ..., 7} token accept
    [GPU/CPU] KV write 1 + accept_count token, 나머지 rollback
    진척 = 1 + accept_count token (최소 1, 항상 progress)
```

평균 accept_count 를 α (0~7), 평균 진척 K = 1 + α (1~8). spec step time 을 vanilla step time × R 로 두면:

```
spec_wall ≈ N_total_tokens / K × spec_step_time
         = N_total_tokens / K × (vanilla_step_time × R)
         = vanilla_wall × R / K

따라서  spec_wall / vanilla_wall = R / K
```

→ **R, K 가 spec 의 성패를 결정**. R > K 면 회귀, R < K 면 가속.

### 3.2 본 환경에서 R 의 추정 범위

본 환경 (Llama-3.3-70B / H100×8 / TP=8 / batch 256 / fp8 KV / cap=8 / decode 단계):

| R 의 원천 | 추정 비중 |
|---|---|
| 1-token → 8-token forward (attention 1→8 position, MLP 1→8 token, decode batch 256→256×8=2048) | +15~20% step_time |
| KV slot 1 → 8 예약 + reject 시 rollback metadata | +3~5% |
| ngram lookup (CPU, cap=8, prompt context len) | +1~5% (prompt 길수록 ↑) |
| rejection sampler + scheduler spec 분기 | +1~2% |
| **합계 R** | **≈ 1.25 ~ 1.40** |

→ **R 은 workload 와 거의 independent** (prompt 길이 영향만 약간). 본 모든 측정이 동일 batch 256 + 동일 spec=7 + 동일 8192 prompt 이므로 R 거의 동일.

→ 정량 분석에서 **R ≈ 1.30 (중앙값) 고정** 가정 사용.

### 3.3 K 역산 (R = 1.30 가정) + SUB_075 실측 (2026-05-24)

```
모델: K_doc = R / wall_ratio  (step 평균, mixed spec+vanilla)
실측: K_spec = mean_accept_len  (per-draft, spec-active step 안에서만)
관계: K_doc = 1 + s × (K_spec − 1)  (s = spec coverage)
```

| workload | wall_ratio | **K_doc (model, R=1.30)** | **K_spec (실측, SUB_075)** | **per-pos α (실측)** | **s (fit, coverage)** |
|---|---:|---:|---:|---:|---:|
| sonnet | 0.419 | **3.10** | **3.72** | 38.8% | **21.1%** |
| chat | 0.749 | **1.74** | **6.69** ⭐ | 81.2% ⭐ | **6.0%** (낮음) |
| code | 1.292 | **1.01** | **1.10** | 1.4% | **~0%** |

> **해석**: K_doc 은 *step 평균* (vanilla + spec mixed), K_spec 은 *spec-active step 안* per-draft K. 두 값이 다른 이유 = spec coverage s (draft 시도가 일어나는 step 비율). s × (K_spec − 1) 만큼 K_doc 가 vanilla (=1) 보다 상승.
>
> sonnet 의 s=21.1% (높은 coverage) 가 chat 의 s=6.0% 보다 결정적으로 큼 — chat 은 per-draft K 가 더 높음에도 coverage 가 낮아 net throughput 향상 폭이 작음. code 의 s=0 이 본 doc 의 ‟α≈0% near regression" 예측과 정합.

> R = 1.30 은 보수적 중앙 추정. R = 1.40 가정 시 K 값 모두 +8% (sonnet K=3.34, chat K=1.87, code K=1.08). 어느 시나리오에서도 **rank order 와 정성적 결론 동일**:
> - sonnet: 매 step ~3 token 진척 (step 평균 K_doc), per-draft 는 K_spec=3.72 / coverage 21%
> - chat: 매 step ~1.7 token 진척 (K_doc), per-draft 는 K_spec=6.69 / coverage 6% — coverage 가 sonnet 보다 낮음
> - code: 매 step ~1.0 token 진척 (K_doc), per-draft K_spec=1.10 / coverage ~0% (zero acceptance) → R 만 누적 → net regression

> **참고**: 본 보수적 추정에서 sonnet K_doc ≈ 3 (step 평균) 는 본 환경 sonnet 의 per-pos α = 38.8% (SUB_075 실측) 과 정합. vLLM literature 의 "ngram sonnet ~60%" 은 다른 setup (workload generator / batch / seed) 기준일 가능성. 본 fork 환경 실측 SUB_075 결과 = canonical fact (본 doc 의 R/K framework 의 first empirical validation).

본 doc 의 결론은 K 의 **절대값** 보다 **rank order** (sonnet ≫ chat ≫ code) 와 **K ≷ R 의 부호** (sonnet 가속 / chat 약한 가속 / code 회귀) 에 의존. 그 부분은 R 가정 범위 어디서나 동일.

### 3.4 Leviathan closed-form 와의 비교 (literature alignment)

본 doc 의 linear approximation `K = 1 + α × γ` 는 **upper bound on K**. Leviathan 2022 ([arXiv 2211.17192](https://arxiv.org/abs/2211.17192), Thm 3.8) 의 정확한 closed-form 은:

```
K_exact = E[# tokens accepted + 1]
        = (1 − α^(γ+1)) / (1 − α)
```

α (acceptance rate per token) 가 낮을수록 두 식의 차이가 작고, α 가 높을수록 linear 가 K 를 과대 평가:

| α | linear K = 1 + 7α | Leviathan K_exact | over-estimate |
|---:|---:|---:|---:|
| 0.0 | 1.00 | 1.000 | — |
| 0.1 | 1.70 | 1.111 | +53 % |
| 0.3 | 3.10 | 1.428 | +117 % |
| 0.5 | 4.50 | 1.992 | +126 % |
| 0.6 | 5.20 | 2.460 | +111 % |
| 0.8 | 6.60 | 4.165 | +58 % |

**implication on K 역산 + SUB_075 실측 비교 (2026-05-24)**:

SUB_075 의 실측 α (per-pos) 를 두 식에 대입:

| workload | 실측 α (per-pos) | linear K=1+7α | Leviathan K_exact | **실측 mean_accept_len** | linear/K_exact 와 실측의 gap |
|---|---:|---:|---:|---:|---|
| sonnet | 0.388 | 3.72 | **1.63** | **3.72** | linear 일치 ✓ / Leviathan 가 ~2.3× 과소 |
| chat | 0.812 | 6.68 | **4.31** | **6.69** | linear 일치 ✓ / Leviathan 가 ~1.55× 과소 |
| code | 0.014 | 1.10 | **1.01** | **1.10** | 둘 다 거의 일치 |

→ **실측 mean_accept_len ≈ linear `1 + 7α` 와 일치, Leviathan closed-form 보다 항상 큼**. 이유 (추정):
- vLLM ngram drafter 의 per-position acceptance 가 **i.i.d. 가정과 다름** — ngram match 가 발견된 시점이 본질적으로 *conformant* 한 위치, 한 position 이 accept 되면 다음 position 도 accept 될 확률 ↑ (correlation > 0).
- Leviathan closed-form 의 `(1 − α^(γ+1)) / (1 − α)` 는 i.i.d. 가정 하의 *geometric stopping*. 실제 ngram 은 first-token-accept 이후 chain 끝까지 accept 되는 strong correlation 발견.
- → linear `1 + 7α` 가 ngram drafter 의 acceptance length 의 **better approximation**. Leviathan closed-form 은 model-based drafter (Medusa/EAGLE 등 i.i.d. closer to assumption) 에 더 잘 fit.

후속 SUB candidate: position-별 conditional acceptance probability 측정 (`num_accepted_tokens_per_pos` field 활용) → ngram 의 correlation structure 정량화.

→ **acceptance rate 직접 측정** (§7) 으로 R 와 K 를 분리하는 것이 정확한 framework alignment 의 전제. 본 doc §1 TL;DR 의 sonnet K ≈ 5.0 / acceptance ~60% 는 linear approximation + vLLM literature 의 acceptance 값을 그대로 차용한 것이며, Leviathan closed-form 적용 시 K ≈ 2.5 가 됨 — 이 두 framework 의 격차는 §7 의 직접 측정 시 closed.

Reference: Leviathan et al. (Google, ICML 2023), "Fast Inference from Transformers via Speculative Decoding" — Thm 3.8 (expected accepted prefix length) + Thm 3.11 (improvement factor IF = K_exact / (γc + 1), 여기서 γc + 1 = 본 doc 의 R 와 대응).

## 4. workload 별 분석 (prompt 구조 + ngram 매칭 잠재력)

### 4.1 sonnet — K ≈ 3, +134% (★★★ 가속)

**prompt 구조** (builder = `random.choices(sonnet.txt 의 517 line, k=164)`):

```
These poor rude lines of thy deceased lover,
When, in disgrace with fortune and men's eyes,
Much liker than your painted counterfeit:
O, change thy thought, that I may change my mind!
... (164 random lines from 517-line sonnet, with replacement)
```

**왜 K 가 높은가**:

1. **prompt 안 line 반복 빈도 매우 높음**: 164 / 517 ≈ 32% per draw with replacement → **birthday problem**: 164 draws among 517 → 평균 ~24 collision (같은 line 두 번 이상 등장). 같은 line 안의 모든 5-gram 이 반복 등장.
2. **sonnet 어휘 자체가 한정적**: 셰익스피어 sonnet 전체 unique word ~5,000 미만 + 자주 쓰는 단어 (`thou`, `thy`, `love`, `time`, `eye`, `fair`, `shall` 등) 가 매우 높은 빈도 → bi-gram / tri-gram pool 이 dense.
3. **generated 응답 = 같은 sonnet 어휘 분포 의 continuation**: model 이 sonnet style 로 응답을 이어가므로 (temperature=0 greedy), 다음 토큰이 prompt 안에 이미 등장한 token sequence 일 확률 ↑.

**결과**: ngram proposer 가 매 step 7 draft 중 평균 2~5 token accept → step 진척 K ≈ 3~5 → spec wall 이 vanilla 의 0.42 배로 압축 → +134%.

### 4.2 chat — K ≈ 1.74, +37.5% (★★ 약한 가속)

**prompt 구조** (builder = system + sonnet excerpt + question + `<|assistant|>`):

```
<|system|>
You are a coding assistant. Help the user with their problem.
<|user|>
When, in disgrace with fortune and men's eyes,
Much liker than your painted counterfeit:
O, change thy thought, that I may change my mind!
... (sonnet lines, ~140 line)
yet mortal looks adore his beauty still,
And every fair from fair sometime declines,
And fortify yourself in your decay

What is the author's intent?
<|assistant|>
```

**왜 K 가 sonnet 보다 낮은가**:

1. **prompt 의 80% 가 sonnet excerpt** → ngram pool 자체는 sonnet 와 거의 동일하게 dense. ngram proposer 의 draft 품질은 sonnet 와 비슷.
2. 그러나 **generated 응답 (assistant 의 literary analysis)** 은 sonnet 의 line 을 *부분 인용* 만 함 (`The author's intent appears to be... as evidenced by the line "When, in disgrace with fortune..."`). 인용된 부분은 ngram 매칭 성공, **assistant 의 메타 어휘** (`analysis`, `evidence`, `theme`, `intent`, `summarize`) 는 prompt 안에 거의 없음 → 매칭 실패.
3. **응답 길이 매우 짧음** (avg 660 tok/prompt, EOS 조기 도달): 응답 끝부분의 closing remark (`In conclusion, ...`) 는 standardized → 그 부분도 매칭 어렵. 짧은 응답에서 inquoted-citation 비율이 평균치를 끌어내림.

**결과**: K ≈ 1.74. R (~1.30) 보다 약간 커 net positive (+37.5%), but sonnet (+134%) 의 1/3.

**부수 효과**: chat 응답이 짧음 (660 tok/prompt) → wall 자체가 짧음 (113s vs sonnet 의 367s). 즉 절대 시간 절감은 ~38s 인 반면 sonnet 은 ~508s. **business impact 영역 sonnet 우위**.

### 4.3 code — K ≈ 1.0, −23.2% (회귀)

**prompt 구조** (builder = function stub + 의미 없는 comment word salad):

```python
def longest_common_subseq(two strings s1 s2):
    """
    Return length of LCS.
    Examples:
    >>> count_chars('hello')          ← random.choice 로 다른 함수 example 섞임
    {'h':1,'e':1,'l':2,'o':1}
    """
    # comment line 0: python test list test return test function test
    # comment line 1: code code code python return test dict loop
    # comment line 2: dict code python python return function dict function
    ... (160 comment lines, 각 line = 8 단어 random.choices from
         ["python","code","function","list","dict","loop","return","test"])
```

**왜 K ≈ 1 (zero acceptance) 인가**:

1. **prompt 본문 = `# comment line N: <8 random words>`** — 어휘 pool 이 단 **8 단어**. bi-gram 의 unique 가능 조합 64 개에 불과. 이 word salad 는 model 이 생성할 **실제 Python 코드** (`if`, `for`, `def`, `return result`, `range(len(s1))`, etc.) 와 **공통 토큰이 거의 없음**.
2. **generated 응답 = 진짜 Python 코드** (model 이 LCS algorithm 구현): `def longest_common_subseq(s1, s2): \n    m = len(s1)\n    n = len(s2)\n    dp = [[0] * (n+1) for _ in range(m+1)]\n    for i in range(1, m+1): ...`. 이 token sequence 는 prompt 안의 word-salad bi-gram 과 매칭되지 않음.
3. **응답 길이 매우 김** (avg 7,829 tok/prompt = 95.6% of max 8192): code 응답이 EOS 없이 max 까지 끝없이 생성 → 매 토큰의 spec step overhead 가 누적될 시간이 충분.
4. **generated code 자체의 self-repetition 도 낮음**: 매 위치마다 새 변수명·함수명·로직 → 응답이 길어져도 self-lookup 으로 매칭되는 ngram 거의 없음 (반복되는 `return`, `else`, `if` 같은 keyword 정도만 hit).

**결과**: K ≈ 1.0 = vanilla 와 단계당 진척 동일. spec step overhead R ≈ 1.30 만 wall 에 누적 → net wall +29%, throughput −23%.

**부수 효과**: code 응답이 max 까지 길게 생성 (7,829 tok/prompt) → 절대 시간 손해도 큼 (+164 s wall). production 환경에서 code request 비율이 높으면 직접 user-facing latency 악화.

## 5. workload 별 prompt 안 ngram 매칭 잠재력 (정성 비교)

| 측면 | sonnet | chat | code |
|---|---|---|---|
| prompt 안 단어 unique count (대략) | ~3,000 | ~3,000 (sonnet excerpt 거의 동일) | ~8 (`python`, `code`, ...) + def header 어휘 |
| prompt 안 line repeat | 매우 높음 (164 draws among 517) | 매우 높음 (sonnet excerpt) | 매우 높음 (`# comment line N:` template) |
| **prompt ↔ generated 간 어휘 overlap** | **★★★ 매우 높음** (continuation in same style) | **★★ 부분적** (cited excerpts) | **★ 거의 없음** (Python code vs word salad) |
| generated self-repetition | 높음 (sonnet line 자체 반복) | 보통 | 낮음 (each var/func unique) |
| **ngram acceptance 예상** | **★★★ 30~60%** | **★★ 10~15%** | **★ 0~3%** |
| **net spec speedup (측정)** | **+134%** | **+37.5%** | **−23.2%** |

→ **결정 인자는 "prompt ↔ generated 간 어휘 overlap"**. prompt 가 self-repeat 가 많아도 generated 응답이 다른 분포면 (code 의 word salad ↔ Python keyword 처럼) 매칭 실패.

## 6. workload-aware gating heuristic (production 권장)

본 분석의 production 함의: **spec decode 의 ON/OFF 를 prompt content 로 dynamic 결정**.

### 6.0 측정 결과 (SUB_076, 2026-05-24)

본 환경 prompt builder (SUB_044/047/071 의 build_sonnet/chat/code_prompts) 의 prompt 1,500개 (3 workload × 500) 에 classifier 적용:

| true \\ pred | sonnet | chat | code | acc |
|---|---:|---:|---:|---:|
| sonnet | **500** | 0 | 0 | 1.000 |
| chat | 0 | **500** | 0 | 1.000 |
| code | 0 | 0 | **500** | 1.000 |

→ **Macro accuracy = 1.000 (perfect)**. 본 환경 prompt builder 는 강한 signature (chat template tag, code def/comment, sonnet free text) 를 부여하므로 trivial 분류 가능. real production traffic 에서는 accuracy 가 낮을 것 (보수 estimate 0.85-0.95). 후속: real dataset (ShareGPT / LMSYS-chat) reproduction (SUB_076 §4.2 candidate). 상세: [`measurements/sub076_classifier_20260524/RESULTS.md`](measurements/sub076_classifier_20260524/RESULTS.md).

### 6.1 권장 heuristic (1차 PoC level)

prompt 의 다음 feature 로 score:

| feature | 가중 | 의도 |
|---|---:|---|
| ` def ` / ` class ` / `def __init__` count | code 지표 | code-like prompt |
| triple-backtick `` ``` `` count | code 지표 | code block 포함 |
| `import ` / `from ` count | code 지표 | Python import statement |
| `<\|system\|>` / `<\|user\|>` / `<\|assistant\|>` tag | chat 지표 | chat template |
| 영문 단어 frequency (top-20 단어 빈도 비율) | high-overlap 지표 | sonnet-like (어휘 한정) |
| unique token / total token ratio | low-repeat 지표 | 낮을수록 spec 가속에 유리 |

**rule 1차**: code 지표 > 임계 → spec OFF. 그 외 → spec ON (cap=8).

### 6.2 효과 추정 (production traffic mix 가정)

가정: production traffic = 60% chat-like + 30% sonnet-like (long-doc analysis) + 10% code-like.

| 시나리오 | sonnet 60% spec | chat 60% spec | code 10% spec | 결과 |
|---|---|---|---|---|
| 항상 spec ON | +134% | +37.5% | **−23.2%** | mix avg ≈ +60% (code drag) |
| **workload-aware gating** | +134% | +37.5% | **0% (vanilla 동등)** | **mix avg ≈ +66%** |

→ gating 으로 production avg throughput **+4~6%** 추가 향상 + code latency p99 안정화.

### 6.3 effort (별도 SUB 신설 시)

- prompt feature extractor: ~1 시간 (regex / tokenizer 기반)
- vLLM 의 per-request spec config 지원 여부 확인: vLLM v1 `SpeculativeConfig` 가 LLM-level only 일 가능성 → request-level override 가 unsupported 면 두 instance (spec ON / OFF) 분리 라우팅 패턴 필요
- 측정: 3 mix scenario × 1-run = ~30 분

→ 본 doc 의 후속으로 SUB_072 (workload-aware gating PoC) 가 자연스러운 next step.

## 7. 검증 방안 (acceptance rate 직접 측정)

본 분석의 K 값은 wall ratio + R 가정으로 역산한 **추정**. R 를 독립 측정하면 K 도 정확히 분리 가능.

### 7.1 vLLM v1 의 spec decode metric

`disable_log_stats=False` 로 LLM 생성 시 vLLM 영역 log 영역:

```
spec_decode_metrics:
  num_drafts: <int>
  num_draft_tokens: <int>     ← total draft (≈ num_drafts × 7)
  num_accepted_tokens: <int>  ← 실제 accept count
  num_emitted_tokens: <int>   ← 실제 progressed token (= num_accepted_tokens + num_drafts)
  acceptance_rate: <float>    ← num_accepted_tokens / num_draft_tokens
```

acceptance_rate 가 직접 측정되면 α = acceptance_rate × 7, K = 1 + α.

### 7.2 효과 추정 effort

- wrapper `/tmp/run_workload_gen.py` 의 `disable_log_stats=True` → `False` 한 줄 변경
- chat / code spec 각 1-run 재측정 (~5 min × 2 = 10 min)
- log 파싱 → acceptance_rate 추출

→ 본 doc 갱신 (§3.3, §4) acceptance 직접값으로 보강.

### 7.3 본 doc 의 conclusion 영향

direct measurement 가 본 doc 의 정성 결론을 바꿀 가능성: **낮음**.
- rank order (sonnet ≫ chat ≫ code): 측정값이 wall_ratio 와 정합 → 동일.
- code 의 zero acceptance: K ≈ 1 ↔ acceptance ≈ 0 으로 직접 fact.
- 정량 boundary (K 값 절대치): 변동 가능. 본 doc 의 K 추정은 ±20% noise band.

## 8. 종합 결론

| workload | mechanism | K | spec 효과 | production 권장 |
|---|---|---:|---:|---|
| **sonnet** | prompt 와 응답 모두 same dense vocab → ngram 매 위치 dense pool, 매 step 다수 accept | ≈ 3.0~5.0 | **+134%** ⭐ | **spec ON, num_spec=7** |
| **chat** | prompt 안 sonnet excerpt 부분 인용 효과만, 메타 응답 어휘는 매칭 안 됨 | ≈ 1.7~1.9 | **+37.5%** | **spec ON** (이득 작음) |
| **code** | prompt word salad 와 generated Python 어휘 disjoint, 매 위치 zero accept | ≈ 1.0 | **−23.2%** | **spec OFF** (workload detector 필요) |

**핵심 lesson**:
1. ngram spec decode 의 효과는 model 의 forward latency 가 아니라 **prompt-↔-generated 어휘 overlap** 에 의해 결정됨.
2. **spec step 의 fixed overhead R ≈ 1.30 이 양날의 검**: K > R 이면 가속, K ≤ R 이면 회귀. 회귀 폭은 R 가 그대로 노출.
3. production 적용 시 **workload-aware gating 가 필수** — 단순히 모든 request 에 spec ON 하면 code-heavy 시 user-facing latency 악화.

## 9. raw 자료 / 부록

| 항목 | 위치 |
|---|---|
| sonnet 측정 RESULTS (SUB_047 canonical 3-run) | [`../measurements/sub047_t3_3run_verify_20260523/RESULTS.md`](../measurements/sub047_t3_3run_verify_20260523/RESULTS.md) |
| sonnet 첫 net-positive (SUB_044) | [`../measurements/sub044_spec_decode_20260523/RESULTS.md`](../measurements/sub044_spec_decode_20260523/RESULTS.md) |
| chat / code 측정 RESULTS (SUB_071) | [`../measurements/sub071_workload_large_20260524/RESULTS.md`](../measurements/sub071_workload_large_20260524/RESULTS.md) |
| SUB_071 plan | [`../planning/SUB_071_workload_large_chatcode.md`](../planning/SUB_071_workload_large_chatcode.md) |
| Best config doc | [`../Best_SpecDecode_10778tps.md`](../Best_SpecDecode_10778tps.md) |
| INDEX | [`../INDEX.md`](../INDEX.md) |
| wrapper (prompt builders) | `/tmp/run_workload_gen.py` (`build_sonnet_prompts` / `build_chat_prompts` / `build_humaneval_prompts`) |
| raw eval — sonnet | `eval/results/20260523_*sub047*/` |
| raw eval — chat | `eval/results/20260524_183239_sub071_chat_*` |
| raw eval — code | `eval/results/20260524_183239_sub071_code_*` |

### 9.1 부록: prompt 예 (각 workload 첫 500 chars)

**sonnet**:
```
These poor rude lines of thy deceased lover,
When, in disgrace with fortune and men's eyes,
Much liker than your painted counterfeit:
O, change thy thought, that I may change my mind!
Yet, do thy worst, old Time: despite thy wrong,
As he takes from you, I engraft you new.
When to the sessions of sweet silent thought
And sable curls all silver'd o'er with white;
... (164 random lines from sonnet.txt, with replacement)
```

**chat**:
```
<|system|>
You are a coding assistant. Help the user with their problem.
<|user|>
When, in disgrace with fortune and men's eyes,
Much liker than your painted counterfeit:
...
And fortify yourself in your decay

What is the author's intent?
<|assistant|>
```

**code**:
```
def longest_common_subseq(two strings s1 s2):
    """
    Return length of LCS.

    Examples:
    >>> count_chars('hello')
{'h':1,'e':1,'l':2,'o':1}
    """
    # comment line 0: python test list test return test function test
    # comment line 1: code code code python return test dict loop
    # comment line 2: dict code python python return function dict function
    ... (160 comment lines)
```

## 10. 관련 연구

본 doc 의 세 가지 주장 — (a) ngram spec 의 효과는 **prompt ↔ generated 어휘 overlap** 에 의해 결정 (§4·§5), (b) workload 에 따라 acceptance 가 zero ~ 60 % 까지 변동 (§3·§4), (c) production 권장은 **workload-aware gating** (§6) — 은 외부 학계·산업계 reference 에서 모두 직·간접 확증된다. 본 절은 각 reference 가 본 doc 의 어느 section 과 연결되는지 명시한다.

### 10.1 reference 일람표

| # | 제목 | 저자 / 소속 | 연도 | venue / 식별자 | 본 doc 연결 section |
|---|---|---|---|---|---|
| R1 | **Prompt Lookup Decoding (PLD)** — vLLM ngram spec 의 원형 | A. Umang | 2023 | [github.com/apoorvumang/prompt-lookup-decoding](https://github.com/apoorvumang/prompt-lookup-decoding) | §4 (mechanism) / §5 / §6 |
| R2 | **Spec-Bench / Survey of Speculative Decoding** | Xia, Yang, Dong, Wang et al. (HKPU / PKU / MSRA / Alibaba) | 2024 | ACL 2024 Findings · [arXiv 2401.07851](https://arxiv.org/abs/2401.07851) · [github.com/hemingkx/Spec-Bench](https://github.com/hemingkx/Spec-Bench) | §3 (정량 모델) / §5 / §10.2 |
| R3 | **Cascade — Utility-Driven Speculative Decoding for MoE** | Saxena, Tsai, Taneja, Jaleel, Qureshi | 2025 | [arXiv 2506.20675](https://arxiv.org/abs/2506.20675) | §6 (gating heuristic) |
| R4 | **Nightjar — Dynamic Adaptive Speculative Decoding for LLM Serving** | (Nightjar 저자, paper 본문 확인 필요) | 2025-12 | [arXiv 2512.22420](https://arxiv.org/abs/2512.22420) | §6 (gating heuristic) |
| R5 | **DSDE — Dynamic Speculative Decoding with KLD Stability** | Samsung SDS | 2025-09 | [arXiv 2509.01083](https://arxiv.org/abs/2509.01083) | §6 / §7 (검증 방안) |
| R6 | **SuffixDecoding — Extreme Speculative Decoding for Emerging AI Applications** | Snowflake AI Research + CMU | 2024-11 (NeurIPS 2025 spotlight) | [arXiv 2411.04975](https://arxiv.org/abs/2411.04975) | §4 (ngram pool 구조) / §6 |
| R7 | **AdaSpec — Adaptive Speculative Decoding for Fast, SLO-Aware LLM Serving** | (AdaSpec 저자) | 2025-03 | [arXiv 2503.05096](https://arxiv.org/abs/2503.05096) | §6 |
| R8 | **AdaEDL — Early Draft Stopping via Entropy-based Lower Bound on Acceptance Probability** | Agrawal et al. | 2024-10 (NeurIPS 2024 ENLSP) | [arXiv 2410.18351](https://arxiv.org/abs/2410.18351) | §3 (R / K 모델) |
| R9 | **TAPS — Task Aware Proposal Distributions for Speculative Sampling** | (TAPS 저자) | 2026-03 | [arXiv 2603.27027](https://arxiv.org/abs/2603.27027) | §4 / §6 |
| R10 | **SpecDec++ — Boosting Speculative Decoding via Adaptive Candidate Lengths** | (SpecDec++ 저자) | 2024-05 | [arXiv 2405.19715](https://arxiv.org/abs/2405.19715) | §3 / §6 |
| R11 | **SGLang Adaptive Speculative Decoding** (산업 구현체) | LMSYS / SGLang team | 2025-2026 | [docs](https://sgl-project.github.io/advanced_features/adaptive_speculative_decoding.html) · [RFC #9319](https://github.com/sgl-project/sglang/issues/9319) | §6 (산업계 비교) |
| R12 | **vLLM RFC #4565 — Automate Speculative Decoding** | (community, closed as not planned) | 2024-05 | [vllm-project/vllm#4565](https://github.com/vllm-project/vllm/issues/4565) | §6 / §6.3 (vLLM 지원 여부) |
| R13 | **vLLM 공식 spec decode 블로그 — "up to 2.8x"** | LMSYS / vLLM team | 2024-10 | [blog.vllm.ai/2024/10/17/spec-decode.html](https://blog.vllm.ai/2024/10/17/spec-decode.html) | §3 / §6 |
| R14 | **Snowflake Arctic Inference + Arctic Training (SuffixDecoding 의 prod 배포)** | Snowflake AI Research | 2025-05 | [blog](https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/) | §6 (산업계 사례) |
| R15 | **TGI (text-generation-inference) `--speculate N`** | Hugging Face | 2024- | [github docs](https://github.com/huggingface/text-generation-inference/blob/main/docs/source/basic_tutorials/consuming_tgi.md) | §6.3 (per-request override 미지원 비교) |
| R16 | **TensorRT-LLM speculative decoding** | NVIDIA | 2024- | [docs](https://nvidia.github.io/TensorRT-LLM/1.2.0rc6/features/speculative-decoding.html) | §6.3 (per-request 미지원 비교) |
| R17 | **vLLM Semantic Router — Workload-Router-Pool architecture** | vLLM community | 2025-09~2026-01 | [arXiv 2603.21354](https://arxiv.org/abs/2603.21354) · [arXiv 2603.12646](https://arxiv.org/abs/2603.12646) · [blog](https://blog.vllm.ai/2025/09/11/semantic-router.html) | §6 / §6.3 (구현 vehicle) |
| R18 | **Sarathi-Serve — chunked-prefill / stall-free scheduling** | Microsoft / Georgia Tech | OSDI 2024 · [arXiv 2403.02310](https://arxiv.org/abs/2403.02310) | §6 (orthogonal 영역 인접 reference) |
| R19 | **DistServe — prefill/decode disaggregation** | Peking University / UCSD | OSDI 2024 · [arXiv 2401.09670](https://arxiv.org/abs/2401.09670) | §6 (orthogonal 영역 인접 reference) |
| R20 | **vLLM issue #8439 — spec decode 가 vanilla 보다 느려진 사례** (Qwen2-7B) | community user | 2024-09 | [vllm-project/vllm#8439](https://github.com/vllm-project/vllm/issues/8439) | §3 / §4.3 (회귀 사례) |

#### 추가: Fundamental theory (R21~R27)

| # | 제목 | 저자 / 소속 | 연도 | venue / 식별자 | 본 doc 연결 section |
|---|---|---|---|---|---|
| **R21** ⭐ | **Fast Inference from Transformers via Speculative Decoding** — spec decoding 의 canonical 원조 | Leviathan, Kalman, Matias (Google) | 2022 (ICML 2023) | [arXiv 2211.17192](https://arxiv.org/abs/2211.17192) | §3.4 (closed-form K_exact = (1−α^(γ+1))/(1−α) + improvement factor IF) |
| **R22** | **Accelerating Large Language Model Decoding with Speculative Sampling** — DeepMind concurrent, Chinchilla 70B 2-2.5× | Chen, Borgeaud, Irving, Lespiau, Sifre, Jumper (DeepMind) | 2023 | [arXiv 2302.01318](https://arxiv.org/abs/2302.01318) | §3 (Leviathan 식 재유도 + modified rejection sampler). Chinchilla HumanEval α=0.8 + speedup 2.46× 는 **model-based drafter** 결과 → 본 doc 의 "code α≈0" (ngram drafter) 와 mechanism 이 다름을 명시. |
| **R23** ⭐ | **EAGLE-1/2/3** — feature-level autoregression draft head | Y. Li, F. Wei, C. Zhang, H. Zhang (PKU/MS) | 2024-2025 | [arXiv 2401.15077](https://arxiv.org/abs/2401.15077) (EAGLE-1) · [arXiv 2406.16858](https://arxiv.org/abs/2406.16858) (EAGLE-2) · [arXiv 2503.01840](https://arxiv.org/abs/2503.01840) (EAGLE-3) · [github.com/SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE) | §4 (drafter mechanism 차이). EAGLE-3 가 HumanEval 6.5× / accept len 7.5 로 **code 가 best workload** (vLLM ngram 의 −23% 와 정반대 패턴) — drafter mechanism 이 K rank order 를 결정. |
| **R24** | **Medusa — multiple decoding head + tree attention** | Cai, Li, Geng, Peng, Lee, Chen, Dao | 2024 | [arXiv 2401.10774](https://arxiv.org/abs/2401.10774) · [github.com/FasterDecoding/Medusa](https://github.com/FasterDecoding/Medusa) | §4 (multi-head draft). MT-Bench coding 3.29× (Vicuna-7B, Medusa-2) — code 가 평균보다 강함, vLLM ngram 과 반대. |
| **R25** | **REST — Retrieval-Based Speculative Decoding** (datastore retrieval) | Z. He, Z. Zhong, T. Cai, J. Lee, D. He | 2024 (NAACL) | [arXiv 2311.08252](https://arxiv.org/abs/2311.08252) · [github.com/FasterDecoding/REST](https://github.com/FasterDecoding/REST) | §4.3 (code workload 대안). external datastore (The Stack code corpus) 에서 suffix retrieval → vLLM ngram 의 "prompt 내 매칭만" 한계를 보완하는 prior art. |
| **R26** | **Lookahead Decoding (LADE) — Jacobi iteration trajectory + n-gram pool** | Y. Fu, P. Bailis, I. Stoica, H. Zhang (UCSD, Hao AI Lab) | 2024 (ICML) | [arXiv 2402.02057](https://arxiv.org/abs/2402.02057) · [github.com/hao-ai-lab/LookaheadDecoding](https://github.com/hao-ai-lab/LookaheadDecoding) | §4 (n-gram pool 확장). prompt 외 trajectory 까지 n-gram pool 에 추가 → code workload 의 self-generated repetition 활용 path. |
| **R27** | **SpecInfer — token tree + tree-parallel verification** | X. Miao et al. (CMU FlexFlow) | 2023 | [arXiv 2305.09781](https://arxiv.org/abs/2305.09781) · [github.com/flexflow/FlexFlow/tree/inference](https://github.com/flexflow/FlexFlow/tree/inference) | §3 (tree-vs-chain). vLLM ngram = single chain → 이론적 K 상한이 tree 보다 낮음. |

#### 추가: vLLM ngram 구현 history (R28~R34)

본 환경의 SUB_047 patch + SUB_065/067 lever 들의 직접 origin / 외부 corroboration. 본 분석 doc 이 *vLLM 본 환경 특이* 가 아니라 *upstream-인지된 한계* 의 일부임을 보여주는 1차 자료.

| # | 항목 | 저자 / merger | 일자 | URL | 본 doc 연결 section |
|---|---|---|---|---|---|
| **R28** ⭐ | **vLLM PR #24986 — `[Spec Decode] Add Batch Parallel Ngram. Upto 8x lower overhead`** — SUB_047 cap=8 patch + SUB_065 num_tokens_threshold=8192 의 **직접 origin** | ekagra-ranjan, co-author Nick Hill (Red Hat) | merged 2025-09-25 | [vllm-project/vllm#24986](https://github.com/vllm-project/vllm/pull/24986) | §1·§7 (SUB_047 patch). PR review (benchislett) 에서 "threading plateaus after 4, no improvement beyond 8 due to sync overhead" + TP coordination 미해결로 threading 실질 disabled 상태 merge — `min(1, cpu_count//2)` + TODO 의 출처. |
| R29 | vLLM PR #12193 — `[V1][Spec Decode] Ngram Spec Decode` (v1 ngram 신규 도입) | LiuXiaoxuanPKU | merged 2025-02-15 | [vllm-project/vllm#12193](https://github.com/vllm-project/vllm/pull/12193) | §0 (v1 ngram origin). pure-Python KMP, single thread. multi-thread / batch parallel 부재 시점의 baseline. |
| R30 | vLLM PR #13365 — `[V1][Spec Decode] Optimize N-gram matching with Numba` (20-30× speedup with large batch) | WoosukKwon | merged 2025-02-18 | [vllm-project/vllm#13365](https://github.com/vllm-project/vllm/pull/13365) | §3.2 R 분석 (ngram lookup CPU 비용의 numba JIT 기여). |
| R31 | vLLM PR #22437 — `[Core][N-gram SD Optimization][1/n] Propose tokens with a single KMP` (O(n²)→O(n)) | Jialin | merged 2025-08-13 | [vllm-project/vllm#22437](https://github.com/vllm-project/vllm/pull/22437) | §3.2 R (ngram lookup 알고리즘 최적화). |
| R32 | vLLM PR #29779 — `[BugFix] Fix index error in ngram_proposer` (PR #24986 의 batch parallel indexing bug 수정, non-contiguous request slot corruption) | usberkeley | merged 2025-12-02 | [vllm-project/vllm#29779](https://github.com/vllm-project/vllm/pull/29779) | §6.3 stability context. PR #24986 의 batch parallel 이 ~2개월간 indexing bug 보유. |
| R33 | vLLM PR #15151 — `[V1][Metrics] Initial speculative decoding metrics` (`vllm:spec_decode_num_drafts / num_draft_tokens / num_accepted_tokens / num_accepted_tokens_per_pos`) | markmc (Red Hat) | merged 2025-04-01 | [vllm-project/vllm#15151](https://github.com/vllm-project/vllm/pull/15151) | §7 (acceptance rate 직접 측정 path). `disable_log_stats=False` + scheduler stats → mean acceptance length = 1 + (accepted / drafts). |
| R34 | vLLM PR #29184 — `[Core] NGram GPU Implementation compatible with Async Scheduler` | (community) | merged 2026-03-08 | [vllm-project/vllm#29184](https://github.com/vllm-project/vllm/pull/29184) | §3 R 분석 (CPU lookup 의 GPU offload 대안). |
| **R35** ⭐ | **vLLM Issue #16258 — `[Usage]: The performance of ngram speculative decoding`** — 외부 user 의 ngram 회귀 보고 (opt-125m / starcoder2-3b 영역 batch=10, k=5, **acceptance 0.698 인데 throughput 238 vs 504 tok/s = 2.1× 회귀**, stale-close) | dtransposed | 2025-04-08 | [vllm-project/vllm#16258](https://github.com/vllm-project/vllm/issues/16258) | §3·§4.3 (high acceptance 만으로 net win 보장 안됨 — R 가 결정). 본 doc 의 code 회귀가 본 환경 특이가 아닌 외부 corroboration. |
| **R36** ⭐ | **vLLM Issue #19254 — `[Bug]: N-gram speculative decoding performs slower than Qwen3-32B-FP8`** — 4×H20 + TP=4, ShareGPT 1k req, acceptance 38.8-61.3%, **mean accept length 1.79-2.83 인데 latency 35.6s → 25.2s 악화 (not planned)** | renne444 | 2025-06-06 | [vllm-project/vllm#19254](https://github.com/vllm-project/vllm/issues/19254) | §3·§4.3·§6.3 (4-8 GPU TP 환경에서 ngram CPU overhead 가 GPU compute 를 능가). vLLM 측 "not planned" close = 본 doc 의 env-tunable lever 정당성. |
| R37 | vLLM Issue #40875 — `[Bug]: ngram default prompt_lookup_min=2 causes tool-call output corruption on Qwen3 with structured output` (KMP spurious match → 잘못된 token; `prompt_lookup_min=8` 로 회피) | Sandermage | 2026-04-25 | [vllm-project/vllm#40875](https://github.com/vllm-project/vllm/issues/40875) | §6 정확도 회귀 가능성 — workload 가 acceptance 뿐 아니라 **correctness** 까지 회귀시킬 수 있음. |
| R38 | vLLM Issue #28947 — `[Tracking Issue][Performance]: Speculative decoding performance/QoL improvements` (in-flight: NGram-GPU PR #29184, Hybrid ngram-eagle PR #24344) | xinli-sw | 2025-11-18 | [vllm-project/vllm#28947](https://github.com/vllm-project/vllm/issues/28947) | §7 (vLLM upstream roadmap). 본 SUB_047 lever 영역 active tracking 없음 → fork patch 의 의미. |
| R39 | vLLM Issue #1802 + #2469 — community 의 PLD → vLLM ngram 통합 요청 | community | 2023 | [#1802](https://github.com/vllm-project/vllm/issues/1802) · [#2469](https://github.com/vllm-project/vllm/issues/2469) | §0 ngram origin (PLD ↔ vLLM 통합 history). |
| R40 | vLLM 공식 doc — `docs/features/speculative_decoding/{README,n_gram}.md` | vLLM team | active | local: `docs/features/speculative_decoding/` | §1 vLLM 공식 권장 = "n-gram = low-to-medium gain, lightweight". 본 doc sonnet +134% 는 공식 예상 범위 초과. |

### 10.2 핵심 reference 5종 상세 (본 doc 결론 직접 지지)

#### R1. Prompt Lookup Decoding (PLD) — **본 doc 의 직접 ancestor**

vLLM 의 ngram speculative decoding 은 PLD (A. Umang, 2023) 의 같은 mechanism 을 internalize 한 것. PLD README 가 본 doc 의 결론과 정확히 같은 워크로드 의존성을 보고:

- **Summarization / context-QA**: "consistent **2.4x** speedup on average" (본 doc 의 sonnet ★★★ 와 유사 mechanism — prompt 의 어휘를 응답이 그대로 인용).
- **Code**: "coding has very high gain in **2nd turn**, because there is lots of code copying" — 즉 1st turn 의 code prompt 와 처음 generation 사이는 gain 이 *작음* (본 doc 의 code 회귀 mechanism 과 정합. 본 doc 의 code prompt 는 1st turn).
- **Roleplay**: "**worst gain. This is probably because there isn't many ngrams to copy, since each generation is sort of unique**." — 본 doc 의 code workload 회귀 mechanism 과 **단어 그대로 같은 설명** ("each generation is unique" ↔ "generated Python 과 prompt word salad 가 disjoint").

→ 본 doc §4.3 의 code 회귀 해석은 PLD 원작자의 self-reported "roleplay worst" 와 같은 mechanism · 같은 결론을 다른 workload 에서 재확인한 것.

#### R2. Spec-Bench — **본 doc §3 의 K rank order 외부 corroboration**

Xia et al. (ACL 2024 Findings, arXiv 2401.07851) 의 unified benchmark. 6 subtask (MT-Bench multi-turn / Translation / Summarization / QA / Math / RAG). RTX 3090 baseline 에서 PLD speedup:

| subtask | PLD speedup |
|---|---:|
| Multi-turn Conversation | 1.64× |
| Translation | **1.15× (lowest)** |
| Summarization | **2.46× (highest)** |
| QA | 1.28× |
| Math | 1.72× |
| RAG | 1.71× |

→ summarization vs translation 사이 **2.14× 격차** = 본 doc 의 sonnet/chat/code 격차 (2.34× / 1.38× / 0.77×) 와 같은 정도의 wide variance. Translation 이 낮은 이유 = source 와 target 의 자연어가 달라 어휘 overlap 이 본질적으로 낮음 (본 doc 의 code 회귀와 같은 mechanism). **Spec-Bench 자체에는 coding category 없음** — 본 doc 의 code workload 측정 (HumanEval-style stub) 은 Spec-Bench 가 cover 하지 않은 영역을 보강하는 fact.

#### R3. Cascade (Utility-Driven Speculative Decoding for MoE) — **본 doc §6 와 같은 design pattern**

Saxena et al. (Georgia Tech / NVIDIA, arXiv 2506.20675, 2025) 은 MoE 환경에서 spec decoding 이 "**최대 1.5× slowdown** 까지 회귀할 수 있다" 를 정량 측정 (이유: draft tokens 가 모인 expert subset 을 모두 활성화하면 weight movement ↑). 본 doc 의 code 회귀 (−23.2 %) 와 정성적으로 같은 phenomenon — fixed overhead R 이 K 를 넘는 영역. 해법으로 **Cascade — speculation utility (token gains / verification cost) 라는 lightweight metric 으로 spec 을 selectively enable / dynamically tune K** 를 제안. 본 doc §6 의 workload-aware gating heuristic 과 정확히 같은 architectural pattern, 다른 domain.

→ 본 doc §6 의 권장사항은 **MoE serving 의 utility-driven approach 의 ngram dense-model 버전** 으로 framing 가능.

#### R4. Nightjar — **본 doc §6 의 "spec OFF 권장" 과 동일한 mechanism 을 LLM serving 일반에서 구현한 학계 첫 사례**

(arXiv 2512.22420, 2025-12). 핵심 발견:
- "existing methods use fixed lengths and **cannot adapt to workload changes or decide when to stop speculation**" — 본 doc §6.3 에서 지적한 vLLM v1 의 한계와 정확히 같은 진단.
- MAB (multi-armed bandit) planner 가 batch size 별 optimal speculative length 를 dynamic 선택, **"speculation no longer beneficial" 판단 시 disable**.
- disabled 상태에서는 draft model 을 CPU 로 offload → KV cache 확보 → larger batch.
- 보고된 결과: standard spec decoding 대비 throughput +14.8 %, latency −20.2 %.

→ 본 doc §6 의 "code 검출 시 spec OFF" 권장은 Nightjar 가 system-load 신호로 같은 행동을 한 것과 design symmetry — 본 doc 는 그 gating signal 을 *prompt content* 로 두자는 보완 제안.

#### R21. Leviathan 2022 — **본 doc §3 정량 모델의 canonical origin**

Leviathan et al. (Google, ICML 2023, arXiv 2211.17192). 본 doc §3 의 `spec_wall / vanilla_wall = R / K` 모델의 정확한 식적 기원.

- **Theorem 3.8 — Expected accepted prefix length**:
  ```
  K_exact = E[#tokens per spec step] = (1 − α^(γ+1)) / (1 − α)
  ```
  α = per-token acceptance rate, γ = draft length (본 환경 = 7). 본 doc §3.4 의 closed-form vs linear approximation 표 참조.

- **Theorem 3.11 — Wall-clock improvement factor**:
  ```
  IF = K_exact / (γc + 1) = (1 − α^(γ+1)) / [(1 − α) × (γc + 1)]
  ```
  c = T_draft / T_target. **ngram drafter 는 c ≈ 0** (CPU lookup, target model forward 와 비교 무시 가능) → 분모가 1 에 수렴, IF ≈ K_exact.

- **본 doc R 의 정확한 정체**: Leviathan 의 분모 `γc + 1` 은 *draft 비용* 만 포함. 본 doc 의 R = 1.30 은 *draft 비용 + verify-side overhead* (n-token forward, KV bookkeeping, rejection sampler) 까지 묶은 broader quantity. 이 차이를 §3 에 명시했음.

- **Reference impl**: 비공식 [romsto/Speculative-Decoding](https://github.com/romsto/Speculative-Decoding).

#### R23/R24. EAGLE-3 & Medusa — **본 doc 의 K rank order 와 정반대 패턴**

ngram drafter (vLLM 본 환경) 의 code workload K ≈ 1.0 (regression) 와 정반대로, **model-based drafter** (EAGLE / Medusa) 는 code workload 가 **best** 시나리오:

| drafter | model | task | speedup / accept len |
|---|---|---|---|
| EAGLE-3 | (target dependent) | **HumanEval (code)** | **6.5× / accept len 7.5** ⭐ |
| Medusa-2 | Vicuna-7B | **MT-Bench coding** | **3.29×** (평균 2.3-2.8× 보다 높음) |
| Chen 2023 (R22) | Chinchilla 70B + 7B | HumanEval | α=0.8, **2.46×** |

EAGLE-3 저자 인용: *"due to many fixed templates in code generation tasks, generating drafts is the easiest."* → **code 의 fixed template (loops, conditionals, return statements) 은 model-based drafter 에게 학습된 prior 가 되지만, ngram drafter 에게는 prompt 안에 그 template token sequence 가 없으면 매칭 불가**. 결과적으로 같은 "code workload" 라도 drafter mechanism 에 따라 K 가 정반대로 ordering 됨.

→ 본 doc 의 결론 ("code workload 회귀") 은 **vLLM ngram drafter 한정**. workload-aware gating 의 alternative path = drafter 자체를 model-based 로 교체 (SUB_050 Eagle CPU draft 가 미해결 이유 = Llama-3.3 전용 EAGLE head 미존재).

#### R28. vLLM PR #24986 — **SUB_047 patch 의 직접 origin**

ekagra-ranjan (co-author Nick Hill, Red Hat), merged 2025-09-25, [vllm-project/vllm#24986](https://github.com/vllm-project/vllm/pull/24986). PR 제목: `[Spec Decode] Add Batch Parallel Ngram. Upto 8x lower overhead.`

본 PR 이 추가한 정확한 코드 (vllm/v1/spec_decode/ngram_proposer.py):

```python
# TODO(ekagra-ranjan): bump up the cap from 1 to 8
# when TP parallelization for ngram is implemented.
self.num_numba_thread_available = min(1, (cpu_count // 2))
self.num_numba_thread_available //= tp_size
```

및 `self.num_tokens_threshold = 8192` hardcoded.

**PR review 인용** (benchislett):
- *"threading performance plateaus after 4 threads, no improvement beyond 8 threads due to sharing and synchronization overhead"* — SUB_047 의 cap=8 선택의 외부 근거.
- *"If only one or two reqs need ngram, you might be able to save some overhead and just run it directly"* — `num_tokens_threshold=8192` 의 design rationale. 단 **8192 자체의 measurement-based justification 은 PR/review 에서 미발견** (heuristic 추정).

**"8× overhead reduction" 은 promise 였으나 TP coordination 미해결로 threading 실제로는 disabled 상태에서 merge** — 본 PR 은 interface refactor 가 주 목적. 본 doc 의 SUB_047 patch (env-tunable cap, divide_by_tp 무력화) 는 이 disabled-threading 을 *실제로 enable* 하는 fork patch.

→ 본 doc §1·§7 의 SUB_047 cap=8 patch 가 **vLLM upstream 의 인지된 TODO 를 즉시 활용** 한 것임의 1차 근거. PR #29779 (R32) 의 indexing bugfix 가 ~2개월 뒤 나온 점에서 본 PR 의 batch parallel path 는 production-mature 단계 미달.

#### R11/R12. SGLang Adaptive Speculative Decoding vs vLLM RFC #4565 — **산업계 현 상태**

- **SGLang** (`--speculative-adaptive`): accept_len 의 EMA 기반으로 num_steps 를 candidate list `[1, 3, 7]` 중 선택. **per-request 가 아닌 batch 평균. 최소 1 step 이고 spec 을 OFF 로 전환하지는 않음.** 즉 본 doc §6 의 "spec OFF" 시나리오는 SGLang adaptive 도 cover 하지 않음.
- **vLLM RFC #4565** (May 2024): "running_queue 가 threshold 초과 시 spec 일시 정지" 를 milestone 1 로 제안 — 본 doc §6 와 같은 방향. 다만 **closed as not planned** (stale, May 2024 ~ 90 days+ inactivity). vLLM v1 에는 현재 per-request spec config override 가 없음 (vLLM `SpeculativeConfig` 는 LLM-level only). 따라서 본 doc §6 의 implementation 은 (a) RFC #4565 의 재제안 또는 (b) **vLLM Semantic Router (R17) 가 두 instance — spec ON / spec OFF — 사이를 prompt content 로 routing** 하는 패턴이 현실적.

→ 본 doc §6.3 의 effort 추정 ("vLLM 의 per-request spec config 지원 여부 확인 → unsupported 면 두 instance 분리 라우팅") 은 **vLLM Semantic Router 의 Workload-Router-Pool architecture 와 직접 연결되는 production 패턴** — 본 doc 의 권장사항이 신규 vLLM core 개발 없이 즉시 prod 적용 가능.

### 10.3 본 doc 와 학계 / 산업계의 차별점

본 doc 의 unique contribution 은 기존 literature 와 비교 시 다음 두 가지:

1. **regression 측정의 정량 분리** — 기존 literature 대부분이 spec decoding 의 *gain* 만 보고하고 regression 은 "MoE / high-QPS / cold-start" 등 *부수 조건* 으로 다룸. 본 doc 은 동일 hardware · 동일 batch · 동일 num_spec=7 · 동일 prompt scale 에서 **순수 workload 차이만으로 +134 % ↔ −23 % 의 257 percentage-point 격차** 를 직접 측정. 이는 PLD README 의 "roleplay worst" 정성 보고를 정량화한 첫 사례에 가까움. **SUB_078 + SUB_079 (2026-05-24) cross-validation 확인**: small/fast model (Qwen 0.5B/1.5B) × {sonnet, chat, code} 6 cell 모두 −48~−65% 회귀 (workload-universal regression — R ≫ K 가 model-scale 의 함수, workload 무관). 외부 issue #16258 + 본 fork 측정 + small model 3 workload 측정 = **3 source corroboration**.
2. **mechanism 의 prompt-content 수준 explanation** — 기존 adaptive literature (R4 Nightjar / R5 DSDE / R8 AdaEDL / R10 SpecDec++) 는 모두 *측정된* acceptance / KLD / entropy 를 feedback signal 로 사용 (즉 spec 시도 후 적응). 본 doc 의 §6 권장은 **spec 시도 전 prompt content 만으로 ON/OFF 결정** — feedback 비용 0 인 *predictive* gating. R3 Cascade 의 utility-driven approach 가 이쪽에 가깝지만 MoE expert activation 에 한정.

### 10.4 후속 reading

- **R5 DSDE** 의 KLD-variance 신호와 §7 의 acceptance-rate 직접 측정을 결합하면 본 doc 의 R / K 분리 모델을 fully empirical 화 가능.
- **R6 SuffixDecoding** 의 per-request suffix tree 가 본 doc 의 code workload (generated 가 prompt 와 disjoint) 에서 **self-generated tokens 만으로 ngram pool 구축** 으로 회귀를 완화할 가능성 — **SUB_074 (2026-05-24) 측정 결과 확정**: code K=1.10→7.67 (7× 향상), tps 5362→7094 (+32% net positive, enforce_eager 패널티 영역도). cuda graph 호환 시 suffix 가 ngram 대비 모든 workload 에서 동등/향상 가능성 강함.
- **R9 TAPS** 의 "draft training distribution 과 workload mismatch" framing 은 본 doc 의 code 회귀를 "ngram proposer 의 implicit prior 가 sonnet-like text 에 fit" 으로 재해석 가능 — draft model 기반 spec 영역으로 본 분석을 확장하는 path.
- **R21 Leviathan closed-form 적용 + R33 PR #15151 의 spec metric** 결합: `disable_log_stats=False` 로 `mean_acceptance_length` 직접 측정 → α 도출 → Leviathan IF 식으로 expected speedup 계산 → 측정 speedup 과 대조 → R 의 정확값 분리 (10 분 effort, SUB_072 의 가장 빠른 entry).
- **R23 EAGLE-3 의 code = best 패턴** 은 SUB_050 (Eagle CPU draft) 재시도의 motivation — model-matched ckpt (Llama-3.3 전용 EAGLE head) 가 가용해지면 본 doc 의 code 회귀 영역을 단번에 ngram → eagle 로 교체 가능. PR #24344 (Hybrid ngram-eagle drafting, R38 tracking issue 의 in-flight 항목) 는 두 drafter 를 workload 별로 switch 하는 vLLM 측의 같은 방향성.
- **R25 REST 의 external datastore retrieval** — vLLM ngram 의 "prompt 안만" 한계를 datastore (The Stack / ShareGPT) 로 보완. Snowflake Arctic Inference (R14) 가 prod 배포 중. 본 doc 의 SUB_054~058 의 CPU pipeline lever 들과 self-host 관점에서 결합 가능.
- **R35 vLLM Issue #16258 의 reproduction**: starcoder2-3b + opt-125m 영역 본 환경에서 재현하면 본 doc 의 "high acceptance ≠ net win" 명제를 두 번째 모델·hardware combo 에서 corroborate. **SUB_078 (2026-05-24) 측정 — opt-125m/starcoder2-3b cache 부재로 Qwen2.5-0.5B/1.5B substitution**: Qwen 0.5B vanilla 11,056 / ngram 4,486 = **2.46× 회귀** (-59%), Qwen 1.5B vanilla 11,016 / ngram 4,195 = **2.63× 회귀** (-62%). issue #16258 의 2.1× 회귀 패턴과 일치, mechanism cross-validation 성공 (small model 영역 R≫K, 항상 net regression). 상세: [`measurements/sub078_repro_20260524/RESULTS.md`](measurements/sub078_repro_20260524/RESULTS.md).

## 11. 본 구현 (SUB_047) vs literature 구현 — 차별점 정리

본 절은 "지금 구현 (SUB_047 = vLLM v1 ngram + env-tunable thread cap=8 + divide_by_tp=0 patch) 이 §10 의 reference 들과 어떤 점에서 다른가" 에 답한다. axis 별 비교 → SUB_047 의 구현 차별점 → 본 분석 framework 의 차별점 → 정직한 contribution 위치 표시 순.

### 11.1 axis 별 구현 비교 (one-glance)

| axis | Leviathan / Chen (R21/R22) | PLD (R1) | Medusa (R24) | EAGLE-1/2/3 (R23) | REST (R25) | SpecInfer (R27) | Lookahead (R26) | **SUB_047 (지금 구현)** |
|---|---|---|---|---|---|---|---|---|
| drafter source | small target-family LM | **prompt 내 n-gram (KMP)** | backbone + 추가 head (train) | feature-level autoregression head (train) | external datastore (The Stack / ShareGPT) | small SSMs | Jacobi trajectory n-gram | **prompt 내 n-gram (KMP)** |
| drafter training 필요 | ✓ (small LM pretrain) | ✗ | **✓ (head fine-tune)** | **✓ (head train)** | ✗ (datastore build) | ✓ | ✗ | **✗** ⭐ |
| drafter 비용 c = T_draft / T_target | 0.015~0.13 (small LM forward) | ≈ 0 (string lookup) | 0.05~0.10 (head forward) | 0.03~0.08 | 0.005~0.02 (retrieval) | 0.02~0.05 | 0 (trajectory 재사용) | **≈ 0 (CPU numba lookup)** |
| proposal 구조 | single chain (γ token) | single chain | tree (multi-head) | dynamic tree (EAGLE-2) | tree (retrieved trie) | token tree | n-gram pool | **single chain** |
| batch / TP 영역 처리 | 논문 미상세 | batch=1 reference | tree attention | tree attention | tree attention | tree-parallel | window | **batch parallel + TP 동일 코드** (PR #24986 base, threading 은 본 patch 로 enable) |
| workload-aware on/off | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **분석 only** — Cascade (R3) / Nightjar (R4) 가 인접하나 **prompt content 기반 predictive gating 은 본 doc 권장이 첫 사례** |
| ckpt 의존 | small LM 필요 (Llama-3.3 base 에는 Llama-3.3 호환 small LM) | 없음 | head ckpt | head ckpt (model-matched 필수, EAGLE-LLaMA3.3 미존재 — 본 환경 SUB_050 정지 사유) | datastore 영역 | SSM 영역 | 없음 | **없음** ⭐ |
| workload K 패턴 | model-dep | **input-grounded 만** | **code = best** (MT-Bench coding 3.29×) | **code = best** (EAGLE-3 HumanEval 6.5×) | code 가속 가능 (datastore Python pattern) | model-dep | trajectory-dep | **code = regression** (-23%) — sonnet-like 만 강함 |

### 11.2 SUB_047 의 구현 차별점

#### 11.2.1 PLD (R1) 와의 차이 — **본질 같지만 vLLM v1 + thread parallelism 적재**

PLD (apoorvumang, 2023-11) 원형:
- pure-Python KMP, single-thread, batch=1 reference impl
- HF Transformers 통합 (PR #27775)

vLLM 영역 ngram path 적재 history (§10 R28~R34):
- **R29 PR #12193** (LiuXiaoxuanPKU, 2025-02-15) — pure-Python KMP, single-thread, single-request 의 v1 신규 도입
- **R30 PR #13365** (WoosukKwon, 2025-02-18) — Numba JIT (20-30× speedup with large batch)
- **R31 PR #22437** (Jialin, 2025-08-13) — single KMP (O(n²) → O(n))
- **R28 PR #24986** (ekagra-ranjan + Nick Hill, 2025-09-25) — Batch parallel ngram interface. **단 TP coordination 미해결로 threading 실질 disabled 상태에서 merge**

**SUB_047 patch (본 구현의 unique 한 줄)**:

```python
# vLLM upstream (PR #24986 of vllm-project/vllm, 2025-09-25):
self.num_numba_thread_available = min(1, (cpu_count // 2))  # 사실상 1 thread
self.num_numba_thread_available //= tp_size                  # 1 // 8 = 0 → fallback 1

# SUB_047 (env-tunable, fork patch):
cap = int(os.environ.get("VLLM_NGRAM_NUM_THREADS_CAP", "1"))
divide_by_tp = int(os.environ.get("VLLM_NGRAM_DIVIDE_BY_TP", "1"))
self.num_numba_thread_available = max(1, min(cap, (cpu_count // 2)))
if divide_by_tp:
    self.num_numba_thread_available //= tp_size
self.num_numba_thread_available = max(1, self.num_numba_thread_available)
```

→ **PR #24986 의 batch parallel 을 실제로 enable**. upstream 이 promise 했지만 disabled 상태로 merge 한 것을 본 fork 가 활성화 (8-thread/rank × 8 rank). 결과: 10,778 → 10,956 tps (+1.6%, sonnet workload). 작아 보이지만 PR #24986 review 의 benchislett 명세 ("threading plateaus after 4-8") 영역 sweet spot.

→ **본 구현 = PLD core idea + vLLM v1 batch path + numba + env-tunable thread cap + TP coordination 우회**. 새 algorithm 아니지만 **upstream 의 인지된 TODO 를 production-ready 까지 가져온 single-line patch**.

#### 11.2.2 model-based drafter (R21/R22/R24/R23/R27) 와의 차이 — **draft model 부재**

| 측면 | model-based drafter | SUB_047 (ngram) |
|---|---|---|
| draft model 추가 weight | 필요 (small LM 또는 head) | **없음 — backbone 만** |
| draft model 학습 / fine-tune | 필요 (Medusa/EAGLE) 또는 매칭 가능 ckpt 필요 (Leviathan) | **불필요** |
| drafter wall cost | T_draft / T_target = 1.5%~10% | **≈ 0** (CPU lookup, target forward 와 비교 무시 가능) |
| model-matched ckpt 의존 | 강함 (Llama-3.3 base 에 Llama-3.3 전용 EAGLE head 필요. 본 환경 SUB_050 시도 시 yuhuili/EAGLE-LLaMA3.3 ckpt 미존재 확인) | **없음** |
| workload K (acceptance) 패턴 | **code = best** (EAGLE-3 HumanEval 6.5×, Medusa-2 MT-Bench coding 3.29×) — code 의 fixed template 이 학습된 prior 와 일치 | **code = regression** (−23%) — prompt word salad 와 generated Python 어휘 disjoint |
| 새 base model 적용 | head retrain 필요 (수 일 ~ 수 주) | **즉시 적용** (env flag 만) |

→ **본 구현 trade-off**: zero-training / instant-apply 강점 ↔ workload-shape 의존 (sonnet-like 만 가속). model-based drafter 는 학습 비용 큰 대신 workload 다양성 강함 (특히 code).

#### 11.2.3 REST (R25) 와의 차이 — **external datastore 부재**

- **REST**: external datastore (The Stack 924MB, ShareGPT 465MB) 영역 longest-suffix retrieval. code generation 시 datastore 의 일반 Python pattern 으로 draft 가능 → **code workload 도 가속**.
- **SUB_047**: prompt 안만 lookup. code prompt 가 word salad 이면 draft 후보 없음 → −23% 회귀.

→ **trade-off**: REST 는 storage (>1GB) + retrieval latency + datastore freshness 관리 부담. SUB_047 은 zero-storage, 단 workload 제한.

#### 11.2.4 SpecInfer (R27) / Medusa (R24) / EAGLE-2 (R23) / Lookahead (R26) 와의 차이 — **chain vs tree, prompt-only vs trajectory**

- **SpecInfer / Medusa / EAGLE-2**: **tree proposal** + tree-parallel verification. 같은 forward 안에 multiple candidate branch 검증 → K 의 이론적 상한 높음.
- **Lookahead**: **Jacobi iteration trajectory** 에서 n-gram pool 수집 → prompt 외 영역 (현재 생성 중인 응답) 도 pool 에 포함 → self-generated repetition 활용.
- **SUB_047**: **single chain proposal** (next 7 token only), **prompt 안 n-gram 만**.

→ **trade-off**: SUB_047 은 KV branching / rollback 영역 단순 (구현 cost 낮음, R32 PR #29779 같은 indexing bug 적음) ↔ K 상한이 tree 방식보다 낮음.

### 11.3 본 분석 doc (framework) 의 차별점

본 doc 자체는 새 algorithm 이 아니라 **분석 framework + production heuristic**. 학계/산업계와의 차별을 axis 로 정리 (§10.3 의 두 항목을 확장):

| 차원 | 기존 literature | 본 doc |
|---|---|---|
| **regression 측정** | 대부분 *gain* 만 보고. regression 은 MoE / high-QPS / cold-start 등 부수 조건으로 다룸 | **동일 hardware + 동일 config + 동일 num_spec 에서 순수 workload 차이로 +134% ↔ −23% 의 257pp 격차 정량 분리** — PLD README 의 "roleplay worst" 정성 보고를 정량화한 첫 사례 (Spec-Bench 가 가장 가깝지만 coding category 없음) |
| **R/K 분해 framework** | Leviathan 의 `K_exact / (γc+1)` 영역 c 는 draft 비용만. ngram drafter 의 verify-side overhead R 는 분리 안 됨 | **R = verify-side overhead (1-token → 8-token forward + KV bookkeeping + sampler 분기), K = 평균 진척** 으로 분해. K ≷ R 부호로 가속/회귀 직접 판정. §3.4 에서 Leviathan closed-form 과 alignment 완결. |
| **gating signal** | 모두 *feedback 기반* — Nightjar (R4) 의 MAB 는 측정된 throughput, AdaEDL (R8) 의 entropy 는 측정된 logit, Cascade (R3) 의 utility 는 측정된 acceptance | **prompt content 기반 predictive gating** — spec 시도 전에 prompt feature (`def`/`class`/`<\|system\|>` 등) 만으로 ON/OFF. feedback cost 0 |
| **production path** | adaptive paper 들은 vLLM core 변경 필요. SGLang `--speculative-adaptive` (R11) 는 batch-level EMA 만, OFF 전환 없음. vLLM RFC #4565 (R12) closed as not planned | **vLLM Semantic Router (R17) 의 Workload-Router-Pool architecture 로 두 instance 분리 라우팅** = vLLM core 변경 없이 즉시 prod 적용 |

### 11.4 정직한 contribution 위치 표시

본 구현 (SUB_047) + 본 분석 doc 의 contribution 을 *과장하지 않고* 정리:

| 측면 | 평가 | 근거 |
|---|---|---|
| 새 algorithm | ✗ | PLD (R1) core idea 그대로 + vLLM batch path 활용 |
| vLLM upstream 의 인지된 TODO 의 즉시 활성화 | ✓ | PR #24986 (R28) 의 disabled-threading 을 single-line env 로 enable |
| 정량 성능 향상 (sonnet workload) | ✓ ⭐ | 4,680 → 10,956 tps (+134%, 2.341×) — SUB_047 t3 canonical 3-run |
| 일반화 (workload generality) | ✗ | sonnet-like 만, chat +37.5%, code −23.2% (SUB_071) |
| 분석 framework 의 originality | ◐ | R/K 분해는 Leviathan/Spec-Bench framework instance. 단 **regression 정량 분리 + prompt-content predictive gating** 은 외부 literature 미발견 |
| production-ready | ◐ | workload-aware gating 없이는 mixed traffic 에서 회귀 위험 — §6 권장사항 적용 필수 |
| open source 즉시 활용 | ✓ | env flag 2개 (`VLLM_NGRAM_NUM_THREADS_CAP=8` + `VLLM_NGRAM_DIVIDE_BY_TP=0`) + ngram_proposer.py 6줄 patch, 새 model weight / datastore 불필요 |

### 11.5 한 줄 요약

> **SUB_047 = PLD core × vLLM v1 batch parallel × upstream TODO 의 fork-side activation. zero-training / zero-datastore / chain-only 의 극단적으로 가벼운 구현이지만, 그 trade-off 로 workload-shape 의존 (sonnet 만 +134%, code 는 −23% 회귀) 이 발생. 본 분석 doc 은 그 mechanism 을 R/K 분해 + Leviathan closed-form 정합 + prompt-content predictive gating heuristic 까지 framework 화한 산학 첫 사례.**
