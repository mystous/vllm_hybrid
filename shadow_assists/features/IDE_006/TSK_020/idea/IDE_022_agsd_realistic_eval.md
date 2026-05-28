# IDE_022 — AGSD realistic-workload + decision-regret evaluation

> **parent backlog**: [`README.md`](README.md) (TSK_020 / SUB_072)
> **선행 idea**: [`IDE_012_workload_aware_gating_poc.md`](IDE_012_workload_aware_gating_poc.md) (SUB_076 classifier first-pass)
> **자식 SUB**: TBD (본 idea 진입 결정 후 신설)
> **발견**: 2026-05-29, AGSD gating 평가 방법론 재검토
> **priority**: ★★ (현 AGSD 결과의 evaluation validity 직결)
> **status**: 활성 (계획)

---

## 1. fact — 현재 AGSD 평가의 두 약점

### 1.1 분류기 자체 — regex pattern matching 의 한계

현 [SUB_076](../planning/SUB_076_workload_aware_gating_classifier.md) 분류기는 prompt 에 다음 rule 적용:

| feature | extractor | 판정 |
|---|---|---|
| ` def ` / ` class ` count | regex | ≥ 1 → code 지표 |
| triple-backtick \`\`\` count | regex | ≥ 2 → code 지표 |
| `import ` / `from ` count | regex | ≥ 2 → code 지표 |
| `<\|system\|>` / `<\|user\|>` tag | regex | ≥ 1 → chat 지표 |

→ 다음 같은 실제 prompt 에서 **brittle**:
- "Explain what this function does: …" (code 블록 없이 자연어로만 설명 요청) → sonnet 분류, 실제론 code-like 응답
- "Here's my chat with the model: `def foo()` …" (chat 인용에 code 포함) → code 오분류
- multi-turn 의 후반부 turn (chat-tag 없는 raw text 입력) → sonnet 으로 분류
- 한국어/일본어 등 비영문 prompt → top-20 영문 단어 빈도 룰 자동 실패

### 1.2 평가 workload — 분류기 룰과 같은 분포

[SUB_076](../planning/SUB_076_workload_aware_gating_classifier.md) §2 의 평가셋:
- sonnet 500 × 3 (SUB_044/047)
- chat 500 (SUB_071)
- code 500 (SUB_071)

→ 모두 본 fork 내 builder 가 합성한 것. **분류기 룰 (`def`/```/chat-tag) 이 잘 hit 되도록 만들어진 prompt 분포** → macro accuracy 1.000 은 자명한 결과 (in-distribution test).

또한 §1.4 의 production mix scenario (M1/M2/M3) 도 같은 3-bucket 의 비율 조합일 뿐 — 분포 shift 없음.

### 1.3 metric 선택의 misalignment

분류기의 실제 목적은 **"label 을 맞히기"** 가 아니라 **"throughput/latency 가 최선이 되는 spec method 를 고르기"** 입니다. label accuracy 가 1.000 이어도 oracle 대비 method 선택이 잘못되면 router 가 의미 없습니다. 반대로 label 이 부정확해도 throughput regret 이 작으면 production 에선 OK.

→ 현 SUB_076 은 metric 자체가 production decision 과 misaligned.

---

## 2. 본 idea — 두 축의 평가 재설계

### 2.1 workload — 합성 3-bucket → 실 trace 로 교체

다음 5 개 실 데이터셋을 main eval corpus 로 채택:

| 데이터셋 | 라이선스 | 사이즈 | 특성 | 다운로드 |
|---|---|---|---|---|
| **LMSYS-Chat-1M** | LMSYS Chat-1M (gated, agree-to-license) | 1M conversation | Chatbot Arena 실 user 입력. 25 model 응답 포함. category 메타데이터 (영어/code/창작 등) 자체 분류 제공. multi-turn. | [huggingface.co/datasets/lmsys/lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) |
| **WildChat-1M** | AI2 Impact License (research) | 1M conversation | GPT-3.5/4 실 user 로그 (Hugging Face Space 수집). toxic / multilingual / NSFW 마스킹. country/state 메타데이터. | [huggingface.co/datasets/allenai/WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) |
| **ShareGPT (RyokoAI 90K)** | CC0 / 사용자 공유 | ~90K conversation | ShareGPT browser extension 으로 수집된 GPT 대화. multi-turn 비율 높음. code/chat mix 가 자연스러움. | [huggingface.co/datasets/RyokoAI/ShareGPT52K](https://huggingface.co/datasets/RyokoAI/ShareGPT52K) |
| **LiveCodeBench** | MIT | 800+ problem | LeetCode/AtCoder/CodeForces 실 문제 (시간 stamped, contamination-free). code generation + execution test. | [huggingface.co/datasets/livecodebench/code_generation_lite](https://huggingface.co/datasets/livecodebench/code_generation_lite) |
| **SWE-Bench Lite** | MIT | 300 issue | GitHub 실제 issue + PR diff. repo context 포함 long-context. | [huggingface.co/datasets/princeton-nlp/SWE-bench_Lite](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite) |

**보조 (cross-check / 분포 보강용)**:

| 데이터셋 | 용도 | 다운로드 |
|---|---|---|
| **Chatbot Arena Conversations** | 33K human-preference 쌍, LMSYS-Chat-1M 의 sibling | [huggingface.co/datasets/lmsys/chatbot_arena_conversations](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations) |
| **OASST1** | 161K message tree, 자원자 작성 (in-the-wild 와는 다른 분포) | [huggingface.co/datasets/OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) |
| **MT-Bench** | 80 multi-turn 평가용 prompt | [huggingface.co/datasets/lmsys/mt_bench_human_judgments](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments) |
| **HumanEval** | 164 small Python 문제 (in-distribution code baseline) | [huggingface.co/datasets/openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval) |
| **Aya Dataset** | 65 언어 human-curated multilingual prompt | [huggingface.co/datasets/CohereForAI/aya_dataset](https://huggingface.co/datasets/CohereForAI/aya_dataset) |

→ **main eval = LMSYS-Chat-1M + WildChat-1M + ShareGPT + LiveCodeBench + SWE-Bench Lite** (실 user / 실 code-task / 자연 mix).
→ 본 fork 의 sonnet/chat/code 500p × 3 셋은 **builder/calibration 용으로만 강등**. 본 corpus 에서 분류기 fitting → 위 5개 셋에서 cross-corpus eval.

### 2.2 metric — accuracy → **decision regret**

분류기의 정답은 "label" 이 아니라 "throughput-optimal method 선택" 입니다. 따라서:

**oracle 측정 (1회, 고정 비용)**:
- 각 prompt p 에 대해 candidate method 집합 M = {vanilla, ngram, suffix, trident-core} 전부 실행
- 각 method m 에서 측정: `tps(p, m)`, `latency(p, m)`, `accept_rate(p, m)`, `peak_mem(p, m)`
- oracle 선택: `m*(p) = argmax_m tps(p, m)` (혹은 latency / regret budget 별 다중 oracle)

**classifier evaluation (재현 가능, GPU 없이도)**:
- 분류기 c 가 prompt p 에 대해 선택한 method `c(p)` 와 `m*(p)` 의 차이
- **regret(p, c) = tps(p, m*(p)) − tps(p, c(p))** ≥ 0
- 분류기 성능 지표:
  - **mean regret** (분포 전체의 평균 손해)
  - **regret CDF** (분포 형태 — fat-tail 여부)
  - **p99 regret** (worst-case 보장)
  - **% prompts with regret = 0** (oracle 일치율)
  - **fraction of prompts where c(p) is *worst* method** (catastrophic mis-route 율)

→ "rule 이 단순해도 regret 작으면 OK", "embedding-based 분류기여도 oracle 대비 손해가 크면 실패" 가 깔끔히 분리됩니다.

### 2.3 candidate 분류기 — 비교 sweep

| 분류기 | 설명 | 학습 데이터 필요? |
|---|---|---|
| **C0: current regex** (SUB_076) | baseline, 룰 그대로 | X |
| **C1: extended regex** | + 자연어 markdown / 한국어 / chat-history 패턴 추가 | X (수동 룰) |
| **C2: bag-of-words + LR** | scikit-learn LogReg, TF-IDF top-1k | LMSYS 의 5-10K subset |
| **C3: distilled MiniLM head** | `sentence-transformers/all-MiniLM-L6-v2` (22M param, CPU-friendly) + 3-class head | LMSYS subset (5-10K) |
| **C4: oracle upper-bound** | label = 각 prompt 의 실측 best method (cheat) | (regret = 0 by construction, ceiling) |

각 분류기 = 같은 corpus 에서 동일 prompt 에 대해 regret 측정. C0~C3 의 regret 곡선이 C4 (=0) 에 얼마나 가까운지가 "classifier quality" 의 정의.

---

## 3. 실험 계획

### 3.1 phase 분할

| phase | 목적 | 머신 | 비용 |
|---|---|---|---|
| **P0: data prep** | 위 5 corpus 다운로드, dedup, length filter (≥32 tok, ≤4096 tok), sampling (각 corpus 2000 prompt 균일 stratified) | dev (CPU only) | ~수 GB 디스크, 1-2 시간 |
| **P1: oracle measurement** | 2000 × 5 corpus = 10000 prompt × 4 method = 40000 generation. **prod (H100×8) 필수** | prod (Xeon SPR + H100×8) | 4 method × 단일 prompt p50 ~2 s × 10000 = ~80 GPU·시간 (Llama-70B TP=8 기준). 작게 시작하려면 corpus 당 500 prompt → 20 GPU·시간. |
| **P2: classifier sweep (offline replay)** | C0~C3 학습 + 4 corpus 에 대한 regret 계산. **GPU 불필요** (oracle table 으로 replay) | dev (CPU only) | 분류기 학습 < 1 시간, regret 계산 < 1 분 |
| **P3: production validation** | best classifier 1 개 + (oracle, always-on, always-off) baseline 을 LMSYS hold-out 으로 e2e 측정 | prod | 4 config × 1000 prompt × p50 2 s = 2.2 GPU·시간 |

### 3.2 step-by-step

**P0 — data prep (dev, CPU)**

```bash
# 1. install
pip install datasets

# 2. download (gated 데이터셋은 huggingface-cli login 후)
python -c "
from datasets import load_dataset
load_dataset('lmsys/lmsys-chat-1m', split='train').to_parquet('data/lmsys.parquet')
load_dataset('allenai/WildChat-1M', split='train').to_parquet('data/wildchat.parquet')
load_dataset('RyokoAI/ShareGPT52K', split='train').to_parquet('data/sharegpt.parquet')
load_dataset('livecodebench/code_generation_lite', split='test').to_parquet('data/livecodebench.parquet')
load_dataset('princeton-nlp/SWE-bench_Lite', split='test').to_parquet('data/swebench.parquet')
"

# 3. 각 corpus 에서 첫 user turn 만 추출, length filter, stratified sample
#    → corpus 당 2000 prompt (총 10000)
#    → metadata 저장: corpus name, original length, language(LMSYS/WildChat 자체 제공), ...
```

**P1 — oracle measurement (prod, GPU)**

각 prompt p, 각 method m ∈ {vanilla, ngram cap=8, suffix spec32, trident-core} 에 대해:
- 동일 SamplingParams (temperature=0, max_tokens=512, seed=42)
- 동일 vLLM build, 동일 모델 (Llama-3.3-70B TP=8)
- 측정값: `tps_decode`, `tps_e2e`, `accept_rate` (spec method), `peak_mem`, `wall_s`

→ 결과: `oracle_table.parquet` (`prompt_id` × `method` × metric)

**P2 — classifier sweep (dev, CPU, replay)**

```python
# C0~C3 정의
classifiers = {
    'C0_regex': RegexClassifier(),  # SUB_076 룰
    'C1_regex_extended': ExtendedRegexClassifier(),
    'C2_tfidf_lr': TfidfLogReg(train_on=lmsys_train_subset),
    'C3_minilm': MinilmHead(base='sentence-transformers/all-MiniLM-L6-v2',
                            train_on=lmsys_train_subset),
}

# regret 계산 (GPU 없이, oracle_table 만으로)
for c_name, c in classifiers.items():
    for p in eval_prompts:
        m_picked = c.predict(p)         # classifier 선택
        m_star   = oracle[p].argmax()    # oracle 선택
        regret[c_name, p] = oracle[p][m_star] - oracle[p][m_picked]

# report: mean / p50 / p99 / CDF / catastrophic-rate
```

**P3 — production validation (prod, GPU, e2e)**

best classifier (C2 or C3) + (always-vanilla, always-best-by-oracle, current C0) baseline 을 LMSYS hold-out 2000 prompt 에 대해 router 모드로 e2e 측정. SUB_094/095 와 동일 multi-instance setup 재사용.

### 3.3 지표 보고 양식

corpus × classifier × metric matrix:

| corpus | classifier | mean regret (tps) | p99 regret | regret=0 율 | catastrophic 율 | e2e tps (P3) |
|---|---|---|---|---|---|---|
| LMSYS-Chat-1M | C0 (current) | … | … | … | … | … |
| LMSYS-Chat-1M | C3 (MiniLM) | … | … | … | … | … |
| WildChat-1M | C0 | … | … | … | … | — |
| WildChat-1M | C3 | … | … | … | … | — |
| … (× 5 corpus × 4 classifier) | | | | | | |

→ 최종 production 권장: P3 e2e tps 1 위 + p99 regret < threshold.

---

## 4. accept gate / kill condition

### 4.1 accept condition

본 idea 진입을 정당화하려면 P1 결과가 다음 중 하나 충족:
- 실 corpus 에서 **C0 (현 regex) 의 mean regret > 5%** of best method tps → 현 분류기가 실 trace 에서 실제로 손해
- C0 의 catastrophic rate (=worst method 선택) **> 10%** of prompts → 분류기가 종종 backward
- 그 외 in-distribution accuracy 1.000 인데 cross-corpus regret 가 0 에 가까운 경우 → 룰이 의외로 robust, 진입 불필요

### 4.2 kill condition

- P1 oracle 측정 결과 method 간 차이가 prompt-level 에서 < 5% (=어차피 누구를 골라도 비슷) → AGSD 가치 자체가 약함, 분류기 정교화 무의미
- C2/C3 의 regret 가 C0 대비 유의하게 개선되지 않음 (예: > 80% prompt 에서 동일 선택) → 룰 분류기로 sufficient, 비용 큰 학습형 무의미

---

## 5. risk & open question

| risk | mitigation |
|---|---|
| LMSYS / WildChat **gated dataset** (HuggingFace 동의 필요) | research use 동의 후 다운로드. 본 fork 공개 시엔 prompt 본문 미공개, hash + oracle metric 만 공유 |
| oracle 측정 비용 (~20-80 GPU·시간) | corpus 당 500-1000 prompt 만으로 시작. method 도 vanilla/ngram/suffix 3개로 축소 가능 |
| 다국어 prompt 비율 (LMSYS/WildChat ~30% 비영어) | per-language regret 분해 보고. 분류기 입력 단계에서 language tag (langdetect) feature 추가 |
| SUB_094/095 의 기존 3-mix 결과와 평가 metric 변경 → 비교 불가 | regret metric 과 별도로 기존 "3-mix avg tps" 도 병행 보고 (backward-compat) |
| oracle 의 stochastic noise (seed 의존) | seed 3 회 run + median tps 사용 |

**open question**:
- corpus 별 prompt 분포가 다른데 (예: LiveCodeBench 는 거의 100% code) corpus 별 가중치를 어떻게 합산할지? → 일단 corpus 별 분리 보고, production 가중치는 LMSYS 분포 기준.
- multi-turn conversation 의 어느 turn 을 classifier 입력으로 쓸지? → 1차는 마지막 user turn 만. multi-turn aware 는 follow-up SUB.
- AGSD 가 LiveCodeBench / SWE-Bench Lite 에서 long-context 가 dominant 일 때 spec 자체가 동작 가능한지 (KV memory pressure)? → P1 oracle 측정 자체가 답을 줌.

---

## 6. 자식 SUB 후보 (id_registry 진입 시 신설)

| SUB 후보 | 내용 | effort |
|---|---|---|
| SUB-X1 (P0) | data prep + 5 corpus stratified sampling tool | 1-2 시간 (dev) |
| SUB-X2 (P1) | oracle measurement runner + table 생성 | 1-2 일 (prod GPU) |
| SUB-X3 (P2) | classifier sweep + regret replay tool + report 생성 | 1 일 (dev) |
| SUB-X4 (P3) | best classifier production e2e validation | 0.5 일 (prod GPU) |

---

## 7. 결과 (TBD)

본 idea 가 SUB 신설 후 측정 진행되면 결과를 본 §7 에 누적.

| 항목 | 결과 |
|---|---|
| P0 data prep | — |
| P1 oracle measurement | — |
| P2 classifier sweep regret | — |
| P3 production e2e | — |
| 본 fork 최종 결정 | — |
