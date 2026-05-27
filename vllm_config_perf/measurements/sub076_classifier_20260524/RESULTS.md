# SUB_076 — workload-aware gating classifier (1차 PoC) RESULTS

> **parent**: TSK_020 / SUB_072 / idea I004
> **plan**: [`../../planning/SUB_076_workload_aware_gating_classifier.md`](../../planning/SUB_076_workload_aware_gating_classifier.md)
> **measurement**: 2026-05-24 15:38 KST
> **scope**: classifier accuracy 만 (routing/serving 통합 후속)
> **prompt set**: SUB_044/047/071 의 build_sonnet_prompts / build_chat_prompts / build_humaneval_prompts, seed=0, n=500 × 3 workload = 1,500 prompt
> **raw**: `eval/results/20260525_003850_sub076_classifier/`

---

## 1. confusion matrix (1,500 prompt)

| true \\ pred | sonnet | chat | code | accuracy |
|---|---:|---:|---:|---:|
| sonnet (n=500) | **500** | 0 | 0 | 1.000 |
| chat (n=500) | 0 | **500** | 0 | 1.000 |
| code (n=500) | 0 | 0 | **500** | 1.000 |

**Macro accuracy: 1.000 (perfect)**

| label | precision | recall | F1 |
|---|---:|---:|---:|
| sonnet | 1.000 | 1.000 | 1.000 |
| chat | 1.000 | 1.000 | 1.000 |
| code | 1.000 | 1.000 | 1.000 |

## 2. classifier rule (`/tmp/workload_classifier.py`)

```python
code_hits = sum([
    n_def_class    >= 1,    # `def ` / `class ` / `async def `
    n_triple_tick  >= 2,    # ```
    n_import       >= 2,    # `import ` / `from `
    n_comment_line >= 10,   # `^\s*#`
    n_py_kw        >= 3,    # return/else/elif/except/raise/yield/lambda/for/while
])
if code_hits >= 2:
    return "code"
if n_chat_tag >= 1:     # `<|system|>` / `<|user|>` / `<|assistant|>`
    return "chat"
return "sonnet"  # default
```

## 3. 해석 — 왜 1.000 인가

본 환경의 prompt builder (`/tmp/run_workload_gen.py`) 가 **분류 가능한 signature 를 강하게 부여**:

| workload | signature feature | 검출 결과 (avg per prompt) |
|---|---|---|
| **sonnet** | `<|...|>` 태그 0 / def·class 0 / triple-tick 0 / comment 0 | code_hits=0, chat_tag=0 → default "sonnet" 분류 |
| **chat** | `<|system|>` + `<|user|>` + `<|assistant|>` 항상 포함 | n_chat_tag=3 → "chat" 분류 |
| **code** | `def f(args):` header + `>>>` example + 160 개 `# comment line N:` | n_def_class=1, n_comment_line=160, n_py_kw 다수 → code_hits ≥ 2 → "code" 분류 |

→ **본 결과는 본 환경 builder 의 trivial classification capability 확인** 의미. real production traffic 의 코드/채팅 mix 는 더 noise 가 있어 accuracy 가 1.000 보다 낮을 것.

## 4. 한계 / 후속 SUB candidate

### 4.1 본 1차 PoC 의 한계

- **trivial dataset**: SUB_044/047/071 의 builder 가 self-classify 가능하도록 일관된 template 사용. 실제 user prompt (예: ShareGPT, MS-MARCO, LMSYS-chat) 의 mixed/ambiguous case 에서는 accuracy 떨어질 것.
- **routing 통합 없음**: 본 PoC 는 classifier accuracy 만. 실제 routing → spec ON/OFF → throughput 측정은 본 SUB scope 외 (idea I004 §2.2 의 옵션 A 가 본 환경 GPU 제약으로 즉시 불가).

### 4.2 후속 SUB candidate

- **SUB_079+** (대기): real user prompt dataset (ShareGPT subset, LMSYS-chat-1M subset) 으로 classifier accuracy 재측정.
- **SUB_080+** (대기): vLLM Semantic Router (R17) 와 결합한 routing PoC — 두 instance 운영 가능 환경 (예: TP=4 × 2 instance) 에서.
- **SUB_081+** (대기): vLLM v1 의 per-request `speculative_config` override 지원 PR (upstream).

## 5. 본 doc 의 §6 갱신 항목

`analysis/workload_acceptance_analysis_20260524.md` §6.1 권장 heuristic 표에 추가:
- "본 환경 builder set 에서 classifier accuracy = 1.000 (SUB_076 측정, 2026-05-24)"
- 실제 traffic 에서는 더 낮을 것임 (보수 estimate ~0.85-0.95).

## 6. raw 자료

| 항목 | 위치 |
|---|---|
| classifier 코드 | `/tmp/workload_classifier.py` |
| launcher | `/tmp/run_sub076_classifier.sh` |
| 생성된 prompt JSON | `eval/results/20260525_003850_sub076_classifier/{sonnet,chat,code}_prompts.json` |
| 분류 결과 JSON | `eval/results/20260525_003850_sub076_classifier/{sonnet,chat,code}_results.json` |
| stdout log | `/tmp/sub076.log` (있는 경우) |
