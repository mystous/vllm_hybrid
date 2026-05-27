# SUB_170 — IDE_018 accuracy gate (phase-burst OFF vs ON)

> **parent**: IDE_018 / TSK_034 (paper main contribution accuracy 검증)
> **scope**: 2026-05-27 KST. CLAUDE.md 운영 해석에 따른 per-token logprob max abs diff < 1e-3 gate.
> **status**: 완료 (2026-05-27 07:56 KST) — gate PASS.

---

## 0. 두괄식

| 발견 | 정량 |
|---|---|
| GPU forward path 의 `mark_phase()` 만 추가, model output mutate 없음 | 이론상 OFF/ON 동일 출력 기대 |
| 8 prompts × 32 tokens × top-5 logprobs 비교 | 256 token-level samples |
| per-token logprob max abs diff | **0.000000** |
| token-level match (top1) | **256/256 (100.00%)** |
| gate PASS / FAIL | **PASS** (threshold < 1e-3) |

---

## 1. 측정 방법

### 1.1 sampling 설정 (greedy)

- model: Qwen2.5-32B-Instruct (TP=4 vanilla endpoint 8001)
- temperature = 0.0 (greedy)
- seed = 42
- max_tokens = 32 (짧게 → cumulative divergence 추적)
- logprobs = 5 (top-5)

### 1.2 prompts (8 짧은 prompt — 다양한 task)

1. "The quick brown fox jumps over the lazy dog. Tell me about reflexes."
2. "Explain quantum entanglement in three short sentences."
3. "Write a haiku about machine learning."
4. "What is the capital of France, and why is it famous?"
5. "Describe how a bubble sort works step by step."
6. "List five common HTTP status codes and their meanings."
7. "Translate to Korean: Hello, how are you today?"
8. "Summarize the plot of Hamlet in two sentences."

### 1.3 gate

```
overall_max_abs_diff_logprob < 1e-3   → PASS
```

CLAUDE.md 운영 해석: token-level bit-exact 동등 아님, distribution-similarity binding metric.

---

## 2. 결과

`accuracy_gate_probe.py compare` 출력 → `gate_summary.json`.

### 2.1 per-prompt summary

| idx | prompt 요약 | n_tokens | tok_match | max_logprob_diff | mean_logprob_diff | text_match |
|---:|---|---:|---:|---:|---:|:--:|
| 0 | reflexes (fox sentence) | 32 | 32/32 (100%) | 0.000000 | 0.000000 | OK |
| 1 | quantum entanglement     | 32 | 32/32 (100%) | 0.000000 | 0.000000 | OK |
| 2 | haiku ML                 | 32 | 32/32 (100%) | 0.000000 | 0.000000 | OK |
| 3 | capital of France        | 32 | 32/32 (100%) | 0.000000 | 0.000000 | OK |
| 4 | bubble sort              | 32 | 32/32 (100%) | 0.000000 | 0.000000 | OK |
| 5 | 5 HTTP codes             | 32 | 32/32 (100%) | 0.000000 | 0.000000 | OK |
| 6 | translate to Korean      | 32 | 32/32 (100%) | 0.000000 | 0.000000 | OK |
| 7 | Hamlet summary           | 32 | 32/32 (100%) | 0.000000 | 0.000000 | OK |

### 2.2 overall

| 지표 | 값 |
|---|---|
| overall_max_abs_diff_logprob | **0.000000** |
| overall_token_match | **256/256 (100.00%)** |
| gate_threshold | 1e-3 |
| **gate_pass** | **PASS** |

binding metric (분포 유사성) 만 본 게이트가 아니라 informational metric (token-level) 까지 **bit-exact 일치**. phase-burst hook 의 GPU forward output 무영향 가설 확인.

---

## 3. 분석 — phase-burst hook 의 model output 영향

`mark_phase()` 의 implementation (`phase_burst/_core.cpp` → `PhaseSignal::update`):

```cpp
void update(uint8_t phase, uint64_t step_id) {
    phase_.store(phase, std::memory_order_release);
    step_id_.store(step_id, std::memory_order_release);
    phase_start_ns_.store(now_ns(), std::memory_order_release);
    // eventfd write 통한 worker 알림
}
```

- atomic store (memory order release) — model state 와 무관
- eventfd write — worker process 알림 (별도 thread/process)
- 본 SUB 의 task pool 은 **stub** 상태 (TSK_032/033 의 실제 task 미부착) — 따라서 vllm 의 sampling output 에 영향 0 이 기대됨

실측에서 OFF/ON 두 capture 의 `token_logprobs` 가 256 token 전부에 대해 **완전 동일** 했으며, top1 token id 도 256/256 일치. 즉 bf16 비결합성 noise 조차 관측되지 않음 (단일 endpoint 의 동일 seed greedy 경로이므로 deterministic). 가설 — `mark_phase()` 의 atomic store + eventfd 이 GPU compute graph 와 메모리 의존성이 없음 — 이 실측으로 confirm.

---

## 4. raw data

- `logprobs_off.json` — OFF capture
- `logprobs_on.json` — ON capture
- `gate_summary.json` — compare output + verdict
- `logs/capture_{off,phase_burst_on}.log`

---

## 5. 한계 + 후속

- 본 gate 는 **vanilla endpoint** (8001) 만 측정 — trident endpoint (8002) 의 speculative decoding 경로는 별도 검증.
- AGSD router 의 routing logic 자체는 phase-burst 와 무관 (라우팅은 backend 선택만, prompt → backend forward 는 phase-burst 영향 가능).
- task pool 의 실제 task (TSK_032/033 land 후) 의 mutation 가능성은 본 gate 범위 외 — task 별 별도 accuracy check 필요.
