# SUB_046 ~ SUB_049 — Tier 1 A/B/C + Tier 3 E CPU+spec 결합 plan

> **parent**: TSK_019 / SUB_044 (spec=7 best 10778 tps) + 사용자 지시 — Tier 1 + Tier 3 모두 시도
> **본 doc**: 본 turn 안에 measurement 가능한 영역 (Tier 3 F = SUB_045) 외의 **vLLM 내부 코드 변경 필요한 lever** 의 plan 정리.
> **선행**: SUB_045 (Tier 3 F = spec=7 + multi-workload) 결과 본 후 진입 결정.

---

## 1. Tier 1 A — SUB_046: CPU draft model (small LLM CPU inference)

### 1.1 mechanism

```
main GPU instance (Llama-70B, spec verify)  ←→  CPU draft instance (Llama-3.2-1B, spec proposer)
   │                                                │
   ├─ accept K spec tokens                          ├─ generate K spec tokens
   └─ reject + re-roll                              │
                                                    └─ smaller model → CPU 영역 inference 가능
```

### 1.2 vLLM 의 한계

- 현 vLLM 의 `speculative_config={"method":"draft_model", "model":"..."}` 는 draft model 도 GPU 영역 에 올림
- `vllm/platforms/cpu.py` 존재 — CPU LLM instance 가능. 단 spec_decode 의 draft model 영역 에 CPU device 지정 인터페이스 없음
- → **vLLM 내부 코드 변경 필요**: `SpeculativeConfig.draft_parallel_config.device_type = "cpu"` 지원 추가

### 1.3 적재 surface

| 파일 | 변경 |
|---|---|
| `vllm/config/speculative.py` | `draft_device_type` field 추가 (default "auto") |
| `vllm/engine/arg_utils.py` | spec_config 에 draft_device_type pass-through |
| `vllm/v1/spec_decode/draft_model.py` (또는 worker) | draft model creation 시 device=cpu 지정 |
| `vllm/distributed/` | GPU main ↔ CPU draft tensor sync 패턴 — 큰 변경 |

### 1.4 effort

- large (vLLM 내부 코드 변경 + GPU↔CPU tensor sync overhead 검증 + 정확도 verify)
- 3-5 일

### 1.5 위험

| risk | mitigation |
|---|---|
| CPU draft 영역 의 ms-level latency 가 GPU verify 영역 의 spec gain 압도 | draft model 매우 작게 (1B) + AMX 활용 |
| GPU↔CPU tensor transfer overhead | pinned mem + async copy |
| 정확도 — small draft model 의 reject rate | num_speculative_tokens 보수적 설정 |

---

## 2. Tier 1 B — SUB_047: ngram lookup CPU thread 분리

### 2.1 mechanism

현 vLLM 의 ngram proposer 가 GPU worker 안에서 동기 실행. CPU 별도 thread 로 분리:

```
GPU forward (verify K tokens)  ←→  CPU thread (next K ngram lookup)
   ├─ accept/reject              ├─ async
   └─ output                     └─ pipeline 다음 step 의 draft 미리 준비
```

### 2.2 surface

| 파일 | 변경 |
|---|---|
| `vllm/v1/spec_decode/ngram_proposer.py` | sync → async (별도 CPU thread pool) |
| `vllm/v1/spec_decode/eagle.py` 또는 base | proposer 가 future return |

### 2.3 effort

- medium (vLLM 의 ngram proposer 코드만)
- 2-3 일

### 2.4 효과 가설

- ngram lookup 자체 = micro-sec (~10-100 μs) → wall 영향 작음
- 가설: spec=7 wall 373s 중 ngram lookup overhead = 80 layer × 51 step × 100 μs ≈ 400 ms (~0.1%) → noise

→ **ROI 작음**. 단 CPU 활용은 늘림 (사용자 목표 정합).

---

## 3. Tier 1 C (기각)

spec sampling/logit CPU offload 원래 plan — negative ROI 가능성 + 본 task 영역에서 진행 안 함. (id_registry SUB_048 = 기각)

---

## 4. Tier 3 E — SUB_049: CPU draft + GPU verify + CPU sample 결합

### 4.1 mechanism

= A + C 결합. **모든 영역 CPU 우선** — GPU 는 pure verify forward 만:

```
CPU draft (small LLM)  →  GPU verify (forward only)  →  CPU rejection sampling  →  CPU accept/reject 결정
```

### 4.2 surface

- A + C 의 sum
- 추가: CPU draft 와 CPU sampling 영역 의 zero-copy pipeline (pinned mem 활용)

### 4.3 effort

- very large (A + C + integration)
- 1-2 주

### 4.4 효과 가설

- 매우 unclear — CPU 영역 의 모든 ops 가 GPU 단일 step 안 끝나야 wall 가속
- CPU bottleneck → wall 손해 가능성

→ **권장 안 함** — A 와 B 의 결과 본 후 재평가.

---

## 5. 권장 sequencing

| 순서 | SUB | 작업 | effort | 가치 |
|---|---|---|:-:|---|
| 1 | **SUB_045 (Tier 3 F)** | spec=7 + CPU BG multi-workload — **본 turn 진행 중** | small | ★★★ CLAUDE.md 목표 직접 검증 |
| 2 | **SUB_046 (Tier 1 A)** | CPU draft model | large (3-5일) | ★★ CPU 활용 적극 |
| 3 | **SUB_047 (Tier 1 B)** | ngram CPU thread | medium (2-3일) | ★ ROI 작음 |
| 4 | SUB_049 (Tier 3 E) | A + C 결합 | very large (1-2주) | ⚪ A 결과 후 재평가 |

## 6. 본 turn 결정

- **SUB_045 (Tier 3 F)** 만 본 turn 진행 — 측정 가능한 영역
- SUB_046-049 = plan 만 등록, 다음 turn 진입 결정
- 사용자 지시 ("Tier 1 + Tier 3 모두") 의 진정한 의미 = **simple measurable 영역 (F) 우선 시도 + 나머지 plan + 결과 보고 후 next turn**
