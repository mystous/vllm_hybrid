# SUB_064 — Dynamic CPU/GPU workload migration

> **parent**: TSK_020 / 카테고리 E (Scheduling)
> **status**: 대기 (plan only)
> **effort**: large (1-2 주)
> **CPU% target**: (다른 lever 와 결합)
> **master plan**: [`SUB_050_to_064_objective_levers.md`](SUB_050_to_064_objective_levers.md) §5

---

## 1. Mechanism

GPU saturated (queue depth ≥ threshold) 시 영역 영역 request 영역 (영역 latency-tolerant 영역) 영역 CPU LLM (smaller model 영역, Qwen 1.5B) 영역 migrate. latency-sensitive 영역 영역 GPU 영역 유지.

LoongServe / Splitwise 영역 phase-split 영역 영역 + 영역 multi-model routing.

```
[scheduler queue depth ↑]
       ↓
[classify pending requests by latency tolerance]
       ↓
   ┌───┴───┐
   ↓        ↓
[GPU main]  [CPU LLM (smaller model)]
       ↓        ↓
   [response]  [response (smaller model 영역 quality 영역 ↓)]
```

본 lever 영역 영역 multi-model serving 영역 dynamic load balance — 영역 cluster-level throughput 영역 향상 + CPU 영역 추가 saturate.

## 2. 출처

| 자료 | 위치 |
|---|---|
| LoongServe | [arXiv 2404.09526](https://arxiv.org/abs/2404.09526) — "LoongServe: Efficiently Serving Long-context Large Language Models with Elastic Sequence Parallelism" |
| Splitwise | [arXiv 2311.18677](https://arxiv.org/abs/2311.18677) — prefill/decode split inference |
| Llumnix | [arXiv 2406.03243](https://arxiv.org/abs/2406.03243) — dynamic LLM serving |
| 관련 idea | OmniServe (gradient-aware), DistServe (분리) |

## 3. Code surface

| 파일 | 변경 |
|---|---|
| `vllm/v1/core/sched/scheduler.py` | queue depth probe + migration logic |
| `vllm/v1/engine/dynamic_routing.py` (신규) | CPU LLM IPC client + request migrate |
| `vllm/v1/core/sched/request_classifier.py` (신규) | latency tolerance 영역 classify (heuristic 또는 ML) |
| CPU LLM wrapper (SUB_049 영역 확장) | IPC server side |

## 4. Effort breakdown

| Phase | 작업 | 예상 |
|---|---|:-:|
| Phase 0 | LoongServe / Splitwise 영역 impl 영역 검토 | 1-2 일 |
| Phase 1 | request classifier (heuristic 영역 start) | 1 일 |
| Phase 2 | queue depth probe + migration trigger | 2 일 |
| Phase 3 | CPU LLM IPC server + protocol | 2 일 |
| Phase 4 | dynamic routing logic + env-gated | 2 일 |
| Phase 5 | 측정 (multi-workload) | 2 일 |
| 총 | | **~11 일 (2 주)** |

## 5. CPU% target / throughput 가설

- 영역 main vLLM (GPU) saturated 시 영역 영역 영역 request 영역 CPU LLM 영역 migrate → CPU 영역 영역 burst saturate ↑
- 영역 quality 영역 trade-off — smaller model 영역 영역 quality ↓
- cluster-level throughput ↑ (영역 영역 영역 영역 영역 영역 영역 영역 영역 영역)

## 6. Risk

| 위험 | 완화 |
|---|---|
| smaller model 영역 quality 영역 영역 영역 영역 영역 | latency-tolerant (background, summarization) 영역 영역 영역 |
| migration overhead 영역 영역 throughput 영역 영역 영역 | classifier 영역 정확도 ↑ |
| classifier 영역 ML 영역 영역 영역 영역 영역 영역 영역 영역 | heuristic-first 영역 시작 (영역 영역 영역 영역 영역 영역 영역) |

## 7. Dependencies

- SUB_049 + SUB_061 (CPU LLM 영역 dedicated saturate)
- SUB_063 (CPU load scheduler) 영역 selectively 결합

## 8. Acceptance criteria

- [ ] migration trigger 정상 동작 (queue depth threshold)
- [ ] CPU LLM 영역 burst 영역 영역 사용 영역 ≥ 60%
- [ ] cluster-level throughput ≥ +5% over solo
- [ ] quality 영역 영역 latency-tolerant 영역 영역 acceptable
