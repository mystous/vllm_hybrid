# SUB_063 — CPU-load aware request scheduler

> **parent**: TSK_020 / 카테고리 E (Scheduling)
> **status**: 대기 (plan only)
> **effort**: medium (2-3 일)
> **CPU% target**: 30-50% (다른 lever 와 결합)
> **master plan**: [`SUB_050_to_064_objective_levers.md`](SUB_050_to_064_objective_levers.md) §5

---

## 1. Mechanism

vLLM scheduler 영역 영역 request 영역 영역 GPU-only path 영역 routing 함. 본 SUB 영역 영역 CPU load (CPU LLM busy %, embedder backlog, reranker queue) 영역 probe → request 영역 영역 CPU-friendly task (preprocess, draft, postprocess) 영역 routing.

```
[incoming request] → [scheduler check CPU load]
                            ↓
                  ┌─────────┴─────────┐
       CPU idle ↓                        ↓ CPU busy
[route to CPU pre/post path]    [route to GPU-only path]
                            ↓
                  [Sarathi-style chunked prefill]
```

영역 SUB_054/055 (CPU embedder/reranker) 영역 SUB_050/051 (CPU draft) 영역 lever 영역 dynamic 영역 사용량 영역 영역 영역 영역 영역 영역 영역 영역 영역 영역.

## 2. 출처

| 자료 | 위치 |
|---|---|
| Sarathi-Serve | [arXiv 2403.02310](https://arxiv.org/abs/2403.02310) — "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve" — chunked prefill |
| DistServe | [arXiv 2401.09670](https://arxiv.org/abs/2401.09670) — "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving" |
| Splitwise | [arXiv 2311.18677](https://arxiv.org/abs/2311.18677) — prefill/decode split idea |
| vLLM 기존 scheduler | `vllm/v1/core/sched/scheduler.py` |

## 3. Code surface

| 파일 | 변경 |
|---|---|
| `vllm/v1/core/sched/scheduler.py` | priority queue 영역 CPU load probe 영역 추가 |
| `vllm/v1/core/sched/cpu_load_probe.py` (신규) | psutil 영역 CPU% / CPU LLM backlog probe |
| `vllm/v1/engine/processor.py` | request routing logic (CPU path / GPU path) |
| `vllm/config/parallel.py` | `enable_cpu_load_scheduling` flag |

## 4. Effort breakdown

| Phase | 작업 | 예상 |
|---|---|:-:|
| Phase 0 | Sarathi-Serve 영역 chunked prefill 영역 검토 | 0.5 일 |
| Phase 1 | CPU load probe (psutil 영역 영역, 외부 IPC) | 0.5 일 |
| Phase 2 | scheduler 영역 priority queue 영역 routing rule 영역 add | 1 일 |
| Phase 3 | env-gated 영역 (default OFF) + 측정 | 0.5 일 |
| 총 | | **2.5 일** |

## 5. CPU% target / throughput 가설

- CPU 영역 idle 시 영역 routing 영역 CPU 사용량 ↑ (다른 lever 결합 영역)
- 본 lever 영역 standalone 영역 throughput 영역 영향 영역 작음 — combinator 역할
- combined with SUB_054 + SUB_050 영역 영역 영역 30-50% CPU 영역 가능

## 6. Risk

| 위험 | 완화 |
|---|---|
| CPU load probe 영역 hot path 영역 진입 시 latency 영역 영향 | probe 영역 frequency 영역 낮게 (1 Hz 영역) |
| scheduler 영역 변경 영역 vLLM upstream 영역 conflict | env-gated default OFF |
| 본 lever 영역 영역 standalone 영역 효과 영역 측정 영역 어려움 | combined 영역 측정 영역 의미 |

## 7. Dependencies

- SUB_054 / SUB_055 / SUB_050 등 CPU path lever 영역 적어도 1개 land
- psutil 영역 영역 영역 영역 IPC mechanism (Unix socket / shared mem)

## 8. Acceptance criteria

- [ ] scheduler 영역 CPU path routing 영역 활성 시 정상 동작
- [ ] env-gated 영역 default 영역 영역 무영향
- [ ] SUB_054 / SUB_050 결합 영역 CPU busy ≥ 30%, throughput ≥ 10,800 tps
