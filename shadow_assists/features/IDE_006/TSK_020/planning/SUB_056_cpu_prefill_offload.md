# SUB_056 — CPU prefill offload for long prompts

> **parent**: TSK_020 / 카테고리 B (Multi-instance CPU pipeline)
> **status**: 대기 (plan only)
> **effort**: large (1-2 주)
> **CPU% target**: 30-50% (prefill 동안)
> **위험**: KV H2D transfer overhead 영역 prefill 절약 시간 압도 가능 (NEO 영역 함정 유사)
> **master plan**: [`SUB_050_to_064_objective_levers.md`](SUB_050_to_064_objective_levers.md) §2

---

## 1. Mechanism

긴 prompt (≥ 4096 token) 의 prefill stage 의 영역 처음 N layers 영역 **CPU 영역 미리 계산** 후 KV 만 GPU 영역 transfer. GPU 영역 영역 나머지 layers 영역 prefill + 영역 decode 영역 transfer.

```
[long prompt (≥ 4096 token)]
  ↓
[CPU: prefill layer 0~N (~10-20 layers)]
  ↓ (KV state H2D, pinned + non_blocking)
[GPU: prefill layer N+1 ~ end + decode]
```

PowerInfer / LLM-in-a-Flash 영역 sparse activation 영역 활용 — 본 lever 영역 (sparse 없이) layer split + KV transfer 영역 단순 변형.

## 2. 출처

| 자료 | 위치 |
|---|---|
| PowerInfer | [arXiv 2312.12456](https://arxiv.org/abs/2312.12456) — "PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU" |
| LLM-in-a-Flash | [arXiv 2312.11514](https://arxiv.org/abs/2312.11514) — "LLM in a Flash: Efficient Large Language Model Inference with Limited Memory" |
| Splitwise | [arXiv 2311.18677](https://arxiv.org/abs/2311.18677) — "Splitwise: Efficient generative LLM inference using phase splitting" |

## 3. Code surface

| 파일 | 변경 |
|---|---|
| `vllm/v1/worker/cpu_prefill.py` (신규) | CPU prefill layer 0~N (transformers 영역 활용 또는 pacpu 영역 확장) |
| `vllm/v1/worker/gpu_model_runner.py` | prefill split logic + KV transfer pattern |
| `vllm/v1/core/sched/scheduler.py` | long prompt 영역 routing — CPU prefill 영역 path |
| `vllm/config/parallel.py` | `cpu_prefill_layers` field 추가 |

## 4. Effort breakdown

| Phase | 작업 | 예상 |
|---|---|:-:|
| Phase 0 | PowerInfer / Splitwise impl 검토 | 2 일 |
| Phase 1 | CPU prefill layer module (transformers Llama layer 영역 CPU forward) | 3 일 |
| Phase 2 | KV state pinned buffer + H2D async pattern | 2 일 |
| Phase 3 | Scheduler routing logic (≥ 4096 prompt → CPU prefill path) | 2 일 |
| Phase 4 | 정확도 verify + throughput 측정 | 2 일 |
| 총 | | **~11 일 (2 주)** |

## 5. CPU% target / throughput 가설

- prompt 8192 token × layer 0~20 (Llama-70B 영역 80 layer 중 25%) 영역 CPU prefill ~5-10 sec
- GPU prefill 영역 절약 time = ~2-3 sec (양 path 영역 wall 차이)
- CPU busy ~30-50% (prefill 동안)
- 가설 (낙관적): throughput 영역 영향 0~+5% (prefill bound workload 영역)
- **위험**: KV H2D transfer (4096 token × 80 layer × 4 KV head × 128 head_dim × bf16 = ~16MB/req) 영역 GPU prefill 절약 시간 압도 가능 — SUB_036/040 NEO 결과 영역 같은 함정 우려

## 6. Risk

| 위험 | 완화 |
|---|---|
| KV transfer overhead 영역 throughput 영역 negative | partial layer (1-5) 만 시도 → KV size ↓ |
| CPU prefill 영역 wall 영역 GPU 영역 압도 — CPU 영역 bottleneck | layer 0~5 정도 영역 small split |
| accuracy: bf16 양 환경 영역 정합 (CPU 영역 numerical 차이) | per-token logprob diff < 0.01 영역 verify |
| NEO 영역 dead path 와 같은 함정 가능 (transfer overhead) | small 영역 시도 후 빨리 측정, negative 시 폐기 |

## 7. Dependencies

- transformers Llama layer 영역 CPU forward (이미 가능)
- pinned mem + non_blocking H2D pattern (vLLM 영역 기존 패턴)

## 8. Acceptance criteria

- [ ] CPU prefill 5 layer + GPU prefill 75 layer 영역 정확도 ≥ 99% top-1
- [ ] throughput ≥ 10,800 tps (SUB_047 의 -1.5% 안)
- [ ] CPU busy ≥ 30% (prefill 동안)
- [ ] **negative ROI 시 폐기 결정 명확화**
