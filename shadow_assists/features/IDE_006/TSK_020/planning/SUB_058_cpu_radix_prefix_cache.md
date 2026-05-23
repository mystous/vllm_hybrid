# SUB_058 — CPU radix-tree prefix KV cache (SGLang style)

> **parent**: TSK_020 / 카테고리 C (vLLM 내부 CPU lever)
> **status**: 대기 (plan only)
> **effort**: large (1-2 주)
> **CPU% target**: 20-40% / **throughput 가설**: repeated prompt 영역 + 큰 효과
> **master plan**: [`SUB_050_to_064_objective_levers.md`](SUB_050_to_064_objective_levers.md) §3

---

## 1. Mechanism

SGLang RadixAttention 영역 patternprefix 영역 KV state 영역 CPU radix tree 영역 cache 한다. 영역 prompt 영역 동일 prefix 영역 가질 때 cache hit → GPU 영역 KV 만 transfer (prefill 영역 skip).

```
[incoming prompt]
  ↓
[CPU: radix tree lookup (prompt prefix → KV cache entry)]
  ↓ (hit)
[KV state CPU pinned → GPU H2D async]
  ↓
[GPU prefill (cache hit prefix 영역 skip) + decode]
```

vLLM 영역 기 prefix caching 영역 있음 (`vllm/v1/core/sched/prefix_cache.py`) — GPU side. 본 SUB 영역 영역 **CPU side radix tree** 영역 추가 → GPU mem 영역 영역 더 많은 prefix cache.

## 2. 출처

| 자료 | 위치 |
|---|---|
| SGLang paper | [arXiv 2312.07104](https://arxiv.org/abs/2312.07104) — "Efficiently Programming Large Language Models using SGLang" |
| RadixAttention | SGLang §3.2 |
| Reference impl | GitHub `sgl-project/sglang` |
| vLLM 기존 prefix cache | `vllm/v1/core/sched/prefix_cache.py` (GPU side, hash-based) |

## 3. Code surface

| 파일 | 변경 |
|---|---|
| `vllm/v1/core/cpu_radix_prefix_cache.py` (신규) | CPU radix tree (Python 또는 cython) |
| `vllm/v1/core/sched/prefix_cache.py` | GPU cache miss 시 CPU radix lookup 추가 |
| `vllm/v1/worker/gpu_model_runner.py` | KV transfer pattern (CPU radix → GPU) |
| `vllm/config/cache.py` | `cpu_prefix_cache_size_gb` field 추가 |

## 4. Effort breakdown

| Phase | 작업 | 예상 |
|---|---|:-:|
| Phase 0 | SGLang RadixAttention impl 검토 | 1 일 |
| Phase 1 | CPU radix tree skeleton (Python prototype) | 2 일 |
| Phase 2 | KV pinned buffer + CPU lookup integration | 2 일 |
| Phase 3 | KV transfer pattern (CPU radix → GPU) — async | 2 일 |
| Phase 4 | cython 또는 C++ 영역 lookup hot path 최적화 | 2 일 |
| Phase 5 | 정확도 + throughput 측정 (chat workload, repeated prompt) | 2 일 |
| 총 | | **~11 일 (2 주)** |

## 5. CPU% target / throughput 가설

- radix tree lookup 영역 CPU busy ~10-20% (single thread lookup)
- KV pinned buffer mgmt 영역 추가 ~10-20% (multi-thread copy)
- 가설: chat workload (영역 chat history 영역 repeat) 영역 cache hit rate ~30-50%
- throughput: prefill skip → +20-40% (chat workload 영역), sonnet 영역 영역 영향 작음

## 6. Risk

| 위험 | 완화 |
|---|---|
| CPU radix tree 영역 lookup latency 영역 critical path 영역 들어가면 영역 throughput 영역 감소 | cython/C++ 영역 hot path |
| KV transfer overhead | pinned + async |
| sonnet workload 영역 영역 cache hit 영역 거의 없음 → 영역 benefit 작음 | chat workload 별도 측정 (sonnet 외) |
| vLLM 기 GPU prefix cache 영역 중복 → 영역 영역 management 영역 복잡 | 명확 영역 분리 (GPU hot / CPU cold) |

## 7. Dependencies

- vLLM `prefix_cache.py` (이미 존재)
- chat workload dataset (sonnet 외 시도 필수)

## 8. Acceptance criteria

- [ ] CPU radix tree lookup 정상
- [ ] chat workload 영역 cache hit rate ≥ 30%
- [ ] throughput ≥ 11,500 tps (chat workload baseline 대비 +20%)
- [ ] CPU busy ≥ 20%
- [ ] sonnet 영역 throughput 영역 영향 ≤ -2%
