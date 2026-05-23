# SUB_055 — CPU re-ranker / safety filter pipeline

> **parent**: TSK_020 / 카테고리 B (Multi-instance CPU pipeline)
> **status**: 대기 (plan only)
> **effort**: small (1-2 일)
> **CPU% target**: 25-45%
> **master plan**: [`SUB_050_to_064_objective_levers.md`](SUB_050_to_064_objective_levers.md) §2

---

## 1. Mechanism

별도 CPU process 가 cross-encoder re-ranker (BGE-reranker-large) + LlamaGuard 7B (INT8 quantized) 를 항시 실행. 영역 prompt/response 영역 safety filter + retrieval 영역 re-ranking 영역 담당.

```
[incoming prompt] → [LlamaGuard CPU INT8 safety check (input)]
                              ↓ (pass)
[main vLLM (GPU spec decode)]
                              ↓ (response)
                    [LlamaGuard CPU (output safety)]
                              ↓ (pass)
[response → user]
                              ↓ (RAG 영역 retrieval 영역 있을 때)
                    [BGE-reranker CPU (top-K rerank)]
```

본 lever 영역 SUB_054 와 complementary — embedder 영역 retrieval, reranker 영역 post-process.

## 2. 출처

| 자료 | 위치 |
|---|---|
| BGE-reranker | [arXiv 2402.03216](https://arxiv.org/abs/2402.03216) — "Making Large Language Models A Better Foundation For Dense Retrieval" |
| LlamaGuard | [arXiv 2312.06674](https://arxiv.org/abs/2312.06674) — "Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations" |
| LlamaGuard ckpt | meta-llama/LlamaGuard-7b (Hugging Face) |
| BGE-reranker ckpt | BAAI/bge-reranker-large |

## 3. Code surface

| 파일 | 변경 |
|---|---|
| `/tmp/cpu_reranker_safety_loop.py` (신규) | BGE-reranker + LlamaGuard INT8 wrapper |
| `/tmp/run_sub055_cpu_reranker.sh` (신규) | launcher |
| **vLLM 변경 없음** |  |

## 4. Effort breakdown

| Phase | 작업 | 예상 |
|---|---|:-:|
| Phase 0 | LlamaGuard 7B INT8 quantize (bitsandbytes 또는 llama.cpp Q4_0) | 0.5 일 |
| Phase 1 | BGE-reranker-large wrapper + batch=16 | 0.5 일 |
| Phase 2 | NUMA1 binding + thread cap | 0.5 일 |
| Phase 3 | SUB_047 best + 본 lever 동시 측정 | 0.5 일 |
| 총 | | **1-2 일** |

## 5. CPU% target / throughput 가설

- BGE-reranker-large 560M params + LlamaGuard 7B INT8 (~7GB)
- LlamaGuard INT8 영역 single token classify ~30-50ms (CPU bf16 instead 영역 100ms)
- BGE-reranker batch=16 영역 ~80ms
- 둘 다 sustained 영역 CPU busy ~25-45%
- main vLLM throughput 영역 영향 -1~-3%

## 6. Risk

| 위험 | 완화 |
|---|---|
| LlamaGuard 7B INT8 영역 ckpt loading time 영역 길음 (~30s) | warmup 단계 분리 |
| CPU memory bandwidth share | NUMA1 dedicated |
| LlamaGuard INT8 영역 정확도 영역 약간 ↓ | safety 영역 conservative 영역 false-positive ↑ 허용 |

## 7. Dependencies

- SUB_054 와 같은 NUMA1 binding pattern
- bitsandbytes (또는 llama.cpp) 영역 LlamaGuard INT8 quantize

## 8. Acceptance criteria

- [ ] LlamaGuard CPU INT8 + BGE-reranker 영역 sustained 영역 active
- [ ] main vLLM throughput ≥ 10,800 tps
- [ ] CPU busy ≥ 25%
- [ ] safety filter precision ≥ 95% (vs ground truth)
