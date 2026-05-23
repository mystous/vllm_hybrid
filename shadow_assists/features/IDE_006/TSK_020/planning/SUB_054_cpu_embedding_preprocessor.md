# SUB_054 — CPU embedding model preprocessor

> **parent**: TSK_020 / 카테고리 B (Multi-instance CPU pipeline)
> **status**: ✓ **측정 완료 (2026-05-23)** — batch sweep 32/64/128 측정 + Phase 1 결합 측정
> **effort**: small (1-2 일) ✓
> **CPU% 결과**: 19~21% (target 30-50% 미달 — SUB_049 영역 영역 영역 영역 영역 영역 영역)
> **★ 최적 batch=64**: main_tps 10,848.1 (-1.0% vs SUB_047) / CPU 21.21% / embedder 36.7 sps
> **master plan**: [`SUB_050_to_064_objective_levers.md`](SUB_050_to_064_objective_levers.md) §2

---

## 0. 측정 결과 종합 (2026-05-23)

| batch | main tps | vs SUB_047 best | CPU% | GPU% | emb sps | wall(s) |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 10,834.2 | -1.1% | 19.70 | 41.0 | ~36 | 371.0 |
| **★ 64** | **10,848.1** | **-1.0%** | 21.21 | 41.1 | 36.7 | 370.5 |
| 128 | 10,616.3 | -3.1% | 21.35 | 41.8 | 42.2 | 378.6 |

**production config**: batch=64 권장 — throughput cost 최소 (-1.0%) + CPU 활성 +15.7pp + embedder sustained 36.7 sps. (batch=128 영역 embedder throughput +16% 영역 main throughput -2.1pp trade-off.)

**raw**: `eval/results/20260523_182152_sub054_cpu_embedder/` (b=32), `..._194810_phase2_sub054_batch64/`, `..._batch128/`

**참고 — Phase 1 결합 (Qwen 1.5B + BGE emb + BGE rerank 동시)**: main_tps 9,635.4 (-12.1%) / CPU 23.85% — CPU contention 으로 단독 합보다 throughput 큰 회귀. 결합 시 자원 분리 (NUMA1 56 core 영역 단일 lever) 가 합리적.

---

---

## 1. Mechanism

별도 CPU process 가 sentence embedding model (BGE-large / E5-large / MiniLM-L6) 을 항시 실행. RAG / semantic search / dedup / 이전 prompt cache 같은 LLM-인접 작업 영역 CPU 영역 담당.

```
[incoming prompt] → [CPU: BGE/E5 embedder (NUMA1 56 thread)]
                                 ↓
                         [redis / in-mem vector store]
                                 ↓
                         [retrieval / dedup result]
                                 ↓
[main vLLM (GPU spec decode)]
```

본 lever 의 SUB_049 와 차이: SUB_049 영역 CPU LLM (Qwen 0.5B/1.5B causal) inference. 본 SUB 영역 encoder (BGE/E5) — encoder 영역 transformer 영역 SIMD 영역 더 친화 (autoregressive 아닌 fully parallel).

## 2. 출처

| 자료 | 위치 |
|---|---|
| BGE paper | [arXiv 2309.07597](https://arxiv.org/abs/2309.07597) — "C-Pack: Packaged Resources To Advance General Chinese Embedding" |
| E5 paper | [arXiv 2212.03533](https://arxiv.org/abs/2212.03533) — "Text Embeddings by Weakly-Supervised Contrastive Pre-training" |
| MTEB benchmark | [arXiv 2210.07316](https://arxiv.org/abs/2210.07316) — "MTEB: Massive Text Embedding Benchmark" |
| BGE GitHub | `FlagOpen/FlagEmbedding` |
| sentence-transformers | GitHub `UKPLab/sentence-transformers` |

## 3. Code surface

| 파일 | 변경 |
|---|---|
| `/tmp/cpu_embedding_loop.py` (신규) | BGE-large-en-v1.5 wrapper (sentence-transformers) + NUMA1 binding |
| `/tmp/run_sub054_cpu_embedding.sh` (신규) | launcher (SUB_049 pattern 따라) |
| **vLLM 변경 없음** | wrapper script 만으로 가능 |

## 4. Effort breakdown

| Phase | 작업 | 예상 |
|---|---|:-:|
| Phase 0 | sentence-transformers + BGE-large-en-v1.5 dependency 설치 | 0.5 일 |
| Phase 1 | `cpu_embedding_loop.py` wrapper (sonnet 영역 chunk 영역 input, batch=32) | 0.5 일 |
| Phase 2 | NUMA1 binding + thread cap (56 core) | 0.5 일 |
| Phase 3 | SUB_047 best + 본 CPU embedder 동시 측정 (3 scenario: solo / +embedder small / +embedder large) | 0.5 일 |
| 총 | | **1-2 일** |

## 5. CPU% target / throughput 가설

- BGE-large-en-v1.5 (335M params, BF16) + batch=32 + seq_len=512 영역 ~50ms/batch
- 56 thread × 80% util → CPU busy ~40-50%
- main vLLM spec decode throughput 영역 영향 거의 없음 (NUMA0/1 분리)
- 가설: CPU 4.66% → **40-50%**, throughput 영향 -1~-3% (memory bandwidth share)

## 6. Risk

| 위험 | 완화 |
|---|---|
| NUMA0/1 영역 memory bandwidth share 영역 spec decode 영향 | NUMA1 dedicated, `numactl --membind=1` |
| BGE 영역 transformers Python 영역 GIL bottleneck | sentence-transformers 영역 internal 영역 ONNX runtime 옵션 |
| 본 lever 자체 가 spec throughput 향상 안 함 | "별도 throughput" 영역 즉, cluster-level throughput 영역 평가 |

## 7. Dependencies

- SUB_049 의 NUMA1 binding pattern (이미 검증)
- sentence-transformers 설치 (uv pip install sentence-transformers)

## 8. Acceptance criteria

- [ ] BGE embedder NUMA1 영역 56 thread 영역 sustained 영역 active
- [ ] main vLLM throughput ≥ 10,800 tps (SUB_047 의 -1.5% 안)
- [ ] CPU busy ≥ 35%
- [ ] embedder throughput ≥ 100 batch/sec (≥ 3,200 sentence/sec)
