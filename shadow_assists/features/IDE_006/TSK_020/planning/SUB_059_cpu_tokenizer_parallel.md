# SUB_059 — CPU tokenizer / stop-string parallel

> **parent**: TSK_020 / 카테고리 C (vLLM 내부 CPU lever)
> **status**: 대기 (plan only)
> **effort**: small (1 일)
> **CPU% target**: 5-15% (tokenizer 만)
> **master plan**: [`SUB_050_to_064_objective_levers.md`](SUB_050_to_064_objective_levers.md) §3

---

## 1. Mechanism

vLLM 의 prefill 단계 영역 tokenizer 영역 단일 thread (PyO3 GIL 안 영역 single-call) 영역 사용. Hugging Face Tokenizers (Rust impl) 영역 batch 영역 native multi-thread 영역 (rayon parallel) 영역 가능 — 본 lever 영역 영역 batch encode 영역 multi-thread 영역 + stop-string matching 영역 parallel.

```
[batch of N prompts]
  ↓
[tokenizer.encode_batch(N) — rayon parallel]  ← 8 thread
  ↓
[GPU prefill]
  ...
[batch of N output streams]
  ↓
[stop-string match parallel (per stream)]  ← 8 thread
```

본 lever 영역 throughput 영역 큰 영역 없지만 (tokenizer 영역 micro-sec) — CPU busy 영역 추가 영역 5-10% + tokenizer latency 영역 ↓ (large batch 영역 효과).

## 2. 출처

| 자료 | 위치 |
|---|---|
| HF Tokenizers | GitHub `huggingface/tokenizers` (Rust impl, PyO3 binding) |
| Rayon parallel | Rust rayon crate (work-stealing thread pool) |
| vLLM tokenizer | `vllm/transformers_utils/tokenizer*.py` |

## 3. Code surface

| 파일 | 변경 |
|---|---|
| `vllm/transformers_utils/tokenizer.py` | `encode_batch` 호출 시 multi-thread 활용 (RAYON_NUM_THREADS env) |
| `vllm/v1/engine/processor.py` | prefill batch tokenize 의 parallel 처리 |
| `vllm/v1/engine/output_processor.py` | stop-string match parallel (per stream thread pool) |
| env | `RAYON_NUM_THREADS=8` |

## 4. Effort breakdown

| Phase | 작업 | 예상 |
|---|---|:-:|
| Phase 0 | 현 tokenizer call pattern 분석 (single vs batch) | 0.25 일 |
| Phase 1 | encode_batch 적용 + RAYON_NUM_THREADS 설정 | 0.25 일 |
| Phase 2 | stop-string match parallel (concurrent.futures 또는 threading pool) | 0.25 일 |
| Phase 3 | 측정 (tokenize time + CPU busy) | 0.25 일 |
| 총 | | **1 일** |

## 5. CPU% target / throughput 가설

- tokenizer batch=256 prompt × 8192 token 영역 영역 single thread ~500ms, 8 thread ~70ms
- 영역 measurement window 영역 throughput 영역 영향 영역 작음 (tokenize 영역 wall ↓ 영역 init 영역 차지하는 영역 ↓)
- CPU busy 영역 영역 추가 영역 +5-10% (sustained 영역 아님 — burst)

## 6. Risk

| 위험 | 완화 |
|---|---|
| GIL 영역 contention | tokenize 영역 영역 Rust 안 영역 — GIL release |
| stop-string matching 영역 영역 small overhead | per-stream lock-free |
| 본 lever 영역 영역 throughput 영역 거의 영역 영향 안 줄 가능 | SUB_054/060 와 결합 영역 의미 |

## 7. Dependencies

- HF tokenizers 영역 ≥ 0.15 (encode_batch + rayon)
- Python concurrent.futures

## 8. Acceptance criteria

- [ ] tokenize wall ≤ 100ms (batch 256 × 8192)
- [ ] CPU busy 영역 +5% (burst)
- [ ] throughput 영역 영향 ≥ +0% (regression 영역 없음)
