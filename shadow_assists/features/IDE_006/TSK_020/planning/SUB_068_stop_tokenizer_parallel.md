# SUB_068 — stop-string + tokenizer parallel (D2/D4 idle CPU)

> **parent**: TSK_020 / bottleneck-driven D2/D4 (SUB_059 영역 확장)
> **status**: 진행 (2026-05-24)
> **effort**: 1 일
> **bottleneck**: output processing 영역 stop-string match 영역 single-thread + tokenizer single-thread

## Mechanism

### D2: tokenizer parallel
HF Tokenizers (Rust) 영역 `encode_batch` 영역 rayon multi-thread 영역. vLLM 영역 영역 single-thread 영역. prefill 영역 batch tokenize 영역 영역 →
```python
tokenizer.encode_batch(prompts, ...)  # auto rayon parallel
os.environ["RAYON_NUM_THREADS"] = "8"  # explicit thread count
```

### D4: stop-string parallel per-stream
영역 generated token 영역 stop string check (output processing). 영역 stream 별 독립. forward 동안 background thread 영역 영역 영역 영역.

## Implementation surface

| 파일 | 변경 |
|---|---|
| `vllm/transformers_utils/tokenizer.py` | `encode_batch` 사용 + RAYON_NUM_THREADS=8 |
| `vllm/v1/engine/output_processor.py` | stop-string match 영역 ThreadPoolExecutor |
| env | `VLLM_TOKENIZER_PARALLEL=1`, `VLLM_STOP_PARALLEL=1` |

## Measurement plan (500p × 8192 × 8192)

| config | 가설 |
|---|---|
| baseline | 10,956 tps (SUB_047) |
| tokenizer parallel | prefill 영역 영역 — total wall 영역 영향 작음 (~1%) |
| stop parallel | output processing 영역 영역 — 영역 작음 |
| 결합 | +1-3% 가능 |

## Risk

- tokenizer parallel 영역 GIL release 영역 (Rust 영역 영역)
- stop parallel 영역 thread overhead 영역 작은 stream count 영역 손해 가능
- 회귀 시 default OFF 유지
