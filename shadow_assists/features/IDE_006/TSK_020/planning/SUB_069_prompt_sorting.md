# SUB_069 — prompt sorting by length (F1 batch density ↑)

> **parent**: TSK_020 / bottleneck-driven F1
> **status**: 진행 (2026-05-24)
> **effort**: 0.5 일 (wrapper 변경 + 측정)
> **bottleneck**: prompt length variance 영역 영역 batch padding waste + ngram cache hit 영역

## Mechanism

영역 prompt length 영역 다양 → continuous batching 영역 short prompt 영역 finish 후 영역 prompt 추가 → batch density 영역 영역 영역.

영역 length 영역 sorted 영역 영역 영역 영역 prompt 묶음 → continuous batching 영역 batch density 영역 + ngram cache hit 영역 (similar workload 영역).

## Implementation surface

wrapper 영역 변경 (vLLM 무수정):

```python
# in run_spec_decode.py
prompts = _build_prompts(...)
# SUB_069: sort by tokenized length DESC
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL_PATH)
prompts_with_len = [(p, len(tok.encode(p))) for p in prompts]
prompts_with_len.sort(key=lambda x: -x[1])  # DESC
prompts = [p for p, _ in prompts_with_len]
```

env `VLLM_PROMPT_SORT=1` (default 0).

## Measurement plan (500p × 8192 × 8192)

| config | 가설 |
|---|---|
| baseline (unsorted) | 10,956 tps (SUB_047) |
| sorted DESC | +1-3% (batch density ↑) |
| sorted ASC | 영역 가능 (영역 영역 묶음) |

## Risk

- sorted DESC 영역 영역 prompt 영역 영역 영역 → scheduler 영역 batch 영역 영역 영역 가능
- 영역 prompt 영역 finish 영역 batch 영역 영역 영역 영역 영역 영역 → padding waste
- 영역 영역 small (+1-3%)
