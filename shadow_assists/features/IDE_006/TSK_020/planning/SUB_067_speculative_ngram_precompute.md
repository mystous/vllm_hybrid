# SUB_067 — speculative ngram precompute (C1 inter-step barrier 해소)

> **parent**: TSK_020 / bottleneck-driven C1
> **status**: 진행 (2026-05-24)
> **effort**: 2-3 일 (scaffold + smoke + measurement)
> **bottleneck**: 영역 GPU forward 70-90ms 동안 CPU idle (5.51%) — next step ngram 영역 wait

## Mechanism

현 pipeline:
```
step N: ngram (10-30ms) → GPU forward (70-90ms) → result → ┐
step N+1:                                                    │ ngram (wait) → forward → ...
```

speculative precompute:
```
step N: ngram → GPU forward (70-90ms)
                 ↓ background CPU thread (overlap)
              [precompute ngram for likely-accepted prefix of N+1]
              ↓
step N+1: 영역 cache hit → forward 즉시 launch (CPU stall 영역)
```

핵심 가정: step N 의 accept 영역 영역 영역 (top-1) 영역 영역 영역. accepted token = ngram chain 의 prefix 영역 영역 영역 영역 token. 영역 영역 영역 영역 영역 hit 가능성 ↑.

## Implementation surface

| 파일 | 변경 |
|---|---|
| `vllm/v1/spec_decode/ngram_proposer.py` | speculative cache 추가, propose 영역 cache hit check |
| `vllm/v1/worker/gpu_model_runner.py` | forward 시작 시 background thread launch (`concurrent.futures.ThreadPoolExecutor`) |
| `vllm/v1/spec_decode/speculative_cache.py` (신규) | dict-based cache (request_id → (last_tokens, ngram_chain)) |
| env | `VLLM_NGRAM_PRECOMPUTE=1` (default 0) |

## Implementation detail

```python
# in gpu_model_runner forward:
if VLLM_NGRAM_PRECOMPUTE:
    # launch background thread to compute next-step ngram (speculative)
    future = self._precompute_executor.submit(
        self._precompute_ngrams,
        current_token_ids,
        accepted_chains_top1,
    )
# ... GPU forward (70-90ms) ...
# next step:
if hit := self._spec_cache.lookup(current_state):
    draft = hit
else:
    draft = standard_lookup()
```

## Measurement plan (500p × 8192 × 8192)

| config | 가설 |
|---|---|
| baseline | 10,956 tps (SUB_047) |
| precompute (cache hit ratio ?) | 영역 cache hit 시 step time -10-20% → +5-15% throughput |

cache hit ratio 측정 영역 — 영역 ngram pattern 영역 영역 영역 hit rate 다름.

## Risk

- background thread 영역 GIL 영역 영역 영역 영역 영역 (numba 영역 GIL release 영역 영역)
- cache miss 시 영역 work 영역 영역 (영역 fallback)
- thread overhead 영역 step time 영역 영역 영역 영역 영역 영역
- 영역 영역 default OFF 유지
