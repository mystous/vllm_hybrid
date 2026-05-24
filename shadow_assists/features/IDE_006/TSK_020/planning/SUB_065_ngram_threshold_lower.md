# SUB_065 — ngram threshold lower (B-4 self-barrier 제거)

> **parent**: TSK_020 / bottleneck-driven B-4
> **status**: 진행 (2026-05-24)
> **effort**: 1 시간 (1 줄 + 5-way sweep)
> **bottleneck**: small decode-only batch (total tokens < 8192) 영역 single-thread fallback → ngram lookup 영역 self-imposed barrier

## Mechanism

`ngram_proposer.py` 의 `num_tokens_threshold=8192`:
```python
if total_tokens >= self.num_tokens_threshold:
    set_num_threads(final_num_threads)  # multi-thread (cap=8)
else:
    set_num_threads(1)  # single thread fallback
```

decode-only step (256 seqs × 7 spec = 1,792 tokens) 영역 8192 영역 영역 영역 → single-thread. multi-thread 영역 작동 안 함. ngram lookup 영역 single thread time → CPU stall.

## Implementation

```python
self.num_tokens_threshold = int(
    os.environ.get("VLLM_NGRAM_THRESHOLD", "8192")
)
```

env-tunable. default 8192 = no break.

## Measurement plan (500p × 8192 × 8192)

| threshold | 효과 가설 |
|---:|---|
| 8192 (baseline) | 현 SUB_047 = 10,956 tps |
| 4096 | half batch 영역 multi-thread enable |
| 2048 | small batch 영역 영역 multi-thread |
| 1024 | 영역 영역 영역 multi-thread |
| 0 | 무조건 multi-thread (overhead 영역 영역) |

5-way × ~8 분 = ~40 분.
