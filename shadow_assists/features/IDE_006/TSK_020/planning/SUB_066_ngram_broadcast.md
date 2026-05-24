# SUB_066 — ngram broadcast from rank 0 (B-2 duplicate work 제거)

> **parent**: TSK_020 / bottleneck-driven B-2
> **status**: 진행 (2026-05-24)
> **effort**: 1-2 일 (scaffold + measurement)
> **bottleneck**: 8 TP rank 영역 동일 ngram lookup 영역 7x duplicate work

## Mechanism

vLLM TP=8: 영역 worker (rank 0~7) 가 자기 batch 의 ngram lookup 수행. 모든 rank 가 동일 input (prompt token_ids_cpu) → 동일 결과. 7x duplicate.

대안: rank 0 만 lookup → NCCL broadcast 또는 shared memory 영역 영역 rank 영역.

## Implementation surface

| 파일 | 변경 |
|---|---|
| `vllm/v1/spec_decode/ngram_proposer.py` | rank check + broadcast logic 추가 |
| `vllm/v1/worker/gpu_worker.py` | rank info access 영역 |
| env | `VLLM_NGRAM_BROADCAST=1` (default 0 = current per-rank) |

## Implementation detail

```python
# in batch_propose:
if VLLM_NGRAM_BROADCAST and rank == 0:
    # compute as usual
    batch_propose_numba(...)
elif VLLM_NGRAM_BROADCAST:
    # wait + receive from rank 0 (via dist.broadcast)
    dist.broadcast(self.valid_ngram_draft, src=0)
    dist.broadcast(self.valid_ngram_num_drafts, src=0)
else:
    # current per-rank behavior
    batch_propose_numba(...)
```

## Measurement plan (500p × 8192 × 8192)

| config | 가설 |
|---|---|
| baseline (per-rank) | 10,956 tps (SUB_047) |
| broadcast (rank 0 only) | 영역 영역 영역 + NCCL broadcast overhead 영역 |

가설: ngram time 자체 영역 작아서 영역 영역 영역 작음 (+1-2%). 단 NCCL barrier overhead 가 broadcast time 영역 영역 영역 영역 회귀 가능.

## Risk

- NCCL broadcast 영역 small payload 영역 latency overhead 큼 (~50-100 μs) → 영역 step 영역 영역 영역 영역
- rank 0 영역 ngram time 영역 영역 영역 영역 → 영역 영역 영역 영역 (TP 영역 영역 영역 영역)
- 회귀 시 default OFF 유지
