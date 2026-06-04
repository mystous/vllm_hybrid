# SUB_201 A2 — Batched DMA microbench results

## Context

Per-call 8 KB push from `verify_dma_wrapper.json` clocked **0.79 GB/s**
(9.6 μs/call) — at Llama-70B (N=80 layers) this means **768 μs/block**, which
is *worse* than the matrix-measured memcpy baseline (336 μs/evt). This is the
A2 lever's regression risk.

Mitigation: amortize per-call CUDA driver overhead by submitting all 80
transfers as a single batch. Three variants implemented in
`src/pinned_pool.cpp`:

| variant | function | how |
| --- | --- | --- |
| A | `push_batch_async`            | C-side for-loop of `cudaMemcpyAsync` + 1 event |
| B | `push_batch_async_native`     | `cudaMemcpyBatchAsync` (CUDA 12.4+)            |
| C | `push_batch_async_staged`     | host-side pack into pinned staging + 1 `cudaMemcpyAsync` (640 KiB) |

## Measurement setup

- GPU: B200 (GPU 0)
- CUDA: 12.8
- Shape: Llama-70B / TP=8 → N=80 layers, per_layer=8 192 B,
  total=655 360 B (640 KiB)
- Iters: 300 (+ 30 warmup); p50 is the primary metric
  (other processes share the host so mean/p99 are noisy)

## Results (300 iters, B200, GPU 0)

| variant         | mean μs | p50 μs | p99 μs | GB/s (p50) | speedup vs naive |
| --------------- | ------: | -----: | -----: | ---------: | ---------------: |
| baseline_loop   | 2504.12 | 821.62 |17029.05|       0.74 |            1.00× |
| batch_loop_A    |  697.94 | 435.34 | 2579.96|       1.40 |            1.89× |
| batch_native_B  |  378.25 | 197.46 | 2400.46|       3.09 |            4.16× |
| **batch_staged_C** | **82.57** | **80.02** | **101.74** | **7.63** | **10.27×** |

Raw JSON: `verify_batch_dma.json`.

## Verdict

- **Regression risk resolved**: best variant (C, staged) hits **80 μs / 7.63 GB/s**,
  which is **4.2× faster** than the matrix baseline budget of 336 μs/block
  (headroom = +256 μs). Net positive guaranteed for the Llama-70B per-block
  transfer.
- Even the simplest C-side batching (A) brings 1.89× speedup and is enough
  to neutralize the per-call regression at p50 (435 μs vs 336 μs is still
  over-budget — A alone is *not* sufficient).
- `cudaMemcpyBatchAsync` (B) gives a real 2.2× win over A (197 μs vs
  435 μs p50) but is still over the matrix budget. B is useful when the per-layer
  destination layout is non-contiguous and we cannot stage.
- **C is the path for engine wiring**: as long as the destination is one
  contiguous device slab (typical KV-block layout when packed per-block), the
  host-side memcpy + 1 cudaMemcpyAsync wins by a wide margin. The host-side
  pack cost (~30 μs for 640 KiB across 80 sources) is dominated by the
  amortized H2D bandwidth at 640 KiB.

## Recommendation

A2 KV tier engine wiring (next dev step) should:
1. Lay out per-block KV as one contiguous host-pinned + device-contiguous
   slab (per-block, all-layers packed). Use **variant C** for push.
2. Use **variant B** (`cudaMemcpyBatchAsync`) only for the (rarer) case
   where layer slices live in non-contiguous device memory.
3. Avoid the per-call API path on the hot block-push path; reserve it for
   single-buffer one-shots.
