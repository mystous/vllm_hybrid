# SUB_201 A2 — Cold KV → DRAM tiering PoC (design + analysis)

> Status: PoC artifacts only. **No engine smoke / boot test** (worktree
> venv constraint per the SUB_201 §5 plan).

## 1. Motivation (recap of SUB_201 §5)

Llama-70B suffix nsys:
- `cudaMemcpyAsync` = **80.4 %** of GPU API time (≈ 65 s in 60 s wall)
- 3 236 evts / s, **avg 336 μs / evt**
- Dominated by inter-GPU H2D / D2H transfers under TP=8
- Matrix throughput: 10 400 tps / wall 96 s / GPU util 83.4 %
  (≈ 16 % gap; part of it is memcpy synchronization)

Hypothesis: cold (old-token) KV blocks parked on HBM contribute to the
suffix attention only as read-only context. Spilling them to pinned DRAM
recovers HBM working set for *active* KV and (more importantly) lets the
cold-KV traffic overlap with the next forward step on a dedicated stream
instead of fighting compute on the default stream.

## 2. PoC scope (this run)

Done in this PoC:
- (a) Built `libpinned_pool.so` from IDE_017 source (g++ + CUDA + numa). Lib re-runnable.
- (b) Re-verified DMA path through a Python wrapper that engine code will use:

  | size  | push p50 | pull p50 | BW (push) |
  |-------|----------|----------|-----------|
  | 64 MB | 1217 μs  | 1223 μs  | **51.34 GB/s** |
  | 16 MB |  313 μs  |  314 μs  | 49.86 GB/s |
  |  1 MB |   30 μs  |   30 μs  | 32.28 GB/s |
  | 640 KiB (Llama-70B/TP=8, all 80 layers) | 22.7 μs | 23.3 μs | 26.83 GB/s |
  |  8 KiB (per-layer slice) |  9.6 μs |  11.4 μs |  0.79 GB/s |

  Matches SUB_166 / SUB_176 baseline. CRC round-trip OK from SUB_176.

- (c) Designed `KVDramTier` (Python prototype, `vllm/v1/core/kv_dram_tiering.py`)
  with evict / fetch / wait / drop / stats API. `py_compile` clean.

- (d) Designed `PinnedPool` Python wrapper
  (`SUB_201_A2_kvtier_poc/pinned_pool_wrapper.py`) — the only ctypes
  surface so the rest of the codebase stays clean.

- (e) Identified all hook sites in vllm core (see §4).

**Explicitly out of scope** (per SUB_201 PoC limits):
- No engine boot / smoke test.
- No actual patch into `block_pool.py` / `kv_cache_manager.py`.
- No real cold-block selection heuristic — leaving it as a policy stub.

## 3. Microbench finding that drives the design

The **per-layer 8 KiB slice runs at 0.79 GB/s** — overhead-bound.
Llama-70B has 80 attention layers; with vLLM's per-layer KV cache
tensor layout (`kv_caches: list[torch.Tensor]`, one tensor per layer),
spilling *one block* means 80 separate cudaMemcpyAsync calls if we
treat per-layer chunks individually.

**Implication**: KVDramTier MUST use `push_batch_async` /
`pull_batch_async` to bundle the 80 per-layer slices into a single
event for one block. Otherwise the 80 × 9.6 μs = 768 μs / block cost
*destroys* the win we hoped to gain from the 336 μs/evt baseline.

The all-layers concatenated size (640 KiB) sustains 26.8 GB/s which
is still a 4× DMA wall but at least not API-overhead-bound. Even better
would be batching K and V slices per layer separately (160 entries) and
ideally using a single contiguous staging buffer per block — but that
requires a layout change in `GPUModelRunner.initialize_kv_cache_tensors`.

## 4. vllm KV manager hook points

### 4.1 Block alloc / free path

| event | file | line | function |
|-------|------|------|----------|
| KVCacheBlock dataclass | `vllm/v1/core/kv_cache_utils.py` | 110 | `class KVCacheBlock` |
| Doubly-linked LRU free queue | `vllm/v1/core/kv_cache_utils.py` | 158 | `class FreeKVCacheBlockQueue` |
| New-block alloc (pop LRU front) | `vllm/v1/core/block_pool.py` | 322 | `BlockPool.get_new_blocks` |
| Evict cached block on reuse | `vllm/v1/core/block_pool.py` | 354 | `BlockPool._maybe_evict_cached_block` |
| Touch (cache hit re-attach) | `vllm/v1/core/block_pool.py` | 391 | `BlockPool.touch` |
| Free / append to queue tail | `vllm/v1/core/block_pool.py` | 408 | `BlockPool.free_blocks` |
| Force-evict by id | `vllm/v1/core/block_pool.py` | 424 | `BlockPool.evict_blocks` |
| Request-level free | `vllm/v1/core/kv_cache_manager.py` | 438 | `KVCacheManager.free` |
| Request-level alloc | `vllm/v1/core/kv_cache_manager.py` | 257 | `KVCacheManager.allocate_slots` |

### 4.2 Tensor allocation (where GPU pointer lives)

| event | file | line | function |
|-------|------|------|----------|
| Raw int8 buffer per layer | `vllm/v1/worker/gpu_model_runner.py` | 8087 | `_allocate_kv_cache_tensors` |
| Reshape to attention shape | `vllm/v1/worker/gpu_model_runner.py` | 8128 | `_reshape_kv_cache_tensors` |
| Public init | `vllm/v1/worker/gpu_model_runner.py` | 8266 | `initialize_kv_cache_tensors` |

So a `block_id` from BlockPool maps to a **slice** of every per-layer
tensor in `GPUModelRunner.kv_caches` at offset `block_id * page_size`.

### 4.3 Current eviction policy

From `FreeKVCacheBlockQueue` docstring + `BlockPool.free_blocks`:
- Effectively **LRU** — when a request is freed, its tail blocks are
  pushed to the queue tail first (because `free_blocks` is called with
  `reversed` lists upstream); the next `popleft_n` pulls the
  least-recently-freed block.
- When `enable_caching=True`, the queue stays populated with cached
  blocks (ref_cnt=0 but still hashed), and `_maybe_evict_cached_block`
  reclaims them on reuse.
- There is **no notion of cold vs warm within a live request** — cold
  prefix blocks of a still-running request stay pinned in HBM.

### 4.4 Cold-block identification options (proposed)

| ID | Policy | Pros | Cons |
|----|--------|------|------|
| (a) | LRU-evicted side-channel (intercept `BlockPool.free_blocks`) | zero scheduler changes | only fires when request ends → too late for active suffix |
| (b) | Per-sequence: for each running request, mark blocks `[0 : n_blocks - hot_window]` as cold every `K` steps | tracks active sequences | needs scheduler hook, race with allocate_slots |
| (c) | Prefix-cache hit driven (cached but ref_cnt==0) | already evictable | only saves cached path; misses uncached cold blocks |
| (d) | (b) + force only on `BlockPool.get_num_free_blocks() < threshold` | conservative | won't fire under low-pressure benchmarks where 70B fits |

For PoC integration the cheapest experiment is (a) + (c): hook
`BlockPool.free_blocks` to spill ref_cnt=0 cached blocks before the
queue append. This preserves the prefix-cache hit path with cold-fetch.

## 5. KV consistency analysis (barriers required)

Let `S_compute` be the model-runner's compute stream and `S_tier` the
KVDramTier dedicated stream.

### 5.1 Evict (S_compute → S_tier → DRAM)

```
write KV by attention kernel        on S_compute
... last reference to block in main model
[free_blocks called]
KVDramTier.evict_to_dram:
    cudaEventRecord(ev_done, S_compute)        # NEEDED — not in PoC
    cudaStreamWaitEvent(S_tier, ev_done)       # NEEDED — not in PoC
    cudaMemcpyAsync(... , S_tier)              # done
    cudaEventRecord(ev_pull, S_tier)
[block returned to FreeKVCacheBlockQueue]
next request: allocate_slots returns this block_id
    writes via attention kernel on S_compute
    !!! data race if ev_pull not yet resolved !!!
```

**Required barrier**: before recycling the GPU block, the next
`get_new_blocks` path must call `KVDramTier.wait_evict(block_id)` OR
the eviction must be synchronous (`wait=True`).

PoC choice: `wait=True` in `evict_to_dram` for correctness. Async
overlap (`wait=False` + `wait_evict` on alloc) is the **next dev
step** and requires either:
1. A `pending_evict_block_ids: set` checked in `get_new_blocks`, OR
2. Holding back the block from `free_block_queue` until pull resolves.

(2) is cleaner — block sits in a `pending_drain` list and only joins
the LRU queue once the event is reaped. Probably needs a poller thread
or a check at each scheduler tick.

### 5.2 Fetch (DRAM → S_tier → S_compute)

```
allocate_slots returns block_id reused for new request
[scheduler detects this block_id was tiered → fetch]
KVDramTier.fetch_to_gpu:
    cudaMemcpyAsync(host_ptr -> gpu_block_ptr, S_tier)
    cudaEventRecord(ev_push, S_tier)
[next forward step issues attention on S_compute]
    cudaStreamWaitEvent(S_compute, ev_push)    # NEEDED — not in PoC
    attention reads gpu_block_ptr              # safe
```

Alternative (PoC currently picks this): issue `push_async` on
`S_compute` directly — CUDA stream semantics serialize push with
subsequent attention on the same stream, no explicit event needed.
**Cost**: kills the overlap with the next-step compute that motivated
A2 in the first place. Trade: correctness now, overlap later.

The "right" answer is the **two-stream + event** path. The PoC has the
hook (`_TierEntry.pending_ev`) but engine wiring isn't done.

### 5.3 Cache lookup correctness

Tiered blocks must NOT be treated as evicted from the prefix cache.
`_maybe_evict_cached_block` currently calls `pop` on the hash map —
KVDramTier integration must *skip* this pop and instead register the
block as "tiered & still cached", so a future `get_cached_block` hit
finds it and triggers `fetch_to_gpu` before handing to the next
request. This is a non-trivial change to `BlockPool.cached_block_hash_to_block`
semantics.

## 6. Obstacles & next dev step estimate

| # | Obstacle | Effort | Risk |
|---|----------|--------|------|
| 1 | Per-layer slicing — 80 separate copies vs 1 batched event | M | High — without batching, the spill cost (768 μs) exceeds the 336 μs/evt baseline; A2 NEGATIVE if not fixed |
| 2 | Two-stream + event for async evict overlap | M | Medium — needs `pending_drain` list and per-scheduler-tick polling |
| 3 | `cached_block_hash_to_block` semantic extension (cached+tiered) | M | Medium — affects every prefix-cache code path |
| 4 | Cold-block selection policy: who decides which live-request blocks are cold? | L (PoC) – L (production) | Low for (a)+(c), High for sequence-aware (b) |
| 5 | GPU pointer extraction (block_id → slice of `kv_caches[layer]`) | S | Low — straightforward stride math |
| 6 | Multi-TP rank coordination (each rank evicts its own shard) | S | Low — KVDramTier is per-rank singleton |
| 7 | Engine refactor surface: KVDramTier needs reference inside `BlockPool` *and* `GPUModelRunner` (cross-cutting) | M | Medium — feature flag gates everything but touches both Scheduler and Worker sides |

**Net assessment**: A2 is technically buildable on top of IDE_017 with
~2-3 weeks of engineering (per-layer batching + two-stream eviction +
cached-tiered semantics). The single largest risk is **obstacle #1**:
without batching the per-layer DMA, the 80x API overhead per block
turns A2 into a regression. The PoC numbers (640 KiB all-layers @
26.8 GB/s, vs 8 KiB per-layer @ 0.79 GB/s) make this concrete and
quantifiable.

## 7. Files in this PoC

| path | purpose |
|------|---------|
| `vllm/v1/core/kv_dram_tiering.py` | KVDramTier prototype (NEW; not wired) |
| `shadow_assists/features/IDE_017_dma_zero_copy/SUB_201_A2_kvtier_poc/pinned_pool_wrapper.py` | Python ctypes wrapper for `libpinned_pool.so` |
| `shadow_assists/features/IDE_017_dma_zero_copy/SUB_201_A2_kvtier_poc/verify_dma_wrapper.py` | DMA bench using the wrapper |
| `shadow_assists/features/IDE_017_dma_zero_copy/SUB_201_A2_kvtier_poc/verify_dma_wrapper.json` | recorded measurements |
| `shadow_assists/features/IDE_017_dma_zero_copy/SUB_201_A2_kvtier_poc/DESIGN.md` | this doc |
| `shadow_assists/features/IDE_017_dma_zero_copy/build/libpinned_pool.so` | rebuilt shared lib (28 384 B) |
