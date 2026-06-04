# SUB_201 A2 — Engine wiring DESIGN (KVDramTier hot-path integration)

> Scope: this doc covers the **second** PoC step — wiring
> `KVDramTier` + `pinned_pool option C (staged batch)` into vLLM v1
> engine **hot-paths** with an env-flag gated, default-OFF, regression-
> safe hook design.
>
> Status: design + minimal hook patch + smoke test only. **No e2e
> vLLM boot** in this worktree (venv constraint per SUB_201 §5 PoC
> guardrail). Follow-up: e2e boot + microbench on prod TP=4 H100×8.

## 0. Cross-reference

| artifact | role |
| --- | --- |
| `DESIGN.md` (this doc) | engine wiring (2nd step) |
| `BATCH_MEASUREMENTS.md` | option-C 80 μs / 7.63 GB/s justification |
| `verify_batch_dma.py` / `.json` | reproducible measurement |
| `pinned_pool_wrapper.py` | ctypes ABI (sole entry point) |
| `vllm/v1/core/kv_dram_tiering.py` | KVDramTier prototype |
| `shadow_assists/features/IDE_022_agsd_realistic_eval/SUB_201_cpu_host_path_bottleneck/profile/LEVER_AUDIT.md` | A2 lever audit |

## 1. Target geometry: Llama-70B / TP=4

The hot-path numbers below are computed for the **prod machine**
(Sapphire Rapids + H100×8, TP=4 per replica).

```
model:    Llama-70B
layers:   80 attention
heads:    8 KV heads (GQA)
head_size:128
dtype:    bf16 (2 B)
TP:       4   → per-rank KV heads = 8 / 4 = 2
block:    16 tokens (vLLM default)
```

Per-block, per-layer (one rank):
```
block_size × num_kv_heads_per_rank × head_size × kv_factor × dtype
= 16 × 2 × 128 × 2 × 2
= 16 384 B = 16 KiB
```

Per-block, all 80 layers (one rank):
```
80 × 16 KiB = 1 280 KiB ≈ 1.25 MiB
```

```mermaid
flowchart TB
    classDef hot fill:#fdd,stroke:#a00
    classDef cold fill:#dfd,stroke:#080
    classDef tier fill:#ffd,stroke:#880

    A["Llama-70B/TP=4 KV block<br/>16 tokens × 2 heads × 128 dim × bf16 × (K,V)<br/>= 16 KiB / layer"]:::hot
    B["80 layers stacked (per-block)<br/>= 1 280 KiB / block / rank"]:::hot
    A --> B

    B --> C{cold-eligible<br/>(cached, ref_cnt=0)}
    C -- yes --> D["KVDramTier.evict_to_dram_batch<br/>(option C staged)"]:::tier
    C -- no  --> E["regular FreeKVCacheBlockQueue"]:::cold

    D --> F["pinned DRAM (pool)<br/>≈80 μs / 1.28 MiB block"]:::tier
    F -.fetch on cache hit.-> G["fetch_to_gpu_batch<br/>(option C reverse)"]:::tier
    G --> H["next allocate_slots / touch"]:::hot
```

Cf. dev box (RTX 3090 + i9) → TP=1 single GPU; per-block = 80 × 64
KiB = 5 MiB. The hook surface is identical; bandwidth ceiling
differs (~25 GB/s host pinned vs 60 GB/s on H100 PCIe Gen5).

## 2. Hook surface (vllm/v1/* changes)

The full design uses three insertion points. **This PoC patches only
(P1)** for risk minimization; P2 + P3 are scaffolded by leaving
explicit TODO markers near the call sites.

```
┌────────────────────────────────────────────────────────────────┐
│ P0 (init / one-time)                                            │
│   KVCacheManager.__init__                                       │
│     └─ if VLLM_KV_TIERING_DRAM: build PinnedPool + KVDramTier  │
│        sized off num_gpu_blocks × per_block_nbytes              │
├────────────────────────────────────────────────────────────────┤
│ P1 (HOT — evict / free)                                         │
│   BlockPool.free_blocks(blocks_list)                            │
│     ├─ existing: ref_cnt--, append_n to LRU queue              │
│     └─ NEW: if cached & ref_cnt==0 & gpu_ptr_known →           │
│              KVDramTier.evict_to_dram_batch(block_ids, ptrs)    │
├────────────────────────────────────────────────────────────────┤
│ P2 (HOT — alloc / fetch)                                        │
│   BlockPool.get_new_blocks(num_blocks)                          │
│     ├─ existing: popleft_n + _maybe_evict_cached_block          │
│     └─ NEW: if KVDramTier.is_tiered(block_id) →                │
│              schedule fetch_to_gpu_batch BEFORE first attn read │
├────────────────────────────────────────────────────────────────┤
│ P3 (HOT — touch / cache hit)                                    │
│   BlockPool.touch(blocks)                                       │
│     └─ NEW: same condition → fetch_to_gpu_batch                 │
└────────────────────────────────────────────────────────────────┘
```

### 2.1 GPU pointer lookup (cross-cutting)

`KVDramTier` needs the **GPU pointer** for each block. In vllm v1,
that lives in `GPUModelRunner.kv_caches` — a list of per-layer
tensors of shape `[num_blocks, ...]`. So for `block_id b`:

```python
# per layer L, per rank:
slice_l = self.kv_caches[L][b]                # torch.Tensor view
ptr_l   = slice_l.data_ptr()                  # int
```

There is no API today for `BlockPool` → `GPUModelRunner.kv_caches`.
Options:

1. **Injection at init**: Worker side, after
   `initialize_kv_cache_tensors`, register `kv_caches` into a
   `KVDramTierBinding` object that the BlockPool can dereference.
   *Chosen.*
2. **Callback** registered on the singleton.

The binding holds a `list[list[int]]` (per-block per-layer ptrs)
computed once. Hot-path lookup is O(num_layers) without any tensor
ops. For option-C batching we also pre-pack a per-block size array
(constant 16 KiB × 80) and a per-block staging slot (lazy).

### 2.2 Cold-block selection policy

This PoC uses the **conservative cached-LRU policy**:

> A block is eligible for DRAM tiering iff at `free_blocks` time
> its `ref_cnt == 0`, `block_hash is not None` (i.e. cached) and
> `KVDramTier.dram_bytes_in_use + per_block_nbytes ≤ max_dram_bytes`.

Rationale: at `free_blocks` the GPU has finished writing this
block (attention kernel completed in the issuing step); the block is
about to enter the LRU queue for potential reuse, and any later
allocate request will overwrite it. By tiering only cached blocks
we keep the prefix-cache hit path correct: when the cached hash
re-attaches via `touch` / `get_cached_block`, P2/P3 issues a
`fetch_to_gpu_batch` before the block is read.

Non-cached free → still goes to LRU directly (no tier interaction).

### 2.3 Stream / sync semantics

Two streams: `S_compute` (vLLM main) and `S_tier` (KVDramTier-owned).

| operation | stream | barrier | reason |
| --- | --- | --- | --- |
| evict pull (GPU→DRAM) | `S_tier` | `cudaEventRecord(ev_done, S_compute) → cudaStreamWaitEvent(S_tier, ev_done)` | data-write completion on compute |
| GPU block re-use | next `allocate_slots` | **MUST** `wait_evict(block_id)` OR `wait=True` in evict | else data race |
| fetch push (DRAM→GPU) | **`S_compute`** for PoC | none (CUDA stream order serializes attention) | PoC trades overlap for correctness |
| fetch push (future) | `S_tier` | `cudaStreamWaitEvent(S_compute, ev_push)` before attn | restores overlap |

**PoC choice**: `wait=True` on evict, push on `S_compute`. This
*guarantees correctness without scheduler changes* but loses the
overlap that motivates A2. The performance win is recovered in the
follow-up step by:
1. Adding `pending_evict_block_ids: set[int]` to BlockPool;
   `get_new_blocks` does `wait_evict(b)` only if `b` is in this set.
2. Pushing fetch on `S_tier` + event-recording for attn-side wait.

### 2.4 Regression guard (env flag OFF)

When `VLLM_KV_TIERING_DRAM` is unset or `0`:
- `KVCacheManager.__init__`: skips PinnedPool / KVDramTier creation,
  sets `self._kv_dram_tier = None`.
- `BlockPool` carries an `Optional[KVDramTier]` reference (default
  `None`); the hot-path branch is one `is None` check. On a 80-layer
  Llama-70B step at 200 TPS this is < 10 ns / call — well under
  noise.
- The smoke test exercises both branches.

## 3. Batched DMA path (option C wiring)

`evict_to_dram_batch(block_ids, layer_ptrs_per_block)` builds, for
each block:

```
host_dest    = pool.alloc(1280 KiB)   # one slab per block
staging_dst  = host_dest              # pinned staging == final
device_srcs  = [layer_ptrs_per_block[0..79]]  # 80 GPU ptrs
sizes        = [16 KiB] * 80
ev = pool.pull_batch_async(           # NOTE: pull variant; PoC currently
       device_srcs, host_dest, sizes, # only push_batch_async_staged exists;
       stream=S_tier, mode="staged")  # the pull-staged counterpart MUST
                                      # be added in src/pinned_pool.cpp
                                      # → tracked as obstacle below.
```

For **push (fetch)** we already have `push_batch_async_staged`
measured at 80 μs / 1 280 KiB. The pull-staged twin is symmetric:
80 cudaMemcpyAsync(device→pinned) packed into a single 1.28 MiB H2D
operation. **Engineering todo (next dev step)**: add
`pinned_pool_pull_batch_async_staged` to `src/pinned_pool.cpp`. For
THIS PoC we use the non-batched `pull_async` per layer (slow but
correct), and **gate full speedup behind a `# TODO(pull-staged)`**.

## 4. Patches in this PoC (minimal)

| file | LOC delta | purpose |
| --- | --- | --- |
| `vllm/v1/core/kv_dram_tiering.py` | +~60 / 0 | add `KVDramTierBinding` + `evict_to_dram_batch` / `fetch_to_gpu_batch` stubs + `try_attach_to_block_pool` helper |
| `vllm/v1/core/block_pool.py` | +~25 / 0 | optional `KVDramTier` reference + hook in `free_blocks` (P1) + hook in `get_new_blocks` (P2) — both gated by `self._kv_dram_tier is not None` |
| `vllm/v1/core/kv_cache_manager.py` | +~20 / 0 | `__init__` builds KVDramTier if env on; passes to BlockPool |
| `tests/v1/spec_decode/test_kv_dram_tiering.py` | +~120 (new file) | smoke test: import, flag toggle, no-op path, hook firing |

**No edits** to `gpu_model_runner.py` in this step — the
`KVDramTierBinding` is created by the **caller of `initialize_kv_cache`**
(worker boot). For PoC we expose a `bind_kv_caches(kv_caches)`
classmethod on `KVDramTier` that the worker can call later. Today
the smoke test calls it from a fake worker.

## 5. Failure modes / acceptance gate

| failure | detection | mitigation |
| --- | --- | --- |
| env flag mistyped | `_is_enabled` returns False → no tier | strict `("1","true","on","yes")` allowlist |
| pool init OOM (DRAM) | `PinnedPool.__init__` raises | manager catches, logs warning, falls back to non-tiered |
| `libpinned_pool.so` missing | ctypes `CDLL` raises | manager catches → fallback |
| tier full | `evict_to_dram` returns False | `_n_evict_skipped_full` counter; block goes to LRU normally |
| race on GPU block re-use | only triggered with `wait=False` | PoC forces `wait=True` |

Acceptance for this PoC step (no e2e boot):
1. `python -c "from vllm.v1.core import kv_dram_tiering"` → OK
2. `python -m py_compile` on all touched files → OK
3. Smoke test: flag OFF path makes zero KVDramTier calls
4. Smoke test: flag ON path with fake pool sees 1 evict + 1 fetch per
   simulated block

## 6. Next dev step (post-PoC)

1. Add `pinned_pool_pull_batch_async_staged` C symbol + Python binding.
2. e2e boot in prod env (`VLLM_USE_PRECOMPILED=1`) with TP=4 H100×4,
   200 prompt benchmark — measure tps delta vs OFF.
3. Two-stream + event for fetch (recover overlap).
4. Replace conservative cached-LRU policy with sequence-aware cold
   window (policy (b) from BATCH_MEASUREMENTS DESIGN §4.4).
5. Add registry entry in `shadow_assists/id_registry.md` (`SUB_201`
   sub-task or new SUB) once the wiring graduates from PoC.
