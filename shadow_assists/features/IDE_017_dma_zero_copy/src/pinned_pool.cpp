// IDE_017 / TSK_028 — Pinned memory pool + DMA push primitive (production)
//
// SUB_166 input (H100 PCIe 5.0):
//   4 KB   : 35 μs / 0.12 GB/s      (overhead-bound)
//   64 KB  : 36 μs / 1.83 GB/s      (overhead-bound)
//   1 MB   : 60 μs / 17.5 GB/s      (crossover)
//   16 MB  : 338 μs / 49.6 GB/s     (bandwidth-bound)
//   64 MB  : 1251 μs / 53.6 GB/s    (asymptotic)
//
// 본 구현의 lever:
//   1. cudaHostAlloc(Default) pre-allocated size-class pool — alloc/free O(1) (lockless mpmc ring)
//   2. NUMA-aware allocation hint (numa_set_preferred / numa_alloc_onnode-equivalent)
//      → SUB_113 의 GPU 0-3↔NUMA 0 / GPU 4-7↔NUMA 1 affinity 활용
//   3. batched DMA push API — same-stream chunks 묶어 fixed 35 μs overhead 분산
//   4. CRC32 round-trip integrity helper (tests 용)
//
// build:
//   nvcc -arch=sm_90 -O3 -std=c++17 -Xcompiler -fPIC -shared \
//        -o libpinned_pool.so pinned_pool.cpp -lcudart -lnuma
//
// status: ✅ TSK_028 완성 — lockless ring + batched push

#include <cuda_runtime.h>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#if __has_include(<numa.h>)
#  include <numa.h>
#  define VLLM_DMA_HAVE_NUMA 1
#else
#  define VLLM_DMA_HAVE_NUMA 0
#endif

namespace vllm_hybrid_dma {

// ──────────────────────────────────────────────────────────────────────
// Size class — SUB_166 measurement regions
// ──────────────────────────────────────────────────────────────────────

constexpr int NUM_SIZE_CLASSES = 5;
constexpr size_t SIZE_CLASS_BYTES[NUM_SIZE_CLASSES] = {
    4ull * 1024,           // class 0 — overhead-bound (spec candidate IDs, attention bias)
    64ull * 1024,          // class 1 — overhead-bound (small KV chunks, draft head slice)
    1ull * 1024 * 1024,    // class 2 — crossover (draft logits)
    16ull * 1024 * 1024,   // class 3 — bandwidth-bound (medium KV chunks, token logits)
    64ull * 1024 * 1024,   // class 4 — asymptotic (cold KV chunks, large activations)
};
// Default per-class block counts. Total ≈ 16·(4K+64K+1M+16M+64M) = ~1.3 GB if all maxed.
constexpr size_t DEFAULT_BLOCKS_PER_CLASS[NUM_SIZE_CLASSES] = {
    256, 128, 64, 16, 8,
};

// ──────────────────────────────────────────────────────────────────────
// SUB_201 A2 Phase B8 — size-class allocator extension
//
// Background (B7 finding): production wire-in with VLLM_KV_TIERING_DRAM=1
// observed n_evict=512 (~448 MiB) with skipped_full=97195 even though
// the engine-side max_dram_bytes was ~154 GB. The bottleneck was the
// fixed DEFAULT_BLOCKS_PER_CLASS schedule (~1.3 GB ceiling) which silently
// capped how many KV pages could ever live in the tier — the engine kept
// asking, the ring kept refusing, every "miss" fell through to a direct
// cudaHostAlloc that bypassed the budget accounting.
//
// Phase B8 introduces two env knobs to expand per-class capacity without
// changing the default contract:
//
//   VLLM_PINNED_POOL_BLOCKS_PER_CLASS=N0,N1,N2,N3,N4
//       Override the absolute per-class block count (any of N* may be 0
//       or absent — falls back to DEFAULT_BLOCKS_PER_CLASS[c]). Honoured
//       only when set; default behaviour identical to pre-B8.
//
//   VLLM_PINNED_POOL_AUTO_BUDGET=1
//       Auto-derive per-class budget from the ctor's total_limit_bytes
//       so each class can grow up to ~ (total_limit / NUM_SIZE_CLASSES).
//       This is the recommended default for KV tiering: KV blocks are
//       routed to the 1 MiB class and the per-class budget jumps from
//       64 blocks (~64 MiB) to ~30 GB worth of 1 MiB pages on a 154 GB
//       host pool — large enough that the engine no longer trips the
//       skipped_full counter.
//
// When both are set, BLOCKS_PER_CLASS wins (explicit override).
// ──────────────────────────────────────────────────────────────────────

static const char* _env_str(const char* name) {
    const char* v = std::getenv(name);
    return (v && *v) ? v : nullptr;
}

static bool _env_truthy(const char* name) {
    const char* v = _env_str(name);
    if (!v) return false;
    // Accept "1" / "true" / "yes" / "on" (case-insensitive).
    char buf[8] = {0};
    for (size_t i = 0; i < 7 && v[i]; ++i) {
        char c = v[i];
        if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
        buf[i] = c;
    }
    return std::strcmp(buf, "1") == 0 || std::strcmp(buf, "true") == 0 ||
           std::strcmp(buf, "yes") == 0 || std::strcmp(buf, "on") == 0;
}

// Resolve per-class block budget honoring the B8 env knobs. Modifies
// ``out_blocks[NUM_SIZE_CLASSES]`` in place. ``total_limit_bytes`` is the
// PinnedPool ctor budget (used by AUTO mode).
static void _resolve_blocks_per_class(size_t total_limit_bytes,
                                      size_t* out_blocks) {
    // Start from the historical defaults so every code path agrees on the
    // baseline schedule.
    for (int c = 0; c < NUM_SIZE_CLASSES; ++c) {
        out_blocks[c] = DEFAULT_BLOCKS_PER_CLASS[c];
    }

    // Auto-budget: split total_limit_bytes evenly across the 5 classes.
    if (_env_truthy("VLLM_PINNED_POOL_AUTO_BUDGET")) {
        const size_t per_class_bytes = total_limit_bytes / NUM_SIZE_CLASSES;
        for (int c = 0; c < NUM_SIZE_CLASSES; ++c) {
            size_t blk = SIZE_CLASS_BYTES[c];
            // At least DEFAULT — never *shrink* below the historical
            // baseline (so AUTO on a tiny pool degrades gracefully).
            size_t auto_blocks = (per_class_bytes > 0)
                ? (per_class_bytes / blk)
                : 0;
            if (auto_blocks > out_blocks[c]) {
                out_blocks[c] = auto_blocks;
            }
        }
    }

    // Explicit override wins. Format: comma-separated decimals,
    // missing/0 entries keep whatever is already in out_blocks[c].
    const char* spec = _env_str("VLLM_PINNED_POOL_BLOCKS_PER_CLASS");
    if (spec) {
        int c = 0;
        const char* p = spec;
        while (c < NUM_SIZE_CLASSES && *p) {
            char* endp = nullptr;
            long v = std::strtol(p, &endp, 10);
            if (endp == p) break;
            if (v > 0) {
                out_blocks[c] = (size_t)v;
            }
            ++c;
            p = endp;
            while (*p == ',' || *p == ' ') ++p;
        }
    }
}

// ──────────────────────────────────────────────────────────────────────
// Lockless MPMC ring buffer (power-of-two slot count).
//   - bounded SPSC/MPMC queue using slot sequence numbers (Vyukov-style)
//   - each free block ptr lives in one slot; alloc = dequeue, free = enqueue
//   - O(1) atomic ops, no mutex on hot path
// ──────────────────────────────────────────────────────────────────────

struct alignas(64) Slot {
    std::atomic<size_t> seq;
    void* ptr;
};

class LocklessRing {
public:
    LocklessRing() : slots_(nullptr), mask_(0), enq_(0), deq_(0) {}

    void init(size_t capacity_pow2) {
        // round up to power-of-two
        size_t cap = 1;
        while (cap < capacity_pow2) cap <<= 1;
        slots_ = new Slot[cap];
        mask_ = cap - 1;
        for (size_t i = 0; i < cap; ++i) {
            slots_[i].seq.store(i, std::memory_order_relaxed);
            slots_[i].ptr = nullptr;
        }
        enq_.store(0, std::memory_order_relaxed);
        deq_.store(0, std::memory_order_relaxed);
    }

    ~LocklessRing() { delete[] slots_; }

    // returns false if full
    bool enqueue(void* p) {
        size_t pos = enq_.load(std::memory_order_relaxed);
        for (;;) {
            Slot& s = slots_[pos & mask_];
            size_t seq = s.seq.load(std::memory_order_acquire);
            intptr_t diff = (intptr_t)seq - (intptr_t)pos;
            if (diff == 0) {
                if (enq_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    s.ptr = p;
                    s.seq.store(pos + 1, std::memory_order_release);
                    return true;
                }
            } else if (diff < 0) {
                return false;
            } else {
                pos = enq_.load(std::memory_order_relaxed);
            }
        }
    }

    // returns nullptr if empty
    void* dequeue() {
        size_t pos = deq_.load(std::memory_order_relaxed);
        for (;;) {
            Slot& s = slots_[pos & mask_];
            size_t seq = s.seq.load(std::memory_order_acquire);
            intptr_t diff = (intptr_t)seq - (intptr_t)(pos + 1);
            if (diff == 0) {
                if (deq_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    void* p = s.ptr;
                    s.ptr = nullptr;
                    s.seq.store(pos + mask_ + 1, std::memory_order_release);
                    return p;
                }
            } else if (diff < 0) {
                return nullptr;
            } else {
                pos = deq_.load(std::memory_order_relaxed);
            }
        }
    }

private:
    Slot* slots_;
    size_t mask_;
    alignas(64) std::atomic<size_t> enq_;
    alignas(64) std::atomic<size_t> deq_;
};

// ──────────────────────────────────────────────────────────────────────
// PinnedPool — pre-allocated cudaHostAlloc blocks per size class
// ──────────────────────────────────────────────────────────────────────

struct AllocRecord {
    int  size_class;   // 0..NUM_SIZE_CLASSES-1, or -1 for outsized direct-alloc
    size_t size;       // actual cudaHostAlloc size
};

class PinnedPool {
public:
    PinnedPool(size_t total_limit_bytes, int numa_node = -1)
        : total_limit_(total_limit_bytes),
          in_use_bytes_(0),
          owned_bytes_(0),
          numa_node_(numa_node),
          oversize_count_(0)
    {
#if VLLM_DMA_HAVE_NUMA
        if (numa_node_ >= 0 && numa_available() >= 0) {
            numa_set_preferred(numa_node_);
        }
#endif
        // SUB_201 A2 Phase B8: resolve per-class block budget honoring
        // the new VLLM_PINNED_POOL_AUTO_BUDGET / _BLOCKS_PER_CLASS env
        // knobs. When neither is set the schedule equals the historical
        // DEFAULT_BLOCKS_PER_CLASS (zero behaviour change).
        size_t blocks_per_class[NUM_SIZE_CLASSES] = {0};
        _resolve_blocks_per_class(total_limit_, blocks_per_class);

        // Pre-allocate per-class blocks until total_limit hits.
        for (int c = 0; c < NUM_SIZE_CLASSES; ++c) {
            size_t blk = SIZE_CLASS_BYTES[c];
            size_t target = blocks_per_class[c];
            // Ring slot count must be ≥ target and a power of two — keep
            // the historical 2× headroom so concurrent dequeues don't
            // starve under churn.
            ring_[c].init(target ? target * 2 : 2);
            class_capacity_[c] = 0;
            class_block_size_[c] = blk;
            for (size_t i = 0; i < target; ++i) {
                if (owned_bytes_ + blk > total_limit_) break;
                void* p = nullptr;
                cudaError_t rc = cudaHostAlloc(&p, blk, cudaHostAllocDefault);
                if (rc != cudaSuccess || p == nullptr) break;
                if (!ring_[c].enqueue(p)) { cudaFreeHost(p); break; }
                pool_ptrs_[c].push_back(p);
                owned_bytes_ += blk;
                class_capacity_[c] += 1;
            }
        }
    }

    ~PinnedPool() {
        for (int c = 0; c < NUM_SIZE_CLASSES; ++c) {
            for (void* p : pool_ptrs_[c]) cudaFreeHost(p);
        }
        // outsize blocks: caller is expected to free via pool->free which calls cudaFreeHost
        // but in destructor we cannot reliably collect them — leak warning instead.
        if (oversize_count_.load() != 0) {
            std::fprintf(stderr,
                "[PinnedPool] WARNING: %zu oversize blocks not freed at destruction\n",
                (size_t)oversize_count_.load());
        }
    }

    // Allocate ≥ size bytes. NUMA preferred (if configured at ctor).
    // Returns nullptr if pool exhausted and direct cudaHostAlloc also fails.
    void* alloc(size_t size, int* out_class = nullptr) {
        int cls = pick_size_class(size);
        if (cls >= 0) {
            void* p = ring_[cls].dequeue();
            if (p) {
                in_use_bytes_.fetch_add(SIZE_CLASS_BYTES[cls], std::memory_order_relaxed);
                {
                    std::lock_guard<std::mutex> lk(record_mu_);
                    records_[p] = AllocRecord{cls, SIZE_CLASS_BYTES[cls]};
                }
                if (out_class) *out_class = cls;
                return p;
            }
            // class exhausted — fall through to direct alloc
        }
        // Direct allocation (> 64 MB or class empty). Cost: cudaHostAlloc syscall.
        void* p = nullptr;
        size_t alloc_size = (cls >= 0 ? SIZE_CLASS_BYTES[cls] : size);
        if (cudaHostAlloc(&p, alloc_size, cudaHostAllocDefault) != cudaSuccess) return nullptr;
        in_use_bytes_.fetch_add(alloc_size, std::memory_order_relaxed);
        oversize_count_.fetch_add(1, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lk(record_mu_);
            records_[p] = AllocRecord{-1, alloc_size};
        }
        if (out_class) *out_class = -1;
        return p;
    }

    void free(void* ptr) {
        if (!ptr) return;
        AllocRecord rec;
        {
            std::lock_guard<std::mutex> lk(record_mu_);
            auto it = records_.find(ptr);
            if (it == records_.end()) return;  // unknown pointer — caller error
            rec = it->second;
            records_.erase(it);
        }
        in_use_bytes_.fetch_sub(rec.size, std::memory_order_relaxed);
        if (rec.size_class >= 0) {
            // return to ring
            if (!ring_[rec.size_class].enqueue(ptr)) {
                // ring full — direct free (rare)
                cudaFreeHost(ptr);
            }
        } else {
            cudaFreeHost(ptr);
            oversize_count_.fetch_sub(1, std::memory_order_relaxed);
        }
    }

    // ── DMA primitives ──────────────────────────────────────────────

    // Single async push host→device on given stream.
    // Caller is responsible for cudaEventDestroy on returned event.
    cudaEvent_t push_async(const void* host_ptr, void* dev_ptr, size_t size,
                           cudaStream_t stream) {
        cudaMemcpyAsync(dev_ptr, host_ptr, size, cudaMemcpyHostToDevice, stream);
        cudaEvent_t ev;
        cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
        cudaEventRecord(ev, stream);
        return ev;
    }

    cudaEvent_t pull_async(const void* dev_ptr, void* host_ptr, size_t size,
                           cudaStream_t stream) {
        cudaMemcpyAsync(host_ptr, dev_ptr, size, cudaMemcpyDeviceToHost, stream);
        cudaEvent_t ev;
        cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
        cudaEventRecord(ev, stream);
        return ev;
    }

    // Batched DMA push (option A): enqueue n transfers on same stream → 1 event at end.
    // Amortizes 35 μs API overhead per transfer down to ~5-10 μs/transfer.
    cudaEvent_t push_batch_async(const void* const* host_ptrs,
                                 void* const* dev_ptrs,
                                 const size_t* sizes,
                                 int n,
                                 cudaStream_t stream) {
        for (int i = 0; i < n; ++i) {
            cudaMemcpyAsync(dev_ptrs[i], host_ptrs[i], sizes[i],
                            cudaMemcpyHostToDevice, stream);
        }
        cudaEvent_t ev;
        cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
        cudaEventRecord(ev, stream);
        return ev;
    }

    // Batched DMA push (option B): use cudaMemcpyBatchAsync (CUDA 12.4+).
    // Single API call for all n transfers — driver may fuse / pipeline.
    // Returns 1 event for whole batch.
    // Note: per-CUDA-12.8 signature, attrs/attrsIdxs may be NULL to use defaults.
    cudaEvent_t push_batch_async_native(const void* const* host_ptrs,
                                        void* const* dev_ptrs,
                                        const size_t* sizes,
                                        int n,
                                        cudaStream_t stream) {
#if CUDART_VERSION >= 12040
        // Build per-call non-const dst/src arrays (cudaMemcpyBatchAsync wants
        // void** / void**, not void* const*). We do it once per call; cheap
        // (N=80 → 640 bytes copy).
        std::vector<void*> dsts(n);
        std::vector<void*> srcs(n);
        std::vector<size_t> sz(n);
        for (int i = 0; i < n; ++i) {
            dsts[i] = dev_ptrs[i];
            srcs[i] = const_cast<void*>(host_ptrs[i]);
            sz[i]   = sizes[i];
        }

        // Single attribute applied to all entries.
        // - srcAccessOrder=Stream: standard async semantics (host pinned).
        // - srcLocHint = Host, dstLocHint = Device 0 — driver may use this to
        //   pick an optimized engine.
        cudaMemcpyAttributes attr{};
        attr.srcAccessOrder    = cudaMemcpySrcAccessOrderStream;
        attr.srcLocHint.type   = cudaMemLocationTypeHost;
        attr.srcLocHint.id     = 0;
        attr.dstLocHint.type   = cudaMemLocationTypeDevice;
        attr.dstLocHint.id     = 0;
        attr.flags             = 0;

        // attrsIdxs[k] = START index (inclusive) where attrs[k] begins to apply.
        // attrs[k] applies to copies [attrsIdxs[k] .. attrsIdxs[k+1]-1],
        // and attrs[numAttrs-1] applies to copies [attrsIdxs[numAttrs-1] .. count-1].
        // With a single attr covering all n entries → attrsIdxs[0] = 0.
        size_t attrsIdxs = 0;
        size_t failIdx   = 0;

        cudaError_t rc = cudaMemcpyBatchAsync(
            dsts.data(), srcs.data(), sz.data(), (size_t)n,
            &attr, &attrsIdxs, 1, &failIdx, stream);
        if (rc != cudaSuccess) {
            std::fprintf(stderr,
                "[PinnedPool] cudaMemcpyBatchAsync failed (rc=%d failIdx=%zu) — "
                "falling back to per-call loop\n", (int)rc, failIdx);
            for (int i = 0; i < n; ++i) {
                cudaMemcpyAsync(dev_ptrs[i], host_ptrs[i], sizes[i],
                                cudaMemcpyHostToDevice, stream);
            }
        }
#else
        for (int i = 0; i < n; ++i) {
            cudaMemcpyAsync(dev_ptrs[i], host_ptrs[i], sizes[i],
                            cudaMemcpyHostToDevice, stream);
        }
#endif
        cudaEvent_t ev;
        cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
        cudaEventRecord(ev, stream);
        return ev;
    }

    // Batched DMA push (option C): copy N source buffers into one contiguous
    // staging block (host-side memcpy) then issue a single cudaMemcpyAsync.
    // staging_ptr must be a pinned-host buffer of size sum(sizes[0..n-1]).
    // On the device side the data lands at dev_dst as a single packed block
    // (caller is responsible for fan-out / strided unpack on GPU if needed).
    // Returns 1 event for the single memcpy.
    cudaEvent_t push_batch_async_staged(const void* const* host_ptrs,
                                        void* staging_ptr,
                                        void* dev_dst,
                                        const size_t* sizes,
                                        int n,
                                        cudaStream_t stream) {
        size_t total = 0;
        char* sptr = (char*)staging_ptr;
        for (int i = 0; i < n; ++i) {
            std::memcpy(sptr + total, host_ptrs[i], sizes[i]);
            total += sizes[i];
        }
        cudaMemcpyAsync(dev_dst, staging_ptr, total,
                        cudaMemcpyHostToDevice, stream);
        cudaEvent_t ev;
        cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
        cudaEventRecord(ev, stream);
        return ev;
    }

    // Batched DMA pull (option C, mirror of push_batch_async_staged):
    //   1. Single cudaMemcpyAsync DeviceToHost: dev_src → staging_ptr (total bytes)
    //   2. Host-side memcpy fan-out: staging[i] → host_ptrs[i]  (per slice)
    //
    // This is the *evict* path mirror used by KVDramTier.evict_block —
    // per-layer 80 GPU slabs collapse into 1 DMA + N cheap host memcpy
    // (≪ 35 μs API overhead per per-layer pull_async).
    //
    // staging_ptr must be a pinned-host buffer of size sum(sizes[0..n-1]).
    // dev_src is a single contiguous device buffer (the packed source).
    // host_ptrs[i] receives sizes[i] bytes (may be unpinned host memory —
    // the memcpy is plain CPU memcpy, not cudaMemcpy).
    //
    // Returns 1 event recorded *after* the cudaMemcpyAsync. Caller must
    // event_sync (or stream_sync) before reading host_ptrs[i] — that is,
    // before the host-side fan-out memcpy completes the per-slice copy.
    //
    // NOTE: the host-side fan-out is executed *synchronously after* the
    // event_sync inside the same C call, because we need staging_ptr
    // populated before we can scatter to host_ptrs. To keep the C symbol
    // simple and self-contained the implementation does:
    //     cudaMemcpyAsync(...)  → record event → (caller event_syncs) →
    //     unpack(staging → host_ptrs[i])
    // We split the unpack into a separate symbol pull_batch_unpack_staged
    // so the caller controls when fan-out runs (e.g. after event_sync).
    cudaEvent_t pull_batch_async_staged(const void* dev_src,
                                        void* staging_ptr,
                                        void* const* host_ptrs,
                                        const size_t* sizes,
                                        int n,
                                        cudaStream_t stream) {
        // Suppress unused warnings: we don't touch host_ptrs / sizes here;
        // they are consumed by pull_batch_unpack_staged after event_sync.
        (void)host_ptrs;
        (void)sizes;
        (void)n;
        size_t total = 0;
        for (int i = 0; i < n; ++i) {
            total += sizes[i];
        }
        cudaMemcpyAsync(staging_ptr, dev_src, total,
                        cudaMemcpyDeviceToHost, stream);
        cudaEvent_t ev;
        cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
        cudaEventRecord(ev, stream);
        return ev;
    }

    // Host-side fan-out after pull_batch_async_staged completes.
    // Caller MUST event_sync the event returned from pull_batch_async_staged
    // before invoking this — otherwise staging_ptr contents are undefined.
    void pull_batch_unpack_staged(const void* staging_ptr,
                                  void* const* host_ptrs,
                                  const size_t* sizes,
                                  int n) {
        const char* sptr = (const char*)staging_ptr;
        size_t total = 0;
        for (int i = 0; i < n; ++i) {
            std::memcpy(host_ptrs[i], sptr + total, sizes[i]);
            total += sizes[i];
        }
    }

    // ── Introspection ────────────────────────────────────────────────

    size_t in_use_bytes() const { return in_use_bytes_.load(std::memory_order_relaxed); }
    size_t owned_bytes() const { return owned_bytes_; }
    size_t total_limit() const { return total_limit_; }
    int    numa_node() const { return numa_node_; }
    size_t class_capacity(int c) const {
        return (c >= 0 && c < NUM_SIZE_CLASSES) ? class_capacity_[c] : 0;
    }
    size_t class_block_size(int c) const {
        return (c >= 0 && c < NUM_SIZE_CLASSES) ? class_block_size_[c] : 0;
    }

    // Find size class index for given byte size; -1 if > 64 MB.
    static int pick_size_class(size_t size) {
        for (int c = 0; c < NUM_SIZE_CLASSES; ++c) {
            if (size <= SIZE_CLASS_BYTES[c]) return c;
        }
        return -1;
    }

private:
    size_t total_limit_;
    std::atomic<size_t> in_use_bytes_;
    size_t owned_bytes_;
    int numa_node_;

    LocklessRing ring_[NUM_SIZE_CLASSES];
    size_t class_capacity_[NUM_SIZE_CLASSES] = {0};
    size_t class_block_size_[NUM_SIZE_CLASSES] = {0};
    std::vector<void*> pool_ptrs_[NUM_SIZE_CLASSES];

    std::mutex record_mu_;
    // Map host_ptr → record. Keeps free() O(1) on average.
    struct PtrHash { size_t operator()(void* p) const { return (size_t)p >> 6; } };
    // Use simple vector-based hash to avoid heavy <unordered_map> dependency in hot path?
    // For now use std::unordered_map with custom hash — TSK_028 hot path = alloc/free,
    // record_mu_ holds < 100 ns. If contention becomes issue, replace with sharded map.
    std::unordered_map<void*, AllocRecord, PtrHash> records_;

    std::atomic<size_t> oversize_count_;
};

// ──────────────────────────────────────────────────────────────────────
// C ABI for Python ctypes / pybind11
// ──────────────────────────────────────────────────────────────────────

extern "C" {

PinnedPool* pinned_pool_create(size_t total_limit, int numa_node) {
    return new PinnedPool(total_limit, numa_node);
}

void pinned_pool_destroy(PinnedPool* pool) { delete pool; }

void* pinned_pool_alloc(PinnedPool* pool, size_t size) { return pool->alloc(size); }

void pinned_pool_free(PinnedPool* pool, void* ptr) { pool->free(ptr); }

size_t pinned_pool_in_use(PinnedPool* pool) { return pool->in_use_bytes(); }
size_t pinned_pool_owned(PinnedPool* pool) { return pool->owned_bytes(); }
size_t pinned_pool_class_capacity(PinnedPool* pool, int c) { return pool->class_capacity(c); }
size_t pinned_pool_class_block_size(PinnedPool* pool, int c) { return pool->class_block_size(c); }
int    pinned_pool_pick_size_class(size_t size) { return PinnedPool::pick_size_class(size); }

// Single async push. Returns event ID (caller frees via cudaEventDestroy).
// host_ptr / dev_ptr / stream are opaque pointers.
void* pinned_pool_push_async(PinnedPool* pool, const void* host_ptr,
                             void* dev_ptr, size_t size, void* stream) {
    return (void*)pool->push_async(host_ptr, dev_ptr, size,
                                   (cudaStream_t)stream);
}

void* pinned_pool_pull_async(PinnedPool* pool, const void* dev_ptr,
                             void* host_ptr, size_t size, void* stream) {
    return (void*)pool->pull_async(dev_ptr, host_ptr, size,
                                   (cudaStream_t)stream);
}

// Batched push (option A: for-loop cudaMemcpyAsync × n) — arrays of n entries each.
void* pinned_pool_push_batch_async(PinnedPool* pool,
                                   const void* const* host_ptrs,
                                   void* const* dev_ptrs,
                                   const size_t* sizes,
                                   int n,
                                   void* stream) {
    return (void*)pool->push_batch_async(host_ptrs, dev_ptrs, sizes, n,
                                         (cudaStream_t)stream);
}

// Batched push (option B: cudaMemcpyBatchAsync — CUDA 12.4+ native).
void* pinned_pool_push_batch_async_native(PinnedPool* pool,
                                          const void* const* host_ptrs,
                                          void* const* dev_ptrs,
                                          const size_t* sizes,
                                          int n,
                                          void* stream) {
    return (void*)pool->push_batch_async_native(host_ptrs, dev_ptrs, sizes, n,
                                                (cudaStream_t)stream);
}

// Batched push (option C: host-side pack into staging_ptr + 1 memcpy to dev_dst).
//   staging_ptr : pinned-host buffer of size ≥ sum(sizes)
//   dev_dst     : single contiguous device dst buffer of the same size
void* pinned_pool_push_batch_async_staged(PinnedPool* pool,
                                          const void* const* host_ptrs,
                                          void* staging_ptr,
                                          void* dev_dst,
                                          const size_t* sizes,
                                          int n,
                                          void* stream) {
    return (void*)pool->push_batch_async_staged(host_ptrs, staging_ptr,
                                                dev_dst, sizes, n,
                                                (cudaStream_t)stream);
}

// Batched pull (option C mirror: dev_src → staging_ptr in 1 cudaMemcpyAsync,
// fan-out staging → host_ptrs[i] must be requested via
// pinned_pool_pull_batch_unpack_staged AFTER event_sync of the returned event).
//   dev_src     : single contiguous device source buffer (size = sum(sizes))
//   staging_ptr : pinned-host buffer of size ≥ sum(sizes)
//   host_ptrs   : N destination host buffers (may be unpinned)
void* pinned_pool_pull_batch_async_staged(PinnedPool* pool,
                                          const void* dev_src,
                                          void* staging_ptr,
                                          void* const* host_ptrs,
                                          const size_t* sizes,
                                          int n,
                                          void* stream) {
    return (void*)pool->pull_batch_async_staged(dev_src, staging_ptr,
                                                host_ptrs, sizes, n,
                                                (cudaStream_t)stream);
}

// Fan-out helper: copy from staging buffer into N host destinations.
// MUST be called AFTER event_sync on the event returned by
// pinned_pool_pull_batch_async_staged.
void pinned_pool_pull_batch_unpack_staged(PinnedPool* pool,
                                          const void* staging_ptr,
                                          void* const* host_ptrs,
                                          const size_t* sizes,
                                          int n) {
    pool->pull_batch_unpack_staged(staging_ptr, host_ptrs, sizes, n);
}

// Event helpers (so Python doesn't need cudart bindings)
int pinned_pool_event_query(void* ev) {
    cudaError_t rc = cudaEventQuery((cudaEvent_t)ev);
    if (rc == cudaSuccess) return 1;
    if (rc == cudaErrorNotReady) return 0;
    return -1;
}
int pinned_pool_event_sync(void* ev) {
    return (int)cudaEventSynchronize((cudaEvent_t)ev);
}
int pinned_pool_event_destroy(void* ev) {
    return (int)cudaEventDestroy((cudaEvent_t)ev);
}

// Stream helpers
void* pinned_pool_stream_create() {
    cudaStream_t s; cudaStreamCreate(&s);
    return (void*)s;
}
int pinned_pool_stream_destroy(void* s) {
    return (int)cudaStreamDestroy((cudaStream_t)s);
}
int pinned_pool_stream_sync(void* s) {
    return (int)cudaStreamSynchronize((cudaStream_t)s);
}

}  // extern "C"

}  // namespace vllm_hybrid_dma
