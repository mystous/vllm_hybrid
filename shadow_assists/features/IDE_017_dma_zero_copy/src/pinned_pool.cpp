// IDE_017 / TSK_028 — Pinned memory pool + DMA push primitive (skeleton)
//
// SUB_166 input: 35 μs overhead per transfer, 1 MB crossover, 54 GB/s asymptotic.
//
// build:
//   nvcc -arch=sm_90 -O3 -Xcompiler -fPIC -shared \
//        -o libpinned_pool.so pinned_pool.cpp -lcudart
//
// status: ⚠ SKELETON — size-class allocator + DMA push 구현 deferred

#include <cuda_runtime.h>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace vllm_hybrid_dma {

// ──────────────────────────────────────────────────────────────────────
// Size class allocator
//   SUB_166 measurement region:
//     class 0: 4 KB     (overhead-bound)
//     class 1: 64 KB    (overhead-bound)
//     class 2: 1 MB     (crossover)
//     class 3: 16 MB    (bandwidth-bound)
//     class 4: 64 MB    (asymptotic)
// ──────────────────────────────────────────────────────────────────────

constexpr int NUM_SIZE_CLASSES = 5;
constexpr size_t SIZE_CLASS_BYTES[NUM_SIZE_CLASSES] = {
    4*1024, 64*1024, 1024*1024, 16*1024*1024, 64*1024*1024
};

struct PoolBlock {
    void* ptr;
    size_t size;
    bool in_use;
};

class PinnedPool {
public:
    PinnedPool(size_t total_limit_bytes)
        : total_limit_(total_limit_bytes), in_use_bytes_(0) {
        // Pre-allocate small pools per size class
        for (int c = 0; c < NUM_SIZE_CLASSES; ++c) {
            // Default: 16 blocks per class (subject to total_limit)
            size_t class_total = 16 * SIZE_CLASS_BYTES[c];
            if (in_use_bytes_ + class_total > total_limit_) break;
            for (int i = 0; i < 16; ++i) {
                void* p;
                if (cudaHostAlloc(&p, SIZE_CLASS_BYTES[c], cudaHostAllocDefault) != cudaSuccess) {
                    break;
                }
                free_lists_[c].push_back({p, SIZE_CLASS_BYTES[c], false});
            }
        }
    }

    ~PinnedPool() {
        for (int c = 0; c < NUM_SIZE_CLASSES; ++c) {
            for (auto& blk : free_lists_[c]) {
                cudaFreeHost(blk.ptr);
            }
        }
    }

    /// Allocate a block ≥ `size` bytes.
    /// Returns nullptr if pool exhausted.
    void* alloc(size_t size) {
        std::lock_guard<std::mutex> lock(mu_);
        int cls = pick_size_class(size);
        if (cls < 0) return nullptr;
        for (auto& blk : free_lists_[cls]) {
            if (!blk.in_use) {
                blk.in_use = true;
                in_use_bytes_ += blk.size;
                return blk.ptr;
            }
        }
        // class exhausted — caller may retry larger class or fall back to direct cudaHostAlloc
        return nullptr;
    }

    void free(void* ptr) {
        std::lock_guard<std::mutex> lock(mu_);
        for (int c = 0; c < NUM_SIZE_CLASSES; ++c) {
            for (auto& blk : free_lists_[c]) {
                if (blk.ptr == ptr && blk.in_use) {
                    blk.in_use = false;
                    in_use_bytes_ -= blk.size;
                    return;
                }
            }
        }
        // Unknown pointer — caller error
    }

    /// Async DMA push: host_ptr → dev_ptr on given stream.
    /// Returns event the caller can wait/sync on.
    cudaEvent_t push_async(void* host_ptr, void* dev_ptr, size_t size, cudaStream_t stream) {
        cudaMemcpyAsync(dev_ptr, host_ptr, size, cudaMemcpyHostToDevice, stream);
        cudaEvent_t ev;
        cudaEventCreate(&ev);
        cudaEventRecord(ev, stream);
        return ev;
    }

    size_t in_use_bytes() const { return in_use_bytes_; }

private:
    int pick_size_class(size_t size) {
        for (int c = 0; c < NUM_SIZE_CLASSES; ++c) {
            if (size <= SIZE_CLASS_BYTES[c]) return c;
        }
        return -1;  // too large for any class
    }

    size_t total_limit_;
    std::atomic<size_t> in_use_bytes_;
    std::vector<PoolBlock> free_lists_[NUM_SIZE_CLASSES];
    std::mutex mu_;
};

// ──────────────────────────────────────────────────────────────────────
// C ABI for Python pybind11 / ctypes
// ──────────────────────────────────────────────────────────────────────

extern "C" {

PinnedPool* pinned_pool_create(size_t total_limit) {
    return new PinnedPool(total_limit);
}

void pinned_pool_destroy(PinnedPool* pool) {
    delete pool;
}

void* pinned_pool_alloc(PinnedPool* pool, size_t size) {
    return pool->alloc(size);
}

void pinned_pool_free(PinnedPool* pool, void* ptr) {
    pool->free(ptr);
}

size_t pinned_pool_in_use(PinnedPool* pool) {
    return pool->in_use_bytes();
}

}  // extern "C"

}  // namespace vllm_hybrid_dma
