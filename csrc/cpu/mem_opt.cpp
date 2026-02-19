// Memory Bandwidth Optimization for CPU Inference
//
// Key optimizations:
// 1. Non-temporal (streaming) memcpy for large copies
//    - Bypasses cache for write-once data
//    - Avoids cache pollution in copy_blocks
// 2. Software prefetch for KV cache blocks
//    - Prefetches next block while processing current
//    - Reduces memory stall cycles
// 3. NUMA-aware allocation utilities
//    - Memory placed on the same NUMA node as the accessing thread
//    - Maximizes memory bandwidth on multi-socket systems

#include "numa_utils.hpp"

#include <immintrin.h>
#include <torch/all.h>

#include <cstdlib>
#include <cstring>

#ifndef VLLM_NUMA_DISABLED
#include <numa.h>
#include <numaif.h>
#include <sched.h>
#endif

namespace vllm {
namespace numa {

// ============================================================================
// NUMA-aware memory allocation
// ============================================================================

void* alloc_on_node(size_t size, int node) {
#ifndef VLLM_NUMA_DISABLED
  if (numa_available() >= 0) {
    void* ptr = numa_alloc_onnode(size, node);
    if (ptr) return ptr;
  }
#endif
  // Fallback: aligned allocation
  return std::aligned_alloc(64, (size + 63) & ~63UL);
}

void* alloc_interleaved(size_t size) {
#ifndef VLLM_NUMA_DISABLED
  if (numa_available() >= 0) {
    void* ptr = numa_alloc_interleaved(size);
    if (ptr) return ptr;
  }
#endif
  return std::aligned_alloc(64, (size + 63) & ~63UL);
}

void free_numa(void* ptr, size_t size) {
  if (!ptr) return;
#ifndef VLLM_NUMA_DISABLED
  if (numa_available() >= 0) {
    numa_free(ptr, size);
    return;
  }
#endif
  std::free(ptr);
}

int bind_thread_to_node(int node) {
#ifndef VLLM_NUMA_DISABLED
  if (numa_available() >= 0) {
    struct bitmask* mask = numa_allocate_cpumask();
    if (numa_node_to_cpus(node, mask) == 0) {
      int ret = numa_sched_setaffinity(0, mask);
      numa_bitmask_free(mask);
      numa_set_preferred(node);
      return ret;
    }
    numa_bitmask_free(mask);
  }
#endif
  (void)node;
  return -1;
}

int get_current_node() {
#ifndef VLLM_NUMA_DISABLED
  if (numa_available() >= 0) {
    return numa_node_of_cpu(sched_getcpu());
  }
#endif
  return 0;
}

int get_num_nodes() {
#ifndef VLLM_NUMA_DISABLED
  if (numa_available() >= 0) {
    return numa_max_node() + 1;
  }
#endif
  return 1;
}

void* alloc_aligned(size_t size) {
  return std::aligned_alloc(64, (size + 63) & ~63UL);
}

void free_aligned(void* ptr) { std::free(ptr); }

}  // namespace numa
}  // namespace vllm

// ============================================================================
// Non-temporal memcpy using AVX-512 streaming stores
// ============================================================================
#ifdef __AVX512F__

// NT memcpy: bypasses cache for large writes
// Best for: copy_blocks where data is written once and read later
// Threshold: use NT stores for copies > 256KB (larger than L2 cache)
static constexpr size_t NT_MEMCPY_THRESHOLD = 256 * 1024;

void nt_memcpy(void* dst, const void* src, size_t n) {
  if (n < NT_MEMCPY_THRESHOLD) {
    std::memcpy(dst, src, n);
    return;
  }

  char* d = static_cast<char*>(dst);
  const char* s = static_cast<const char*>(src);

  // Handle unaligned prefix
  size_t prefix = reinterpret_cast<uintptr_t>(d) & 63;
  if (prefix) {
    prefix = 64 - prefix;
    if (prefix > n) prefix = n;
    std::memcpy(d, s, prefix);
    d += prefix;
    s += prefix;
    n -= prefix;
  }

  // Main loop: 512 bytes per iteration (8 cache lines)
  while (n >= 512) {
    // Stream load (if source is aligned) or regular load
    __m512i v0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(s + 0));
    __m512i v1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(s + 64));
    __m512i v2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(s + 128));
    __m512i v3 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(s + 192));
    __m512i v4 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(s + 256));
    __m512i v5 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(s + 320));
    __m512i v6 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(s + 384));
    __m512i v7 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(s + 448));

    // Non-temporal stores (bypass cache)
    _mm512_stream_si512(reinterpret_cast<__m512i*>(d + 0), v0);
    _mm512_stream_si512(reinterpret_cast<__m512i*>(d + 64), v1);
    _mm512_stream_si512(reinterpret_cast<__m512i*>(d + 128), v2);
    _mm512_stream_si512(reinterpret_cast<__m512i*>(d + 192), v3);
    _mm512_stream_si512(reinterpret_cast<__m512i*>(d + 256), v4);
    _mm512_stream_si512(reinterpret_cast<__m512i*>(d + 320), v5);
    _mm512_stream_si512(reinterpret_cast<__m512i*>(d + 384), v6);
    _mm512_stream_si512(reinterpret_cast<__m512i*>(d + 448), v7);

    d += 512;
    s += 512;
    n -= 512;
  }

  // Handle remaining 64-byte chunks
  while (n >= 64) {
    __m512i v = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(s));
    _mm512_stream_si512(reinterpret_cast<__m512i*>(d), v);
    d += 64;
    s += 64;
    n -= 64;
  }

  // Handle tail
  if (n > 0) {
    std::memcpy(d, s, n);
  }

  // Ensure all NT stores are visible
  _mm_sfence();
}

// ============================================================================
// KV Cache Block Prefetch
// Prefetches KV cache blocks that will be accessed next
// ============================================================================
void prefetch_kv_blocks(const void* kv_cache, const int32_t* block_table,
                        int num_blocks, int block_stride) {
  const char* base = static_cast<const char*>(kv_cache);

  for (int i = 0; i < num_blocks; ++i) {
    const char* block_ptr = base + static_cast<int64_t>(block_table[i]) *
                                       block_stride;

    // Prefetch first few cache lines of the block
    // T0: prefetch to L1 and L2
    _mm_prefetch(block_ptr, _MM_HINT_T0);
    _mm_prefetch(block_ptr + 64, _MM_HINT_T0);
    _mm_prefetch(block_ptr + 128, _MM_HINT_T0);
    _mm_prefetch(block_ptr + 192, _MM_HINT_T0);

    // T1: prefetch to L2 only (for blocks further ahead)
    if (i + 1 < num_blocks) {
      const char* next_block_ptr =
          base + static_cast<int64_t>(block_table[i + 1]) * block_stride;
      _mm_prefetch(next_block_ptr, _MM_HINT_T1);
      _mm_prefetch(next_block_ptr + 64, _MM_HINT_T1);
    }

    // NTA: prefetch with non-temporal hint (for blocks far ahead)
    if (i + 2 < num_blocks) {
      const char* far_block_ptr =
          base + static_cast<int64_t>(block_table[i + 2]) * block_stride;
      _mm_prefetch(far_block_ptr, _MM_HINT_NTA);
    }
  }
}

// ============================================================================
// Torch entry points
// ============================================================================

// NT memcpy wrapper for use from Python/Torch
void nt_memcpy_tensor(torch::Tensor& dst, const torch::Tensor& src) {
  TORCH_CHECK(dst.nbytes() == src.nbytes(), "dst and src size mismatch");
  TORCH_CHECK(dst.is_contiguous() && src.is_contiguous(),
              "tensors must be contiguous");
  nt_memcpy(dst.data_ptr(), src.data_ptr(), dst.nbytes());
}

// Prefetch KV blocks for upcoming attention computation
void prefetch_kv_cache_blocks(const torch::Tensor& kv_cache,
                              const torch::Tensor& block_table,
                              int num_blocks) {
  TORCH_CHECK(block_table.dtype() == torch::kInt32,
              "block_table must be INT32");
  const int block_stride =
      kv_cache.stride(0) * kv_cache.element_size();
  prefetch_kv_blocks(kv_cache.data_ptr(), block_table.data_ptr<int32_t>(),
                     num_blocks, block_stride);
}

#endif  // __AVX512F__
