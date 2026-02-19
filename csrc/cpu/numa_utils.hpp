// NUMA Utilities for CPU Memory Optimization
//
// Provides NUMA-aware memory allocation and thread binding for
// optimal memory bandwidth utilization on multi-socket systems.

#pragma once

#include <cstddef>
#include <cstdint>

namespace vllm {
namespace numa {

// ============================================================================
// NUMA-aware memory allocation
// ============================================================================

// Allocate memory on a specific NUMA node
// Falls back to standard malloc if NUMA is not available
void* alloc_on_node(size_t size, int node);

// Allocate memory interleaved across all NUMA nodes
// Useful for data accessed by all threads equally
void* alloc_interleaved(size_t size);

// Free NUMA-allocated memory
void free_numa(void* ptr, size_t size);

// ============================================================================
// Thread binding
// ============================================================================

// Bind the calling thread to a specific NUMA node
// Returns 0 on success, -1 on failure
int bind_thread_to_node(int node);

// Get the NUMA node the calling thread is currently on
int get_current_node();

// Get total number of NUMA nodes
int get_num_nodes();

// ============================================================================
// Cache-aligned allocation
// ============================================================================

// Allocate cache-line aligned memory (64-byte alignment)
void* alloc_aligned(size_t size);

// Free cache-line aligned memory
void free_aligned(void* ptr);

}  // namespace numa
}  // namespace vllm
