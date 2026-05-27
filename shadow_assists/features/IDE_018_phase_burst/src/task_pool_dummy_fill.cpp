// IDE_018 / SUB_184 — Dummy task pool fill for phase-burst overlap verification.
//
// goal: validate the paper §4 main hypothesis — "CPU 가 GPU phase 와 overlap 만
//       하면 critical path 는 변경 없이 CPU util 을 paper target 30%+ 로 끌어
//       올릴 수 있다" — via a heavy-but-real CPU compute that mirrors actual
//       AVX-512 task density (no GIL, no Python callback).
//
// design:
//   - per-task = touch a 64 KB working set with a fused multiply-add loop.
//   - heavy_burst_count = N tasks enqueued per (ATTENTION|LINEAR) mark.
//   - controlled by ENV via python wrapper, not hardcoded here.
//   - executed entirely inside C++ worker threads pinned to 80-99.
//
// non-goals:
//   - actual numerical correctness (this is overlap probe, not a real kernel).
//   - persistence across process restarts.

#include "scheduler.h"
#include "task_pool.h"

#include <atomic>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace vllm_hybrid_phase {
namespace dummy_fill {

// thread-local working buffer to avoid cross-thread cache invalidation.
// 64 KB ≈ L2 fit per task. 2 buffer (A,B) → 128 KB total per thread, well in L2.
static thread_local std::vector<float>* tls_A = nullptr;
static thread_local std::vector<float>* tls_B = nullptr;
static constexpr size_t kBufElems = 16 * 1024;   // 64 KB float32

std::atomic<uint64_t> g_dummy_invocation_count{0};
std::atomic<uint64_t> g_dummy_total_iters{0};

// One dummy compute pass. iters controls how heavy.
// Default iters = 64 → ~5 ms on 1 core on Alder Lake / Sapphire Rapids
// (memcpy 64 KB + fma loop 16K × 64 ≈ 1 M flops, ≈ 1 ms; the spin overhead
// brings it to several ms; tunable via VLLM_PHASE_BURST_DUMMY_ITERS).
static void dummy_compute(int iters) {
    if (tls_A == nullptr) {
        tls_A = new std::vector<float>(kBufElems, 1.000001f);
        tls_B = new std::vector<float>(kBufElems, 0.999999f);
    }
    float* A = tls_A->data();
    float* B = tls_B->data();
    // Force compiler to keep work.
    volatile float acc_sink = 0.0f;
    for (int it = 0; it < iters; ++it) {
        // 1) memcpy-ish blend
        for (size_t i = 0; i < kBufElems; ++i) {
            A[i] = A[i] * 1.0000001f + B[i] * 0.0000001f;
        }
        // 2) reduce (vectorizable)
        float acc = 0.0f;
        for (size_t i = 0; i < kBufElems; ++i) {
            acc += A[i];
        }
        acc_sink = acc;
    }
    (void)acc_sink;
    g_dummy_invocation_count.fetch_add(1, std::memory_order_relaxed);
    g_dummy_total_iters.fetch_add(iters, std::memory_order_relaxed);
}

// Build a Task object for a single dummy unit. kind / mask routes it to the
// appropriate phase queue. Uses TASK_E_KV_PREFETCH for LINEAR pool, TASK_B for
// ATTENTION pool — preserves stats categorization sanity.
static Task make_dummy_task(TaskKind kind, uint8_t mask,
                            uint64_t step_id, int iters) {
    return Task{
        kind,
        step_id,
        mask,
        /*fn=*/ [iters]() { dummy_compute(iters); },
        /*enqueued_ns=*/ 0,
    };
}

// Bulk enqueue — N dummy tasks for ATTENTION phase.
size_t enqueue_dummy_attention_burst(PhaseBurstScheduler& sched,
                                     uint64_t step_id, int count, int iters) {
    if (count <= 0) return 0;
    if (iters <= 0) iters = 1;
    for (int i = 0; i < count; ++i) {
        sched.enqueue(make_dummy_task(TASK_B_DETOKENIZE,
                                      MASK_ATTN | MASK_IDLE,
                                      step_id, iters));
    }
    return static_cast<size_t>(count);
}

// Bulk enqueue — N dummy tasks for LINEAR phase.
size_t enqueue_dummy_linear_burst(PhaseBurstScheduler& sched,
                                  uint64_t step_id, int count, int iters) {
    if (count <= 0) return 0;
    if (iters <= 0) iters = 1;
    for (int i = 0; i < count; ++i) {
        sched.enqueue(make_dummy_task(TASK_E_KV_PREFETCH,
                                      MASK_LINEAR | MASK_TP_AR | MASK_IDLE,
                                      step_id, iters));
    }
    return static_cast<size_t>(count);
}

uint64_t invocation_count() {
    return g_dummy_invocation_count.load(std::memory_order_relaxed);
}

uint64_t total_iters() {
    return g_dummy_total_iters.load(std::memory_order_relaxed);
}

}  // namespace dummy_fill
}  // namespace vllm_hybrid_phase
