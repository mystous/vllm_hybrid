// IDE_018 / TSK_031 — Phase detection header
//
// PhaseSignal: shared atomic counter + eventfd, ABI 단일 출처.
// scheduler.cpp, task_pool_*.cpp, Python binding 이 본 header 만 include.

#pragma once

#include <atomic>
#include <cstdint>
#include <poll.h>

#ifdef ENABLE_CUDA
#  include <cuda_runtime.h>
#endif

namespace vllm_hybrid_phase {

// Phase enum — paper Table 1a/1b (SUB_167/168) 의 5 phase + IDLE/POST.
enum Phase : uint8_t {
    PHASE_IDLE       = 0,
    PHASE_ATTENTION  = 1,
    PHASE_LINEAR     = 2,
    PHASE_SAMPLE     = 3,
    PHASE_TP_ALLRED  = 4,
    PHASE_POST_STEP  = 5,
    PHASE_COUNT      = 6,
};

inline const char* phase_name(uint8_t p) {
    switch (p) {
        case PHASE_IDLE:       return "IDLE";
        case PHASE_ATTENTION:  return "ATTN";
        case PHASE_LINEAR:     return "LINEAR";
        case PHASE_SAMPLE:     return "SAMPLE";
        case PHASE_TP_ALLRED:  return "TP_AR";
        case PHASE_POST_STEP:  return "POST";
        default:               return "?";
    }
}

// Shared phase signal — vLLM forward thread (writer) + CPU task pool (readers).
// Single cacheline-aligned (64 B) atomics + eventfd for blocking wake.
struct alignas(64) PhaseSignal {
    std::atomic<uint64_t> step_id;
    std::atomic<uint8_t>  phase;
    std::atomic<uint64_t> phase_start_ns;
    std::atomic<uint64_t> total_updates;
    std::atomic<uint64_t> signal_drops;
    int eventfd_ = -1;

    // factory: anonymous shared mmap (same-process producer + consumer)
    static PhaseSignal* create_anonymous();
    static void destroy(PhaseSignal*);

    // Writer side — vLLM forward thread.
    // new_step_id = 0 means "keep current step_id".
    void update(uint8_t new_phase, uint64_t new_step_id = 0);

    // Reader side — CPU task pool worker.
    // timeout_us < 0 = block forever, 0 = non-blocking poll.
    uint8_t wait_next(int timeout_us = -1);
    uint8_t current() const;
    uint64_t current_step() const;
    uint64_t ns_in_phase() const;

    static uint64_t current_time_ns();
};

// Global singleton helpers (Python binding 의 simple access path).
PhaseSignal* get_or_create_global_signal();
void release_global_signal();

#ifdef ENABLE_CUDA
struct PhaseHookCtx {
    PhaseSignal* signal;
    uint8_t      phase;
    uint64_t     step_id;
};

void CUDART_CB phase_signal_callback(cudaStream_t, cudaError_t, void*);

void insert_phase_signal(cudaStream_t stream,
                         PhaseSignal* signal,
                         uint8_t phase,
                         uint64_t step_id);

struct PhaseEventPair {
    cudaEvent_t start;
    cudaEvent_t stop;
    uint8_t     phase;

    static PhaseEventPair* create();
    static void destroy(PhaseEventPair*);
};

void record_phase_start(cudaStream_t stream, PhaseEventPair* p, uint8_t phase);
void record_phase_stop(cudaStream_t stream, PhaseEventPair* p);
bool poll_phase_done(PhaseEventPair* p);
#endif

}  // namespace vllm_hybrid_phase
