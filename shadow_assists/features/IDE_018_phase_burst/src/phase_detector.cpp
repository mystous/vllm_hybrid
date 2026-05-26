// IDE_018 / TSK_031 — Phase detection mechanism (skeleton)
//
// CUDA event hook + IPC primitive for phase-burst scheduler.
// vLLM forward path 에 patch 로 삽입 (별도 turn).
//
// Phase A input:
//   - SUB_148: VLLM thread default full-mask — OS-coordination 가능
//   - SUB_162: 96% S — CPU pool 에 dispatch 안전
//
// status: ⚠ SKELETON — vLLM patch + CUDA event timing 별도 turn

#include <cuda_runtime.h>
#include <atomic>
#include <cstdint>
#include <sys/eventfd.h>
#include <unistd.h>

namespace vllm_hybrid_phase {

// ──────────────────────────────────────────────────────────────────────
// Phase enum + shared signal
// ──────────────────────────────────────────────────────────────────────

enum Phase : uint8_t {
    PHASE_IDLE       = 0,
    PHASE_ATTENTION  = 1,
    PHASE_LINEAR     = 2,
    PHASE_SAMPLE     = 3,
    PHASE_TP_ALLRED  = 4,
    PHASE_POST_STEP  = 5,
};

/// Shared phase signal — mapped between vLLM forward thread (writer) + CPU task pool (reader).
/// Allocated in shared memory (e.g., shm_open + mmap).
struct PhaseSignal {
    std::atomic<uint64_t> step_id;
    std::atomic<uint8_t>  phase;
    std::atomic<uint64_t> phase_start_ns;
    int                    eventfd;       // for blocking wait by CPU task pool

    /// Update phase + notify (called from vLLM forward thread on cuda event callback).
    void update(uint8_t new_phase) {
        uint64_t now = current_time_ns();
        phase_start_ns.store(now, std::memory_order_relaxed);
        phase.store(new_phase, std::memory_order_release);
        // notify (1 = wake one waiter, non-blocking write)
        uint64_t one = 1;
        ::write(eventfd, &one, sizeof(one));
    }

    /// Wait for next phase signal (CPU task pool side).
    /// Returns the new phase. Blocks if no signal.
    uint8_t wait_next() {
        uint64_t val;
        ::read(eventfd, &val, sizeof(val));   // blocks until signal
        return phase.load(std::memory_order_acquire);
    }

    static uint64_t current_time_ns() {
        // monotonic ns timer
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return uint64_t(ts.tv_sec) * 1'000'000'000ULL + ts.tv_nsec;
    }
};


// ──────────────────────────────────────────────────────────────────────
// CUDA event callback wrapper
// ──────────────────────────────────────────────────────────────────────

struct PhaseHookCtx {
    PhaseSignal* signal;
    uint8_t phase;
};

/// cudaStreamAddCallback compatible — called when stream reaches this point.
/// Runs on cuda driver-managed thread, MUST be fast (no blocking, no cuda calls).
void CUDART_CB phase_signal_callback(cudaStream_t stream, cudaError_t status, void* userData) {
    auto* ctx = static_cast<PhaseHookCtx*>(userData);
    ctx->signal->update(ctx->phase);
}

/// Insert a phase signal at this point in the stream.
void insert_phase_signal(cudaStream_t stream, PhaseSignal* signal, uint8_t phase) {
    auto* ctx = new PhaseHookCtx{signal, phase};   // freed in callback? — caller arena better
    cudaStreamAddCallback(stream, phase_signal_callback, ctx, 0);
}


// ──────────────────────────────────────────────────────────────────────
// Alternative — CUDA event timestamp poll (no callback overhead)
// ──────────────────────────────────────────────────────────────────────

/// Lower-overhead alternative: record CUDA events at phase boundaries,
/// poll them from CPU task pool thread (no callback context switch).
struct PhaseEventPair {
    cudaEvent_t start;
    cudaEvent_t stop;
    uint8_t phase;
};

void record_phase_start(cudaStream_t stream, PhaseEventPair* p, uint8_t phase) {
    p->phase = phase;
    cudaEventRecord(p->start, stream);
}

void record_phase_stop(cudaStream_t stream, PhaseEventPair* p) {
    cudaEventRecord(p->stop, stream);
}

/// Poll-based check (called from CPU task pool).
/// Returns true if phase has ended (event completed).
bool poll_phase_done(PhaseEventPair* p) {
    return cudaEventQuery(p->stop) == cudaSuccess;
}

}  // namespace vllm_hybrid_phase
