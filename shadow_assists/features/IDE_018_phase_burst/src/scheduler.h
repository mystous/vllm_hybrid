// IDE_018 / TSK_034 — Phase-burst scheduler header
//
// Public ABI for scheduler + Task abstraction.

#pragma once

#include "phase_detector.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace vllm_hybrid_phase {

// Task kinds — paper Table 1a/1b (SUB_167/168) 10-task matrix.
enum TaskKind : uint8_t {
    TASK_A_SCHEDULE    = 0,   // attention-phase: schedule next batch
    TASK_B_DETOKENIZE  = 1,   // attention-phase: detokenize prev output
    TASK_C_GRAMMAR     = 2,   // attention-phase: grammar check
    TASK_D_CLASSIFY    = 3,   // attention-phase: request classifier
    TASK_E_KV_PREFETCH = 4,   // linear-phase: KV cache DMA prefetch
    TASK_F_DRAFT       = 5,   // linear-phase: AMX draft head
    TASK_G_COLDKV      = 6,   // linear-phase: cold-KV decompress
    TASK_H_SAMPLE      = 7,   // sample-phase: AVX-512 sampling kernel
    TASK_I_LOGITS      = 8,   // sample-phase: logit processor
    TASK_J_PRECOMPUTE  = 9,   // tp_allreduce-phase: logit pre-compute
    TASK_KIND_COUNT    = 10,
};

inline const char* task_kind_name(uint8_t k) {
    switch (k) {
        case TASK_A_SCHEDULE:    return "A_schedule";
        case TASK_B_DETOKENIZE:  return "B_detok";
        case TASK_C_GRAMMAR:     return "C_grammar";
        case TASK_D_CLASSIFY:    return "D_classify";
        case TASK_E_KV_PREFETCH: return "E_kv_prefetch";
        case TASK_F_DRAFT:       return "F_draft";
        case TASK_G_COLDKV:      return "G_coldkv";
        case TASK_H_SAMPLE:      return "H_sample";
        case TASK_I_LOGITS:      return "I_logits";
        case TASK_J_PRECOMPUTE:  return "J_precompute";
        default:                 return "?";
    }
}

// Bitmask helpers for applicable_phases.
static constexpr uint8_t phase_mask(uint8_t p) { return uint8_t(1) << p; }

static constexpr uint8_t MASK_ATTN   = phase_mask(PHASE_ATTENTION);
static constexpr uint8_t MASK_LINEAR = phase_mask(PHASE_LINEAR);
static constexpr uint8_t MASK_SAMPLE = phase_mask(PHASE_SAMPLE);
static constexpr uint8_t MASK_TP_AR  = phase_mask(PHASE_TP_ALLRED);
static constexpr uint8_t MASK_IDLE   = phase_mask(PHASE_IDLE);
static constexpr uint8_t MASK_POST   = phase_mask(PHASE_POST_STEP);
static constexpr uint8_t MASK_ANY    = 0x3F;  // all 6 phases

struct Task {
    TaskKind kind;
    uint64_t step_id;
    uint8_t  applicable_phases;
    std::function<void()> fn;
    uint64_t enqueued_ns;
};

// Per-phase priority queue. Linear scan dequeue (queue depth ≤ tens).
class PhaseQueue {
public:
    void enqueue(Task t);
    bool try_dequeue_for_phase(Task& out, uint8_t current_phase);
    size_t pending() const;
    void wake_all();

private:
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<Task> queue_;
};

// Stats snapshot — used by monitor.py + paper Figure 5 generation.
struct PhaseBurstStats {
    int num_workers = 0;
    size_t pending_tasks = 0;
    uint64_t tasks_executed[PHASE_COUNT] = {};
    uint64_t tasks_skipped[PHASE_COUNT] = {};
    uint64_t avg_dispatch_latency_ns[PHASE_COUNT] = {};
};

class PhaseBurstScheduler {
public:
    PhaseBurstScheduler(PhaseSignal* signal, int num_workers, int cpu_base);
    ~PhaseBurstScheduler();

    void start();
    void stop();
    void enqueue(Task t);

    PhaseBurstStats snapshot_stats() const;
    PhaseSignal* signal() { return signal_; }

private:
    void worker_loop(int worker_id);
    static bool pin_to_cpu(int cpu);
    static void CPU_RELAX_HELPER();

    PhaseSignal* signal_;
    int num_workers_;
    int cpu_base_;
    std::atomic<bool> stopped_;
    std::vector<std::thread> workers_;
    PhaseQueue pool_;

    // per-phase stats
    std::atomic<uint64_t> per_phase_tasks_executed_[PHASE_COUNT];
    std::atomic<uint64_t> per_phase_tasks_skipped_[PHASE_COUNT];
    std::atomic<uint64_t> per_phase_dispatch_latency_ns_sum_[PHASE_COUNT];
    std::atomic<uint64_t> per_phase_dispatch_count_[PHASE_COUNT];
};

}  // namespace vllm_hybrid_phase
