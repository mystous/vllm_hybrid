// IDE_018 / TSK_034 — Phase-burst scheduler (skeleton)
//
// paper main contribution. CPU task pool with phase-aware dispatch.
//
// Phase A input:
//   - 20 worker pinned to cpu 80-99 (사용자 100 core max, SUB_165)
//   - 10.24 TFLOPS available (SUB_117 → scale to 20 worker = ~6.4 TFLOPS)
//   - sampler 44%, logits 27%, penalties 23% (SUB_161)
//
// status: ⚠ SKELETON — task pool runtime + vLLM integration 별도 turn

#include "phase_detector.cpp"   // PhaseSignal
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <sched.h>

namespace vllm_hybrid_phase {

// ──────────────────────────────────────────────────────────────────────
// Task abstraction
// ──────────────────────────────────────────────────────────────────────

enum TaskKind : uint8_t {
    TASK_A_SCHEDULE = 0,        // attention-phase: schedule next batch
    TASK_B_DETOKENIZE = 1,      // attention-phase: detokenize prev output
    TASK_C_GRAMMAR = 2,         // attention-phase: grammar check
    TASK_D_CLASSIFY = 3,        // attention-phase: request classifier
    TASK_E_KV_PREFETCH = 4,     // linear-phase: KV cache prefetch
    TASK_F_DRAFT = 5,           // linear-phase: AMX draft head
    TASK_G_COLDKV = 6,          // linear-phase: cold-KV decompress
    TASK_H_SAMPLE = 7,          // sample-phase: AVX-512 sampling rewrite
    TASK_I_LOGITS = 8,          // sample-phase: logit processor
    TASK_J_PRECOMPUTE = 9,      // tp-allreduce-phase: logit pre-compute
};

struct Task {
    TaskKind kind;
    uint64_t step_id;
    uint8_t  applicable_phases;  // bitmask of Phase
    std::function<void()> fn;
    uint64_t enqueued_ns;
};

/// Per-phase priority queue.
class TaskPool {
public:
    void enqueue(Task t) {
        std::lock_guard<std::mutex> lock(mu_);
        queue_.push(std::move(t));
        cv_.notify_one();
    }

    bool try_dequeue(Task& out, uint8_t current_phase) {
        std::lock_guard<std::mutex> lock(mu_);
        // O(n) scan for first task applicable to current_phase
        std::deque<Task> tmp;
        bool found = false;
        while (!queue_.empty()) {
            Task t = std::move(queue_.front());
            queue_.pop();
            if (!found && (t.applicable_phases & (1 << current_phase))) {
                out = std::move(t);
                found = true;
            } else {
                tmp.push_back(std::move(t));
            }
        }
        for (auto& t : tmp) queue_.push(std::move(t));
        return found;
    }

    size_t pending() const {
        std::lock_guard<std::mutex> lock(mu_);
        return queue_.size();
    }

private:
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::queue<Task> queue_;
};

// ──────────────────────────────────────────────────────────────────────
// Phase-burst scheduler
// ──────────────────────────────────────────────────────────────────────

class PhaseBurstScheduler {
public:
    PhaseBurstScheduler(PhaseSignal* signal, int num_workers, int cpu_base)
        : signal_(signal), num_workers_(num_workers), cpu_base_(cpu_base), stopped_(false) {}

    void start() {
        for (int i = 0; i < num_workers_; ++i) {
            workers_.emplace_back(&PhaseBurstScheduler::worker_loop, this, i);
        }
    }

    void stop() {
        stopped_.store(true);
        for (auto& t : workers_) {
            if (t.joinable()) t.join();
        }
    }

    void enqueue(Task t) { pool_.enqueue(std::move(t)); }

private:
    void worker_loop(int worker_id) {
        // Pin to physical core (사용자 100 core max constraint — cpu 80-99)
        int cpu = cpu_base_ + worker_id;
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        sched_setaffinity(0, sizeof(set), &set);

        while (!stopped_.load()) {
            uint8_t phase = signal_->phase.load(std::memory_order_acquire);

            Task t;
            if (pool_.try_dequeue(t, phase)) {
                // Execute task
                t.fn();
            } else {
                // No applicable task — short spin or wait
                // 50 μs spin then yield (latency target from task.md)
                std::this_thread::yield();
            }
        }
    }

    PhaseSignal* signal_;
    int num_workers_;
    int cpu_base_;
    std::atomic<bool> stopped_;
    std::vector<std::thread> workers_;
    TaskPool pool_;
};

}  // namespace vllm_hybrid_phase
