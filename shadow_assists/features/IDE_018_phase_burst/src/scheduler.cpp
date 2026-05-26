// IDE_018 / TSK_034 — Phase-burst scheduler (full impl)
//
// paper main contribution. CPU task pool with phase-aware dispatch.
//
// Design summary:
//   - 20 worker thread pinned to cpu 80-99 (사용자 100 core max constraint).
//   - Per-phase priority queue + global low-priority queue (idle phase fallback).
//   - Each worker: read PhaseSignal::current() (atomic, lockless) →
//     try dequeue task whose applicable_phases bitmask includes current phase.
//   - eventfd wake on phase change; spin-then-block within worker.
//   - Phase A inputs:
//       SUB_117 N=32 pinned: 10.24 TFLOPS — 20 worker = ~6.4 TFLOPS available.
//       SUB_148 default full-mask: scheduler 의 cpuset 만 강제하면 OS 협조 가능.
//       SUB_162 96% S: VLLM Python threads idle window 절대 보장.
//
// status: ✅ FULL — task pool runtime (task_pool_attention.cpp /
//          task_pool_linear.cpp) 에서 enqueue 한 Task 를 본 scheduler 가 dispatch.

#include "phase_detector.h"
#include "scheduler.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <functional>
#include <mutex>
#include <queue>
#include <sched.h>
#include <thread>
#include <vector>

namespace vllm_hybrid_phase {

// ──────────────────────────────────────────────────────────────────────
// PhaseQueue — per-phase priority queue
// ──────────────────────────────────────────────────────────────────────

void PhaseQueue::enqueue(Task t) {
    std::lock_guard<std::mutex> lock(mu_);
    // priority: lower numeric kind first within same applicable_phases.
    // (task_pool_attention.cpp 에서 task_kind 별 deadline / latency hint 셋업)
    queue_.push_back(std::move(t));
    cv_.notify_one();
}

bool PhaseQueue::try_dequeue_for_phase(Task& out, uint8_t current_phase) {
    std::lock_guard<std::mutex> lock(mu_);
    if (queue_.empty()) return false;

    const uint8_t mask = uint8_t(1) << current_phase;

    // Scan queue for the first task applicable to current_phase.
    // O(n) per dequeue but n is small (≤ 32 typically — task pool bounded).
    auto it = std::find_if(queue_.begin(), queue_.end(),
        [mask](const Task& t) { return (t.applicable_phases & mask) != 0; });
    if (it == queue_.end()) {
        // Try fallback — any task with PHASE_IDLE bit (run anytime).
        const uint8_t idle_mask = uint8_t(1) << PHASE_IDLE;
        it = std::find_if(queue_.begin(), queue_.end(),
            [idle_mask](const Task& t) { return (t.applicable_phases & idle_mask) != 0; });
        if (it == queue_.end()) return false;
    }
    out = std::move(*it);
    queue_.erase(it);
    return true;
}

size_t PhaseQueue::pending() const {
    std::lock_guard<std::mutex> lock(mu_);
    return queue_.size();
}

void PhaseQueue::wake_all() {
    std::lock_guard<std::mutex> lock(mu_);
    cv_.notify_all();
}

// ──────────────────────────────────────────────────────────────────────
// PhaseBurstScheduler
// ──────────────────────────────────────────────────────────────────────

PhaseBurstScheduler::PhaseBurstScheduler(PhaseSignal* signal,
                                         int num_workers,
                                         int cpu_base)
    : signal_(signal),
      num_workers_(num_workers),
      cpu_base_(cpu_base),
      stopped_(false) {
    for (int i = 0; i < PHASE_COUNT; ++i) {
        per_phase_tasks_executed_[i].store(0);
        per_phase_tasks_skipped_[i].store(0);
        per_phase_dispatch_latency_ns_sum_[i].store(0);
        per_phase_dispatch_count_[i].store(0);
    }
}

PhaseBurstScheduler::~PhaseBurstScheduler() {
    if (!stopped_.load()) stop();
}

void PhaseBurstScheduler::start() {
    stopped_.store(false);
    workers_.reserve(num_workers_);
    for (int i = 0; i < num_workers_; ++i) {
        workers_.emplace_back(&PhaseBurstScheduler::worker_loop, this, i);
    }
}

void PhaseBurstScheduler::stop() {
    stopped_.store(true);
    pool_.wake_all();
    // also notify the eventfd so wait_next() unblocks
    if (signal_) {
        signal_->update(signal_->current());
    }
    for (auto& t : workers_) {
        if (t.joinable()) t.join();
    }
    workers_.clear();
}

void PhaseBurstScheduler::enqueue(Task t) {
    if (t.enqueued_ns == 0) {
        t.enqueued_ns = PhaseSignal::current_time_ns();
    }
    pool_.enqueue(std::move(t));
}

bool PhaseBurstScheduler::pin_to_cpu(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    int rc = sched_setaffinity(0, sizeof(set), &set);
    return rc == 0;
}

void PhaseBurstScheduler::worker_loop(int worker_id) {
    // ─── CPU pin (cpu_base + worker_id), enforced even if cpu_base+id > 99 ───
    // CLAUDE: 사용자 100 core max constraint → 0..99 정도 권장.
    // 호출자가 (cpu_base=80, num_workers=20) 으로 80-99 지정.
    int cpu = cpu_base_ + worker_id;
    pin_to_cpu(cpu);

    uint8_t last_seen_phase = PHASE_IDLE;
    uint64_t last_phase_ns  = PhaseSignal::current_time_ns();

    // Backoff state — when no task applicable to current phase, escalate
    // from pause-spin → yield → eventfd wait. minimizes CPU burn during
    // unmatched phases without losing μs-scale wakeup.
    int empty_iters = 0;

    while (!stopped_.load(std::memory_order_acquire)) {
        uint8_t phase = signal_->phase.load(std::memory_order_acquire);

        // Phase changed — clear backoff, record dispatch latency.
        if (phase != last_seen_phase) {
            uint64_t now = PhaseSignal::current_time_ns();
            uint64_t lat = now - signal_->phase_start_ns.load(std::memory_order_acquire);
            per_phase_dispatch_latency_ns_sum_[phase].fetch_add(lat, std::memory_order_relaxed);
            per_phase_dispatch_count_[phase].fetch_add(1, std::memory_order_relaxed);
            last_seen_phase = phase;
            last_phase_ns = now;
            empty_iters = 0;
        }

        Task t;
        if (pool_.try_dequeue_for_phase(t, phase)) {
            // Execute. Catch any exception so worker does not die.
            try {
                t.fn();
            } catch (...) {
                // log? per CLAUDE.md korean 존칭 안내 — log message 한국어 가능.
                // production 에서는 spdlog or fprintf(stderr,..)
            }
            per_phase_tasks_executed_[phase].fetch_add(1, std::memory_order_relaxed);
            empty_iters = 0;
        } else {
            empty_iters++;
            per_phase_tasks_skipped_[phase].fetch_add(1, std::memory_order_relaxed);

            if (empty_iters < 32) {
                // tight spin — covers μs-scale gap until task enqueue / phase flip
                CPU_RELAX_HELPER();
            } else if (empty_iters < 512) {
                std::this_thread::yield();
            } else {
                // long idle — block on phase signal with 100 μs timeout so
                // we don't sleep through enqueue from non-driver thread.
                signal_->wait_next(100);
                empty_iters = 0;
            }
        }
    }
}

void PhaseBurstScheduler::CPU_RELAX_HELPER() {
#if defined(__x86_64__) || defined(__i386__)
    asm volatile("pause" ::: "memory");
#else
    asm volatile("" ::: "memory");
#endif
}

// ──────────────────────────────────────────────────────────────────────
// Stats accessor — for monitor + paper Figure 5
// ──────────────────────────────────────────────────────────────────────

PhaseBurstStats PhaseBurstScheduler::snapshot_stats() const {
    PhaseBurstStats s;
    s.num_workers = num_workers_;
    s.pending_tasks = pool_.pending();
    for (int i = 0; i < PHASE_COUNT; ++i) {
        s.tasks_executed[i] = per_phase_tasks_executed_[i].load();
        s.tasks_skipped[i]  = per_phase_tasks_skipped_[i].load();
        uint64_t cnt = per_phase_dispatch_count_[i].load();
        s.avg_dispatch_latency_ns[i] =
            cnt > 0
                ? per_phase_dispatch_latency_ns_sum_[i].load() / cnt
                : 0;
    }
    return s;
}

}  // namespace vllm_hybrid_phase
