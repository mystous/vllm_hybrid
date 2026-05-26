// IDE_018 / TSK_034 — Scheduler unit test (C++, no GPU / no Python)
//
// Verifies:
//   1. Tasks enqueued during attention phase are executed only by workers
//      observing PHASE_ATTENTION.
//   2. Phase change → workers pick up the previously-blocked tasks.
//   3. Per-phase stats counters reflect actual execution.
//
// Pass criteria: all assertions hold, return 0.

#include "phase_detector.h"
#include "scheduler.h"
#include "task_pool.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

using namespace vllm_hybrid_phase;

#define REQUIRE(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL [%s:%d] %s — %s\n", __FILE__, __LINE__, #cond, msg); \
        std::exit(1); \
    } \
} while (0)

static void test_attention_dispatch() {
    fprintf(stderr, "[test] attention phase dispatch\n");
    PhaseSignal* sig = PhaseSignal::create_anonymous();
    sig->update(PHASE_IDLE);

    // 4 workers, cpu_base 0 (test env, not pinned cpu 80-99).
    PhaseBurstScheduler sched(sig, /*num_workers=*/4, /*cpu_base=*/0);
    sched.start();

    std::atomic<int> attn_count{0}, linear_count{0};

    // Enqueue 8 attention-only tasks while phase=IDLE → should NOT execute yet.
    for (int i = 0; i < 8; ++i) {
        Task t{
            TASK_A_SCHEDULE, /*step_id=*/uint64_t(i),
            /*applicable_phases=*/ MASK_ATTN,
            [&attn_count]() { attn_count.fetch_add(1); },
            0
        };
        sched.enqueue(std::move(t));
    }

    // Wait briefly — assert no execution.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    REQUIRE(attn_count.load() == 0, "attn tasks fired during IDLE phase");

    // Flip phase to ATTENTION → workers should drain queue.
    sig->update(PHASE_ATTENTION);
    auto t_start = std::chrono::steady_clock::now();
    while (attn_count.load() < 8 &&
           std::chrono::steady_clock::now() - t_start < std::chrono::seconds(2)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    REQUIRE(attn_count.load() == 8, "attn tasks not drained after phase=ATTENTION");

    // Enqueue 4 linear-only tasks during attention → should NOT execute.
    for (int i = 0; i < 4; ++i) {
        Task t{
            TASK_E_KV_PREFETCH, 100 + uint64_t(i),
            /*applicable_phases=*/ MASK_LINEAR,
            [&linear_count]() { linear_count.fetch_add(1); },
            0
        };
        sched.enqueue(std::move(t));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    REQUIRE(linear_count.load() == 0, "linear tasks fired during ATTENTION");

    // Flip → LINEAR
    sig->update(PHASE_LINEAR);
    t_start = std::chrono::steady_clock::now();
    while (linear_count.load() < 4 &&
           std::chrono::steady_clock::now() - t_start < std::chrono::seconds(2)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    REQUIRE(linear_count.load() == 4, "linear tasks not drained after phase=LINEAR");

    auto stats = sched.snapshot_stats();
    REQUIRE(stats.tasks_executed[PHASE_ATTENTION] == 8, "executed[ATTN] != 8");
    REQUIRE(stats.tasks_executed[PHASE_LINEAR]    == 4, "executed[LINEAR] != 4");

    sched.stop();
    PhaseSignal::destroy(sig);
    fprintf(stderr, "[test] OK — attention/linear dispatch\n");
}

static void test_any_phase_task() {
    fprintf(stderr, "[test] MASK_ANY task runs in any phase\n");
    PhaseSignal* sig = PhaseSignal::create_anonymous();
    sig->update(PHASE_SAMPLE);
    PhaseBurstScheduler sched(sig, 2, 0);
    sched.start();

    std::atomic<int> cnt{0};
    for (int i = 0; i < 5; ++i) {
        Task t{
            TASK_D_CLASSIFY, uint64_t(i), MASK_ANY,
            [&cnt]() { cnt.fetch_add(1); },
            0
        };
        sched.enqueue(std::move(t));
    }
    auto t_start = std::chrono::steady_clock::now();
    while (cnt.load() < 5 &&
           std::chrono::steady_clock::now() - t_start < std::chrono::seconds(1)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    REQUIRE(cnt.load() == 5, "ANY-phase tasks not drained");
    sched.stop();
    PhaseSignal::destroy(sig);
    fprintf(stderr, "[test] OK — MASK_ANY task\n");
}

static void test_stub_handles() {
    fprintf(stderr, "[test] stub handle wiring\n");
    PhaseSignal* sig = PhaseSignal::create_anonymous();
    sig->update(PHASE_ATTENTION);
    PhaseBurstScheduler sched(sig, 4, 0);
    sched.start();

    auto attn_handles = attention_pool::make_stub_handles();
    AttentionStepInput in;
    int32_t prev[4] = { 1, 2, 3, 4 };
    in.prev_tokens = prev;
    in.prev_token_count = 4;

    uint64_t before = attention_pool::stub_invocation_count();
    size_t n = attention_pool::enqueue_attention_phase_tasks(
        sched, attn_handles, /*step_id=*/1, in);
    REQUIRE(n >= 2, "expected at least task A + B enqueued");

    auto t_start = std::chrono::steady_clock::now();
    while (attention_pool::stub_invocation_count() - before < n &&
           std::chrono::steady_clock::now() - t_start < std::chrono::seconds(2)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    REQUIRE(attention_pool::stub_invocation_count() - before == n,
            "stub invocation count mismatch");

    sched.stop();
    PhaseSignal::destroy(sig);
    fprintf(stderr, "[test] OK — stub handles\n");
}

int main() {
    test_attention_dispatch();
    test_any_phase_task();
    test_stub_handles();
    fprintf(stderr, "[test] ALL PASSED\n");
    return 0;
}
