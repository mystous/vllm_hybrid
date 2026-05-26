// IDE_018 / TSK_031 — Phase signal latency microbench
//
// Measures eventfd-based phase signal latency:
//   writer thread: PhaseSignal::update(phase)
//   reader thread: PhaseSignal::wait_next()
//
// target: p50 < 50 μs.
//
// Run:
//   ./phase_burst_bench [n_iters]

#include "phase_detector.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <pthread.h>
#include <sched.h>
#include <thread>
#include <vector>

using namespace vllm_hybrid_phase;

static void pin_self(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    sched_setaffinity(0, sizeof(set), &set);
}

int main(int argc, char** argv) {
    int n_iters = (argc > 1) ? std::atoi(argv[1]) : 10000;
    fprintf(stderr, "[bench] iters=%d\n", n_iters);

    PhaseSignal* sig = PhaseSignal::create_anonymous();

    std::vector<uint64_t> latencies(n_iters);
    std::atomic<bool> reader_ready{false};
    std::atomic<int>  iter_done{0};
    std::atomic<uint64_t> write_ts{0};

    std::thread reader([&]() {
        pin_self(2);   // physical core
        reader_ready.store(true);
        for (int i = 0; i < n_iters; ++i) {
            // wait for next signal — blocks on eventfd
            sig->wait_next(-1);
            uint64_t now = PhaseSignal::current_time_ns();
            uint64_t w = write_ts.load(std::memory_order_acquire);
            latencies[i] = (w > 0 && now > w) ? (now - w) : 0;
            iter_done.fetch_add(1);
        }
    });

    // Wait for reader pin
    while (!reader_ready.load()) std::this_thread::yield();

    pin_self(4);  // physical core (different cpu)

    for (int i = 0; i < n_iters; ++i) {
        // small inter-signal interval so we measure cold wake path
        std::this_thread::sleep_for(std::chrono::microseconds(200));
        uint64_t now = PhaseSignal::current_time_ns();
        write_ts.store(now, std::memory_order_release);
        uint8_t new_phase = (i & 1) ? PHASE_ATTENTION : PHASE_LINEAR;
        sig->update(new_phase, uint64_t(i + 1));
        // Wait until reader processed before next iter (so latency is per-signal)
        while (iter_done.load() <= i) {
            asm volatile("pause" ::: "memory");
        }
    }

    reader.join();

    // Stats
    std::vector<uint64_t> sorted = latencies;
    std::sort(sorted.begin(), sorted.end());
    auto pct = [&](double q) {
        size_t idx = size_t(q * (sorted.size() - 1));
        return sorted[idx];
    };
    uint64_t p50 = pct(0.50);
    uint64_t p90 = pct(0.90);
    uint64_t p99 = pct(0.99);
    uint64_t p999 = pct(0.999);
    uint64_t mn = sorted.front();
    uint64_t mx = sorted.back();

    fprintf(stderr, "[bench] phase signal latency (ns):\n");
    fprintf(stderr, "  min  = %8lu (%.2f μs)\n",   (unsigned long)mn,   mn/1000.0);
    fprintf(stderr, "  p50  = %8lu (%.2f μs)\n",   (unsigned long)p50,  p50/1000.0);
    fprintf(stderr, "  p90  = %8lu (%.2f μs)\n",   (unsigned long)p90,  p90/1000.0);
    fprintf(stderr, "  p99  = %8lu (%.2f μs)\n",   (unsigned long)p99,  p99/1000.0);
    fprintf(stderr, "  p999 = %8lu (%.2f μs)\n",   (unsigned long)p999, p999/1000.0);
    fprintf(stderr, "  max  = %8lu (%.2f μs)\n",   (unsigned long)mx,   mx/1000.0);
    fprintf(stderr, "  drops= %lu\n",
            (unsigned long)sig->signal_drops.load());

    // JSON
    fprintf(stdout,
        "{\"p50_ns\": %lu, \"p90_ns\": %lu, \"p99_ns\": %lu, "
        "\"min_ns\": %lu, \"max_ns\": %lu, \"iters\": %d, "
        "\"drops\": %lu}\n",
        (unsigned long)p50, (unsigned long)p90, (unsigned long)p99,
        (unsigned long)mn, (unsigned long)mx, n_iters,
        (unsigned long)sig->signal_drops.load());

    int rc = (p50 < 50000) ? 0 : 1;   // p50 < 50 μs target
    PhaseSignal::destroy(sig);
    return rc;
}
