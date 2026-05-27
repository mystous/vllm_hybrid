// SUB_196 cell B — branchy hash-table probe × 100ms cycle (low-rate branchy work)
// Position in 2×2 grid: row=100ms, col=branchy.  Missing cell in SUB_188 / SUB_189 / SUB_190.
//
// Work: open-addressing hash probe with miss → linear search fallback.  Branchy code
//       (per-key probe count varies, branch predictor misses), data-dependent loads,
//       inner-loop ↑ to land duty cycle 2-5% over 100ms cycle.
// 16 worker × cores 80-95 pinned. ENV vllm 미접촉.
//
// Build: g++ -O3 -fopenmp -march=native -pthread hash_branchy_100ms.cpp -o hash_branchy_100ms

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <thread>
#include <vector>

#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <omp.h>

namespace {

// hash table (open addressing), ~512 KiB / worker → per-worker L2 pressure
constexpr int kHashSize = 131072;   // 2^17 slots
constexpr int kNumKeysPerCycle = 16384;
constexpr int kInnerReplicas = 8;   // 8 hash sweeps per cycle → ~3-4 ms per cycle (3-4% duty)
constexpr int kCycleMs = 100;
constexpr int kCpuBase = 80;
constexpr int kNumWorkers = 16;
constexpr uint32_t kSentinel = 0xFFFFFFFFu;

std::atomic<bool> g_stop{false};
std::atomic<uint64_t> g_cycles{0};
std::atomic<uint64_t> g_total_ns{0};

void on_signal(int /*sig*/) { g_stop.store(true); }

void pin_omp_threads_to_cores() {
#pragma omp parallel num_threads(kNumWorkers)
    {
        const int tid = omp_get_thread_num();
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(kCpuBase + tid, &set);
        pthread_t self = pthread_self();
        pthread_setaffinity_np(self, sizeof(set), &set);
    }
}

inline uint32_t mix32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// initialize a per-worker open-addressing table (load factor ~50%)
void init_table(std::vector<uint32_t>& tbl, uint32_t seed) {
    for (auto& s : tbl) s = kSentinel;
    const int fill = kHashSize / 2;
    uint32_t k = seed;
    for (int i = 0; i < fill; ++i) {
        k = mix32(k + 0x9E3779B9u);
        uint32_t pos = k & (kHashSize - 1);
        // open addressing: linear probe
        for (int p = 0; p < kHashSize; ++p) {
            uint32_t idx = (pos + p) & (kHashSize - 1);
            if (tbl[idx] == kSentinel) {
                tbl[idx] = k;
                break;
            }
        }
    }
}

// per-cycle work: probe N keys with a 50% miss rate (branch-heavy)
uint64_t probe_keys(const std::vector<uint32_t>& tbl, uint32_t key_seed) {
    uint64_t hits = 0;
    uint32_t k = key_seed;
    for (int r = 0; r < kInnerReplicas; ++r) {
        for (int i = 0; i < kNumKeysPerCycle; ++i) {
            k = mix32(k + 0xC2B2AE35u + static_cast<uint32_t>(i));
            // 50% of keys are "miss" by xoring high bit
            uint32_t probe = (i & 1) ? (k ^ 0x80000000u) : k;
            uint32_t pos = probe & (kHashSize - 1);
            // linear-search probe path — branchy (variable hop count)
            for (int p = 0; p < 32; ++p) {
                uint32_t idx = (pos + p) & (kHashSize - 1);
                uint32_t v = tbl[idx];
                if (v == kSentinel) break;          // miss path
                if (v == probe) { ++hits; break; }  // hit path
            }
        }
    }
    return hits;
}

void run_one_cycle(const std::vector<std::vector<uint32_t>>& tables,
                   std::vector<uint64_t>& sinks) {
#pragma omp parallel num_threads(kNumWorkers)
    {
        const int tid = omp_get_thread_num();
        const uint32_t seed = static_cast<uint32_t>(tid) * 0xDEADBEEFu;
        sinks[tid] += probe_keys(tables[tid], seed);
    }
}

}  // namespace

int main(int argc, char** argv) {
    (void)argc; (void)argv;

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    omp_set_num_threads(kNumWorkers);
    pin_omp_threads_to_cores();

    std::vector<std::vector<uint32_t>> tables(kNumWorkers, std::vector<uint32_t>(kHashSize));
    for (int t = 0; t < kNumWorkers; ++t) {
        init_table(tables[t], static_cast<uint32_t>(t) * 0xABCDEF11u + 1u);
    }
    std::vector<uint64_t> sinks(kNumWorkers, 0);

    std::fprintf(stderr,
                 "[hash_br100] start workers=%d cores=%d-%d hash_size=%d keys=%d replicas=%d cycle=%dms pid=%d\n",
                 kNumWorkers, kCpuBase, kCpuBase + kNumWorkers - 1,
                 kHashSize, kNumKeysPerCycle, kInnerReplicas, kCycleMs, getpid());
    std::fflush(stderr);

    using clock = std::chrono::steady_clock;
    while (!g_stop.load()) {
        const auto t0 = clock::now();
        run_one_cycle(tables, sinks);
        const auto t1 = clock::now();

        const uint64_t elapsed_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        g_total_ns.fetch_add(elapsed_ns);
        const uint64_t c = g_cycles.fetch_add(1) + 1;

        if (c % 50 == 0) {
            const double avg_ms = (static_cast<double>(g_total_ns.load()) / 1e6) / static_cast<double>(c);
            uint64_t total_hits = 0;
            for (auto s : sinks) total_hits += s;
            std::fprintf(stderr, "[hash_br100] cycles=%lu avg=%.3f ms/cycle total_hits=%lu\n",
                         static_cast<unsigned long>(c), avg_ms,
                         static_cast<unsigned long>(total_hits));
            std::fflush(stderr);
        }

        const uint64_t target_ns = static_cast<uint64_t>(kCycleMs) * 1'000'000ull;
        if (elapsed_ns < target_ns) {
            std::this_thread::sleep_for(std::chrono::nanoseconds(target_ns - elapsed_ns));
        }
    }

    const uint64_t cyc = g_cycles.load();
    const double avg_ms = cyc ? (static_cast<double>(g_total_ns.load()) / 1e6) / static_cast<double>(cyc) : 0.0;
    std::fprintf(stderr, "[hash_br100] stop cycles=%lu avg=%.3f ms/cycle\n",
                 static_cast<unsigned long>(cyc), avg_ms);
    return 0;
}
