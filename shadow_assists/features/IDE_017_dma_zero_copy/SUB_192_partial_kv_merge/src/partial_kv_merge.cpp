// SUB_192 — side-channel partial KV merge / dequant worker (NEW workload)
// follow-up to SUB_185 cold-KV proxy noise + SUB_188 small positive + SUB_190 confirmation.
//
// Hypothesis: regular memory access pattern (BF16 mul-add over contiguous KV block) +
//   moderate fire rate (50 ms cycle, 20 Hz) + true side-channel (no vllm hook) ≈
//   side-channel small net positive (SUB_188 / SUB_190 regular-work signature).
//
// Work per cycle:
//   - 1 KV block = 256 tokens × 4096 hidden = ~1 MB (BF16 input, FP32 accumulator)
//   - "INT8 dequant → BF16 → fused mul-add" simulated as float mul-add over 1 MB
//   - 16 worker × cores 80-95 pinned (vllm vanilla 0-49 / trident 56-105 disjoint)
//   - 50 ms cycle (20 Hz) — duty cycle target 3-5%
//   - inner replicas 32 → push compute to that target without growing memory
//
// NOTE: pure side-channel — does NOT inject into vllm KV cache. Independent CPU work
//   that simulates the *shape* of a real KV merge cost. No GIL, no shared cache lines
//   with vllm critical path.
//
// Build: g++ -O3 -fopenmp -march=native -pthread partial_kv_merge.cpp -o partial_kv_merge

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <thread>
#include <vector>

#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <omp.h>

namespace {

constexpr int kBlockTokens = 256;     // KV block size
constexpr int kHidden = 4096;         // hidden dim
constexpr int kBlockBytes = kBlockTokens * kHidden * 2;  // BF16 = 2 bytes — 2 MB
constexpr int kElems = kBlockTokens * kHidden;           // 1 048 576 elements
constexpr int kReplicas = 64;         // inner replicas per cycle (3-5% duty target)
constexpr int kCycleMs = 50;          // 20 Hz
constexpr int kCpuBase = 80;
constexpr int kNumWorkers = 16;

std::atomic<bool> g_stop{false};
std::atomic<uint64_t> g_cycles{0};
std::atomic<uint64_t> g_total_ns{0};

void on_signal(int /*sig*/) { g_stop.store(true); }

struct KVBlock {
    // BF16 → FP32 simulated: we keep two FP32 arrays.
    // src = "dequantized" K block (FP32 weights from INT8 source)
    // scale = per-token scale (FP32) — simulates dequant scale factor
    // dst = accumulator for merge result (FP32)
    std::vector<float> src;     // [kElems]
    std::vector<float> scale;   // [kBlockTokens]
    std::vector<float> dst;     // [kElems]

    KVBlock()
        : src(static_cast<size_t>(kElems)),
          scale(static_cast<size_t>(kBlockTokens)),
          dst(static_cast<size_t>(kElems)) {
        // deterministic seed (cold-look distribution)
        for (size_t i = 0; i < src.size(); ++i) {
            src[i] = static_cast<float>((i * 2654435761u) % 1024) / 1024.0f - 0.5f;
        }
        for (size_t i = 0; i < scale.size(); ++i) {
            scale[i] = 0.05f + static_cast<float>((i * 40503u) % 100) / 1000.0f;
        }
        std::memset(dst.data(), 0, dst.size() * sizeof(float));
    }
};

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

// fused dequant-mul-add: dst = dst + src * scale[t]
// regular contiguous memory access, no branches in inner loop.
inline void dequant_mul_add(KVBlock& kv) {
    const float* src = kv.src.data();
    const float* scl = kv.scale.data();
    float* dst = kv.dst.data();
#pragma omp parallel for schedule(static) num_threads(kNumWorkers)
    for (int t = 0; t < kBlockTokens; ++t) {
        const float s = scl[t];
        const float* row_src = src + static_cast<size_t>(t) * kHidden;
        float* row_dst = dst + static_cast<size_t>(t) * kHidden;
        for (int h = 0; h < kHidden; ++h) {
            row_dst[h] = row_dst[h] + row_src[h] * s;
        }
    }
}

void run_one_cycle(KVBlock& kv) {
    // Reset accumulator each cycle to keep numerical range bounded.
    std::memset(kv.dst.data(), 0, kv.dst.size() * sizeof(float));
    for (int r = 0; r < kReplicas; ++r) {
        dequant_mul_add(kv);
    }
}

}  // namespace

int main(int argc, char** argv) {
    (void)argc; (void)argv;

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    omp_set_num_threads(kNumWorkers);
    pin_omp_threads_to_cores();

    KVBlock kv;

    std::fprintf(stderr,
                 "[kv_merge] start workers=%d cores=%d-%d block_tokens=%d hidden=%d "
                 "elems=%d replicas=%d cycle=%dms (block=%.1fMB) pid=%d\n",
                 kNumWorkers, kCpuBase, kCpuBase + kNumWorkers - 1,
                 kBlockTokens, kHidden, kElems, kReplicas, kCycleMs,
                 static_cast<double>(kBlockBytes) / (1024.0 * 1024.0),
                 getpid());
    std::fflush(stderr);

    using clock = std::chrono::steady_clock;
    while (!g_stop.load()) {
        const auto t0 = clock::now();
        run_one_cycle(kv);
        const auto t1 = clock::now();

        const uint64_t elapsed_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        g_total_ns.fetch_add(elapsed_ns);
        const uint64_t c = g_cycles.fetch_add(1) + 1;

        // checksum to prevent optimization removal
        volatile float sink = kv.dst[0] + kv.dst[kElems - 1];
        (void)sink;

        if (c % 100 == 0) {
            const double avg_ms = (static_cast<double>(g_total_ns.load()) / 1e6) / static_cast<double>(c);
            std::fprintf(stderr, "[kv_merge] cycles=%lu avg=%.3f ms/cycle\n",
                         static_cast<unsigned long>(c), avg_ms);
            std::fflush(stderr);
        }

        const uint64_t target_ns = static_cast<uint64_t>(kCycleMs) * 1'000'000ull;
        if (elapsed_ns < target_ns) {
            const uint64_t sleep_ns = target_ns - elapsed_ns;
            std::this_thread::sleep_for(std::chrono::nanoseconds(sleep_ns));
        }
    }

    const uint64_t cyc = g_cycles.load();
    const double avg_ms = cyc ? (static_cast<double>(g_total_ns.load()) / 1e6) / static_cast<double>(cyc) : 0.0;
    std::fprintf(stderr, "[kv_merge] stop cycles=%lu avg=%.3f ms/cycle\n",
                 static_cast<unsigned long>(cyc), avg_ms);
    return 0;
}
