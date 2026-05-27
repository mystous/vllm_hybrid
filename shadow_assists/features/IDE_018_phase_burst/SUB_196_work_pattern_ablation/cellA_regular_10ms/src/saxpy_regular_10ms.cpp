// SUB_196 cell A — regular vector SAXPY × 10ms cycle (high-rate regular work)
// Position in 2×2 grid: row=10ms, col=regular  →  missing cell of SUB_188 / SUB_189 / SUB_190.
//
// Work: AVX-512 SAXPY  y[i] = a*x[i] + y[i]  over a 1 MiB working-set per worker,
//       branch-free, fully predictable memory access, target duty-cycle 2-5% over 10ms cycle.
// 16 worker × cores 80-95 pinned. ENV vllm 미접촉 (true side-channel).
//
// Build: g++ -O3 -fopenmp -march=native -pthread -mavx512f saxpy_regular_10ms.cpp -o saxpy_regular_10ms

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
#include <immintrin.h>

namespace {

// 1 MiB / 4 B = 262 144 floats per worker, L2-private (≈1 MiB L2 per core on SPR/SKX)
constexpr int kNumFloats = 262144;
constexpr int kInnerReplicas = 2;   // 2 SAXPY passes per cycle → ~0.3 ms per cycle (3% duty)
constexpr int kCycleMs = 10;
constexpr int kCpuBase = 80;
constexpr int kNumWorkers = 16;

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

inline void saxpy_avx512(float* y, const float* x, float a, int n) {
    const __m512 va = _mm512_set1_ps(a);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 vx = _mm512_loadu_ps(x + i);
        __m512 vy = _mm512_loadu_ps(y + i);
        __m512 vr = _mm512_fmadd_ps(va, vx, vy);
        _mm512_storeu_ps(y + i, vr);
    }
    for (; i < n; ++i) {
        y[i] = a * x[i] + y[i];
    }
}

void run_one_cycle(std::vector<float>& Xbuf, std::vector<float>& Ybuf) {
#pragma omp parallel num_threads(kNumWorkers)
    {
        const int tid = omp_get_thread_num();
        float* X = Xbuf.data() + static_cast<size_t>(tid) * kNumFloats;
        float* Y = Ybuf.data() + static_cast<size_t>(tid) * kNumFloats;
        const float a = 1.0001f;
        for (int r = 0; r < kInnerReplicas; ++r) {
            saxpy_avx512(Y, X, a, kNumFloats);
        }
        // checksum to prevent DCE
        if (Y[0] > 1e20f) Y[0] = 0.0f;
    }
}

}  // namespace

int main(int argc, char** argv) {
    (void)argc; (void)argv;

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    omp_set_num_threads(kNumWorkers);
    pin_omp_threads_to_cores();

    const size_t total = static_cast<size_t>(kNumWorkers) * kNumFloats;
    std::vector<float> Xbuf(total), Ybuf(total);
    for (size_t i = 0; i < total; ++i) {
        Xbuf[i] = static_cast<float>(i % 1024) / 1024.0f - 0.5f;
        Ybuf[i] = 0.001f;
    }

    std::fprintf(stderr,
                 "[saxpy_reg10] start workers=%d cores=%d-%d floats/worker=%d replicas=%d cycle=%dms pid=%d\n",
                 kNumWorkers, kCpuBase, kCpuBase + kNumWorkers - 1,
                 kNumFloats, kInnerReplicas, kCycleMs, getpid());
    std::fflush(stderr);

    using clock = std::chrono::steady_clock;
    while (!g_stop.load()) {
        const auto t0 = clock::now();
        run_one_cycle(Xbuf, Ybuf);
        const auto t1 = clock::now();

        const uint64_t elapsed_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        g_total_ns.fetch_add(elapsed_ns);
        const uint64_t c = g_cycles.fetch_add(1) + 1;

        if (c % 500 == 0) {
            const double avg_ms = (static_cast<double>(g_total_ns.load()) / 1e6) / static_cast<double>(c);
            std::fprintf(stderr, "[saxpy_reg10] cycles=%lu avg=%.3f ms/cycle\n",
                         static_cast<unsigned long>(c), avg_ms);
            std::fflush(stderr);
        }

        const uint64_t target_ns = static_cast<uint64_t>(kCycleMs) * 1'000'000ull;
        if (elapsed_ns < target_ns) {
            std::this_thread::sleep_for(std::chrono::nanoseconds(target_ns - elapsed_ns));
        }
    }

    const uint64_t cyc = g_cycles.load();
    const double avg_ms = cyc ? (static_cast<double>(g_total_ns.load()) / 1e6) / static_cast<double>(cyc) : 0.0;
    std::fprintf(stderr, "[saxpy_reg10] stop cycles=%lu avg=%.3f ms/cycle\n",
                 static_cast<unsigned long>(cyc), avg_ms);
    return 0;
}
