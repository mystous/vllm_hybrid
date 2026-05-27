// SUB_188 — side-channel batch precompute worker
// 가설: vllm 의 main critical path (prefill/decode) 와 직접 닿지 않는 CPU work 를
//       미리 batch 으로 계산해두면, cache pollution / scheduler migration / GIL 등
//       indirect contention 만 잘 피하면 throughput 영향 없이 CPU util 끌어올림 가능.
//
// Work: logprob softmax + log
//   - shape: [batch=32, vocab=152064] BF16 input → softmax → log-softmax (FP32)
//   - 매 100ms cycle 마다 1회 fire
//   - 16 worker × cores 80-95 pinned (vllm vanilla 0-49 / trident 56-105 와 cpu_base 분리)
//
// Build: g++ -O3 -fopenmp -march=native -pthread side_channel_precompute.cpp -o side_channel_precompute

#include <atomic>
#include <chrono>
#include <cmath>
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

constexpr int kBatch = 32;
constexpr int kVocab = 152064;
constexpr int kCycleMs = 10;
constexpr int kCpuBase = 80;
constexpr int kNumWorkers = 16;

std::atomic<bool> g_stop{false};
std::atomic<uint64_t> g_cycles{0};
std::atomic<uint64_t> g_total_ns{0};

void on_signal(int /*sig*/) { g_stop.store(true); }

// Pack of FP32 logits (we simulate BF16 by ignoring low 16 bits during init,
// but compute path is FP32 — matches the typical CPU softmax kernel cost.)
struct Tensor {
    std::vector<float> data;  // batch * vocab
    Tensor() : data(static_cast<size_t>(kBatch) * kVocab) {
        // simple seed pattern; not measured contributor to runtime
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = static_cast<float>((i * 2654435761u) % 1024) / 1024.0f - 0.5f;
        }
    }
};

inline void softmax_logsoftmax_row(const float* in, float* out_logsm, int n) {
    // 1) max
    float mx = in[0];
    for (int i = 1; i < n; ++i) {
        if (in[i] > mx) mx = in[i];
    }
    // 2) sum of exp(x - mx)
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += std::exp(in[i] - mx);
    }
    const float lse = std::log(sum) + mx;
    // 3) log-softmax = x - lse
    for (int i = 0; i < n; ++i) {
        out_logsm[i] = in[i] - lse;
    }
}

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

void run_one_cycle(const Tensor& in, std::vector<float>& out) {
#pragma omp parallel for schedule(static) num_threads(kNumWorkers)
    for (int b = 0; b < kBatch; ++b) {
        const float* row_in = in.data.data() + static_cast<size_t>(b) * kVocab;
        float* row_out = out.data() + static_cast<size_t>(b) * kVocab;
        softmax_logsoftmax_row(row_in, row_out, kVocab);
    }
}

}  // namespace

int main(int argc, char** argv) {
    (void)argc; (void)argv;

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    // Pin OMP threads
    omp_set_num_threads(kNumWorkers);
    pin_omp_threads_to_cores();

    Tensor input;
    std::vector<float> output(static_cast<size_t>(kBatch) * kVocab);

    std::fprintf(stderr,
                 "[cellA_reg_10ms] start workers=%d cores=%d-%d batch=%d vocab=%d cycle=%dms pid=%d\n",
                 kNumWorkers, kCpuBase, kCpuBase + kNumWorkers - 1,
                 kBatch, kVocab, kCycleMs, getpid());
    std::fflush(stderr);

    using clock = std::chrono::steady_clock;
    while (!g_stop.load()) {
        const auto t0 = clock::now();
        run_one_cycle(input, output);
        const auto t1 = clock::now();

        const uint64_t elapsed_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        g_total_ns.fetch_add(elapsed_ns);
        const uint64_t c = g_cycles.fetch_add(1) + 1;

        // checksum to prevent optimization out
        volatile float sink = output[0] + output[kVocab - 1];
        (void)sink;

        if (c % 50 == 0) {
            const double avg_ms = (static_cast<double>(g_total_ns.load()) / 1e6) / static_cast<double>(c);
            std::fprintf(stderr, "[cellA_reg_10ms] cycles=%lu avg=%.2f ms/cycle\n",
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
    std::fprintf(stderr, "[cellA_reg_10ms] stop cycles=%lu avg=%.2f ms/cycle\n",
                 static_cast<unsigned long>(cyc), avg_ms);
    return 0;
}
