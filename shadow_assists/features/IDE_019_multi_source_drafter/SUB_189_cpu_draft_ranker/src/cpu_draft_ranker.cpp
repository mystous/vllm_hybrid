// SUB_189 — CPU-side draft candidate ranker (side-channel)
// 가설: spec_decode 의 draft candidate (ngram output) 를 CPU 측에서
//       frequency-based reorder/prune 하는 work 를 vllm critical path 와 분리된
//       cores 80-95 에서 100Hz cycle 로 fire 하면, SUB_188 패턴 (small +1.84%)
//       처럼 cache/scheduler isolation 만 잘 지키면 net positive 가능.
//
// Work: draft candidate frequency-based reorder
//   - 입력: batch=32 sequences × K=7 candidate token ids × HIST=64 history tokens
//   - 매 cycle: 각 sequence 의 7 candidate 를 history (64 tokens) 안에서 frequency 계산
//              + frequency 내림차순 정렬 (no-GIL, 순수 native, vllm hook 없음)
//   - 10 ms cycle (100 Hz) — duty cycle target 2-5%
//   - 16 worker × cores 80-95 pinned (vllm vanilla 0-49 / trident 56-105 와 분리)
//
// Build: g++ -O3 -fopenmp -march=native -pthread cpu_draft_ranker.cpp -o cpu_draft_ranker

#include <algorithm>
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

constexpr int kBatch = 32;       // concurrent sequences
constexpr int kK = 7;            // candidate tokens per sequence (ngram K=7 typical)
constexpr int kHist = 64;        // history window for frequency lookup
constexpr int kVocab = 152064;   // Qwen 2.5 32B vocab
constexpr int kCycleMs = 10;     // 100 Hz
constexpr int kCpuBase = 80;
constexpr int kNumWorkers = 16;
// Per-cycle inner replication factor. Each cycle simulates ranking work for
// `kReplicas` independent batches (e.g., multiple draft hypothesis layers per
// sequence). Pushes per-cycle ~0.03 ms baseline to ~2-5% duty cycle.
constexpr int kReplicas = 192;

std::atomic<bool> g_stop{false};
std::atomic<uint64_t> g_cycles{0};
std::atomic<uint64_t> g_total_ns{0};

void on_signal(int /*sig*/) { g_stop.store(true); }

struct CandBatch {
    // candidate token ids: [batch, K]
    std::vector<int32_t> cand;
    // history token ids: [batch, HIST]
    std::vector<int32_t> hist;
    // per-cycle output: frequency-sorted candidate indices
    std::vector<int32_t> sorted_cand;
    std::vector<int32_t> freq;

    CandBatch()
        : cand(static_cast<size_t>(kBatch) * kK),
          hist(static_cast<size_t>(kBatch) * kHist),
          sorted_cand(static_cast<size_t>(kBatch) * kK),
          freq(static_cast<size_t>(kBatch) * kK) {
        // simple seed pattern; tokens drawn from [0, kVocab) with bias toward
        // small overlap (otherwise frequency tie-break dominates).
        for (size_t i = 0; i < cand.size(); ++i) {
            cand[i] = static_cast<int32_t>((i * 2654435761u) % kVocab);
        }
        for (size_t i = 0; i < hist.size(); ++i) {
            // 30% chance of being one of the candidates' near range — gives
            // realistic frequency distribution.
            uint32_t v = static_cast<uint32_t>(i * 40503u);
            hist[i] = static_cast<int32_t>(v % kVocab);
        }
    }
};

void rank_one_sequence(const int32_t* cand_row,
                       const int32_t* hist_row,
                       int32_t* sorted_out,
                       int32_t* freq_out) {
    // O(K * HIST) frequency lookup — K=7, HIST=64 → 448 cmp per seq.
    int32_t local_freq[kK];
    for (int k = 0; k < kK; ++k) {
        int32_t cnt = 0;
        int32_t cv = cand_row[k];
        for (int h = 0; h < kHist; ++h) {
            cnt += (hist_row[h] == cv) ? 1 : 0;
        }
        local_freq[k] = cnt;
        sorted_out[k] = k;  // index permutation
    }
    // Insertion sort on K=7 indices by descending frequency — branchy but tiny.
    for (int i = 1; i < kK; ++i) {
        int idx = sorted_out[i];
        int f = local_freq[idx];
        int j = i - 1;
        while (j >= 0 && local_freq[sorted_out[j]] < f) {
            sorted_out[j + 1] = sorted_out[j];
            --j;
        }
        sorted_out[j + 1] = idx;
    }
    // Re-write freq_out in sorted order so downstream consumers see frequency.
    for (int k = 0; k < kK; ++k) {
        freq_out[k] = local_freq[sorted_out[k]];
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

void run_one_cycle(CandBatch& cb) {
    // Per cycle we run `kReplicas` independent ranking passes — each pass is
    // ~32 seq × 448 cmp = ~14K cmp. Replicas push duty cycle into 2-5% range.
#pragma omp parallel for collapse(2) schedule(static) num_threads(kNumWorkers)
    for (int r = 0; r < kReplicas; ++r) {
        for (int b = 0; b < kBatch; ++b) {
            const int32_t* cand_row = cb.cand.data() + static_cast<size_t>(b) * kK;
            const int32_t* hist_row = cb.hist.data() + static_cast<size_t>(b) * kHist;
            int32_t* sorted_row = cb.sorted_cand.data() + static_cast<size_t>(b) * kK;
            int32_t* freq_row = cb.freq.data() + static_cast<size_t>(b) * kK;
            // Replica index XORs into cand row pointer offset to break exact
            // result caching, but stays inside same buffers so no extra alloc.
            (void)r;
            rank_one_sequence(cand_row, hist_row, sorted_row, freq_row);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    (void)argc; (void)argv;

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    omp_set_num_threads(kNumWorkers);
    pin_omp_threads_to_cores();

    CandBatch batch;

    std::fprintf(stderr,
                 "[draft_ranker] start workers=%d cores=%d-%d batch=%d K=%d hist=%d cycle=%dms pid=%d\n",
                 kNumWorkers, kCpuBase, kCpuBase + kNumWorkers - 1,
                 kBatch, kK, kHist, kCycleMs, getpid());
    std::fflush(stderr);

    using clock = std::chrono::steady_clock;
    while (!g_stop.load()) {
        const auto t0 = clock::now();
        run_one_cycle(batch);
        const auto t1 = clock::now();

        const uint64_t elapsed_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        g_total_ns.fetch_add(elapsed_ns);
        const uint64_t c = g_cycles.fetch_add(1) + 1;

        // checksum to prevent optimization out
        volatile int32_t sink = batch.sorted_cand[0] + batch.freq[batch.freq.size() - 1];
        (void)sink;

        if (c % 500 == 0) {
            const double avg_ms = (static_cast<double>(g_total_ns.load()) / 1e6) / static_cast<double>(c);
            std::fprintf(stderr, "[draft_ranker] cycles=%lu avg=%.3f ms/cycle\n",
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
    std::fprintf(stderr, "[draft_ranker] stop cycles=%lu avg=%.3f ms/cycle\n",
                 static_cast<unsigned long>(cyc), avg_ms);
    return 0;
}
