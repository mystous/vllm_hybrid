// SUB_190 — async tokenizer worker pool (side-channel)
// 가설: 다음 turn 의 chat template + system prompt 를 미리 batch tokenize 해두는
//       work 를 vllm critical path 와 분리된 cores 80-95 에서 50Hz cycle 로 fire 하면,
//       SUB_188 패턴 (small +1.84%) 처럼 cache/scheduler isolation 만 잘 지키면
//       net positive 가능.
//
// Work: 시뮬레이션된 tokenize encode pass
//   - 입력: batch=4 chat 문장 (avg 100 char) × HF BPE-style lookup table reuse
//   - 매 cycle: 4 문장 × 100 byte → token-id 시퀀스 변환 + lookup table hash
//   - 20 ms cycle (50 Hz) — duty cycle target 2-5%
//   - 16 worker × cores 80-95 pinned (vllm vanilla 0-49 / trident 56-105 와 분리)
//
// 실제 HF tokenizer 와의 차이: 본 worker 는 *진짜 모델 vocab 에 inject 하지 않음*
//   — pure side-channel work (CPU cycles 만 소모). vllm 의 tokenizer 와 무관.
//
// Build: g++ -O3 -fopenmp -march=native -pthread async_tokenizer_worker.cpp -o async_tokenizer_worker

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

constexpr int kBatch = 4;             // chat template batch size (next-turn lookahead)
constexpr int kStrLen = 100;          // avg char per sentence
constexpr int kVocabBits = 17;        // 128K-style lookup table (HF tokenizer)
constexpr int kVocabMask = (1 << kVocabBits) - 1;
constexpr int kCycleMs = 20;          // 50 Hz
constexpr int kCpuBase = 80;
constexpr int kNumWorkers = 16;
// Per-cycle inner replication: simulate multi-sentence × multi-template passes.
// Pushes per-cycle work to ~2-5% duty cycle in 20ms window.
constexpr int kReplicas = 4096;

std::atomic<bool> g_stop{false};
std::atomic<uint64_t> g_cycles{0};
std::atomic<uint64_t> g_total_ns{0};

void on_signal(int /*sig*/) { g_stop.store(true); }

struct TokBatch {
    // Input: batch × StrLen bytes (ASCII-ish characters)
    std::vector<uint8_t> in;
    // Reusable lookup table (BPE-style hash → token id), 128K entries
    std::vector<int32_t> vocab_table;
    // Output: batch × StrLen token-ids (worst case = one per char)
    std::vector<int32_t> tok_ids;

    TokBatch()
        : in(static_cast<size_t>(kBatch) * kStrLen),
          vocab_table(static_cast<size_t>(1u << kVocabBits)),
          tok_ids(static_cast<size_t>(kBatch) * kStrLen) {
        // Seed input bytes (deterministic, but byte stream resembles natural text)
        for (size_t i = 0; i < in.size(); ++i) {
            uint32_t v = static_cast<uint32_t>(i * 2654435761u);
            in[i] = static_cast<uint8_t>(32 + (v % 95));  // printable ASCII
        }
        // Seed vocab table (lookup chain)
        for (size_t i = 0; i < vocab_table.size(); ++i) {
            vocab_table[i] = static_cast<int32_t>((i * 40503u) & kVocabMask);
        }
    }
};

// BPE-style tokenize: rolling 3-byte hash → vocab lookup → emit token id.
// One sentence = StrLen bytes, emits up to StrLen tokens (one per starting position).
inline void tokenize_sentence(const uint8_t* in, int32_t* out,
                              const int32_t* vocab) {
    int32_t acc = 0;
    for (int i = 0; i < kStrLen; ++i) {
        // 3-byte rolling hash (with wrap at boundaries)
        uint32_t h = static_cast<uint32_t>(in[i]) * 2654435761u;
        if (i >= 1) h ^= static_cast<uint32_t>(in[i - 1]) * 40503u;
        if (i >= 2) h ^= static_cast<uint32_t>(in[i - 2]) * 19349663u;
        const int32_t tok = vocab[h & kVocabMask];
        out[i] = tok;
        acc ^= tok;
    }
    // Prevent optimization removal
    out[0] ^= acc & 0;  // identity
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

void run_one_cycle(TokBatch& tb) {
    // Per cycle we run `kReplicas` independent tokenize passes (e.g., multiple
    // chat templates × multi-turn lookahead). Replicas keep duty cycle in 2-5%.
#pragma omp parallel for collapse(2) schedule(static) num_threads(kNumWorkers)
    for (int r = 0; r < kReplicas; ++r) {
        for (int b = 0; b < kBatch; ++b) {
            const uint8_t* row_in = tb.in.data() + static_cast<size_t>(b) * kStrLen;
            int32_t* row_out = tb.tok_ids.data() + static_cast<size_t>(b) * kStrLen;
            (void)r;
            tokenize_sentence(row_in, row_out, tb.vocab_table.data());
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

    TokBatch batch;

    std::fprintf(stderr,
                 "[tokenizer] start workers=%d cores=%d-%d batch=%d strlen=%d vocab=%d cycle=%dms pid=%d\n",
                 kNumWorkers, kCpuBase, kCpuBase + kNumWorkers - 1,
                 kBatch, kStrLen, 1 << kVocabBits, kCycleMs, getpid());
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

        // checksum to prevent optimization removal
        volatile int32_t sink = batch.tok_ids[0] + batch.tok_ids[batch.tok_ids.size() - 1];
        (void)sink;

        if (c % 250 == 0) {
            const double avg_ms = (static_cast<double>(g_total_ns.load()) / 1e6) / static_cast<double>(c);
            std::fprintf(stderr, "[tokenizer] cycles=%lu avg=%.3f ms/cycle\n",
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
    std::fprintf(stderr, "[tokenizer] stop cycles=%lu avg=%.3f ms/cycle\n",
                 static_cast<unsigned long>(cyc), avg_ms);
    return 0;
}
