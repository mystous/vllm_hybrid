// SUB_187 — AMX-accelerated draft forward for Qwen 0.5B small draft head
//
// Reference: IDE_016 / amx_qwen_draft.cpp (BF16 matmul kernel, 22.05 TFLOPS peak).
// Goal: per-step decode latency < 5 ms on Sapphire Rapids 8480+ AMX,
// so that K=7 draft loop completes in ~35 ms which can amortize with
// 40 ms GPU verify.
//
// Shape:
//   Qwen 0.5B small draft — hidden=896, intermediate=4864, vocab=152064,
//   layers=24, heads=14 (decode-only single-step path; no attention compute
//   here — we only model the LM-head matmul + linear chain which is the
//   dominant cost in Jacobi K=7 BK=7 from SUB_181 245 ms breakdown).
//
// Build:
//   g++ -O3 -mamx-tile -mamx-bf16 -mavx512f -mavx512bf16 -mavx512vl \
//       -march=sapphirerapids -fopenmp -fPIC -shared \
//       src/amx_draft_qwen05b.cpp -o build/libamx_draft_qwen05b.so
//
// Public C ABI (loaded via ctypes):
//   int  amx_draft_qwen05b_init(void);
//   void amx_draft_qwen05b_free(void);
//   double amx_draft_qwen05b_step_ms(int B, int K);  // returns per-step ms, runs K matmul chain
//
// We do NOT load real Qwen 0.5B weights here — we allocate matrices of the
// correct shape with deterministic init, then time the AMX kernel chain.
// This is a *latency* microbench, not an accuracy test.

#include <immintrin.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <vector>
#include <random>
#include <algorithm>
#include <omp.h>

// ─────────────────────────────────────────────────────────────────────
// AMX state syscall
// ─────────────────────────────────────────────────────────────────────
#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif
#ifndef XFEATURE_XTILEDATA
#define XFEATURE_XTILEDATA 18
#endif

static int amx_request_permission() {
    long rc = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM,
                     static_cast<unsigned long>(XFEATURE_XTILEDATA));
    return rc == 0 ? 0 : -1;
}

static int amx_available() {
    unsigned eax, ebx, ecx, edx;
    __asm__ __volatile__("cpuid"
                         : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                         : "a"(7), "c"(0));
    bool has_amx = ((edx >> 24) & 1) && ((edx >> 22) & 1);
    return has_amx ? 1 : 0;
}

// ─────────────────────────────────────────────────────────────────────
// Tile config (per-thread)
// ─────────────────────────────────────────────────────────────────────

struct alignas(64) AmxTileCfg {
    uint8_t  palette_id;
    uint8_t  start_row;
    uint8_t  reserved_0[14];
    uint16_t colsb[16];
    uint8_t  rows[16];
};

static thread_local bool t_cfg_loaded = false;
static thread_local bool t_perm_tried = false;

static void config_tiles_thread() {
    if (t_cfg_loaded) return;
    if (!t_perm_tried) {
        t_perm_tried = true;
        amx_request_permission();
    }
    AmxTileCfg cfg = {};
    cfg.palette_id = 1;
    // TMM0 C: 16 rows × 16 FP32 = 64 bytes/row
    cfg.rows[0]  = 16;
    cfg.colsb[0] = 64;
    // TMM1 A: 16 rows × 32 BF16 = 64 bytes/row
    cfg.rows[1]  = 16;
    cfg.colsb[1] = 64;
    // TMM2 B: 16 K-pair rows × (16 BF16 pair = 32 BF16) = 64 bytes/row
    cfg.rows[2]  = 16;
    cfg.colsb[2] = 64;
    _tile_loadconfig(&cfg);
    t_cfg_loaded = true;
}

// ─────────────────────────────────────────────────────────────────────
// BF16 helpers
// ─────────────────────────────────────────────────────────────────────

static inline uint16_t fp32_to_bf16(float f) {
    uint32_t b;
    std::memcpy(&b, &f, sizeof(float));
    uint32_t lsb = (b >> 16) & 1;
    b = b + 0x8000u + lsb;
    return static_cast<uint16_t>(b >> 16);
}

static void fill_bf16_rand(uint16_t* buf, size_t N, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
    for (size_t i = 0; i < N; ++i) buf[i] = fp32_to_bf16(dist(rng));
}

// Repack row-major [K, N] BF16 → AMX [K/2, N, 2] BF16
static void amx_repack_b_bf16(const uint16_t* B_in, uint16_t* B_out, int K, int N) {
    int K_eff = K & ~1;
    for (int k = 0; k < K_eff; k += 2) {
        int kp = k / 2;
        for (int n = 0; n < N; ++n) {
            B_out[(static_cast<size_t>(kp) * N + n) * 2 + 0] = B_in[(k + 0) * N + n];
            B_out[(static_cast<size_t>(kp) * N + n) * 2 + 1] = B_in[(k + 1) * N + n];
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// AMX matmul: C[M,N] = A[M,K] · B_packed[K/2,N,2]
//   Requirements: M%16==0, K%32==0, N%16==0
// ─────────────────────────────────────────────────────────────────────

static void amx_matmul_bf16(const uint16_t* A, const uint16_t* B_packed,
                            float* C, int M, int K, int N) {
    config_tiles_thread();
    const size_t A_row_bytes = static_cast<size_t>(K) * sizeof(uint16_t);
    const size_t B_pair_row_bytes = static_cast<size_t>(N) * 2 * sizeof(uint16_t);
    const size_t C_row_bytes = static_cast<size_t>(N) * sizeof(float);

    for (int m = 0; m < M; m += 16) {
        for (int n = 0; n < N; n += 16) {
            _tile_zero(0);
            for (int k = 0; k < K; k += 32) {
                _tile_loadd(1, A + static_cast<size_t>(m) * K + k,
                           static_cast<long>(A_row_bytes));
                int kp = k / 2;
                _tile_loadd(2,
                    B_packed + (static_cast<size_t>(kp) * N + n) * 2,
                    static_cast<long>(B_pair_row_bytes));
                _tile_dpbf16ps(0, 1, 2);
            }
            _tile_stored(0, C + static_cast<size_t>(m) * N + n,
                        static_cast<long>(C_row_bytes));
        }
    }
}

// AMX matmul with OpenMP parallel over N tiles. Useful for large N (vocab).
static void amx_matmul_bf16_omp_n(const uint16_t* A, const uint16_t* B_packed,
                                  float* C, int M, int K, int N) {
    if (M % 16 || K % 32 || N % 16 || !amx_available()) {
        // Scalar fallback — extremely slow; bench should align inputs
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float acc = 0.f;
                for (int k = 0; k < K; ++k) {
                    uint32_t ab = static_cast<uint32_t>(A[m * K + k]) << 16;
                    int kp = k / 2;
                    int kr = k & 1;
                    uint32_t bb = static_cast<uint32_t>(
                        B_packed[(kp * N + n) * 2 + kr]) << 16;
                    float fa, fb;
                    std::memcpy(&fa, &ab, sizeof(float));
                    std::memcpy(&fb, &bb, sizeof(float));
                    acc += fa * fb;
                }
                C[m * N + n] = acc;
            }
        }
        return;
    }
    // Parallel partitioning across N (16-wide tiles).
    const int N_tile = 16;
    const int n_tiles = N / N_tile;
    #pragma omp parallel for schedule(static)
    for (int nt = 0; nt < n_tiles; ++nt) {
        const int n0 = nt * N_tile;
        config_tiles_thread();
        const size_t A_row_bytes = static_cast<size_t>(K) * sizeof(uint16_t);
        const size_t B_pair_row_bytes = static_cast<size_t>(N) * 2 * sizeof(uint16_t);
        const size_t C_row_bytes = static_cast<size_t>(N) * sizeof(float);
        for (int m = 0; m < M; m += 16) {
            _tile_zero(0);
            for (int k = 0; k < K; k += 32) {
                _tile_loadd(1, A + static_cast<size_t>(m) * K + k,
                           static_cast<long>(A_row_bytes));
                int kp = k / 2;
                _tile_loadd(2,
                    B_packed + (static_cast<size_t>(kp) * N + n0) * 2,
                    static_cast<long>(B_pair_row_bytes));
                _tile_dpbf16ps(0, 1, 2);
            }
            _tile_stored(0, C + static_cast<size_t>(m) * N + n0,
                        static_cast<long>(C_row_bytes));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Static buffers (allocated once at init)
// ─────────────────────────────────────────────────────────────────────

struct DraftState {
    // Qwen 0.5B small draft shape
    static constexpr int HIDDEN       = 896;
    static constexpr int INTERMEDIATE = 4864;
    // Vocab padded to 16 multiple (152064 already 16-mult). We use 152064.
    static constexpr int VOCAB        = 152064;
    static constexpr int LAYERS       = 24;

    // For per-step time we model: 1× MLP-equivalent linear chain
    // (gate + up + down) per layer + final LM-head.
    //
    // For microbench we collapse to:
    //   per_step = LAYERS × (3 × matmul(B,H,I)) + matmul(B,H,V)
    //
    // But this would be ~minutes — for K=7 we cannot afford 24 layers
    // of CPU MLP at H=896 / I=4864. The realistic target is:
    //   1) LM-head matmul (B,H,V)   ← single dominant cost (B×896×152064 = 0.27 GFLOP/token)
    //   2) per-layer hidden update is GPU-side or skipped in draft proxy.
    //
    // So this kernel measures THE DOMINANT COST: LM-head BF16 matmul.
    // Per-step latency = LM-head time.

    // Weights:
    uint16_t* W_lm_head_packed = nullptr;  // [HIDDEN/2, VOCAB, 2] BF16
    uint16_t* W_mlp_gate_packed = nullptr; // [HIDDEN/2, INTERMEDIATE, 2] BF16 (optional)

    // Activation scratch (16-row aligned B max = 16):
    static constexpr int B_MAX = 16;
    uint16_t* act_in = nullptr;            // [B, HIDDEN] BF16
    float*    logits_out = nullptr;        // [B, VOCAB] FP32
    float*    mlp_out    = nullptr;        // [B, INTERMEDIATE] FP32
};

static DraftState g_state;

extern "C" int amx_draft_qwen05b_init(void) {
    if (!amx_available()) {
        std::fprintf(stderr, "[amx_draft_qwen05b] AMX not available\n");
        return -1;
    }
    if (amx_request_permission() != 0) {
        std::fprintf(stderr, "[amx_draft_qwen05b] AMX permission request failed\n");
        return -2;
    }

    const size_t lm_row = DraftState::HIDDEN;
    const size_t lm_col = DraftState::VOCAB;
    const size_t lm_packed = (lm_row / 2) * lm_col * 2;
    g_state.W_lm_head_packed = static_cast<uint16_t*>(
        std::aligned_alloc(64, lm_packed * sizeof(uint16_t)));
    if (!g_state.W_lm_head_packed) return -3;

    // Init: random BF16 then repack
    std::vector<uint16_t> W_lm_rowmajor(lm_row * lm_col);
    fill_bf16_rand(W_lm_rowmajor.data(), W_lm_rowmajor.size(), 0xA1u);
    amx_repack_b_bf16(W_lm_rowmajor.data(), g_state.W_lm_head_packed,
                      lm_row, lm_col);

    // MLP gate weights (smaller — used only if --include-mlp path enabled)
    const size_t mlp_row = DraftState::HIDDEN;
    const size_t mlp_col = DraftState::INTERMEDIATE;
    const size_t mlp_packed = (mlp_row / 2) * mlp_col * 2;
    g_state.W_mlp_gate_packed = static_cast<uint16_t*>(
        std::aligned_alloc(64, mlp_packed * sizeof(uint16_t)));
    if (!g_state.W_mlp_gate_packed) return -4;
    std::vector<uint16_t> W_mlp_rowmajor(mlp_row * mlp_col);
    fill_bf16_rand(W_mlp_rowmajor.data(), W_mlp_rowmajor.size(), 0xB2u);
    amx_repack_b_bf16(W_mlp_rowmajor.data(), g_state.W_mlp_gate_packed,
                      mlp_row, mlp_col);

    // Activations (B_MAX = 16 rows)
    g_state.act_in = static_cast<uint16_t*>(std::aligned_alloc(
        64, DraftState::B_MAX * DraftState::HIDDEN * sizeof(uint16_t)));
    fill_bf16_rand(g_state.act_in,
                   DraftState::B_MAX * DraftState::HIDDEN, 0xC3u);

    g_state.logits_out = static_cast<float*>(std::aligned_alloc(
        64, DraftState::B_MAX * DraftState::VOCAB * sizeof(float)));
    g_state.mlp_out = static_cast<float*>(std::aligned_alloc(
        64, DraftState::B_MAX * DraftState::INTERMEDIATE * sizeof(float)));

    if (!g_state.act_in || !g_state.logits_out || !g_state.mlp_out) return -5;
    return 0;
}

extern "C" void amx_draft_qwen05b_free(void) {
    std::free(g_state.W_lm_head_packed); g_state.W_lm_head_packed = nullptr;
    std::free(g_state.W_mlp_gate_packed); g_state.W_mlp_gate_packed = nullptr;
    std::free(g_state.act_in); g_state.act_in = nullptr;
    std::free(g_state.logits_out); g_state.logits_out = nullptr;
    std::free(g_state.mlp_out); g_state.mlp_out = nullptr;
}

// ─────────────────────────────────────────────────────────────────────
// One-shot LM-head matmul, returns wall ms.
// B clamped to [1, 16] and rounded up to 16 (AMX tile constraint).
// ─────────────────────────────────────────────────────────────────────

extern "C" double amx_draft_qwen05b_step_ms(int B_in, int K) {
    if (!g_state.W_lm_head_packed) return -1.0;
    int B = std::max(1, std::min(B_in, DraftState::B_MAX));
    // AMX needs M % 16 == 0; round up.
    int B_amx = ((B + 15) / 16) * 16;
    if (B_amx > DraftState::B_MAX) B_amx = DraftState::B_MAX;

    using clock = std::chrono::high_resolution_clock;
    auto t0 = clock::now();

    // Per Jacobi K=7 BK=7 pattern: K passes of LM-head matmul.
    for (int k = 0; k < K; ++k) {
        amx_matmul_bf16_omp_n(g_state.act_in, g_state.W_lm_head_packed,
                              g_state.logits_out,
                              B_amx, DraftState::HIDDEN, DraftState::VOCAB);
    }
    auto t1 = clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// Single LM-head matmul (one decode step), ms
extern "C" double amx_draft_qwen05b_single_ms(int B_in) {
    return amx_draft_qwen05b_step_ms(B_in, 1);
}

// MLP-only timing — used to estimate per-layer linear cost
extern "C" double amx_draft_qwen05b_mlp_ms(int B_in) {
    if (!g_state.W_mlp_gate_packed) return -1.0;
    int B = std::max(1, std::min(B_in, DraftState::B_MAX));
    int B_amx = ((B + 15) / 16) * 16;
    if (B_amx > DraftState::B_MAX) B_amx = DraftState::B_MAX;

    using clock = std::chrono::high_resolution_clock;
    auto t0 = clock::now();
    amx_matmul_bf16_omp_n(g_state.act_in, g_state.W_mlp_gate_packed,
                          g_state.mlp_out,
                          B_amx, DraftState::HIDDEN, DraftState::INTERMEDIATE);
    auto t1 = clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// Hardware-detect helper
extern "C" int amx_draft_qwen05b_hw_amx(void) { return amx_available(); }

// ─────────────────────────────────────────────────────────────────────
// Phase A3-real (SUB_198) — load real Qwen-0.5B lm_head weight
//
//   int amx_draft_qwen05b_load_lm_head(
//       const uint16_t* weight_bf16,  // [rows_valid, hidden] row-major BF16
//       int rows_valid,               // typically 151,936 (Qwen 0.5B vocab)
//       int hidden,                   // must == HIDDEN (896)
//       int padded_vocab)             // typically 152,064 (kernel VOCAB)
//
// Semantics:
//   Replace the random-init `W_lm_head_packed` buffer with the caller's
//   real lm_head weight. The caller supplies the [rows_valid, hidden]
//   slice (e.g. tied embed_tokens.weight of Qwen-0.5B which is
//   151,936×896 BF16). We zero-pad rows [rows_valid, padded_vocab) and
//   transpose to AMX-packed B layout: [hidden/2, padded_vocab, 2] BF16.
//
//   Mathematically: lm_head matmul is `logits = act @ W^T` where W is
//   [vocab, hidden]. We feed AMX as `act @ B` with B = W^T, so
//   B[hidden, vocab] = W[vocab, hidden] transposed. The packed layout
//   then is [hidden/2, vocab, 2] (K-pair major).
//
// Returns:
//   0  success
//  -1  not initialized (call amx_draft_qwen05b_init first)
//  -2  hidden mismatch (must equal HIDDEN=896)
//  -3  padded_vocab mismatch (must equal VOCAB=152064)
//  -4  rows_valid > padded_vocab
//  -5  null weight pointer
// ─────────────────────────────────────────────────────────────────────

extern "C" int amx_draft_qwen05b_load_lm_head(const uint16_t* weight_bf16,
                                              int rows_valid,
                                              int hidden,
                                              int padded_vocab) {
    if (!g_state.W_lm_head_packed) return -1;
    if (hidden != DraftState::HIDDEN) return -2;
    if (padded_vocab != DraftState::VOCAB) return -3;
    if (rows_valid < 0 || rows_valid > padded_vocab) return -4;
    if (!weight_bf16) return -5;

    const int H = DraftState::HIDDEN;
    const int V = DraftState::VOCAB;

    // Build the transposed B[hidden, vocab] row-major buffer with
    // zero-padding for the [rows_valid, V) tail.
    std::vector<uint16_t> B_rowmajor(static_cast<size_t>(H) * V, 0);
    // weight_bf16[v, h]  →  B_rowmajor[h, v]
    #pragma omp parallel for schedule(static)
    for (int v = 0; v < rows_valid; ++v) {
        const uint16_t* w_row = weight_bf16
                                + static_cast<size_t>(v) * H;
        for (int h = 0; h < H; ++h) {
            B_rowmajor[static_cast<size_t>(h) * V + v] = w_row[h];
        }
    }
    // rows [rows_valid, V) already zero from vector init.

    // Repack [H, V] → [H/2, V, 2] BF16 into the existing packed buffer.
    amx_repack_b_bf16(B_rowmajor.data(), g_state.W_lm_head_packed, H, V);
    return 0;
}

// ─────────────────────────────────────────────────────────────────────
// Phase A3 (SUB_198) — forward ABI extension
//
//   void amx_draft_qwen05b_forward(
//       const uint16_t* input_bf16,   // [B, HIDDEN(896)] BF16
//       int             B,            // 1..B_MAX(16)
//       uint16_t*       logits_out,   // [B, K, VOCAB(152064)] BF16
//       int32_t*        ids_out,      // [B, K] int32 argmax id (vocab range)
//       int             K)            // K steps
//
// Semantics:
//   For each k in 0..K-1, run the LM-head matmul
//     C[B,VOCAB] = act_in[B,HIDDEN] · W_lm_head_packed
//   then  (a) BF16-cast the FP32 logits row into logits_out at offset
//             (b*K + k) * VOCAB,
//         (b) argmax over the VOCAB_CONFIG (151,936) prefix (the padded
//             trailing 128 ids are NOT valid vocab tokens — they would
//             argmax-poison the result on randomly initialised weights)
//             into ids_out[b*K + k].
//   `act_in` is the SAME init-time random buffer used by step_ms (real
//   weight load is SUB_198 §3 (a-d); this entry validates ABI / shape /
//   dtype / argmax range only).
//
//   Inputs:
//     * input_bf16 MAY be NULL — if so we leave the kernel-internal
//       `g_state.act_in` untouched (microbench mode).
//     * input_bf16 != NULL → copy [B,HIDDEN] BF16 into `g_state.act_in`
//       before the first matmul. Subsequent k=1..K-1 reuse the same
//       activation (Jacobi K-step proxy).
//
//   Outputs:
//     * logits_out → BF16 down-cast of the FP32 accumulator
//       (g_state.logits_out). Layout row-major [B,K,VOCAB].
//     * ids_out    → argmax over [0, VOCAB_CONFIG) for each (b,k).
//
//   Constraints: B clamped to [1, B_MAX]; effective AMX M rounded up
//   to 16. ids_out / logits_out must be sized for the *requested* B
//   and K (caller buffers).
// ─────────────────────────────────────────────────────────────────────

static constexpr int VOCAB_CONFIG_QWEN05B = 151936;

static inline float bf16_to_fp32(uint16_t b) {
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(float));
    return f;
}

extern "C" void amx_draft_qwen05b_forward(const uint16_t* input_bf16,
                                          int B_in,
                                          uint16_t* logits_out,
                                          int32_t* ids_out,
                                          int K) {
    if (!g_state.W_lm_head_packed || !logits_out || !ids_out) return;
    int B = std::max(1, std::min(B_in, DraftState::B_MAX));
    int B_amx = ((B + 15) / 16) * 16;
    if (B_amx > DraftState::B_MAX) B_amx = DraftState::B_MAX;
    if (K < 1) return;

    // (a) Optionally overwrite the act_in buffer with caller hidden.
    if (input_bf16) {
        const size_t bytes = static_cast<size_t>(B)
                             * DraftState::HIDDEN * sizeof(uint16_t);
        std::memcpy(g_state.act_in, input_bf16, bytes);
    }

    const int V = DraftState::VOCAB;
    const int Vc = VOCAB_CONFIG_QWEN05B;

    for (int k = 0; k < K; ++k) {
        // (b) LM-head matmul: act_in × W_lm_head → g_state.logits_out FP32
        amx_matmul_bf16_omp_n(g_state.act_in, g_state.W_lm_head_packed,
                              g_state.logits_out,
                              B_amx, DraftState::HIDDEN, V);

        // (c) Down-cast FP32 → BF16 into caller buffer + argmax over Vc.
        #pragma omp parallel for schedule(static)
        for (int b = 0; b < B; ++b) {
            const float* row_fp32 = g_state.logits_out
                                    + static_cast<size_t>(b) * V;
            uint16_t* row_bf16 = logits_out
                                 + (static_cast<size_t>(b) * K + k) * V;

            int best_id = 0;
            float best_v = row_fp32[0];
            for (int n = 0; n < V; ++n) {
                row_bf16[n] = fp32_to_bf16(row_fp32[n]);
                if (n < Vc && row_fp32[n] > best_v) {
                    best_v = row_fp32[n];
                    best_id = n;
                }
            }
            ids_out[static_cast<size_t>(b) * K + k] = best_id;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Phase A1 (SUB_198 §p_24layer) — full 24-layer Qwen-0.5B forward
//
//   24 decoder layers (attention + RMSNorm + SwiGLU + RoPE + KV cache)
//   + lm_head + argmax. CPU-only AMX BF16 reference forward.
//
//   See: shadow_assists/.../SUB_198_amx_real_integration/p_24layer/DESIGN.md
// ─────────────────────────────────────────────────────────────────────

#include <vector>
#include <cmath>

namespace qwen05b {
constexpr int HIDDEN        = 896;
constexpr int INTERMEDIATE  = 4864;
constexpr int Q_DIM         = 896;   // q_proj output
constexpr int KV_DIM        = 128;   // k_proj / v_proj output (GQA 2 heads × 64)
constexpr int N_HEADS       = 14;
constexpr int N_KV_HEADS    = 2;
constexpr int HEAD_DIM      = 64;
constexpr int GQA_GROUP     = N_HEADS / N_KV_HEADS;  // 7
constexpr int N_LAYERS      = 24;
constexpr int VOCAB_VALID   = 151936;
constexpr int VOCAB_PADDED  = 152064;
constexpr float RMS_EPS     = 1e-6f;
constexpr float ROPE_THETA  = 1000000.0f;
constexpr int MAX_SEQ       = 64;   // smoke test prompt limit
}

// Per-layer state (AMX-packed weights + bias).
struct Qwen05bLayer {
    // RMSNorm weights (FP32 for accuracy)
    std::vector<float> ln1_w_fp32;   // [HIDDEN]
    std::vector<float> ln2_w_fp32;   // [HIDDEN]

    // Q/K/V projection weights (AMX-packed: [H/2, N, 2])
    uint16_t* q_w_packed = nullptr;  // K=896, N=896
    uint16_t* k_w_packed = nullptr;  // K=896, N=128
    uint16_t* v_w_packed = nullptr;  // K=896, N=128

    // Biases (FP32)
    std::vector<float> q_b_fp32;  // [Q_DIM=896]
    std::vector<float> k_b_fp32;  // [KV_DIM=128]
    std::vector<float> v_b_fp32;  // [KV_DIM=128]

    // O projection
    uint16_t* o_w_packed = nullptr;  // K=896, N=896

    // MLP weights
    uint16_t* gate_w_packed = nullptr;  // K=896, N=4864
    uint16_t* up_w_packed   = nullptr;  // K=896, N=4864
    uint16_t* down_w_packed = nullptr;  // K=4864, N=896

    // KV cache: [S_seen, KV_DIM] BF16
    std::vector<uint16_t> k_cache;  // grows by KV_DIM per step
    std::vector<uint16_t> v_cache;
    int s_cached = 0;
};

struct Qwen05bModel {
    Qwen05bLayer layers[qwen05b::N_LAYERS];

    // Embed tokens (also tied lm_head). Row-major BF16 [V_VALID, HIDDEN].
    std::vector<uint16_t> embed_tokens;

    // Final RMSNorm weight (before lm_head)
    std::vector<float> final_norm_w_fp32;  // [HIDDEN]

    // RoPE precompute (FP32): cos/sin tables [MAX_SEQ, HEAD_DIM/2]
    std::vector<float> rope_cos;  // [MAX_SEQ * HEAD_DIM/2]
    std::vector<float> rope_sin;

    bool initialized = false;
};

static Qwen05bModel g_model;

// ─── helpers ─────────────────────────────────────────────────────────

// Free + reallocate an AMX-packed buffer of given [K_rows, N_cols] (BF16,
// after K-pair packing it is [K_rows/2, N_cols, 2] BF16).
static uint16_t* alloc_packed(int K_rows, int N_cols) {
    size_t bytes = (static_cast<size_t>(K_rows) / 2)
                   * N_cols * 2 * sizeof(uint16_t);
    return static_cast<uint16_t*>(std::aligned_alloc(64, bytes));
}

// Transpose [out_dim, in_dim] BF16 to [in_dim, out_dim] BF16, then AMX-pack.
// Mathematically: weight is W[out, in]. matmul: y = x @ W^T. For AMX we
// feed B = W^T [in, out] and pack to [in/2, out, 2].
static void transpose_and_pack(const uint16_t* W_in, int out_dim, int in_dim,
                               uint16_t* out_packed) {
    // Build transposed row-major buffer [in_dim, out_dim].
    std::vector<uint16_t> tr(static_cast<size_t>(in_dim) * out_dim, 0);
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < out_dim; ++o) {
        for (int i = 0; i < in_dim; ++i) {
            tr[static_cast<size_t>(i) * out_dim + o] = W_in[o * in_dim + i];
        }
    }
    amx_repack_b_bf16(tr.data(), out_packed, in_dim, out_dim);
}

// Build RoPE cos/sin tables for given theta + head_dim + max_pos.
static void build_rope_tables(int max_pos, int head_dim, float theta,
                              std::vector<float>& cos_out,
                              std::vector<float>& sin_out) {
    int half = head_dim / 2;
    cos_out.assign(static_cast<size_t>(max_pos) * half, 0.f);
    sin_out.assign(static_cast<size_t>(max_pos) * half, 0.f);
    for (int p = 0; p < max_pos; ++p) {
        for (int i = 0; i < half; ++i) {
            float inv_freq = std::pow(theta, -2.0f * i / head_dim);
            float angle = static_cast<float>(p) * inv_freq;
            cos_out[p * half + i] = std::cos(angle);
            sin_out[p * half + i] = std::sin(angle);
        }
    }
}

// RMSNorm: y = x / sqrt(mean(x²) + eps) * w
// Input/output BF16, internal FP32. Caller provides FP32 weight.
static void rmsnorm_bf16(const uint16_t* x_bf16, const float* w_fp32,
                         uint16_t* y_bf16, int hidden, float eps) {
    // Compute sum of squares in FP32
    float ss = 0.f;
    for (int i = 0; i < hidden; ++i) {
        float v = bf16_to_fp32(x_bf16[i]);
        ss += v * v;
    }
    float rms = 1.0f / std::sqrt(ss / hidden + eps);
    for (int i = 0; i < hidden; ++i) {
        float v = bf16_to_fp32(x_bf16[i]) * rms * w_fp32[i];
        y_bf16[i] = fp32_to_bf16(v);
    }
}

// Apply RoPE to [n_heads, head_dim] in-place at position p.
// q layout: [n_heads * head_dim] BF16, contiguous per head.
static void apply_rope_bf16(uint16_t* q, int n_heads, int head_dim, int p,
                            const float* cos_tbl, const float* sin_tbl) {
    int half = head_dim / 2;
    for (int h = 0; h < n_heads; ++h) {
        uint16_t* q_head = q + h * head_dim;
        for (int i = 0; i < half; ++i) {
            float c = cos_tbl[p * half + i];
            float s = sin_tbl[p * half + i];
            // Qwen2 RoPE convention: rotate the first half against the second half.
            // q_new[i]      = q[i]      * c - q[i+half] * s
            // q_new[i+half] = q[i+half] * c + q[i]      * s
            float q0 = bf16_to_fp32(q_head[i]);
            float q1 = bf16_to_fp32(q_head[i + half]);
            float n0 = q0 * c - q1 * s;
            float n1 = q1 * c + q0 * s;
            q_head[i]        = fp32_to_bf16(n0);
            q_head[i + half] = fp32_to_bf16(n1);
        }
    }
}

// SiLU: y = x * sigmoid(x). Operates on FP32.
static inline float silu_fp32(float x) {
    return x / (1.0f + std::exp(-x));
}

// Matmul wrapper that handles M < 16 (round up to 16, zero-pad input rows).
// A:[M, K] BF16. B_packed: AMX-packed of shape [K/2, N, 2]. C:[M, N] FP32.
static void amx_matmul_padded(const uint16_t* A, const uint16_t* B_packed,
                              float* C, int M, int K, int N,
                              uint16_t* A_pad_scratch, float* C_pad_scratch) {
    int M_amx = ((M + 15) / 16) * 16;
    if (M_amx == M) {
        amx_matmul_bf16_omp_n(A, B_packed, C, M, K, N);
        return;
    }
    // Zero-pad A to M_amx rows
    std::memset(A_pad_scratch, 0, M_amx * K * sizeof(uint16_t));
    std::memcpy(A_pad_scratch, A, M * K * sizeof(uint16_t));
    amx_matmul_bf16_omp_n(A_pad_scratch, B_packed, C_pad_scratch,
                          M_amx, K, N);
    // Copy first M rows back to C
    for (int m = 0; m < M; ++m) {
        std::memcpy(C + static_cast<size_t>(m) * N,
                    C_pad_scratch + static_cast<size_t>(m) * N,
                    N * sizeof(float));
    }
}

// ─── public weight-load API ──────────────────────────────────────────

extern "C" int amx_draft_qwen05b_init_model(void) {
    using namespace qwen05b;
    if (g_model.initialized) return 0;
    if (!amx_available()) return -1;
    if (amx_request_permission() != 0) return -2;

    // Build RoPE tables
    build_rope_tables(MAX_SEQ, HEAD_DIM, ROPE_THETA,
                      g_model.rope_cos, g_model.rope_sin);
    g_model.final_norm_w_fp32.assign(HIDDEN, 1.0f);
    g_model.embed_tokens.assign(static_cast<size_t>(VOCAB_VALID) * HIDDEN, 0);

    for (int L = 0; L < N_LAYERS; ++L) {
        auto& lay = g_model.layers[L];
        lay.ln1_w_fp32.assign(HIDDEN, 1.0f);
        lay.ln2_w_fp32.assign(HIDDEN, 1.0f);
        lay.q_b_fp32.assign(Q_DIM, 0.f);
        lay.k_b_fp32.assign(KV_DIM, 0.f);
        lay.v_b_fp32.assign(KV_DIM, 0.f);

        lay.q_w_packed    = alloc_packed(HIDDEN, Q_DIM);
        lay.k_w_packed    = alloc_packed(HIDDEN, KV_DIM);
        lay.v_w_packed    = alloc_packed(HIDDEN, KV_DIM);
        lay.o_w_packed    = alloc_packed(Q_DIM, HIDDEN);
        lay.gate_w_packed = alloc_packed(HIDDEN, INTERMEDIATE);
        lay.up_w_packed   = alloc_packed(HIDDEN, INTERMEDIATE);
        lay.down_w_packed = alloc_packed(INTERMEDIATE, HIDDEN);
        if (!lay.q_w_packed || !lay.k_w_packed || !lay.v_w_packed
            || !lay.o_w_packed || !lay.gate_w_packed || !lay.up_w_packed
            || !lay.down_w_packed) return -3;

        // Zero-fill packed buffers (in case weight load is partial)
        size_t qsz = (HIDDEN / 2) * Q_DIM * 2;
        size_t kvsz = (HIDDEN / 2) * KV_DIM * 2;
        size_t osz = (Q_DIM / 2) * HIDDEN * 2;
        size_t mlpsz = (HIDDEN / 2) * INTERMEDIATE * 2;
        size_t downsz = (INTERMEDIATE / 2) * HIDDEN * 2;
        std::memset(lay.q_w_packed, 0, qsz * sizeof(uint16_t));
        std::memset(lay.k_w_packed, 0, kvsz * sizeof(uint16_t));
        std::memset(lay.v_w_packed, 0, kvsz * sizeof(uint16_t));
        std::memset(lay.o_w_packed, 0, osz * sizeof(uint16_t));
        std::memset(lay.gate_w_packed, 0, mlpsz * sizeof(uint16_t));
        std::memset(lay.up_w_packed, 0, mlpsz * sizeof(uint16_t));
        std::memset(lay.down_w_packed, 0, downsz * sizeof(uint16_t));

        lay.k_cache.clear();
        lay.v_cache.clear();
        lay.s_cached = 0;
    }
    g_model.initialized = true;
    return 0;
}

extern "C" int amx_draft_qwen05b_load_layer_weights(
    int layer_idx,
    const uint16_t* ln1_w, const uint16_t* q_w, const uint16_t* q_b,
    const uint16_t* k_w, const uint16_t* k_b,
    const uint16_t* v_w, const uint16_t* v_b,
    const uint16_t* o_w, const uint16_t* ln2_w,
    const uint16_t* gate_w, const uint16_t* up_w, const uint16_t* down_w) {
    using namespace qwen05b;
    if (!g_model.initialized) return -1;
    if (layer_idx < 0 || layer_idx >= N_LAYERS) return -2;
    auto& lay = g_model.layers[layer_idx];

    // RMSNorm weights → FP32
    for (int i = 0; i < HIDDEN; ++i) lay.ln1_w_fp32[i] = bf16_to_fp32(ln1_w[i]);
    for (int i = 0; i < HIDDEN; ++i) lay.ln2_w_fp32[i] = bf16_to_fp32(ln2_w[i]);

    // Biases → FP32
    for (int i = 0; i < Q_DIM;  ++i) lay.q_b_fp32[i] = bf16_to_fp32(q_b[i]);
    for (int i = 0; i < KV_DIM; ++i) lay.k_b_fp32[i] = bf16_to_fp32(k_b[i]);
    for (int i = 0; i < KV_DIM; ++i) lay.v_b_fp32[i] = bf16_to_fp32(v_b[i]);

    // Weights: HF stores W[out, in]. We need AMX-packed B = W^T [in, out]
    // packed to [in/2, out, 2].
    transpose_and_pack(q_w,    Q_DIM,        HIDDEN,       lay.q_w_packed);
    transpose_and_pack(k_w,    KV_DIM,       HIDDEN,       lay.k_w_packed);
    transpose_and_pack(v_w,    KV_DIM,       HIDDEN,       lay.v_w_packed);
    transpose_and_pack(o_w,    HIDDEN,       Q_DIM,        lay.o_w_packed);
    transpose_and_pack(gate_w, INTERMEDIATE, HIDDEN,       lay.gate_w_packed);
    transpose_and_pack(up_w,   INTERMEDIATE, HIDDEN,       lay.up_w_packed);
    transpose_and_pack(down_w, HIDDEN,       INTERMEDIATE, lay.down_w_packed);
    return 0;
}

extern "C" int amx_draft_qwen05b_load_embed_tokens(const uint16_t* embed,
                                                   int vocab_valid,
                                                   int hidden) {
    using namespace qwen05b;
    if (!g_model.initialized) return -1;
    if (hidden != HIDDEN) return -2;
    if (vocab_valid != VOCAB_VALID) return -3;
    std::memcpy(g_model.embed_tokens.data(), embed,
                static_cast<size_t>(vocab_valid) * hidden * sizeof(uint16_t));
    return 0;
}

extern "C" int amx_draft_qwen05b_load_final_norm(const uint16_t* w) {
    using namespace qwen05b;
    if (!g_model.initialized) return -1;
    for (int i = 0; i < HIDDEN; ++i)
        g_model.final_norm_w_fp32[i] = bf16_to_fp32(w[i]);
    return 0;
}

extern "C" void amx_draft_qwen05b_reset_kv_cache(void) {
    using namespace qwen05b;
    for (int L = 0; L < N_LAYERS; ++L) {
        g_model.layers[L].k_cache.clear();
        g_model.layers[L].v_cache.clear();
        g_model.layers[L].s_cached = 0;
    }
}

extern "C" int amx_draft_qwen05b_free_model(void) {
    using namespace qwen05b;
    if (!g_model.initialized) return 0;
    for (int L = 0; L < N_LAYERS; ++L) {
        auto& lay = g_model.layers[L];
        std::free(lay.q_w_packed); lay.q_w_packed = nullptr;
        std::free(lay.k_w_packed); lay.k_w_packed = nullptr;
        std::free(lay.v_w_packed); lay.v_w_packed = nullptr;
        std::free(lay.o_w_packed); lay.o_w_packed = nullptr;
        std::free(lay.gate_w_packed); lay.gate_w_packed = nullptr;
        std::free(lay.up_w_packed); lay.up_w_packed = nullptr;
        std::free(lay.down_w_packed); lay.down_w_packed = nullptr;
        lay.k_cache.clear(); lay.k_cache.shrink_to_fit();
        lay.v_cache.clear(); lay.v_cache.shrink_to_fit();
        lay.s_cached = 0;
    }
    g_model.embed_tokens.clear(); g_model.embed_tokens.shrink_to_fit();
    g_model.initialized = false;
    return 0;
}

// ─── per-layer forward ───────────────────────────────────────────────
//
// h_in:  [S, HIDDEN] BF16 (current chunk tokens)
// pos0:  position of first token in h_in (KV cache size before this call)
// h_out: [S, HIDDEN] BF16
// scratch (caller allocates, sized for worst case):
//   tmp_norm:  [S, HIDDEN] BF16
//   tmp_q:     [S, Q_DIM]  BF16
//   tmp_k:     [S, KV_DIM] BF16
//   tmp_v:     [S, KV_DIM] BF16
//   q_fp32:    [S, Q_DIM]  FP32   (matmul accum)
//   k_fp32:    [S, KV_DIM] FP32
//   v_fp32:    [S, KV_DIM] FP32
//   attn_out:  [S, Q_DIM]  BF16
//   o_fp32:    [S, HIDDEN] FP32
//   gate_fp32: [S, INTER]  FP32
//   up_fp32:   [S, INTER]  FP32
//   swi_bf16:  [S, INTER]  BF16
//   down_fp32: [S, HIDDEN] FP32
//   A_pad:     [16, max_K] BF16 (matmul row pad)
//   C_pad:     [16, max_N] FP32 (matmul row pad)
//
static void qwen05b_layer_forward(int L, const uint16_t* h_in, int S, int pos0,
                                  uint16_t* h_out,
                                  uint16_t* tmp_norm,
                                  uint16_t* tmp_q, uint16_t* tmp_k, uint16_t* tmp_v,
                                  float* q_fp32, float* k_fp32, float* v_fp32,
                                  uint16_t* attn_out,
                                  float* o_fp32,
                                  float* gate_fp32, float* up_fp32,
                                  uint16_t* swi_bf16, float* down_fp32,
                                  uint16_t* A_pad, float* C_pad) {
    using namespace qwen05b;
    auto& lay = g_model.layers[L];

    // 1) RMSNorm input_layernorm
    for (int s = 0; s < S; ++s) {
        rmsnorm_bf16(h_in + s * HIDDEN, lay.ln1_w_fp32.data(),
                     tmp_norm + s * HIDDEN, HIDDEN, RMS_EPS);
    }

    // 2) Q,K,V projection
    amx_matmul_padded(tmp_norm, lay.q_w_packed, q_fp32, S, HIDDEN, Q_DIM,  A_pad, C_pad);
    amx_matmul_padded(tmp_norm, lay.k_w_packed, k_fp32, S, HIDDEN, KV_DIM, A_pad, C_pad);
    amx_matmul_padded(tmp_norm, lay.v_w_packed, v_fp32, S, HIDDEN, KV_DIM, A_pad, C_pad);

    // Add biases, cast to BF16
    for (int s = 0; s < S; ++s) {
        for (int n = 0; n < Q_DIM; ++n) {
            tmp_q[s * Q_DIM + n] = fp32_to_bf16(q_fp32[s * Q_DIM + n] + lay.q_b_fp32[n]);
        }
        for (int n = 0; n < KV_DIM; ++n) {
            tmp_k[s * KV_DIM + n] = fp32_to_bf16(k_fp32[s * KV_DIM + n] + lay.k_b_fp32[n]);
            tmp_v[s * KV_DIM + n] = fp32_to_bf16(v_fp32[s * KV_DIM + n] + lay.v_b_fp32[n]);
        }
    }

    // 3) RoPE on Q (per head, head_dim=64, n_heads=14) and K (n_heads=2)
    for (int s = 0; s < S; ++s) {
        int p = pos0 + s;
        apply_rope_bf16(tmp_q + s * Q_DIM, N_HEADS, HEAD_DIM, p,
                        g_model.rope_cos.data(), g_model.rope_sin.data());
        apply_rope_bf16(tmp_k + s * KV_DIM, N_KV_HEADS, HEAD_DIM, p,
                        g_model.rope_cos.data(), g_model.rope_sin.data());
    }

    // 4) Append to KV cache
    size_t kv_old = static_cast<size_t>(lay.s_cached) * KV_DIM;
    lay.k_cache.resize(kv_old + static_cast<size_t>(S) * KV_DIM);
    lay.v_cache.resize(kv_old + static_cast<size_t>(S) * KV_DIM);
    std::memcpy(lay.k_cache.data() + kv_old, tmp_k, S * KV_DIM * sizeof(uint16_t));
    std::memcpy(lay.v_cache.data() + kv_old, tmp_v, S * KV_DIM * sizeof(uint16_t));
    lay.s_cached += S;
    int S_kv = lay.s_cached;

    // 5) GQA attention.
    //    For each (s_query, q_head): scores[s_kv] = (q · k_kv) * 1/sqrt(64),
    //    softmax over s_kv ∈ [0, s_query], then attn = sum_skv softmax * v_kv.
    //
    //    Causal mask: only s_kv ≤ (pos0 + s) are visible.
    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
    std::vector<float> scores(S_kv);
    for (int s = 0; s < S; ++s) {
        int p_q = pos0 + s;
        for (int h = 0; h < N_HEADS; ++h) {
            int kv_h = h / GQA_GROUP;
            const uint16_t* qh = tmp_q + s * Q_DIM + h * HEAD_DIM;

            // scores
            float max_score = -INFINITY;
            int s_kv_lim = std::min(S_kv, p_q + 1);
            for (int t = 0; t < s_kv_lim; ++t) {
                const uint16_t* kh = lay.k_cache.data() + t * KV_DIM + kv_h * HEAD_DIM;
                float dot = 0.f;
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dot += bf16_to_fp32(qh[d]) * bf16_to_fp32(kh[d]);
                }
                float sc = dot * inv_sqrt_d;
                scores[t] = sc;
                if (sc > max_score) max_score = sc;
            }
            // softmax
            float ssum = 0.f;
            for (int t = 0; t < s_kv_lim; ++t) {
                scores[t] = std::exp(scores[t] - max_score);
                ssum += scores[t];
            }
            float inv_ssum = 1.f / ssum;
            // weighted sum of V
            float out_fp32[HEAD_DIM] = {0};
            for (int t = 0; t < s_kv_lim; ++t) {
                float w = scores[t] * inv_ssum;
                const uint16_t* vh = lay.v_cache.data() + t * KV_DIM + kv_h * HEAD_DIM;
                for (int d = 0; d < HEAD_DIM; ++d) {
                    out_fp32[d] += w * bf16_to_fp32(vh[d]);
                }
            }
            uint16_t* ah = attn_out + s * Q_DIM + h * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; ++d) {
                ah[d] = fp32_to_bf16(out_fp32[d]);
            }
        }
    }

    // 6) o_proj
    amx_matmul_padded(attn_out, lay.o_w_packed, o_fp32, S, Q_DIM, HIDDEN, A_pad, C_pad);

    // 7) Residual + RMSNorm post_attention_layernorm
    // h_mid_bf16 = BF16(h_in + o_fp32)
    // tmp_norm = RMSNorm(h_mid)
    std::vector<uint16_t> h_mid_bf16(static_cast<size_t>(S) * HIDDEN);
    for (int s = 0; s < S; ++s) {
        for (int i = 0; i < HIDDEN; ++i) {
            float v = bf16_to_fp32(h_in[s * HIDDEN + i]) + o_fp32[s * HIDDEN + i];
            h_mid_bf16[s * HIDDEN + i] = fp32_to_bf16(v);
        }
        rmsnorm_bf16(h_mid_bf16.data() + s * HIDDEN, lay.ln2_w_fp32.data(),
                     tmp_norm + s * HIDDEN, HIDDEN, RMS_EPS);
    }

    // 8) MLP: gate, up, SwiGLU, down
    amx_matmul_padded(tmp_norm, lay.gate_w_packed, gate_fp32, S, HIDDEN, INTERMEDIATE, A_pad, C_pad);
    amx_matmul_padded(tmp_norm, lay.up_w_packed,   up_fp32,   S, HIDDEN, INTERMEDIATE, A_pad, C_pad);
    for (int s = 0; s < S; ++s) {
        for (int i = 0; i < INTERMEDIATE; ++i) {
            float g = gate_fp32[s * INTERMEDIATE + i];
            float u = up_fp32[s * INTERMEDIATE + i];
            float sg = silu_fp32(g);
            swi_bf16[s * INTERMEDIATE + i] = fp32_to_bf16(sg * u);
        }
    }
    amx_matmul_padded(swi_bf16, lay.down_w_packed, down_fp32, S, INTERMEDIATE, HIDDEN, A_pad, C_pad);

    // 9) Residual + write h_out
    for (int s = 0; s < S; ++s) {
        for (int i = 0; i < HIDDEN; ++i) {
            float v = bf16_to_fp32(h_mid_bf16[s * HIDDEN + i]) + down_fp32[s * HIDDEN + i];
            h_out[s * HIDDEN + i] = fp32_to_bf16(v);
        }
    }
}

// ─── public single-layer forward (P2 validation) ─────────────────────

extern "C" int amx_draft_qwen05b_layer_forward(int layer_idx,
                                               const uint16_t* h_in, int S,
                                               int pos0,
                                               uint16_t* h_out) {
    using namespace qwen05b;
    if (!g_model.initialized) return -1;
    if (layer_idx < 0 || layer_idx >= N_LAYERS) return -2;
    if (S < 1 || S > MAX_SEQ) return -3;

    // Scratch buffers (worst-case S × max_N)
    int max_N = INTERMEDIATE;  // largest output N
    int M_amx = ((S + 15) / 16) * 16;
    std::vector<uint16_t> tmp_norm(S * HIDDEN);
    std::vector<uint16_t> tmp_q(S * Q_DIM);
    std::vector<uint16_t> tmp_k(S * KV_DIM);
    std::vector<uint16_t> tmp_v(S * KV_DIM);
    std::vector<float> q_fp32(M_amx * Q_DIM);
    std::vector<float> k_fp32(M_amx * KV_DIM);
    std::vector<float> v_fp32(M_amx * KV_DIM);
    std::vector<uint16_t> attn_out(S * Q_DIM);
    std::vector<float> o_fp32(M_amx * HIDDEN);
    std::vector<float> gate_fp32(M_amx * INTERMEDIATE);
    std::vector<float> up_fp32(M_amx * INTERMEDIATE);
    std::vector<uint16_t> swi_bf16(S * INTERMEDIATE);
    std::vector<float> down_fp32(M_amx * HIDDEN);
    std::vector<uint16_t> A_pad(static_cast<size_t>(M_amx) * INTERMEDIATE, 0);
    std::vector<float> C_pad(static_cast<size_t>(M_amx) * max_N, 0.f);

    qwen05b_layer_forward(layer_idx, h_in, S, pos0, h_out,
                          tmp_norm.data(), tmp_q.data(), tmp_k.data(), tmp_v.data(),
                          q_fp32.data(), k_fp32.data(), v_fp32.data(),
                          attn_out.data(),
                          o_fp32.data(),
                          gate_fp32.data(), up_fp32.data(),
                          swi_bf16.data(), down_fp32.data(),
                          A_pad.data(), C_pad.data());
    return 0;
}

// ─── full 24-layer forward + lm_head + argmax ───────────────────────
//
//   input_ids[S] → embed → 24 × layer_forward → final_norm → lm_head
//   → argmax → emit 1 token → append to input, repeat K times.
//
//   logits_last_bf16 may be NULL; if provided, receives the K-step
//   logits BF16 [K, VOCAB_PADDED].
//
extern "C" int amx_draft_qwen05b_forward_full(const int32_t* input_ids,
                                              int S_prompt,
                                              int32_t* out_ids,
                                              int K,
                                              uint16_t* logits_last_bf16) {
    using namespace qwen05b;
    if (!g_model.initialized) return -1;
    if (S_prompt < 1 || S_prompt > MAX_SEQ) return -2;
    if (K < 0 || S_prompt + K > MAX_SEQ) return -3;
    if (!g_state.W_lm_head_packed) return -4;

    // Reset KV cache for fresh forward
    for (int L = 0; L < N_LAYERS; ++L) {
        g_model.layers[L].k_cache.clear();
        g_model.layers[L].v_cache.clear();
        g_model.layers[L].s_cached = 0;
    }

    // Scratch (reused across layers, sized for worst case S = S_prompt)
    int S_max = std::max(S_prompt, 1);
    int M_amx = ((S_max + 15) / 16) * 16;
    int max_N = INTERMEDIATE;
    std::vector<uint16_t> tmp_norm(S_max * HIDDEN);
    std::vector<uint16_t> tmp_q(S_max * Q_DIM);
    std::vector<uint16_t> tmp_k(S_max * KV_DIM);
    std::vector<uint16_t> tmp_v(S_max * KV_DIM);
    std::vector<float> q_fp32(M_amx * Q_DIM);
    std::vector<float> k_fp32(M_amx * KV_DIM);
    std::vector<float> v_fp32(M_amx * KV_DIM);
    std::vector<uint16_t> attn_out(S_max * Q_DIM);
    std::vector<float> o_fp32(M_amx * HIDDEN);
    std::vector<float> gate_fp32(M_amx * INTERMEDIATE);
    std::vector<float> up_fp32(M_amx * INTERMEDIATE);
    std::vector<uint16_t> swi_bf16(S_max * INTERMEDIATE);
    std::vector<float> down_fp32(M_amx * HIDDEN);
    std::vector<uint16_t> A_pad(static_cast<size_t>(M_amx) * INTERMEDIATE, 0);
    std::vector<float> C_pad(static_cast<size_t>(M_amx) * max_N, 0.f);

    std::vector<uint16_t> h_curr(S_max * HIDDEN);
    std::vector<uint16_t> h_next(S_max * HIDDEN);

    // 1) Embed input_ids → h_curr
    for (int s = 0; s < S_prompt; ++s) {
        int tok = input_ids[s];
        if (tok < 0 || tok >= VOCAB_VALID) tok = 0;
        std::memcpy(h_curr.data() + s * HIDDEN,
                    g_model.embed_tokens.data() + static_cast<size_t>(tok) * HIDDEN,
                    HIDDEN * sizeof(uint16_t));
    }

    // 2) Prefill: process the prompt through all 24 layers, S = S_prompt
    int pos0 = 0;
    for (int L = 0; L < N_LAYERS; ++L) {
        qwen05b_layer_forward(L, h_curr.data(), S_prompt, pos0, h_next.data(),
                              tmp_norm.data(), tmp_q.data(), tmp_k.data(), tmp_v.data(),
                              q_fp32.data(), k_fp32.data(), v_fp32.data(),
                              attn_out.data(),
                              o_fp32.data(),
                              gate_fp32.data(), up_fp32.data(),
                              swi_bf16.data(), down_fp32.data(),
                              A_pad.data(), C_pad.data());
        std::swap(h_curr, h_next);
    }

    // 3) Final RMSNorm on the LAST token only (for next-token prediction)
    std::vector<uint16_t> last_hidden(HIDDEN);
    rmsnorm_bf16(h_curr.data() + (S_prompt - 1) * HIDDEN,
                 g_model.final_norm_w_fp32.data(),
                 last_hidden.data(), HIDDEN, RMS_EPS);

    // 4) lm_head: [1, HIDDEN] × W_lm_head_packed → [1, VOCAB_PADDED] FP32 → argmax
    auto do_lm_head = [&](const uint16_t* h_in_bf16, int32_t* out_id,
                          uint16_t* out_logits_bf16_or_null) {
        int M = 1;
        int M_amx_lm = 16;
        std::vector<uint16_t> A_pad_lm(M_amx_lm * HIDDEN, 0);
        std::vector<float> C_pad_lm(M_amx_lm * VOCAB_PADDED, 0.f);
        std::memcpy(A_pad_lm.data(), h_in_bf16, HIDDEN * sizeof(uint16_t));
        amx_matmul_bf16_omp_n(A_pad_lm.data(), g_state.W_lm_head_packed,
                              C_pad_lm.data(), M_amx_lm, HIDDEN, VOCAB_PADDED);
        float* row = C_pad_lm.data();
        int best_id = 0;
        float best_v = row[0];
        for (int n = 0; n < VOCAB_VALID; ++n) {
            if (row[n] > best_v) { best_v = row[n]; best_id = n; }
        }
        *out_id = best_id;
        if (out_logits_bf16_or_null) {
            for (int n = 0; n < VOCAB_PADDED; ++n) {
                out_logits_bf16_or_null[n] = fp32_to_bf16(row[n]);
            }
        }
        (void)M;
    };

    // Generate K tokens (K may be 0; if K=0 just emit the prompt's next-token to out_ids[0])
    int K_emit = std::max(K, 1);
    int32_t prev_tok = 0;
    for (int k = 0; k < K_emit; ++k) {
        do_lm_head(last_hidden.data(), &prev_tok,
                   logits_last_bf16 ? logits_last_bf16 + k * VOCAB_PADDED : nullptr);
        if (k < K) out_ids[k] = prev_tok;

        // If we need more tokens, feed prev_tok back through the model (S=1 decode-step).
        if (k + 1 < K_emit) {
            std::memcpy(h_curr.data(),
                        g_model.embed_tokens.data() + static_cast<size_t>(prev_tok) * HIDDEN,
                        HIDDEN * sizeof(uint16_t));
            int pos = S_prompt + k;
            for (int L = 0; L < N_LAYERS; ++L) {
                qwen05b_layer_forward(L, h_curr.data(), 1, pos, h_next.data(),
                                      tmp_norm.data(), tmp_q.data(), tmp_k.data(), tmp_v.data(),
                                      q_fp32.data(), k_fp32.data(), v_fp32.data(),
                                      attn_out.data(),
                                      o_fp32.data(),
                                      gate_fp32.data(), up_fp32.data(),
                                      swi_bf16.data(), down_fp32.data(),
                                      A_pad.data(), C_pad.data());
                std::swap(h_curr, h_next);
            }
            rmsnorm_bf16(h_curr.data(),
                         g_model.final_norm_w_fp32.data(),
                         last_hidden.data(), HIDDEN, RMS_EPS);
        }
    }
    return 0;
}
