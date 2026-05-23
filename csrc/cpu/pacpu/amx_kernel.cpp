// SUB_015-Phase 3 A: AMX BF16 qk-product host C++ kernel.
//
// 목적: NEO paged-attention 의 qk_product (8.75% cycle on prod SPR) 를 AMX
//       _tile_dpbf16ps 으로 대체. FP16 → BF16 변환 + K^T pre-pack 후 tile
//       matmul. softmax / av_product 는 ISPC 유지.
//
// 호출 contract (ISPC qk_product 와 동일):
//   qk_amx(cur_layer, num_blocks, seq_len, q[NUM_Q_HEADS, HEAD_DIM] FP16,
//          k_cache[num_layers, num_blocks, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM] FP16,
//          block_table[seq_len], a[seq_len, NUM_KV_HEADS, NUM_Q_HEADS] FP32)
//
// Llama-3.3-70B TP=8 hyper-param (dtype.h):
//   NUM_Q_HEADS=8, NUM_KV_HEADS=1, QH_PER_KVH=8, HEAD_DIM=128, BLOCK_SIZE=16
//
// AMX tile cfg (sanity-test 검증된 colsb=64 패턴 — B tile padding):
//   A: rows=16, colsb=64 — Q[16, 32 BF16] (M=8 valid + 8 unused rows)
//   B: rows=16, colsb=64 — K^T 1 round of K_pair=16 packed pairs, padded
//   C: rows=16, colsb=64 — output FP32 [16, 16] (M=8 valid)
//
// K=128 BF16 = 4 rounds (32 BF16 each).
#include <immintrin.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include "dtype.h"

#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILEDATA 18

namespace amx_kernel {

struct alignas(64) TileConfig {
    uint8_t  palette;
    uint8_t  start_row;
    uint8_t  reserved[14];
    uint16_t colsb[16];
    uint8_t  rows[16];
};

// Thread-local AMX init flag.
static thread_local bool amx_initialized = false;

// SUB_015-Phase 3 B (Strategy B thread-level): Q BF16 cache.
//   같은 thread 의 consecutive task 가 same-seq 의 chunk 처리 시 Q 동일 — 변환 skip.
//   tasks[] partition 이 seq 순으로 ordered (core.h line 282-292) → consecutive
//   same-seq task 빈도 ~30-50% (ws=14, batch~30 typical).
//   Cache size: 16 rows × 128 BF16 = 4 KB / thread (L1 fit).
static thread_local const data_t* _last_q_ptr = nullptr;
alignas(64) static thread_local uint16_t _cached_Q_bf16[16 * 128];

bool ensure_amx_init() {
    if (amx_initialized) return true;
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) != 0) {
        return false;
    }
    TileConfig cfg = {};
    cfg.palette = 1;
    // Step 5 (best): 3 tile config — A=0 (Q), B=1 (K^T), C=2 (FP32 result).
    // Step 6 (5-tile + G + C' 통합) tried — -3.7% regression. Reverted.
    for (int i = 0; i < 3; ++i) {
        cfg.rows[i] = 16;
        cfg.colsb[i] = 64;
    }
    _tile_loadconfig(&cfg);
    amx_initialized = true;
    return true;
}

// Convert FP32 → BF16 (round-to-nearest-even).
static inline uint16_t f32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return static_cast<uint16_t>((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

// Convert _Float16 → BF16 via FP32.
static inline uint16_t fp16_to_bf16(_Float16 h) {
    float f = static_cast<float>(h);
    return f32_to_bf16(f);
}

// Step 5 (vectorized): AVX-512 FP16 → BF16 batch conversion.
//   16 elem at a time via _mm512_cvtph_ps + _mm512_cvtneps_pbh.
//   src/dst should be 32-byte aligned for max speed; loadu/storeu used for safety.
static inline void fp16_to_bf16_batch(const _Float16* src, uint16_t* dst, int count) {
    int i = 0;
    for (; i + 15 < count; i += 16) {
        __m256i fp16_v = _mm256_loadu_si256((const __m256i*)(src + i));
        __m512  fp32_v = _mm512_cvtph_ps(fp16_v);
        __m256bh bf16_v = _mm512_cvtneps_pbh(fp32_v);
        _mm256_storeu_si256((__m256i*)(dst + i), (__m256i)bf16_v);
    }
    for (; i < count; ++i) {
        dst[i] = fp16_to_bf16(src[i]);
    }
}

// SUB_039: Vectorized FP32 → BF16 batch conversion.
//   16 elem at a time via _mm512_cvtneps_pbh.
static inline void fp32_to_bf16_batch(const float* src, uint16_t* dst, int count) {
    int i = 0;
    for (; i + 15 < count; i += 16) {
        __m512 fp32_v = _mm512_loadu_ps(src + i);
        __m256bh bf16_v = _mm512_cvtneps_pbh(fp32_v);
        _mm256_storeu_si256((__m256i*)(dst + i), (__m256i)bf16_v);
    }
    for (; i < count; ++i) {
        dst[i] = f32_to_bf16(src[i]);
    }
}

// qk_amx — replaces ispc::qk_product for ONE sequence segment.
//   Layout matches dtype.h: data_t = _Float16, itmd_t = float.
//   M = NUM_Q_HEADS = 8, K = HEAD_DIM = 128, N = BLOCK_SIZE = 16.
//   per-block matmul: output[8, 16] = Q[8, 128] @ K_block[16, 128]^T
extern "C" void qk_amx(
    int cur_layer,
    int num_blocks,
    int seq_len,
    const data_t* q,           // [NUM_Q_HEADS, HEAD_DIM] FP16
    const data_t* k_cache,     // [num_layers, num_blocks, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM] FP16
    const int* block_table,
    itmd_t* a                  // [seq_len, NUM_KV_HEADS, NUM_Q_HEADS] FP32
) {
    if (!ensure_amx_init()) {
        // Fallback path absent — caller should check via ensure_amx_init separately.
        return;
    }

    // Strategy B (thread-level): skip Q FP16→BF16 conversion if same q pointer.
    // 한 thread 의 consecutive same-seq task 가 같은 Q 사용 → 1 회 변환 후 cache.
    if (q != _last_q_ptr) {
        std::memset(_cached_Q_bf16, 0, sizeof(_cached_Q_bf16));
        for (int m = 0; m < NUM_Q_HEADS; ++m) {
            for (int k = 0; k < HEAD_DIM; ++k) {
                _cached_Q_bf16[m * 128 + k] = fp16_to_bf16(q[m * HEAD_DIM + k]);
            }
        }
        _last_q_ptr = q;
    }
    uint16_t* Q_bf16 = _cached_Q_bf16;

    int imax = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Strategy A (cheap variant): K^T pre-pack outer hoist.
    //   기존 매 (block × round) 마다 K^T BF16 변환 + tile_loadd + dpbf16 interleaved.
    //   변경 = (1) 모든 imax × 4 round 의 K^T BF16 변환을 outer loop 으로 hoist,
    //         (2) inner AMX loop 은 cache 에서 tile_loadd 만.
    //   기대: BTB / prefetch locality 개선 (변환 hot + AMX hot 분리).
    //
    //   Cache size: imax × 4 round × 16 row × 64 byte = imax × 4096 byte.
    //   imax=128 (seq_len 2048) 시 512 KB / thread (L2 2MB fit). thread_local heap.
    static thread_local std::vector<uint16_t> _kt_cache;
    constexpr int B_TILE_BYTES = 16 * 64;  // 1024 byte per B tile (padded)
    size_t needed_u16 = (size_t)imax * 4 * (B_TILE_BYTES / 2);  // u16 units
    if (_kt_cache.size() < needed_u16) {
        _kt_cache.resize(needed_u16);
    }

    // Outer pre-pack: convert all K^T blocks BF16 + pad to tile layout.
    // Step 5 (best): AVX-512 vectorized FP16→BF16 (4-8× faster than scalar).
    // Step 6 (+ G SW prefetch + C' 2-block fused) tried — -3.7% regression. Reverted.
    alignas(64) uint16_t K_bf16_rowmajor[BLOCK_SIZE * HEAD_DIM];  // 16 × 128 × 2 = 4 KB stack
    for (int i = 0; i < imax; ++i) {
        const data_t* k_block = k_cache +
            (1ll * cur_layer * num_blocks + block_table[i]) * BLOCK_NELEM;
        int tmax = std::min(BLOCK_SIZE, seq_len - i * BLOCK_SIZE);

        // Step 1: vectorized K[n, 0..127] FP16→BF16 for valid n.
        for (int n = 0; n < tmax; ++n) {
            fp16_to_bf16_batch(k_block + n * HEAD_DIM,
                               K_bf16_rowmajor + n * HEAD_DIM, HEAD_DIM);
        }
        // Zero-pad n = tmax..BLOCK_SIZE-1 for partial last block.
        for (int n = tmax; n < BLOCK_SIZE; ++n) {
            std::memset(K_bf16_rowmajor + n * HEAD_DIM, 0, HEAD_DIM * 2);
        }

        // Step 2: interleave to AMX tile layout (4 round × 16 K_pair × 16 N pair × 2 byte).
        for (int round = 0; round < 4; ++round) {
            int k_off = round * 32;
            uint16_t* B_tile = _kt_cache.data() +
                ((size_t)i * 4 + round) * (B_TILE_BYTES / 2);
            std::memset(B_tile, 0, B_TILE_BYTES);
            for (int k_pair = 0; k_pair < 16; ++k_pair) {
                int k_lo = k_off + 2 * k_pair;
                for (int n = 0; n < BLOCK_SIZE; ++n) {
                    B_tile[k_pair * 32 + n * 2 + 0] = K_bf16_rowmajor[n * HEAD_DIM + k_lo];
                    B_tile[k_pair * 32 + n * 2 + 1] = K_bf16_rowmajor[n * HEAD_DIM + k_lo + 1];
                }
            }
        }
    }

    // Inner AMX hot loop: tile_loadd from pre-packed cache, no conversion.
    alignas(64) float C[16 * 16];
    for (int i = 0; i < imax; ++i) {
        int tmax = std::min(BLOCK_SIZE, seq_len - i * BLOCK_SIZE);
        _tile_zero(2);
        for (int round = 0; round < 4; ++round) {
            int k_off = round * 32;
            const uint16_t* B_tile = _kt_cache.data() +
                ((size_t)i * 4 + round) * (B_TILE_BYTES / 2);
            _tile_loadd(0, Q_bf16 + k_off, 128 * 2);
            _tile_loadd(1, B_tile, 64);
            _tile_dpbf16ps(2, 0, 1);
        }
        _tile_stored(2, C, 16 * 4);

        // Write C[M=8, N=tmax] → a[(i*BS + t) * NUM_Q_HEADS + h]
        int a_base = i * BLOCK_SIZE * NUM_Q_HEADS;
        for (int t = 0; t < tmax; ++t) {
            for (int h = 0; h < NUM_Q_HEADS; ++h) {
                a[a_base + t * NUM_Q_HEADS + h] = C[h * 16 + t];
            }
        }
    }
}

// External ISPC declarations (use after pacpu_ispc.h is generated).
namespace ispc {
extern "C" void softmax(int seq_len, float softmax_scale, float* a, float* asb);
extern "C" void softmax_online(int seq_len, float softmax_scale, float* a, float* asb);
extern "C" void av_product(int cur_layer, int num_blocks, int seq_len,
                            const float* a, const _Float16* v_cache,
                            const int* block_table, float* o);
}

// SUB_033 B3: env-gated dispatch (1회 결정, thread_local cached)
//   VLLM_NEO_SOFTMAX_ONLINE=1 → ispc::softmax_online (2-pass)
//   default                  → ispc::softmax        (3-pass, NEO 원본)
typedef void (*softmax_fn_t)(int, float, float*, float*);
static inline softmax_fn_t _softmax_dispatch() {
    static thread_local softmax_fn_t fn = nullptr;
    if (fn == nullptr) {
        const char* env = std::getenv("VLLM_NEO_SOFTMAX_ONLINE");
        fn = (env && env[0] && env[0] != '0')
           ? (softmax_fn_t)&ispc::softmax_online
           : (softmax_fn_t)&ispc::softmax;
    }
    return fn;
}

// attn_one_seq_amx — replaces ispc::attn_one_seq when VLLM_NEO_USE_AMX is set.
//   qk = AMX BF16 (host C++), softmax = ISPC FP32, av = ISPC FP32.
extern "C" void attn_one_seq_amx(
    int cur_layer,
    int num_blocks,
    int seq_len,
    float softmax_scale,
    const data_t* q,
    const data_t* k_cache,
    const data_t* v_cache,
    const int* block_table,
    itmd_t* a,
    otpt_t* o,
    itmd_t* asb
) {
    qk_amx(cur_layer, num_blocks, seq_len, q, k_cache, block_table, a);
    _softmax_dispatch()(seq_len, softmax_scale, a, asb);
    ispc::av_product(cur_layer, num_blocks, seq_len, a, v_cache, block_table, o);
}

// P3 (F3) — BF16-native qk_amx variant. host K 가 이미 BF16 store 인 경우
//   Step 1 (FP16→BF16 vec conv) skip → memory bandwidth saved.
//   K^T pre-pack (Step 2) 의 interleave 영역은 동일 유지.
//   Q 변환 영역 (Strategy B thread_local cache) 도 동일.
extern "C" void qk_amx_bf16(
    int cur_layer,
    int num_blocks,
    int seq_len,
    const data_t* q,                   // FP16, AMX path 내부 BF16 conv
    const uint16_t* k_cache_bf16,      // BF16 bits, host store 그대로
    const int* block_table,
    itmd_t* a
) {
    if (!ensure_amx_init()) return;

    // Strategy B: Q FP16 → BF16 thread_local cache (동일).
    if (q != _last_q_ptr) {
        std::memset(_cached_Q_bf16, 0, sizeof(_cached_Q_bf16));
        for (int m = 0; m < NUM_Q_HEADS; ++m) {
            for (int k = 0; k < HEAD_DIM; ++k) {
                _cached_Q_bf16[m * 128 + k] = fp16_to_bf16(q[m * HEAD_DIM + k]);
            }
        }
        _last_q_ptr = q;
    }
    uint16_t* Q_bf16 = _cached_Q_bf16;

    int imax = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Strategy A: K^T pre-pack outer hoist (동일).
    static thread_local std::vector<uint16_t> _kt_cache;
    constexpr int B_TILE_BYTES = 16 * 64;
    size_t needed_u16 = (size_t)imax * 4 * (B_TILE_BYTES / 2);
    if (_kt_cache.size() < needed_u16) {
        _kt_cache.resize(needed_u16);
    }

    // P3 영역: K 가 이미 BF16. partial last block 만 local buffer 로 zero-pad.
    alignas(64) uint16_t K_bf16_local[BLOCK_SIZE * HEAD_DIM];
    for (int i = 0; i < imax; ++i) {
        const uint16_t* k_block = k_cache_bf16 +
            (1ll * cur_layer * num_blocks + block_table[i]) * BLOCK_NELEM;
        int tmax = std::min(BLOCK_SIZE, seq_len - i * BLOCK_SIZE);

        // P3 (F3): Step 1 vec K FP16→BF16 conv 제거. k_block 이 이미 BF16.
        const uint16_t* K_bf16_rowmajor;
        if (tmax == BLOCK_SIZE) {
            K_bf16_rowmajor = k_block;   // alias, no copy
        } else {
            // Partial block — local buffer 에 copy + zero-pad.
            std::memcpy(K_bf16_local, k_block, tmax * HEAD_DIM * 2);
            for (int n = tmax; n < BLOCK_SIZE; ++n) {
                std::memset(K_bf16_local + n * HEAD_DIM, 0, HEAD_DIM * 2);
            }
            K_bf16_rowmajor = K_bf16_local;
        }

        // Step 2: interleave to AMX tile layout (동일).
        for (int round = 0; round < 4; ++round) {
            int k_off = round * 32;
            uint16_t* B_tile = _kt_cache.data() +
                ((size_t)i * 4 + round) * (B_TILE_BYTES / 2);
            std::memset(B_tile, 0, B_TILE_BYTES);
            for (int k_pair = 0; k_pair < 16; ++k_pair) {
                int k_lo = k_off + 2 * k_pair;
                for (int n = 0; n < BLOCK_SIZE; ++n) {
                    B_tile[k_pair * 32 + n * 2 + 0] = K_bf16_rowmajor[n * HEAD_DIM + k_lo];
                    B_tile[k_pair * 32 + n * 2 + 1] = K_bf16_rowmajor[n * HEAD_DIM + k_lo + 1];
                }
            }
        }
    }

    // Inner AMX hot loop (동일).
    alignas(64) float C[16 * 16];
    for (int i = 0; i < imax; ++i) {
        int tmax = std::min(BLOCK_SIZE, seq_len - i * BLOCK_SIZE);
        _tile_zero(2);
        for (int round = 0; round < 4; ++round) {
            int k_off = round * 32;
            const uint16_t* B_tile = _kt_cache.data() +
                ((size_t)i * 4 + round) * (B_TILE_BYTES / 2);
            _tile_loadd(0, Q_bf16 + k_off, 128 * 2);
            _tile_loadd(1, B_tile, 64);
            _tile_dpbf16ps(2, 0, 1);
        }
        _tile_stored(2, C, 16 * 4);

        int a_base = i * BLOCK_SIZE * NUM_Q_HEADS;
        for (int t = 0; t < tmax; ++t) {
            for (int h = 0; h < NUM_Q_HEADS; ++h) {
                a[a_base + t * NUM_Q_HEADS + h] = C[h * 16 + t];
            }
        }
    }
}

// SUB_039: av_amx — AMX BF16 matmul for av_product.
//   Shape: A[seq_len, NUM_Q_HEADS=64] @ V_per_kvhead[seq_len, HEAD_DIM=128]
//          → O[NUM_Q_HEADS=64, HEAD_DIM=128]
//   Per kv_head j (0..NUM_KV_HEADS=8), QH_PER_KVH=8 q heads share V[:, j, :].
//   AMX tile config: M=16 (QH_PER_KVH=8 padded to 16), K=32 BF16, N=16.
//
//   K rounds: ceil(seq_len/32). N rounds: HEAD_DIM/16 = 8.
//   Inner loop: tile_dpbf16ps(C, A_tile, V_tile) accumulates.
//   FP32 accumulator → store to O.
//
//   CAUTION (turn N+2): 본 함수는 SKELETON. 실제 AMX matmul 검증 필요:
//     (1) A_bf16 [QH_PER_KVH × K_padded] 의 tile_loadd stride
//     (2) V_bf16 [K_padded × N=16] 의 interleave (qk_amx_bf16 의 K_pre 패턴 참조)
//     (3) tile_stored → output[q_head*HEAD_DIM + n_offset:] padding 처리
//     (4) BF16 cast 정확도 (FP32 A + FP16 V → BF16 의 relative error 검증)
//   env-gated (VLLM_NEO_AV_AMX=1) default OFF — 정확도 verify 전까지 미사용.
extern "C" void av_amx_bf16(
    int cur_layer,
    int num_blocks,
    int seq_len,
    const float* a,            // FP32 [seq_len, NUM_Q_HEADS=64]
    const _Float16* v_cache,   // FP16
    const int* block_table,
    float* o                   // FP32 [NUM_Q_HEADS=64, HEAD_DIM=128]
) {
    if (!ensure_amx_init()) return;

    int imax = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Step 1: V FP16 → BF16 cast (per block, all kv_heads + all positions).
    //   layout target: V_bf16[i_block, j_kv, t, l] linearly
    //   thread_local cache, resize on demand.
    static thread_local std::vector<uint16_t> _v_bf16_cache;
    size_t v_needed = (size_t)imax * NUM_KV_HEADS * BLOCK_SIZE * HEAD_DIM;
    if (_v_bf16_cache.size() < v_needed) _v_bf16_cache.resize(v_needed);
    for (int i = 0; i < imax; ++i) {
        int tmax = std::min(BLOCK_SIZE, seq_len - i * BLOCK_SIZE);
        const _Float16* v_block = v_cache +
            (1ll * cur_layer * num_blocks + block_table[i]) * BLOCK_NELEM;
        // V block layout: [NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM]
        for (int j = 0; j < NUM_KV_HEADS; ++j) {
            for (int t = 0; t < tmax; ++t) {
                fp16_to_bf16_batch(
                    v_block + (j * BLOCK_SIZE + t) * HEAD_DIM,
                    _v_bf16_cache.data() + (((i * NUM_KV_HEADS + j) * BLOCK_SIZE + t) * HEAD_DIM),
                    HEAD_DIM);
            }
            // zero-pad if partial block
            for (int t = tmax; t < BLOCK_SIZE; ++t) {
                std::memset(
                    _v_bf16_cache.data() + (((i * NUM_KV_HEADS + j) * BLOCK_SIZE + t) * HEAD_DIM),
                    0, HEAD_DIM * sizeof(uint16_t));
            }
        }
    }

    // Step 2: A FP32 → BF16 cast (entire [seq_len, NUM_Q_HEADS] matrix).
    static thread_local std::vector<uint16_t> _a_bf16_cache;
    size_t a_needed = (size_t)seq_len * NUM_Q_HEADS;
    if (_a_bf16_cache.size() < a_needed) _a_bf16_cache.resize(a_needed);
    fp32_to_bf16_batch(a, _a_bf16_cache.data(), (int)a_needed);

    // Step 3: AMX matmul per kv_head.
    //   For each kv_head j: compute A^T[j_head_group, seq_len] @ V_per_j[seq_len, HEAD_DIM]
    //   AMX inner: tile_dpbf16ps with K=32 BF16 lanes.
    //
    //   TODO (SUB_039 next turn): Implement actual AMX tile loop here.
    //   Skeleton: fallback to ispc::av_product for correctness.
    //   When VLLM_NEO_AV_AMX=1 is set + this fallback removed, the AMX matmul
    //   replaces ISPC av_product.

    // FALLBACK — call ISPC av_product for now (correctness preserved).
    // This branch will be replaced by AMX inner loop after verification.
    ispc::av_product(cur_layer, num_blocks, seq_len, a, v_cache, block_table, o);
}

// SUB_039: env-gated dispatch for av_product.
//   VLLM_NEO_AV_AMX=1 → av_amx_bf16 (currently skeleton with fallback)
//   default          → ispc::av_product (NEO 원본)
typedef void (*av_fn_t)(int, int, int, const float*, const _Float16*, const int*, float*);
static inline av_fn_t _av_dispatch() {
    static thread_local av_fn_t fn = nullptr;
    if (fn == nullptr) {
        const char* env = std::getenv("VLLM_NEO_AV_AMX");
        fn = (env && env[0] && env[0] != '0')
           ? (av_fn_t)&av_amx_bf16
           : (av_fn_t)&ispc::av_product;
    }
    return fn;
}

// P3 (F3) — BF16-K variant of attn_one_seq_amx. K BF16, V FP16 (ISPC kernel
//   요구). Q FP16 그대로.
extern "C" void attn_one_seq_amx_bf16(
    int cur_layer,
    int num_blocks,
    int seq_len,
    float softmax_scale,
    const data_t* q,                   // FP16
    const uint16_t* k_cache_bf16,      // BF16
    const data_t* v_cache,             // FP16 그대로 (ISPC av_product)
    const int* block_table,
    itmd_t* a,
    otpt_t* o,
    itmd_t* asb
) {
    qk_amx_bf16(cur_layer, num_blocks, seq_len, q, k_cache_bf16, block_table, a);
    _softmax_dispatch()(seq_len, softmax_scale, a, asb);
    ispc::av_product(cur_layer, num_blocks, seq_len, a, v_cache, block_table, o);
}

}  // namespace amx_kernel
