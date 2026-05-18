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
    // sanity-test verified cfg: all 3 tiles rows=16 colsb=64
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
    // Strategy G (SW prefetch) tried in Step 3 — -4% regression (block_table indirection
    // makes next-block addr unpredictable, prefetch overhead > win). Reverted.
    for (int i = 0; i < imax; ++i) {
        const data_t* k_block = k_cache +
            (1ll * cur_layer * num_blocks + block_table[i]) * BLOCK_NELEM;
        int tmax = std::min(BLOCK_SIZE, seq_len - i * BLOCK_SIZE);
        for (int round = 0; round < 4; ++round) {
            int k_off = round * 32;
            uint16_t* B_tile = _kt_cache.data() +
                ((size_t)i * 4 + round) * (B_TILE_BYTES / 2);
            std::memset(B_tile, 0, B_TILE_BYTES);
            for (int k_pair = 0; k_pair < 16; ++k_pair) {
                for (int n = 0; n < tmax; ++n) {
                    int k_lo = k_off + 2 * k_pair;
                    int k_hi = k_lo + 1;
                    B_tile[k_pair * 32 + n * 2 + 0] = fp16_to_bf16(k_block[n * HEAD_DIM + k_lo]);
                    B_tile[k_pair * 32 + n * 2 + 1] = fp16_to_bf16(k_block[n * HEAD_DIM + k_hi]);
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
extern "C" void av_product(int cur_layer, int num_blocks, int seq_len,
                            const float* a, const _Float16* v_cache,
                            const int* block_table, float* o);
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
    ispc::softmax(seq_len, softmax_scale, a, asb);
    ispc::av_product(cur_layer, num_blocks, seq_len, a, v_cache, block_table, o);
}

}  // namespace amx_kernel
