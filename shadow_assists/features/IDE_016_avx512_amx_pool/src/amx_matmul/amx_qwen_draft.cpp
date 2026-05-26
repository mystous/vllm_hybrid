// IDE_016 / TSK_026 — AMX BF16 matmul + Qwen MLP forward
//
// SUB_106 input: 22.05 TFLOPS peak BF16 (Qwen 7B B=256, 20.79× vs FP32).
// SUB_117 input: 10.24 TFLOPS available (N=32 pinned).
//
// AMX tile pattern (palette_id = 1):
//   TMM0 : C accumulator    16 rows × 64 bytes   (16 × FP32 = 16 elements/row, 16 rows)
//   TMM1 : A operand        16 rows × 64 bytes   (32 × BF16 = 32 elements/row, 16 rows)
//   TMM2 : B operand        16 rows × 64 bytes   (16 × (BF16,BF16) pair = 32 elements/row, 16 K-pair rows = 32 K)
//
//   _tile_dpbf16ps semantics:
//     For each row m of TMM0 (FP32 acc):
//       For each col n of TMM0:
//         C[m,n] += sum over k_pair in TMM1[m] of (A[m,2k]·B[2k,n] + A[m,2k+1]·B[2k+1,n])
//
// build (prod Sapphire Rapids 8480+):
//   g++ -O3 -mamx-tile -mamx-bf16 -mavx512f -mavx512vl \
//       -march=sapphirerapids -fPIC -c amx_qwen_draft.cpp
//
// 빌드 검증 :
//   g++ < 11 의 일부는 _tile_dpbf16ps prototype 부재. g++-11.4 / clang 14 + OK.

#include "../amx_matmul/amx_kernels.h"

#include <immintrin.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace vllm_hybrid_amx {

// AMX state syscall (Linux 5.16+)
#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif
#ifndef ARCH_GET_XCOMP_PERM
#define ARCH_GET_XCOMP_PERM 0x1022
#endif
#ifndef XFEATURE_XTILEDATA
#define XFEATURE_XTILEDATA 18
#endif

int amx_request_permission() {
    // arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)
    long rc = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM,
                     static_cast<unsigned long>(XFEATURE_XTILEDATA));
    return rc == 0 ? 0 : -1;
}

int amx_available() {
    // Check cpuid leaf 0x7 sub-leaf 0 EDX bit 24 (AMX_TILE)
    unsigned eax, ebx, ecx, edx;
    __asm__ __volatile__(
        "cpuid"
        : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
        : "a"(7), "c"(0));
    // AMX_TILE = EDX[24], AMX_BF16 = EDX[22]
    bool has_amx = ((edx >> 24) & 1) && ((edx >> 22) & 1);
    return has_amx ? 1 : 0;
}

// ──────────────────────────────────────────────────────────────────────
// Tile config (per-thread cached)
// ──────────────────────────────────────────────────────────────────────

struct alignas(64) AmxTileCfg {
    uint8_t  palette_id;
    uint8_t  start_row;
    uint8_t  reserved_0[14];
    uint16_t colsb[16];
    uint8_t  rows[16];
};

static thread_local bool t_cfg_loaded = false;
static thread_local bool t_perm_tried = false;

static void config_tiles_bf16_16x32_thread() {
    if (t_cfg_loaded) return;
    if (!t_perm_tried) {
        t_perm_tried = true;
        amx_request_permission();   // best-effort; may fail on dev hw
    }
    AmxTileCfg cfg = {};
    cfg.palette_id = 1;
    // TMM0: C accumulator — 16 rows × 16 FP32 elements (64 bytes / row)
    cfg.rows[0]  = 16;
    cfg.colsb[0] = 64;
    // TMM1: A operand — 16 rows × 32 BF16 elements (64 bytes / row)
    cfg.rows[1]  = 16;
    cfg.colsb[1] = 64;
    // TMM2: B operand — 16 K-pair rows × 16 (BF16,BF16) pair columns
    //   = 16 rows × 64 bytes (per row = 32 BF16 = 16 pairs)
    cfg.rows[2]  = 16;
    cfg.colsb[2] = 64;
    _tile_loadconfig(&cfg);
    t_cfg_loaded = true;
}

static void release_tiles_thread() {
    if (t_cfg_loaded) {
        _tile_release();
        t_cfg_loaded = false;
    }
}

// ──────────────────────────────────────────────────────────────────────
// Core BF16 matmul
// ──────────────────────────────────────────────────────────────────────

// Scalar fallback for shapes not satisfying AMX alignment.
static void scalar_matmul_bf16(const uint16_t* A, const uint16_t* B_packed,
                              float* C, int M, int K, int N) {
    // B_packed [K/2, N, 2] BF16
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
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
}

void amx_matmul_bf16(const uint16_t* A, const uint16_t* B_packed,
                    float* C, int M, int K, int N) {
    if (M <= 0 || K <= 0 || N <= 0) return;
    if ((M % 16) || (K % 32) || (N % 16) || !amx_available()) {
        scalar_matmul_bf16(A, B_packed, C, M, K, N);
        return;
    }

    config_tiles_bf16_16x32_thread();

    const int K_PACKED = K / 2;   // K-pair rows in B_packed
    const size_t A_row_bytes = static_cast<size_t>(K) * sizeof(uint16_t);
    const size_t B_pair_row_bytes = static_cast<size_t>(N) * 2 * sizeof(uint16_t);
    const size_t C_row_bytes = static_cast<size_t>(N) * sizeof(float);

    for (int m = 0; m < M; m += 16) {
        for (int n = 0; n < N; n += 16) {
            _tile_zero(0);
            for (int k = 0; k < K; k += 32) {
                // A tile: 16 rows × 32 BF16 at A[m..m+16, k..k+32]
                _tile_loadd(1, A + static_cast<size_t>(m) * K + k,
                           static_cast<long>(A_row_bytes));
                // B tile: 16 K-pair rows × 16 (BF16,BF16) pair cols
                //  at B_packed[k/2 .. k/2+16, n..n+16, *]
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
    // Note: thread-local tile config 보존 — 다음 호출 시 reload 안 함.
    // application 종료 시 thread join 으로 자연 해제됨.
    (void)release_tiles_thread;   // referenced for documentation
}

// ──────────────────────────────────────────────────────────────────────
// B-matrix repack (row-major [K, N] → AMX [K/2, N, 2])
// ──────────────────────────────────────────────────────────────────────

void amx_repack_b_bf16(const uint16_t* B_in, uint16_t* B_out, int K, int N) {
    // K 가 홀수면 마지막 row pad 0
    int K_eff = K & ~1;
    for (int k = 0; k < K_eff; k += 2) {
        int kp = k / 2;
        for (int n = 0; n < N; ++n) {
            B_out[(static_cast<size_t>(kp) * N + n) * 2 + 0] = B_in[(k + 0) * N + n];
            B_out[(static_cast<size_t>(kp) * N + n) * 2 + 1] = B_in[(k + 1) * N + n];
        }
    }
    if (K & 1) {
        int kp = K_eff / 2;
        for (int n = 0; n < N; ++n) {
            B_out[(static_cast<size_t>(kp) * N + n) * 2 + 0] = B_in[K_eff * N + n];
            B_out[(static_cast<size_t>(kp) * N + n) * 2 + 1] = 0;
        }
    }
}


// ──────────────────────────────────────────────────────────────────────
// FP32 → BF16 cast (round-to-nearest-even) via AVX-512F
// ──────────────────────────────────────────────────────────────────────

static void fp32_to_bf16_array(const float* in, uint16_t* out, size_t N) {
    size_t i = 0;
    for (; i + 16 <= N; i += 16) {
        __m512 x = _mm512_loadu_ps(in + i);
        __m512i u32 = _mm512_castps_si512(x);
        __m512i lsb = _mm512_and_si512(_mm512_srli_epi32(u32, 16),
                                       _mm512_set1_epi32(1));
        __m512i rounded = _mm512_add_epi32(
            _mm512_add_epi32(u32, _mm512_set1_epi32(0x8000)), lsb);
        __m512i bf = _mm512_srli_epi32(rounded, 16);
        __m256i bf16 = _mm512_cvtepi32_epi16(bf);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), bf16);
    }
    for (; i < N; ++i) {
        uint32_t b;
        std::memcpy(&b, in + i, sizeof(float));
        uint32_t lsb = (b >> 16) & 1;
        b = b + 0x8000 + lsb;
        out[i] = static_cast<uint16_t>(b >> 16);
    }
}


// ──────────────────────────────────────────────────────────────────────
// Qwen MLP forward — gate / up / down with SiLU
// ──────────────────────────────────────────────────────────────────────
//
// 두 matmul (gate / up) 후 element-wise silu(g) * u → BF16 → 1 matmul (down).
// scratch FP32 area layout:
//   [0, B*I)         : gate_proj FP32
//   [B*I, 2*B*I)     : up_proj   FP32
//   [2*B*I, 3*B*I)   : down_in   FP32 (silu(gate)*up)
// + down_in_bf16 : 별도 ~B*I*2 byte 영역 — caller 의 scratch 의 마지막 B*I/2 floats
//   = B*I uint16_t 를 reinterpret.

void qwen_mlp_forward_bf16(const uint16_t* input,
                          const uint16_t* W_gate_packed,
                          const uint16_t* W_up_packed,
                          const uint16_t* W_down_packed,
                          uint16_t* output,
                          float* scratch,
                          int B,
                          const QwenMLPShape& s) {
    const int H = s.hidden;
    const int I = s.intermediate;
    const size_t BI = static_cast<size_t>(B) * I;

    float* gate_proj = scratch;
    float* up_proj   = scratch + BI;
    float* down_in_f = scratch + 2 * BI;

    // Wgate / Wup : input · W → FP32
    amx_matmul_bf16(input, W_gate_packed, gate_proj, B, H, I);
    amx_matmul_bf16(input, W_up_packed,   up_proj,   B, H, I);

    // silu(gate) * up — vectorize the scalar pointwise.
    // sigmoid(x) = 1 / (1 + exp(-x)) ; silu(x) = x * sigmoid(x)
    size_t i = 0;
    for (; i + 16 <= BI; i += 16) {
        __m512 g = _mm512_loadu_ps(gate_proj + i);
        __m512 u = _mm512_loadu_ps(up_proj + i);
        // scalar fallback for exp
        alignas(64) float gtmp[16];
        _mm512_storeu_ps(gtmp, g);
        for (int j = 0; j < 16; ++j) {
            float gv = gtmp[j];
            float sig = 1.0f / (1.0f + std::exp(-gv));
            gtmp[j] = gv * sig;
        }
        __m512 silu_g = _mm512_loadu_ps(gtmp);
        __m512 res = _mm512_mul_ps(silu_g, u);
        _mm512_storeu_ps(down_in_f + i, res);
    }
    for (; i < BI; ++i) {
        float gv = gate_proj[i];
        float sig = 1.0f / (1.0f + std::exp(-gv));
        down_in_f[i] = gv * sig * up_proj[i];
    }

    // FP32 → BF16 cast in-place (overwrite down_in_f's lower half)
    uint16_t* down_in_bf = reinterpret_cast<uint16_t*>(down_in_f);
    fp32_to_bf16_array(down_in_f, down_in_bf, BI);

    // Down projection: down_in_bf [B, I] · W_down_packed → output_f, then cast to BF16
    // Reuse gate_proj area for output_f (no longer needed)
    float* output_f = gate_proj;
    amx_matmul_bf16(down_in_bf, W_down_packed, output_f, B, I, H);
    fp32_to_bf16_array(output_f, output, static_cast<size_t>(B) * H);
}

}  // namespace vllm_hybrid_amx
