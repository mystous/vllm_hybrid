// AVX-512 VNNI INT8 GEMM Kernel
// Provides high-performance INT8 GEMM for environments with AVX512F+VNNI
// but without AVX512BF16 or AMX support (e.g., Cascade Lake, Ice Lake).
//
// Uses u8s8 format with s8s8 compensation:
//   a * b = (a + 128) * b - 128 * b
//   s   s       u       s    u    s

#pragma once

#include <immintrin.h>
#include <torch/all.h>

#include <cstdint>

#if defined(__AVX512F__) && defined(__AVX512VNNI__)

namespace vllm {
namespace vnni {

// Cache blocking parameters tuned for Xeon L2/L3 cache sizes
constexpr int MC = 72;   // Row tile (multiple of MR=6)
constexpr int NC = 256;  // Column tile (multiple of NR=16)
constexpr int KC = 256;  // K tile (multiple of 4 for VNNI)
constexpr int MR = 6;    // Micro-kernel rows (uses 6 ZMM accumulators)
constexpr int NR = 16;   // Micro-kernel columns (1 ZMM register width)

// ============================================================================
// Weight packing: row-major [N,K] s8 -> VNNI format [N/NR, K/4, NR, 4]
// Also computes s8s8 compensation: comp[n] = sum_k(B[n][k]) * 128
// ============================================================================
void pack_weight_vnni(const int8_t* src, int8_t* dst,
                      int32_t* comp, int N, int K);

// ============================================================================
// 6x16 micro-kernel: C[6,16] += A_u8[6,K] * B_vnni[K/4,16,4]
// A is u8 (= s8 + 128), B is VNNI-packed s8
// comp[16] is s8s8 compensation: sum_k(B[n][k]) * 128
// ============================================================================
void vnni_micro_kernel_6x16(const uint8_t* A, const int8_t* B_packed,
                            int32_t* C, int K, int lda, int ldc,
                            const int32_t* comp);

// ============================================================================
// INT8 GEMM with cache blocking
// A: [M, K] uint8 (pre-shifted: original_s8 + 128)
// B_packed: VNNI-packed [N/16, K/4, 16, 4] int8
// comp: [N] int32 s8s8 compensation
// C: [M, N] int32 accumulator output
// ============================================================================
void int8_gemm_vnni(const uint8_t* A, const int8_t* B_packed,
                    const int32_t* comp, int32_t* C,
                    int M, int N, int K, int lda, int ldc);

// ============================================================================
// M=1 specialized GEMV (avoids cache blocking overhead)
// ============================================================================
void int8_gemv_vnni(const uint8_t* x, const int8_t* B_packed,
                    const int32_t* comp, int32_t* y, int N, int K);

// ============================================================================
// Dequantize int32 accumulator to output dtype (BF16 or FP32)
// Supports per-token activation scales and per-channel weight scales
// output = (int32_acc - comp) * a_scale * b_scale + bias
// ============================================================================
void dequant_int32_to_output(const int32_t* input, void* output,
                             at::ScalarType dtype, const float* a_scales,
                             const float* b_scales, const float* bias,
                             int M, int N, int ldc, bool per_token_a,
                             bool per_channel_b);

// ============================================================================
// Torch entry points
// ============================================================================

// Full VNNI INT8 GEMM with dequantization
// out: [M, N] BF16/FP32
// a: [M, K] INT8 (will be shifted to u8 internally)
// b_packed: [N/16, K/4, 16, 4] INT8 (VNNI packed)
// b_comp: [N] INT32
// a_scales: [M] or [1] float
// b_scales: [N] or [1] float
// bias: optional [N] float
void vnni_int8_gemm(torch::Tensor& out, const torch::Tensor& a,
                    const torch::Tensor& b_packed,
                    const torch::Tensor& b_comp,
                    const torch::Tensor& a_scales,
                    const torch::Tensor& b_scales,
                    const std::optional<torch::Tensor>& bias);

// Pack weight from row-major s8 to VNNI format
// packed: [N/16, K/4, 16, 4] INT8 (output)
// comp: [N] INT32 (output)
// weight: [N, K] INT8 (input)
void vnni_pack_weight(torch::Tensor& packed, torch::Tensor& comp,
                      const torch::Tensor& weight);

}  // namespace vnni
}  // namespace vllm

#endif  // __AVX512F__ && __AVX512VNNI__
