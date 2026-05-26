// IDE_016 / TSK_026 — AMX matmul public API
//
// SUB_106 reference: 22.05 TFLOPS BF16 peak on Sapphire Rapids (single worker,
// Qwen 7B shape B=256). 본 모듈은 Qwen 0.5B/1.5B/32B draft + main path 양쪽
// 모두 지원하는 generic BF16 matmul + Qwen MLP forward 의 fused entry.
//
// AMX 사용 흐름:
//   1) one-time tile config (palette_id=1, BF16 16x32)
//   2) for each (M, N) tile:
//        _tile_zero(C_acc)
//        for k_block in K:
//            _tile_loadd(A_tile)
//            _tile_loadd(B_tile)   ← K-major packed
//            _tile_dpbf16ps(C_acc, A_tile, B_tile)
//        _tile_stored(C_acc)
//   3) _tile_release after batch
//
// 본 API 는 caller 가 multi-thread 호출할 때 *thread-local* tile cfg 유지.
// __thread bool 로 tile cfg loaded 여부 캐싱.

#pragma once

#include <cstdint>

namespace vllm_hybrid_amx {

// ──────────────────────────────────────────────────────────────────────
// Core BF16 matmul
// ──────────────────────────────────────────────────────────────────────

/// C[M,N] = A[M,K] · B[K,N]  (BF16 × BF16 → FP32 accumulate)
///
/// A   : row-major [M, K] BF16 (uint16_t storage)
/// B   : AMX-packed [K/2, N, 2] BF16 — see amx_repack_b_bf16 helper.
/// C   : row-major [M, N] FP32
///
/// Requirements: M % 16 == 0, K % 32 == 0, N % 16 == 0.
/// Non-aligned shapes 는 caller 가 pad (or fallback scalar).
///
/// Thread safety: 함수 안에서 thread-local _tile_loadconfig 만 1 회 호출.
/// 여러 thread 에서 동시 호출 안전.
void amx_matmul_bf16(const uint16_t* A,
                    const uint16_t* B_packed,
                    float* C,
                    int M, int K, int N);

/// Row-major [K, N] BF16 → AMX-packed [K/2, N, 2] BF16. K must be even.
void amx_repack_b_bf16(const uint16_t* B_in, uint16_t* B_out,
                      int K, int N);

/// AMX availability probe (cpuid + AMX_TILE state). Returns 1 if usable.
int amx_available();

/// One-time XCR0 enable for AMX. Returns 0 on success, -1 if denied.
int amx_request_permission();

// ──────────────────────────────────────────────────────────────────────
// Qwen MLP forward (gate / up / down + SiLU)
// ──────────────────────────────────────────────────────────────────────

struct QwenMLPShape {
    int hidden;        // 896 / 1536 / 5120
    int intermediate;  // 4864 / 8960 / 27648
};

/// Standard Qwen MLP: out = (silu(input · W_gate) ⊙ (input · W_up)) · W_down
///
/// input    [B, hidden] BF16
/// W_gate   AMX-packed [hidden/2, intermediate, 2] BF16
/// W_up     AMX-packed [hidden/2, intermediate, 2] BF16
/// W_down   AMX-packed [intermediate/2, hidden, 2] BF16
/// output   [B, hidden] BF16
/// scratch  [B * intermediate * 3] FP32 workspace (caller allocates, ≥ 64-byte aligned)
void qwen_mlp_forward_bf16(const uint16_t* input,
                          const uint16_t* W_gate_packed,
                          const uint16_t* W_up_packed,
                          const uint16_t* W_down_packed,
                          uint16_t* output,
                          float* scratch,
                          int B,
                          const QwenMLPShape& s);

/// Predefined Qwen shapes for convenience.
constexpr QwenMLPShape QWEN_0_5B { 896, 4864 };
constexpr QwenMLPShape QWEN_1_5B { 1536, 8960 };
constexpr QwenMLPShape QWEN_32B  { 5120, 27648 };

}  // namespace vllm_hybrid_amx
