// IDE_019 / TSK_035 — Jacobi parallel decoding AVX-512 kernel (skeleton)
//
// Lookahead decoding (USENIX OSDI'24) 의 CPU AVX-512 변형.
// K parallel hypothesis lane 으로 candidate token 을 iterate.
//
// status: ⚠ SKELETON — lossless proof + AVX-512 intrinsic 별도 turn

#include <immintrin.h>
#include <cstdint>
#include <vector>

namespace vllm_hybrid_jacobi {

/// Jacobi iteration kernel.
///
/// @param prev_hidden    [B, K, hidden] previous hidden states (K candidates per batch)
/// @param W_lm_head      [hidden, vocab] LM head weights
/// @param candidates_out [B, K] int32 candidate token IDs after iteration
/// @param B              batch size
/// @param K              number of parallel candidates (typically 5-7)
/// @param hidden         hidden size
/// @param vocab          vocab size
/// @param max_iters      max Jacobi iterations (typically 5-10 until convergence)
///
/// Algorithm:
///   iter=0: initialize K candidates randomly or from previous step
///   iter>0:
///     1. compute logits[B, K, vocab] = hidden @ W_lm_head (matmul)
///     2. argmax along vocab dim for each (b, k)
///     3. if all K converged (no change vs prev iter): break
///     4. else update candidates and continue
///
/// Lossless guarantee:
///   converged candidates must match autoregressive (Jacobi 의 fixed-point theorem)
void jacobi_iterate_avx512(
    const __bf16* prev_hidden,
    const __bf16* W_lm_head,
    int32_t* candidates_out,
    int B, int K, int hidden, int vocab,
    int max_iters
) {
    // SKELETON: scalar baseline (slow, for correctness reference)
    std::vector<int32_t> prev_cand(B * K, 0);
    std::vector<int32_t> cur_cand(B * K, 0);

    for (int iter = 0; iter < max_iters; ++iter) {
        bool all_converged = true;
        for (int b = 0; b < B; ++b) {
            for (int k = 0; k < K; ++k) {
                // logits[b, k, :] = prev_hidden[b, k, :] @ W_lm_head
                const __bf16* h = prev_hidden + (b * K + k) * hidden;
                float max_v = -1e30f;
                int max_i = 0;
                for (int v = 0; v < vocab; ++v) {
                    float dot = 0.0f;
                    for (int d = 0; d < hidden; ++d) {
                        dot += static_cast<float>(h[d]) * static_cast<float>(W_lm_head[d * vocab + v]);
                    }
                    if (dot > max_v) { max_v = dot; max_i = v; }
                }
                cur_cand[b * K + k] = max_i;
                if (max_i != prev_cand[b * K + k]) all_converged = false;
            }
        }
        // copy cur → prev for next iter
        prev_cand = cur_cand;
        if (all_converged) break;
    }

    // emit
    for (int i = 0; i < B * K; ++i) {
        candidates_out[i] = cur_cand[i];
    }

    // TODO: AVX-512 vectorize the matmul + argmax
    //   - K parallel lanes for vocab argmax (top-1 via _mm512_reduce_max_ps)
    //   - hidden 차원 16-wide BF16 SIMD
    //   - vocab tiling (152K / 16 = 9504 tiles)
}

}  // namespace vllm_hybrid_jacobi
