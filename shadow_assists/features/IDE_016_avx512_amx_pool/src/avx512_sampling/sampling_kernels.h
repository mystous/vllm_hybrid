// IDE_016 / TSK_025 — AVX-512 sampling kernel public API
//
// SUB_161 lever: trident TP0 의 44.3% CPU 시간 = sampler.py.
// 본 헤더는 topk_topp_kernel.cpp / logit_processor.cpp / penalty_ops.cpp 의
// 외부 노출 함수 시그너처를 한곳에 모은다 (python_bindings.cpp 가 include).
//
// dtype 약속:
//   - logits BF16 (vLLM v1 sampler 의 가장 흔한 dtype). FP32 fast-path 도 제공.
//   - probs / output FP32.
//   - indices int32 (vocab ≤ 2³¹).

#pragma once

#include <cstdint>

namespace vllm_hybrid_avx512 {

// ──────────────────────────────────────────────────────────────────────
// top-k / top-p (topk_topp_kernel.cpp)
// ──────────────────────────────────────────────────────────────────────

/// top-k partial sort, BF16 logits in. AVX-512 streaming threshold + compress.
/// indices_out [B, K] int32, values_out [B, K] FP32 (descending).
void topk_avx512_bf16(const uint16_t* logits_bf16,
                     int B, int V, int K,
                     int32_t* indices_out,
                     float* values_out);

/// top-k partial sort, FP32 logits in. Same algorithm but BF16 upcast skipped.
void topk_avx512_fp32(const float* logits_fp32,
                     int B, int V, int K,
                     int32_t* indices_out,
                     float* values_out);

/// top-p (nucleus) cutoff. Probs must be FP32 descending-sorted per batch.
/// cutoff_out [B] int32 = number of tokens to keep (≥1).
void topp_avx512_fp32(const float* sorted_probs,
                     int B, int V, float p,
                     int32_t* cutoff_out);

/// Fused softmax + top-k + top-p, BF16 logits in. Returns sampled token.
///
/// Pipeline (single pass over vocab where possible):
///   1. apply temperature: logits / T  (in-place workspace)
///   2. find max (stability) → AVX-512 reduce
///   3. exp(logits - max) → workspace FP32
///   4. running threshold for top-K (compress + partial sort)
///   5. normalize to probs over top-K
///   6. apply top-p truncation on top-K subset
///   7. categorical sample (Gumbel-max with uniform RNG seed)
///
/// rng_seed: per-call seed (caller may pass torch.Generator state).
/// sampled_token_out [B] int64.
void fused_sample_avx512_bf16(const uint16_t* logits_bf16,
                             int B, int V, int K, float p,
                             float temperature,
                             uint64_t rng_seed,
                             int64_t* sampled_token_out);

/// Fused softmax + top-k + top-p, FP32 logits.
void fused_sample_avx512_fp32(const float* logits_fp32,
                             int B, int V, int K, float p,
                             float temperature,
                             uint64_t rng_seed,
                             int64_t* sampled_token_out);

// ──────────────────────────────────────────────────────────────────────
// logit processors (logit_processor.cpp)
// ──────────────────────────────────────────────────────────────────────

/// Apply temperature in-place. `out = in / T` (T≤0 → identity).
/// BF16 in, BF16 out — converts via FP32 intermediate.
void apply_temperature_avx512_bf16(uint16_t* logits, int N, float temperature);

/// Apply temperature in-place, FP32.
void apply_temperature_avx512_fp32(float* logits, int N, float temperature);

/// Apply logit bias: logits[i] += bias[i]. Both [B, V].
/// bias may be null (no-op).
void apply_logit_bias_avx512_fp32(float* logits, const float* bias,
                                 int B, int V);

/// Apply additive bias on selected token IDs (sparse path).
/// idx [Nb] / bias [Nb] : logits[batch_row, idx[i]] += bias[i].
void apply_logit_bias_sparse_avx512(float* logits, int V,
                                   const int32_t* idx,
                                   const float* bias,
                                   int batch_row, int Nb);

/// Stable softmax over each row in-place. probs_out can alias logits.
/// FP32 logits, FP32 probs out.
void softmax_avx512_fp32(const float* logits, float* probs, int B, int V);

// ──────────────────────────────────────────────────────────────────────
// penalty operations (penalty_ops.cpp)
// ──────────────────────────────────────────────────────────────────────

/// Repetition penalty (HuggingFace formula):
///   if logits[i] > 0: logits[i] /= penalty
///   else            : logits[i] *= penalty
/// applied at positions listed in `token_ids` (already-seen tokens).
/// logits [B, V], token_ids [B, max_seen], lengths [B] (per-row valid count).
void apply_repetition_penalty_avx512(float* logits, int B, int V,
                                    const int32_t* token_ids,
                                    const int32_t* lengths,
                                    int max_seen,
                                    float penalty);

/// Frequency penalty: logits[i] -= freq[i] * alpha.
/// freq [B, V] int32 (per-token occurrence count).
void apply_frequency_penalty_avx512(float* logits, int B, int V,
                                   const int32_t* freq, float alpha);

/// Presence penalty: for each token with freq > 0, logits[i] -= alpha.
/// (binary version of frequency penalty.)
void apply_presence_penalty_avx512(float* logits, int B, int V,
                                  const int32_t* freq, float alpha);

}  // namespace vllm_hybrid_avx512
