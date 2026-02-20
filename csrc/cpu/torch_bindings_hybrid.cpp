// CPU hybrid extension: Phase 1-5 kernels for CUDA+CPU hybrid builds.
//
// This file registers only Phase 1-5 CPU kernels as the _C_cpu_ops
// extension, separate from the main _C CUDA extension.
//
// Unlike torch_bindings.cpp (which includes ops.h, cache.h, shm.h, etc.),
// this file has minimal dependencies - only the Phase 1-5 kernel headers.

#include <torch/all.h>
#include <torch/library.h>

#include "core/registration.h"

// Phase 1: VNNI INT8 GEMM
#if defined(__AVX512F__) && defined(__AVX512VNNI__)
#include "gemm_vnni.hpp"

// Phase 2: Q8_0 quantization
void q8_0_linear(torch::Tensor& output, const torch::Tensor& input,
                 const torch::Tensor& qweight,
                 const std::optional<torch::Tensor>& bias);
void q8_0_quantize_weight(torch::Tensor& qweight,
                          const torch::Tensor& weight);
#endif

// Phase 3: Decode GEMV
#ifdef __AVX512F__
void decode_gemv(torch::Tensor& output, const torch::Tensor& input,
                 const torch::Tensor& weight,
                 const std::optional<torch::Tensor>& bias);

// Phase 4: Batch-16 paged attention
void batch16_paged_attention_v1(
    torch::Tensor& output, const torch::Tensor& query,
    const torch::Tensor& key_cache, const torch::Tensor& value_cache,
    const torch::Tensor& block_tables, const torch::Tensor& context_lens,
    int num_heads, int head_size, int block_size, int max_blocks_per_seq,
    float scale, int num_kv_heads);

// Phase 5: Memory bandwidth optimization
void nt_memcpy_tensor(torch::Tensor& dst, const torch::Tensor& src);
void prefetch_kv_cache_blocks(const torch::Tensor& kv_cache,
                              const torch::Tensor& block_table,
                              int num_blocks);
#endif

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
#ifdef __AVX512F__
  // Phase 3: Decode GEMV (BF16/FP32, optimized for M<=32)
  ops.def(
      "decode_gemv(Tensor! output, Tensor input, "
      "Tensor weight, Tensor? bias) -> ()");
  ops.impl("decode_gemv", torch::kCPU, &decode_gemv);

  // Phase 5: Memory bandwidth optimization
  ops.def("nt_memcpy_tensor(Tensor! dst, Tensor src) -> ()");
  ops.impl("nt_memcpy_tensor", torch::kCPU, &nt_memcpy_tensor);
  ops.def(
      "prefetch_kv_cache_blocks(Tensor kv_cache, Tensor block_table, "
      "int num_blocks) -> ()");
  ops.impl("prefetch_kv_cache_blocks", torch::kCPU,
           &prefetch_kv_cache_blocks);
#endif

  // AVX-512 VNNI kernels (no AVX512BF16/AMX required)
#if defined(__AVX512F__) && defined(__AVX512VNNI__)
  // Phase 1: VNNI INT8 GEMM
  ops.def(
      "vnni_int8_gemm(Tensor! out, Tensor a, Tensor b_packed, "
      "Tensor b_comp, Tensor a_scales, Tensor b_scales, "
      "Tensor? bias) -> ()");
  ops.impl("vnni_int8_gemm", torch::kCPU, &vllm::vnni::vnni_int8_gemm);
  ops.def(
      "vnni_pack_weight(Tensor! packed, Tensor! comp, "
      "Tensor weight) -> ()");
  ops.impl("vnni_pack_weight", torch::kCPU, &vllm::vnni::vnni_pack_weight);

  // Phase 2: Q8_0 quantization (llama.cpp compatible)
  ops.def(
      "q8_0_linear(Tensor! output, Tensor input, "
      "Tensor qweight, Tensor? bias) -> ()");
  ops.impl("q8_0_linear", torch::kCPU, &q8_0_linear);
  ops.def(
      "q8_0_quantize_weight(Tensor! qweight, Tensor weight) -> ()");
  ops.impl("q8_0_quantize_weight", torch::kCPU, &q8_0_quantize_weight);
#endif
}

TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _cpu), cpu_ops) {
#ifdef __AVX512F__
  // Phase 4: Batch-16 paged attention
  cpu_ops.def(
      "batch16_paged_attention_v1("
      "   Tensor! output, Tensor query, Tensor key_cache,"
      "   Tensor value_cache, Tensor block_tables, Tensor context_lens,"
      "   int num_heads, int head_size, int block_size,"
      "   int max_blocks_per_seq, float scale, int num_kv_heads) -> ()");
  cpu_ops.impl("batch16_paged_attention_v1", torch::kCPU,
               &batch16_paged_attention_v1);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
