// IDE_016 / SUB_201 §5 B1 PoC — C ABI shim for ctypes
//
// Wraps the C++ entry points of batch_bpe_kernel.cpp so they can be loaded
// directly via Python ctypes (no mangled name lookup required).
//
// Build (in the kernel build dir, alongside libavx512_tokenizer.so):
//   g++ -O3 -mavx512f -mavx512bw -mavx512vbmi -mavx512bf16 \
//       -march=sapphirerapids -fopenmp -fPIC -shared \
//       -I src/avx512_tokenizer \
//       src/avx512_tokenizer/batch_bpe_kernel.cpp \
//       src/avx512_tokenizer/c_shim.cpp \
//       -o build/avx512_tokenizer/libavx512_tokenizer.so

#include "tokenizer_kernels.h"

extern "C" {

void avx512_batch_detokenize_bytes(
    const uint8_t* pieces,
    const int32_t* offsets,
    const int32_t* sizes,
    int32_t        V,
    int32_t        total_bytes,
    const int32_t* token_ids,
    const int32_t* seq_offsets,
    int            B,
    uint8_t*       out_bytes,
    int32_t*       out_byte_offsets,
    int32_t*       out_byte_lengths,
    int            use_avx512) {

    vllm_hybrid_tok::VocabTable table{pieces, offsets, sizes, V, total_bytes};
    if (use_avx512) {
        vllm_hybrid_tok::batch_detokenize_bytes_avx512(
            table, token_ids, seq_offsets, B,
            out_bytes, out_byte_offsets, out_byte_lengths);
    } else {
        vllm_hybrid_tok::batch_detokenize_bytes_scalar(
            table, token_ids, seq_offsets, B,
            out_bytes, out_byte_offsets, out_byte_lengths);
    }
}

int64_t avx512_batch_detokenize_byte_total(
    const uint8_t* pieces,
    const int32_t* offsets,
    const int32_t* sizes,
    int32_t        V,
    int32_t        total_bytes,
    const int32_t* token_ids,
    int            total_tokens) {

    vllm_hybrid_tok::VocabTable table{pieces, offsets, sizes, V, total_bytes};
    return vllm_hybrid_tok::batch_detokenize_byte_total(
        table, token_ids, total_tokens);
}

}  // extern "C"
