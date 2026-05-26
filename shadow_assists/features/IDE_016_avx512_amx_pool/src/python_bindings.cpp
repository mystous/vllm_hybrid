// IDE_016 — Python bindings for AVX-512 sampling + AMX matmul.
//
// import path: `avx512_amx_pool._core` (CMake target name).
//
// ABI 결정: torch::Tensor 직접 받는 대신 numpy ndarray 만 사용한다 (`pybind11/numpy.h`).
// 이유: torch 2.11 은 GCC 13.3 + libstdc++ 다른 ABI 로 빌드되어 GCC 11.4 host
// 와 type_caster symbol mismatch 발생. numpy ndarray 인터페이스만 쓰면 torch
// extension 의존성이 사라져 어떤 GCC/Python 조합에서도 link 가능.
//
// Python 측 wrapper 가 torch.Tensor ↔ numpy.ndarray view 변환을 zero-copy 로
// 처리 (torch.from_numpy / tensor.numpy()).
//
// 지원 dtype:
//   logits BF16 : numpy uint16 (BF16 비트 패턴 그대로)
//   logits FP32 : numpy float32
//   indices     : numpy int64 출력
//   probs       : numpy float32

#include "avx512_sampling/sampling_kernels.h"
#include "amx_matmul/amx_kernels.h"
#include "avx512_tokenizer/tokenizer_kernels.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <string>

namespace py = pybind11;
using namespace vllm_hybrid_avx512;
using namespace vllm_hybrid_amx;
namespace tok = vllm_hybrid_tok;

// ──────────────────────────────────────────────────────────────────────
// Runtime capability probe (CPU + ISA)
// ──────────────────────────────────────────────────────────────────────

static bool cpu_has_avx512() {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512f")
        && __builtin_cpu_supports("avx512bw")
        && __builtin_cpu_supports("avx512vl")
        && __builtin_cpu_supports("avx512dq");
}

static bool cpu_has_amx_pybind() {
    return amx_available() != 0;
}


// ──────────────────────────────────────────────────────────────────────
// numpy view validators
// ──────────────────────────────────────────────────────────────────────

template <typename T>
static py::array_t<T, py::array::c_style> ensure_c(py::array_t<T> arr,
                                                  int expected_ndim,
                                                  const char* name) {
    py::array_t<T, py::array::c_style> out = arr;   // forces contiguous
    if (out.ndim() != expected_ndim) {
        throw std::invalid_argument(std::string(name) + " expected ndim "
                                   + std::to_string(expected_ndim));
    }
    return out;
}


// ──────────────────────────────────────────────────────────────────────
// top-k / top-p / fused sample bindings
// ──────────────────────────────────────────────────────────────────────

// Top-K: returns (indices [B, K] int64, values [B, K] float32)
static py::tuple py_topk_bf16(py::array_t<uint16_t, py::array::c_style> logits, int K) {
    auto buf = logits.request();
    if (buf.ndim != 2) throw std::invalid_argument("logits must be 2D");
    int B = static_cast<int>(buf.shape[0]);
    int V = static_cast<int>(buf.shape[1]);
    if (K <= 0 || K > V) throw std::invalid_argument("K out of range");

    py::array_t<int64_t> idx({B, K});
    py::array_t<float>   val({B, K});

    // need int32 temp then upcast to int64 for return
    std::vector<int32_t> idx_i32(static_cast<size_t>(B) * K);
    topk_avx512_bf16(static_cast<const uint16_t*>(buf.ptr),
                    B, V, K,
                    idx_i32.data(),
                    val.mutable_data());
    int64_t* idx_ptr = idx.mutable_data();
    for (size_t i = 0; i < idx_i32.size(); ++i) idx_ptr[i] = idx_i32[i];
    return py::make_tuple(idx, val);
}

static py::tuple py_topk_fp32(py::array_t<float, py::array::c_style> logits, int K) {
    auto buf = logits.request();
    if (buf.ndim != 2) throw std::invalid_argument("logits must be 2D");
    int B = static_cast<int>(buf.shape[0]);
    int V = static_cast<int>(buf.shape[1]);
    if (K <= 0 || K > V) throw std::invalid_argument("K out of range");

    py::array_t<int64_t> idx({B, K});
    py::array_t<float>   val({B, K});
    std::vector<int32_t> idx_i32(static_cast<size_t>(B) * K);
    topk_avx512_fp32(static_cast<const float*>(buf.ptr), B, V, K,
                    idx_i32.data(), val.mutable_data());
    int64_t* idx_ptr = idx.mutable_data();
    for (size_t i = 0; i < idx_i32.size(); ++i) idx_ptr[i] = idx_i32[i];
    return py::make_tuple(idx, val);
}

static py::array_t<int64_t> py_topp(py::array_t<float, py::array::c_style> probs, float p) {
    auto buf = probs.request();
    if (buf.ndim != 2) throw std::invalid_argument("probs must be 2D");
    int B = static_cast<int>(buf.shape[0]);
    int V = static_cast<int>(buf.shape[1]);
    py::array_t<int64_t> cut(B);
    std::vector<int32_t> cut_i32(B);
    topp_avx512_fp32(static_cast<const float*>(buf.ptr), B, V, p, cut_i32.data());
    int64_t* cp = cut.mutable_data();
    for (int b = 0; b < B; ++b) cp[b] = cut_i32[b];
    return cut;
}

static py::array_t<int64_t> py_fused_sample_bf16(
        py::array_t<uint16_t, py::array::c_style> logits,
        int K, float p, float temperature, uint64_t rng_seed) {
    auto buf = logits.request();
    if (buf.ndim != 2) throw std::invalid_argument("logits must be 2D");
    int B = static_cast<int>(buf.shape[0]);
    int V = static_cast<int>(buf.shape[1]);
    if (K <= 0) K = V;
    if (K > V) K = V;
    py::array_t<int64_t> out(B);
    fused_sample_avx512_bf16(static_cast<const uint16_t*>(buf.ptr),
                            B, V, K, p, temperature, rng_seed,
                            out.mutable_data());
    return out;
}

static py::array_t<int64_t> py_fused_sample_fp32(
        py::array_t<float, py::array::c_style> logits,
        int K, float p, float temperature, uint64_t rng_seed) {
    auto buf = logits.request();
    if (buf.ndim != 2) throw std::invalid_argument("logits must be 2D");
    int B = static_cast<int>(buf.shape[0]);
    int V = static_cast<int>(buf.shape[1]);
    if (K <= 0) K = V;
    if (K > V) K = V;
    py::array_t<int64_t> out(B);
    fused_sample_avx512_fp32(static_cast<const float*>(buf.ptr),
                            B, V, K, p, temperature, rng_seed,
                            out.mutable_data());
    return out;
}


// ──────────────────────────────────────────────────────────────────────
// Logit processor bindings (FP32 only — BF16 is rare for these ops)
// ──────────────────────────────────────────────────────────────────────

static void py_apply_temperature_fp32(py::array_t<float, py::array::c_style> logits,
                                      float temperature) {
    auto buf = logits.request();
    apply_temperature_avx512_fp32(static_cast<float*>(buf.ptr),
                                 static_cast<int>(buf.size), temperature);
}

static void py_apply_temperature_bf16(py::array_t<uint16_t, py::array::c_style> logits,
                                      float temperature) {
    auto buf = logits.request();
    apply_temperature_avx512_bf16(static_cast<uint16_t*>(buf.ptr),
                                 static_cast<int>(buf.size), temperature);
}

static void py_apply_logit_bias(py::array_t<float, py::array::c_style> logits,
                                py::array_t<float, py::array::c_style> bias) {
    auto lb = logits.request();
    auto bb = bias.request();
    if (lb.ndim != 2 || bb.ndim != 2) throw std::invalid_argument("2D required");
    if (lb.shape[0] != bb.shape[0] || lb.shape[1] != bb.shape[1])
        throw std::invalid_argument("logits and bias shape mismatch");
    apply_logit_bias_avx512_fp32(static_cast<float*>(lb.ptr),
                                static_cast<const float*>(bb.ptr),
                                static_cast<int>(lb.shape[0]),
                                static_cast<int>(lb.shape[1]));
}

static py::array_t<float> py_softmax(py::array_t<float, py::array::c_style> logits) {
    auto buf = logits.request();
    if (buf.ndim != 2) throw std::invalid_argument("2D required");
    int B = static_cast<int>(buf.shape[0]);
    int V = static_cast<int>(buf.shape[1]);
    py::array_t<float> out({B, V});
    softmax_avx512_fp32(static_cast<const float*>(buf.ptr),
                       out.mutable_data(), B, V);
    return out;
}


// ──────────────────────────────────────────────────────────────────────
// Penalty bindings
// ──────────────────────────────────────────────────────────────────────

static void py_apply_repetition_penalty(
        py::array_t<float, py::array::c_style> logits,
        py::array_t<int32_t, py::array::c_style> token_ids,
        py::array_t<int32_t, py::array::c_style> lengths,
        float penalty) {
    auto lb = logits.request();
    auto tb = token_ids.request();
    auto lbn = lengths.request();
    if (lb.ndim != 2 || tb.ndim != 2 || lbn.ndim != 1)
        throw std::invalid_argument("shape mismatch");
    if (lb.shape[0] != tb.shape[0] || lb.shape[0] != lbn.shape[0])
        throw std::invalid_argument("B mismatch");
    apply_repetition_penalty_avx512(
        static_cast<float*>(lb.ptr),
        static_cast<int>(lb.shape[0]), static_cast<int>(lb.shape[1]),
        static_cast<const int32_t*>(tb.ptr),
        static_cast<const int32_t*>(lbn.ptr),
        static_cast<int>(tb.shape[1]),
        penalty);
}

static void py_apply_frequency_penalty(
        py::array_t<float, py::array::c_style> logits,
        py::array_t<int32_t, py::array::c_style> freq,
        float alpha) {
    auto lb = logits.request();
    auto fb = freq.request();
    if (lb.ndim != 2 || fb.ndim != 2) throw std::invalid_argument("2D required");
    if (lb.shape != fb.shape) throw std::invalid_argument("shape mismatch");
    apply_frequency_penalty_avx512(
        static_cast<float*>(lb.ptr),
        static_cast<int>(lb.shape[0]), static_cast<int>(lb.shape[1]),
        static_cast<const int32_t*>(fb.ptr), alpha);
}

static void py_apply_presence_penalty(
        py::array_t<float, py::array::c_style> logits,
        py::array_t<int32_t, py::array::c_style> freq,
        float alpha) {
    auto lb = logits.request();
    auto fb = freq.request();
    if (lb.ndim != 2 || fb.ndim != 2) throw std::invalid_argument("2D required");
    if (lb.shape != fb.shape) throw std::invalid_argument("shape mismatch");
    apply_presence_penalty_avx512(
        static_cast<float*>(lb.ptr),
        static_cast<int>(lb.shape[0]), static_cast<int>(lb.shape[1]),
        static_cast<const int32_t*>(fb.ptr), alpha);
}


// ──────────────────────────────────────────────────────────────────────
// AMX matmul bindings
// ──────────────────────────────────────────────────────────────────────

static py::array_t<float> py_amx_matmul(
        py::array_t<uint16_t, py::array::c_style> A,
        py::array_t<uint16_t, py::array::c_style> B_packed) {
    auto ab = A.request();
    auto bb = B_packed.request();
    if (ab.ndim != 2 || bb.ndim != 2) throw std::invalid_argument("2D");
    int M = static_cast<int>(ab.shape[0]);
    int K = static_cast<int>(ab.shape[1]);
    int K_pair = static_cast<int>(bb.shape[0]);
    int N2 = static_cast<int>(bb.shape[1]);
    if (K_pair * 2 != K)
        throw std::invalid_argument("B_packed K/2 rows mismatch A K");
    if (N2 % 2 != 0)
        throw std::invalid_argument("B_packed inner dim must be 2N");
    int N = N2 / 2;
    py::array_t<float> C({M, N});
    amx_matmul_bf16(static_cast<const uint16_t*>(ab.ptr),
                   static_cast<const uint16_t*>(bb.ptr),
                   C.mutable_data(), M, K, N);
    return C;
}

static py::array_t<uint16_t> py_amx_repack_b(
        py::array_t<uint16_t, py::array::c_style> B) {
    auto bb = B.request();
    if (bb.ndim != 2) throw std::invalid_argument("2D");
    int K = static_cast<int>(bb.shape[0]);
    int N = static_cast<int>(bb.shape[1]);
    int K_pair = (K + 1) / 2;
    py::array_t<uint16_t> out({K_pair, N * 2});
    std::memset(out.mutable_data(), 0,
               static_cast<size_t>(K_pair) * N * 2 * sizeof(uint16_t));
    amx_repack_b_bf16(static_cast<const uint16_t*>(bb.ptr),
                     out.mutable_data(), K, N);
    return out;
}


// ──────────────────────────────────────────────────────────────────────
// Tokenizer bindings (SUB_171)
// ──────────────────────────────────────────────────────────────────────
//
// VocabTable 은 호출자(Python) 가 한 번 build 한 뒤 재사용한다. pybind11
// capsule 을 통해 host-side raw pointer 4개 (pieces, offsets, sizes, V) 를
// 묶어 보관 — Python lifecycle 은 numpy array 가 보존.

static py::tuple py_batch_detokenize_bytes(
        py::array_t<uint8_t,  py::array::c_style> pieces,
        py::array_t<int32_t,  py::array::c_style> offsets,
        py::array_t<int32_t,  py::array::c_style> sizes,
        py::array_t<int32_t,  py::array::c_style> token_ids,
        py::array_t<int32_t,  py::array::c_style> seq_offsets,
        bool use_avx512) {

    auto pb = pieces.request();
    auto ob = offsets.request();
    auto sb = sizes.request();
    auto tb = token_ids.request();
    auto qb = seq_offsets.request();

    if (ob.ndim != 1 || sb.ndim != 1 || pb.ndim != 1)
        throw std::invalid_argument("pieces/offsets/sizes must be 1D");
    if (tb.ndim != 1 || qb.ndim != 1)
        throw std::invalid_argument("token_ids/seq_offsets must be 1D");

    tok::VocabTable table;
    table.pieces      = static_cast<const uint8_t*>(pb.ptr);
    table.offsets     = static_cast<const int32_t*>(ob.ptr);
    table.sizes       = static_cast<const int32_t*>(sb.ptr);
    table.V           = static_cast<int32_t>(sb.shape[0]);
    table.total_bytes = static_cast<int32_t>(pb.shape[0]);

    int B = static_cast<int>(qb.shape[0]) - 1;
    if (B < 0) throw std::invalid_argument("seq_offsets length must be >= 1");

    int total_tokens = static_cast<int>(tb.shape[0]);
    int64_t total_bytes_estimate = tok::batch_detokenize_byte_total(
        table, static_cast<const int32_t*>(tb.ptr), total_tokens);

    // alloc output buffer (conservative — exact size from estimate)
    py::array_t<uint8_t> out_bytes(static_cast<py::ssize_t>(total_bytes_estimate));
    py::array_t<int32_t> out_byte_offsets(B + 1);
    py::array_t<int32_t> out_byte_lengths(B);

    if (use_avx512) {
        tok::batch_detokenize_bytes_avx512(
            table,
            static_cast<const int32_t*>(tb.ptr),
            static_cast<const int32_t*>(qb.ptr),
            B,
            out_bytes.mutable_data(),
            out_byte_offsets.mutable_data(),
            out_byte_lengths.mutable_data());
    } else {
        tok::batch_detokenize_bytes_scalar(
            table,
            static_cast<const int32_t*>(tb.ptr),
            static_cast<const int32_t*>(qb.ptr),
            B,
            out_bytes.mutable_data(),
            out_byte_offsets.mutable_data(),
            out_byte_lengths.mutable_data());
    }

    return py::make_tuple(out_bytes, out_byte_offsets, out_byte_lengths);
}

static py::list py_batch_detokenize_strings(
        py::array_t<uint8_t,  py::array::c_style> pieces,
        py::array_t<int32_t,  py::array::c_style> offsets,
        py::array_t<int32_t,  py::array::c_style> sizes,
        py::array_t<int32_t,  py::array::c_style> token_ids,
        py::array_t<int32_t,  py::array::c_style> seq_offsets,
        bool use_avx512) {
    // 편의 wrapper — bytes 를 List[str] 로 변환 (UTF-8 decode 는 Python side
    // 가 처리해도 되지만 본 entrypoint 가 vLLM hook 의 최종 형태와 가까움).
    py::tuple t = py_batch_detokenize_bytes(pieces, offsets, sizes,
                                             token_ids, seq_offsets, use_avx512);
    py::array_t<uint8_t> ob = t[0].cast<py::array_t<uint8_t>>();
    py::array_t<int32_t> oo = t[1].cast<py::array_t<int32_t>>();

    auto ob_buf = ob.request();
    auto oo_buf = oo.request();
    int B = static_cast<int>(oo_buf.shape[0]) - 1;
    const uint8_t* bytes_ptr = static_cast<const uint8_t*>(ob_buf.ptr);
    const int32_t* off_ptr   = static_cast<const int32_t*>(oo_buf.ptr);

    py::list result;
    for (int b = 0; b < B; ++b) {
        int32_t lo = off_ptr[b];
        int32_t hi = off_ptr[b + 1];
        result.append(py::str(
            reinterpret_cast<const char*>(bytes_ptr + lo),
            static_cast<size_t>(hi - lo)));
    }
    return result;
}

static py::array_t<int32_t> py_batch_bpe_min_rank(
        py::array_t<int32_t,  py::array::c_style> rank_table,
        py::array_t<int32_t,  py::array::c_style> pair_ids) {
    auto rb = rank_table.request();
    auto pb = pair_ids.request();
    if (rb.ndim != 2 || rb.shape[0] != rb.shape[1])
        throw std::invalid_argument("rank_table must be square 2D");
    if (pb.ndim != 2)
        throw std::invalid_argument("pair_ids must be 2D");

    int32_t dim = static_cast<int32_t>(rb.shape[0]);
    int B = static_cast<int>(pb.shape[0]);
    int num_pairs = static_cast<int>(pb.shape[1]);

    py::array_t<int32_t> out(B);
    tok::batch_bpe_min_rank_avx512(
        static_cast<const int32_t*>(rb.ptr), dim,
        static_cast<const int32_t*>(pb.ptr),
        B, num_pairs,
        out.mutable_data());
    return out;
}


// ──────────────────────────────────────────────────────────────────────
// Module definition
// ──────────────────────────────────────────────────────────────────────

PYBIND11_MODULE(_core, m) {
    m.doc() = "IDE_016 AVX-512 sampling + AMX matmul kernels (numpy iface)";

    // capability probes
    m.def("cpu_has_avx512", &cpu_has_avx512);
    m.def("cpu_has_amx", &cpu_has_amx_pybind);
    m.def("amx_request_permission", &amx_request_permission);

    // sampling — bf16 (uint16 view)
    m.def("topk_bf16", &py_topk_bf16, py::arg("logits"), py::arg("k"));
    m.def("topk_fp32", &py_topk_fp32, py::arg("logits"), py::arg("k"));
    m.def("topp_cutoff", &py_topp, py::arg("sorted_probs"), py::arg("p"));
    m.def("fused_sample_bf16", &py_fused_sample_bf16,
          py::arg("logits"), py::arg("k"), py::arg("p"),
          py::arg("temperature") = 1.0f, py::arg("rng_seed") = 0);
    m.def("fused_sample_fp32", &py_fused_sample_fp32,
          py::arg("logits"), py::arg("k"), py::arg("p"),
          py::arg("temperature") = 1.0f, py::arg("rng_seed") = 0);

    // logit processors
    m.def("apply_temperature_fp32", &py_apply_temperature_fp32,
          py::arg("logits"), py::arg("temperature"));
    m.def("apply_temperature_bf16", &py_apply_temperature_bf16,
          py::arg("logits"), py::arg("temperature"));
    m.def("apply_logit_bias", &py_apply_logit_bias,
          py::arg("logits"), py::arg("bias"));
    m.def("softmax", &py_softmax, py::arg("logits"));

    // penalties
    m.def("apply_repetition_penalty", &py_apply_repetition_penalty,
          py::arg("logits"), py::arg("token_ids"), py::arg("lengths"),
          py::arg("penalty"));
    m.def("apply_frequency_penalty", &py_apply_frequency_penalty,
          py::arg("logits"), py::arg("freq"), py::arg("alpha"));
    m.def("apply_presence_penalty", &py_apply_presence_penalty,
          py::arg("logits"), py::arg("freq"), py::arg("alpha"));

    // AMX matmul
    m.def("amx_matmul", &py_amx_matmul,
          py::arg("A"), py::arg("B_packed"));
    m.def("amx_repack_b", &py_amx_repack_b, py::arg("B"));

    // Tokenizer (SUB_171)
    m.def("batch_detokenize_bytes", &py_batch_detokenize_bytes,
          py::arg("pieces"), py::arg("offsets"), py::arg("sizes"),
          py::arg("token_ids"), py::arg("seq_offsets"),
          py::arg("use_avx512") = true);
    m.def("batch_detokenize_strings", &py_batch_detokenize_strings,
          py::arg("pieces"), py::arg("offsets"), py::arg("sizes"),
          py::arg("token_ids"), py::arg("seq_offsets"),
          py::arg("use_avx512") = true);
    m.def("batch_bpe_min_rank", &py_batch_bpe_min_rank,
          py::arg("rank_table"), py::arg("pair_ids"));
}
