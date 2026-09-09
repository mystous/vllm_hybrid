#!/usr/bin/env python3
"""IDE_046-b: prefill 의 A 양자화 (q_input / q_down) 를 expert 단위 (1 스레드) → expert × 32행 블록 단위로 병렬화. env KT_QA_PAR=1."""
PB = "/sgl-workspace/ktransformers/kt-kernel/operators/amx/la/amx_buffers.hpp"
PM = "/sgl-workspace/ktransformers/kt-kernel/operators/amx/moe_base.hpp"
b = open(PB).read()
if "from_mat_block" not in b:
    i0 = b.index("struct BufferAImpl {")
    anchor = "  void from_mat(int m, ggml_bf16_t* src, int ith, int nth) {"
    i1 = b.index(anchor, i0)
    block = r'''  // IDE_046-b: 행 블록 [m_begin, m_begin+M_STEP) 만 양자화 (from_mat 과 동일 산술, 행 독립)
  void from_mat_block(int m, ggml_bf16_t* src, int m_begin) {
    assert(m <= max_m);
    for (int i = 0; i < M_STEP && m_begin + i < m; i++) {
      __m512 amax_v0 = _mm512_setzero_ps();
      __m512 amax_v1 = _mm512_setzero_ps();
      for (int j = 0; j < k; j += 32) {
        __m512 f0, f1;
        avx512_32xbf16_to_32xfp32((__m512i*)(src + (m_begin + i) * k + j), &f0, &f1);
        amax_v0 = vector_abs_max(amax_v0, f0);
        amax_v1 = vector_abs_max(amax_v1, f1);
      }
      amax_v0 = vector_abs_max(amax_v0, amax_v1);
      float amax = _mm512_reduce_max_ps(amax_v0);
      d[m_begin + i] = amax / ((1 << 7) - 1);
    }
    int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
    for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K_BLOCK) {
      int k_block_size = std::min(K_BLOCK, k - k_block_begin);
      for (int k_begin = 0; k_begin < k_block_size; k_begin += K_STEP) {
        for (int i = 0; i < M_STEP && m_begin + i < m; i++) {
          __m512 id = _mm512_set1_ps(d[m_begin + i] ? 1.0f / d[m_begin + i] : 0.0f);
          int8_t* dst = a + k_block_begin * m_block_size + m_begin * k_block_size + k_begin * M_STEP + i * K_STEP;
          __m512 f0, f1, f2, f3;
          avx512_32xbf16_to_32xfp32((__m512i*)(src + (m_begin + i) * k + k_block_begin + k_begin), &f0, &f1);
          avx512_32xbf16_to_32xfp32((__m512i*)(src + (m_begin + i) * k + k_block_begin + k_begin) + 1, &f2, &f3);
          __m512i i0 = _mm512_cvtps_epi32(_mm512_mul_ps(f0, id));
          __m512i i1 = _mm512_cvtps_epi32(_mm512_mul_ps(f1, id));
          __m512i i2 = _mm512_cvtps_epi32(_mm512_mul_ps(f2, id));
          __m512i i3 = _mm512_cvtps_epi32(_mm512_mul_ps(f3, id));
          __m128i s0 = _mm512_cvtsepi32_epi8(i0);
          __m128i s1 = _mm512_cvtsepi32_epi8(i1);
          __m128i s2 = _mm512_cvtsepi32_epi8(i2);
          __m128i s3 = _mm512_cvtsepi32_epi8(i3);
          if constexpr (requires(__m128i value) { K::prepare_a(value); }) {
            s0 = K::prepare_a(s0);
            s1 = K::prepare_a(s1);
            s2 = K::prepare_a(s2);
            s3 = K::prepare_a(s3);
          }
          _mm_store_si128((__m128i*)dst, s0);
          _mm_store_si128((__m128i*)(dst + 16), s1);
          _mm_store_si128((__m128i*)(dst + 32), s2);
          _mm_store_si128((__m128i*)(dst + 48), s3);
        }
      }
    }
  }

'''
    b = b[:i1] + block + b[i1:]
    open(PB, "w").write(b); print("buffers: from_mat_block added")
else:
    print("buffers: already")

m = open(PM).read()
if "kt_qa_par_" in m:
    print("moe_base: already"); raise SystemExit
oldA = """    direct_or_pool(activated_expert, [this](int task_id) {
      int expert_idx = m_expert_id_map_[task_id];
      gate_up_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_input_ptr_[expert_idx], 0, 1);
    });"""
newA = """    if (kt_qa_par_() && qlen >= 10) {
      const int mb = (max_local_num + T::M_STEP - 1) / T::M_STEP;
      pool->do_work_stealing_job(activated_expert * mb, nullptr, [this, mb](int t) {
        int expert_idx = m_expert_id_map_[t / mb]; int m_begin = (t % mb) * T::M_STEP;
        if (m_begin < m_local_num_[expert_idx])
          gate_up_ba_[expert_idx]->from_mat_block(m_local_num_[expert_idx], m_local_input_ptr_[expert_idx], m_begin);
      }, nullptr);
    } else
    direct_or_pool(activated_expert, [this](int task_id) {
      int expert_idx = m_expert_id_map_[task_id];
      gate_up_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_input_ptr_[expert_idx], 0, 1);
    });"""
assert m.count(oldA) == 1, m.count(oldA)
# max_local_num 이 q_input 앞에서 계산되는지 확인
iA = m.index(oldA); assert "max_local_num" in m[:iA], "max_local_num not computed before q_input"
m = m.replace(oldA, newA)
oldB = """    pool->do_work_stealing_job(
        activated_expert, nullptr,
        [this](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          down_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_gate_output_ptr_[expert_idx], 0, 1);
        },
        nullptr);"""
newB = """    if (kt_qa_par_() && qlen >= 10) {
      const int mb = (max_local_num + T::M_STEP - 1) / T::M_STEP;
      pool->do_work_stealing_job(activated_expert * mb, nullptr, [this, mb](int t) {
        int expert_idx = m_expert_id_map_[t / mb]; int m_begin = (t % mb) * T::M_STEP;
        if (m_begin < m_local_num_[expert_idx])
          down_ba_[expert_idx]->from_mat_block(m_local_num_[expert_idx], m_local_gate_output_ptr_[expert_idx], m_begin);
      }, nullptr);
    } else
    pool->do_work_stealing_job(
        activated_expert, nullptr,
        [this](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          down_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_gate_output_ptr_[expert_idx], 0, 1);
        },
        nullptr);"""
assert m.count(oldB) == 1, m.count(oldB)
m = m.replace(oldB, newB)
# helper: amx_min_rows_ 옆에
anchor = "  void forward_decode(int k, const int64_t* expert_ids, const float* weights, const void* input, void* output) {"
assert m.count(anchor) == 1
m = m.replace(anchor, """  bool kt_qa_par_() const {
    static bool v = std::getenv("KT_QA_PAR") != nullptr;
    return v;
  }
""" + anchor)
open(PM, "w").write(m); print("moe_base: q_input/q_down parallel patch applied")
