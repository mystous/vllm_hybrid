#!/usr/bin/env python3
"""IDE_051: forward_prefill 의 cpy_input(토큰별 gather memcpy) + q_input(expert 별 A 양자화) 두 위상을 한 위상으로 융합.
토큰 i 의 행을 expert e 의 BufferA 슬롯 (m_local_pos_[i][j]) 에 직접 양자화해 쓴다 (행 독립 스케일 → bit 동일). bf16 gather 복사본 불필요.
env KT_FUSE_QIN=1 (qlen>=10 일 때만; 기본 꺼짐)."""
PB = "/sgl-workspace/ktransformers/kt-kernel/operators/amx/la/amx_buffers.hpp"
PM = "/sgl-workspace/ktransformers/kt-kernel/operators/amx/moe_base.hpp"
b = open(PB).read()
if "from_mat_row" not in b:
    i0 = b.index("struct BufferAImpl {")
    anchor = "  // IDE_046-b: 행 블록 [m_begin, m_begin+M_STEP) 만 양자화 (from_mat 과 동일 산술, 행 독립)"
    i1 = b.index(anchor, i0)
    row = r'''  // IDE_051: 행 1개 (row) 만 src_row (bf16, 길이 k) 에서 양자화 (from_mat 과 동일 산술)
  void from_mat_row(int m, const ggml_bf16_t* src_row, int row) {
    assert(row < m && m <= max_m);
    __m512 amax_v0 = _mm512_setzero_ps();
    __m512 amax_v1 = _mm512_setzero_ps();
    for (int j = 0; j < k; j += 32) {
      __m512 f0, f1;
      avx512_32xbf16_to_32xfp32((__m512i*)(src_row + j), &f0, &f1);
      amax_v0 = vector_abs_max(amax_v0, f0);
      amax_v1 = vector_abs_max(amax_v1, f1);
    }
    amax_v0 = vector_abs_max(amax_v0, amax_v1);
    float amax = _mm512_reduce_max_ps(amax_v0);
    d[row] = amax / ((1 << 7) - 1);
    int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
    int m_begin = row / M_STEP * M_STEP; int i = row - m_begin;
    __m512 id = _mm512_set1_ps(d[row] ? 1.0f / d[row] : 0.0f);
    for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K_BLOCK) {
      int k_block_size = std::min(K_BLOCK, k - k_block_begin);
      for (int k_begin = 0; k_begin < k_block_size; k_begin += K_STEP) {
        int8_t* dst = a + k_block_begin * m_block_size + m_begin * k_block_size + k_begin * M_STEP + i * K_STEP;
        __m512 f0, f1, f2, f3;
        avx512_32xbf16_to_32xfp32((__m512i*)(src_row + k_block_begin + k_begin), &f0, &f1);
        avx512_32xbf16_to_32xfp32((__m512i*)(src_row + k_block_begin + k_begin) + 1, &f2, &f3);
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

'''
    b = b[:i1] + row + b[i1:]
    open(PB, "w").write(b); print("buffers: from_mat_row added")
else:
    print("buffers: already")

m = open(PM).read()
if "kt_fuse_qin_" in m:
    print("moe_base: already"); raise SystemExit
oldC = """    direct_or_pool(qlen, [&](int i) {
      for (int j = 0; j < k; j++) {
        if (config_.should_skip_expert(expert_ids[i * k + j])) {
          continue;
        }
        memcpy(m_local_input_ptr_[expert_ids[i * k + j]] + m_local_pos_[i][j] * config_.hidden_size,
               (ggml_bf16_t*)input + i * config_.hidden_size, sizeof(ggml_bf16_t) * config_.hidden_size);
      }
    });"""
newC = """    const bool fuse_qin_ = kt_fuse_qin_() && qlen >= 10;
    if constexpr (requires(typename T::BufferA& x_, const ggml_bf16_t* p_) { x_.from_mat_row(1, p_, 0); }) if (fuse_qin_) {
      pool->do_work_stealing_job(qlen, nullptr, [&](int i) {
        for (int j = 0; j < k; j++) {
          int e = expert_ids[i * k + j];
          if (config_.should_skip_expert(e)) continue;
          gate_up_ba_[e]->from_mat_row(m_local_num_[e], (const ggml_bf16_t*)input + i * config_.hidden_size, m_local_pos_[i][j]);
        }
      }, nullptr);
    } else
    direct_or_pool(qlen, [&](int i) {
      for (int j = 0; j < k; j++) {
        if (config_.should_skip_expert(expert_ids[i * k + j])) {
          continue;
        }
        memcpy(m_local_input_ptr_[expert_ids[i * k + j]] + m_local_pos_[i][j] * config_.hidden_size,
               (ggml_bf16_t*)input + i * config_.hidden_size, sizeof(ggml_bf16_t) * config_.hidden_size);
      }
    });"""
assert m.count(oldC) == 1, m.count(oldC)
m = m.replace(oldC, newC)
# q_input 위상: fuse 시 생략
oldQ = """    if constexpr (requires(typename T::BufferA& x_, ggml_bf16_t* p_) { x_.from_mat_block(1, p_, 0); }) if (kt_qa_par_() && qlen >= 10) {
      const int mb = (max_local_num + T::M_STEP - 1) / T::M_STEP;
      pool->do_work_stealing_job(activated_expert * mb, nullptr, [this, mb](int t) {
        int expert_idx = m_expert_id_map_[t / mb]; int m_begin = (t % mb) * T::M_STEP;
        if (m_begin < m_local_num_[expert_idx])
          gate_up_ba_[expert_idx]->from_mat_block(m_local_num_[expert_idx], m_local_input_ptr_[expert_idx], m_begin);
      }, nullptr);
    } else"""
newQ = """    if (fuse_qin_) {
      // IDE_051: gather 단계에서 이미 양자화됨
    } else
""" + oldQ
assert m.count(oldQ) == 1, m.count(oldQ)
m = m.replace(oldQ, newQ)
anchor = "  bool kt_qa_par_() const {"
assert m.count(anchor) == 1
m = m.replace(anchor, """  bool kt_fuse_qin_() const {
    static bool v = std::getenv("KT_FUSE_QIN") != nullptr;
    return v;
  }
""" + anchor)
open(PM, "w").write(m); print("moe_base: fuse gather+quantize patch applied")
