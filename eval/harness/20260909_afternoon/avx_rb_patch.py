#!/usr/bin/env python3
"""IDE_048: GemmKernel224Int4 AVX-512 (vec_mul) 경로 레지스터 블로킹.
원본 avx_kernel 은 누산기 c512[] 가 메모리에 있고 행마다 B 를 다시 언팩 → dpbssd 체인이 latency-bound (행당 18µs).
새 커널: k-group 마다 B 4 벡터를 한 번 언팩, 행 블록 (8/4/2/1) 의 누산기를 zmm 에 유지, 1~2행은 K 분할 누산기로 체인 수 확보.
env KT_AVX_RB=1 (기본 꺼짐)."""
P = "/sgl-workspace/ktransformers/kt-kernel/operators/amx/la/amx_kernels.hpp"
s = open(P).read()
if "avx_kernel_rb" in s:
    print("already"); raise SystemExit

i0 = s.index("struct GemmKernel224Int4 {")
i1 = s.index("  static void amx_kernel(int m, int n, int k, int m_begin, int n_begin, int k_block_begin, float* c, BufferA* ba,", i0)
kernel = r'''  // ---- IDE_048: 레지스터 블로킹 AVX-512 커널. 정수 누산이므로 원본과 결과 bit 동일.
  static bool avx_rb_on() {
    static bool v = std::getenv("KT_AVX_RB") != nullptr;
    return v;
  }
  template <int RB, int SPLIT>
  static inline void avx_rb_rows(int m, int n, int k, int m_begin, int n_begin, int mb, int k_block_begin, __m512i* c512,
                                 BufferA* ba, BufferB* bb) {
    using K = GemmKernel224Int4;
    __m512i acc[RB][2][SPLIT];
#pragma GCC unroll 8
    for (int r = 0; r < RB; r++) {
#pragma GCC unroll 2
      for (int ni = 0; ni < 2; ni++) {
        acc[r][ni][0] = (k_block_begin == 0) ? _mm512_setzero_si512() : c512[(mb + r) * 2 + ni];
#pragma GCC unroll 4
        for (int j = 1; j < SPLIT; j++) acc[r][ni][j] = _mm512_setzero_si512();
      }
    }
    const __m512i lo = K::lo_mask(), hi = K::hi_mask();
    for (int k_begin = 0; k_begin < K::K_BLOCK && k_block_begin + k_begin < k; k_begin += K::BufferB::B_K_STEP) {
      const int32_t* a32_lo = (const int32_t*)ba->get_submat(m, k, m_begin, k_block_begin + k_begin) + mb * 16;
      const int32_t* a32_hi = (const int32_t*)ba->get_submat(m, k, m_begin, k_block_begin + k_begin + K::K_STEP) + mb * 16;
      const __m512i* b512 = (const __m512i*)bb->get_submat(n, k, n_begin, k_block_begin + k_begin);
#pragma GCC unroll 16
      for (int k_i = 0; k_i < 16; k_i++) {
        const __m512i braw0 = _mm512_loadu_si512((const void*)(b512 + k_i));
        const __m512i braw1 = _mm512_loadu_si512((const void*)(b512 + 16 + k_i));
        const __m512i b_lo0 = _mm512_slli_epi32(_mm512_and_si512(lo, braw0), 4);
        const __m512i b_hi0 = _mm512_and_si512(hi, braw0);
        const __m512i b_lo1 = _mm512_slli_epi32(_mm512_and_si512(lo, braw1), 4);
        const __m512i b_hi1 = _mm512_and_si512(hi, braw1);
        const int j = k_i % SPLIT;
#pragma GCC unroll 8
        for (int r = 0; r < RB; r++) {
          const __m512i ma_lo = _mm512_set1_epi32(a32_lo[r * 16 + k_i]);
          const __m512i ma_hi = _mm512_set1_epi32(a32_hi[r * 16 + k_i]);
          acc[r][0][j] = _mm512_dpbssd_epi32(acc[r][0][j], ma_lo, b_lo0);
          acc[r][0][j] = _mm512_dpbssd_epi32(acc[r][0][j], ma_hi, b_hi0);
          acc[r][1][j] = _mm512_dpbssd_epi32(acc[r][1][j], ma_lo, b_lo1);
          acc[r][1][j] = _mm512_dpbssd_epi32(acc[r][1][j], ma_hi, b_hi1);
        }
      }
    }
#pragma GCC unroll 8
    for (int r = 0; r < RB; r++) {
#pragma GCC unroll 2
      for (int ni = 0; ni < 2; ni++) {
        __m512i t = acc[r][ni][0];
#pragma GCC unroll 4
        for (int j = 1; j < SPLIT; j++) t = _mm512_add_epi32(t, acc[r][ni][j]);
        c512[(mb + r) * 2 + ni] = t;
      }
    }
  }
  static void avx_kernel_rb(int m, int n, int k, int m_begin, int n_begin, int k_block_begin, float* c, BufferA* ba,
                            BufferB* bb) {
    __m512i* c512 = (__m512i*)c;
    int m_block_end = std::min(m - m_begin, M_STEP);
    int mb = 0;
    for (; mb + 8 <= m_block_end; mb += 8) avx_rb_rows<8, 1>(m, n, k, m_begin, n_begin, mb, k_block_begin, c512, ba, bb);
    for (; mb + 4 <= m_block_end; mb += 4) avx_rb_rows<4, 1>(m, n, k, m_begin, n_begin, mb, k_block_begin, c512, ba, bb);
    for (; mb + 2 <= m_block_end; mb += 2) avx_rb_rows<2, 2>(m, n, k, m_begin, n_begin, mb, k_block_begin, c512, ba, bb);
    for (; mb < m_block_end; mb += 1) avx_rb_rows<1, 4>(m, n, k, m_begin, n_begin, mb, k_block_begin, c512, ba, bb);
  }

'''
s = s[:i1] + kernel + s[i1:]

# integer_mat_mul: vec (amx_or_avx=false) + Int4 + env → rb 커널
anchor = "  if constexpr (amx_or_avx && AMX_AVAILABLE && std::is_same_v<K, GemmKernel224Int4>) {"
assert s.count(anchor) == 1
branch = r'''  if constexpr (!amx_or_avx && std::is_same_v<K, GemmKernel224Int4>) {
    if (K::avx_rb_on()) {
      for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K::K_BLOCK) {
        for (int m_begin = 0; m_begin < m; m_begin += K::M_STEP) {
          for (int n_begin = n_start; n_begin < n_end; n_begin += K::N_STEP) {
            float* c = bc->get_submat(m, n, m_begin, n_begin);
            K::avx_kernel_rb(m, n, k, m_begin, n_begin, k_block_begin, c, ba, bb);
            if (k_block_begin + K::K_BLOCK >= k) {
              K::apply_scale(m, n, m_begin, n_begin, c, ba, bb);
            }
          }
        }
      }
      return;
    }
  }
'''
s = s.replace(anchor, branch + anchor)
open(P, "w").write(s)
print("IDE_048 avx rb patch applied")
