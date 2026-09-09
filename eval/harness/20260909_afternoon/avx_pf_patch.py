#!/usr/bin/env python3
"""IDE_050: avx_rb_rows 의 B 스트림에 소프트웨어 prefetch (KT_AVX_PF=<k_begin 단위 거리, 기본 0=끔>).
B 블록 (32행×64B = 2 KB) 단위로 k_begin 앞쪽 PF 블록을 _mm_prefetch(T0)."""
P = "/sgl-workspace/ktransformers/kt-kernel/operators/amx/la/amx_kernels.hpp"
s = open(P).read()
if "avx_pf_dist" in s:
    print("already"); raise SystemExit
old = """    const __m512i lo = K::lo_mask(), hi = K::hi_mask();
    for (int k_begin = 0; k_begin < K::K_BLOCK && k_block_begin + k_begin < k; k_begin += K::BufferB::B_K_STEP) {
      const int32_t* a32_lo = (const int32_t*)ba->get_submat(m, k, m_begin, k_block_begin + k_begin) + mb * 16;
      const int32_t* a32_hi = (const int32_t*)ba->get_submat(m, k, m_begin, k_block_begin + k_begin + K::K_STEP) + mb * 16;
      const __m512i* b512 = (const __m512i*)bb->get_submat(n, k, n_begin, k_block_begin + k_begin);
"""
new = """    const __m512i lo = K::lo_mask(), hi = K::hi_mask();
    const int pf = avx_pf_dist();
    for (int k_begin = 0; k_begin < K::K_BLOCK && k_block_begin + k_begin < k; k_begin += K::BufferB::B_K_STEP) {
      const int32_t* a32_lo = (const int32_t*)ba->get_submat(m, k, m_begin, k_block_begin + k_begin) + mb * 16;
      const int32_t* a32_hi = (const int32_t*)ba->get_submat(m, k, m_begin, k_block_begin + k_begin + K::K_STEP) + mb * 16;
      const __m512i* b512 = (const __m512i*)bb->get_submat(n, k, n_begin, k_block_begin + k_begin);
      if (pf > 0) {
        // 같은 (n_begin, k_block) 안에서 B 블록은 k_begin 순으로 연속 (2 KB 간격) → pf 블록 앞을 미리 적재
        const char* pfp = (const char*)(b512 + 32 * pf);
        for (int l = 0; l < 32; l++) _mm_prefetch(pfp + l * 64, _MM_HINT_T0);
      }
"""
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)
anchor = "  template <int RB, int SPLIT>\n  static inline void avx_rb_rows("
assert s.count(anchor) == 1
s = s.replace(anchor, """  static int avx_pf_dist() {
    static int v = [] { const char* e = std::getenv("KT_AVX_PF"); return e ? std::atoi(e) : 0; }();
    return v;
  }
""" + anchor)
open(P, "w").write(s)
print("IDE_050 prefetch patch applied")
