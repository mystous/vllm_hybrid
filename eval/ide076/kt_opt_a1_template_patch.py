#!/usr/bin/env python3
"""IDE_076 A1 v3 — avx_rb_rows 의 A1 분기를 런타임 bool 에서 템플릿 인자 <RB,SPLIT,A1> 로 바꿔 OFF 인스턴스가 원본과 같은 코드가 되게 한다 (PLAN.md §3.3: 플래그 OFF 경로 = 기존 경로).
avx_kernel_rb 가 kt_opt::a1_enabled() 를 호출당 1회 읽어 <…,true>/<…,false> 를 선택. proof 카운터는 유지. apply | revert | status (컨테이너)."""
import sys, os, shutil, hashlib
ROOT = "/sgl-workspace/ktransformers/kt-kernel"; BK = "/sgl-workspace/ide076_backup"; KH = f"{ROOT}/operators/amx/la/amx_kernels.hpp"
OLD_T = "  template <int RB, int SPLIT>\n  static inline void avx_rb_rows("
NEW_T = "  template <int RB, int SPLIT, bool A1 = false>\n  static inline void avx_rb_rows("
OLD_H = """    const bool a1 = kt_opt::a1_enabled();   // IDE_076 A1 정책 (정적 1회 파싱; OFF 이면 기존 경로 그대로)
    const __m512i gfm = _mm512_set1_epi64((long long)0x0000000001020408LL);"""
NEW_H = """    constexpr bool a1 = A1;   // IDE_076 A1 정책: 템플릿 인자 (OFF 인스턴스는 원본 코드와 동일)
    [[maybe_unused]] const __m512i gfm = _mm512_set1_epi64((long long)0x0000000001020408LL);"""
OLD_L1 = "        const __m512i b_lo0 = a1 ? _mm512_gf2p8affine_epi64_epi8(braw0, gfm, 0) : _mm512_slli_epi32(_mm512_and_si512(lo, braw0), 4);"
NEW_L1 = "        __m512i b_lo0; if constexpr (A1) b_lo0 = _mm512_gf2p8affine_epi64_epi8(braw0, gfm, 0); else b_lo0 = _mm512_slli_epi32(_mm512_and_si512(lo, braw0), 4);"
OLD_L2 = "        const __m512i b_lo1 = a1 ? _mm512_gf2p8affine_epi64_epi8(braw1, gfm, 0) : _mm512_slli_epi32(_mm512_and_si512(lo, braw1), 4);"
NEW_L2 = "        __m512i b_lo1; if constexpr (A1) b_lo1 = _mm512_gf2p8affine_epi64_epi8(braw1, gfm, 0); else b_lo1 = _mm512_slli_epi32(_mm512_and_si512(lo, braw1), 4);"
OLD_D = """    int mb = 0;
    for (; mb + 8 <= m_block_end; mb += 8) avx_rb_rows<8, 1>(m, n, k, m_begin, n_begin, mb, k_block_begin, c512, ba, bb);
    for (; mb + 4 <= m_block_end; mb += 4) avx_rb_rows<4, 1>(m, n, k, m_begin, n_begin, mb, k_block_begin, c512, ba, bb);
    for (; mb + 2 <= m_block_end; mb += 2) avx_rb_rows<2, 2>(m, n, k, m_begin, n_begin, mb, k_block_begin, c512, ba, bb);
    for (; mb < m_block_end; mb += 1) avx_rb_rows<1, 4>(m, n, k, m_begin, n_begin, mb, k_block_begin, c512, ba, bb);"""
NEW_D = """    int mb = 0;
    if (kt_opt::a1_enabled()) {   // IDE_076 A1-a: GFNI 언팩 인스턴스 (호출당 정적 bool 1회 읽기)
      for (; mb + 8 <= m_block_end; mb += 8) avx_rb_rows<8, 1, true>(m, n, k, m_begin, n_begin, mb, k_block_begin, c512, ba, bb);
      for (; mb + 4 <= m_block_end; mb += 4) avx_rb_rows<4, 1, true>(m, n, k, m_begin, n_begin, mb, k_block_begin, c512, ba, bb);
      for (; mb + 2 <= m_block_end; mb += 2) avx_rb_rows<2, 2, true>(m, n, k, m_begin, n_begin, mb, k_block_begin, c512, ba, bb);
      for (; mb < m_block_end; mb += 1) avx_rb_rows<1, 4, true>(m, n, k, m_begin, n_begin, mb, k_block_begin, c512, ba, bb);
      return;
    }
    for (; mb + 8 <= m_block_end; mb += 8) avx_rb_rows<8, 1>(m, n, k, m_begin, n_begin, mb, k_block_begin, c512, ba, bb);
    for (; mb + 4 <= m_block_end; mb += 4) avx_rb_rows<4, 1>(m, n, k, m_begin, n_begin, mb, k_block_begin, c512, ba, bb);
    for (; mb + 2 <= m_block_end; mb += 2) avx_rb_rows<2, 2>(m, n, k, m_begin, n_begin, mb, k_block_begin, c512, ba, bb);
    for (; mb < m_block_end; mb += 1) avx_rb_rows<1, 4>(m, n, k, m_begin, n_begin, mb, k_block_begin, c512, ba, bb);"""


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def apply():
    b = f"{BK}/amx_kernels.hpp.pre_a1tmpl"
    if not os.path.exists(b): shutil.copy(KH, b)
    s = open(KH).read()
    if "bool A1 = false" in s: print("already"); return
    for o in (OLD_T, OLD_H, OLD_L1, OLD_L2, OLD_D): assert s.count(o) == 1, o[:60]
    s = s.replace(OLD_T, NEW_T).replace(OLD_H, NEW_H).replace(OLD_L1, NEW_L1).replace(OLD_L2, NEW_L2).replace(OLD_D, NEW_D); open(KH, "w").write(s); print("applied", sha(KH))


def revert():
    b = f"{BK}/amx_kernels.hpp.pre_a1tmpl"
    if os.path.exists(b): shutil.copy(b, KH); print("reverted", sha(KH))


def status(): print("kernels", sha(KH), "tmpl" if "bool A1 = false" in open(KH).read() else "runtime-branch")


if __name__ == "__main__": {"apply": apply, "revert": revert, "status": status}[sys.argv[1]]()
