#!/usr/bin/env python3
"""IDE_076 A1 v4 — avx_rb_rows 본문에서 proof 카운터·gfm 상수를 제거해 OFF 인스턴스가 원본 본문과 동일하게 되도록 하고, 카운팅은 avx_kernel_rb(호출당 1회) 로 올린다. apply | revert | status (컨테이너)."""
import sys, os, shutil, hashlib
ROOT = "/sgl-workspace/ktransformers/kt-kernel"; BK = "/sgl-workspace/ide076_backup"; KH = f"{ROOT}/operators/amx/la/amx_kernels.hpp"
OLD_H = """    constexpr bool a1 = A1;   // IDE_076 A1 정책: 템플릿 인자 (OFF 인스턴스는 원본 코드와 동일)
    [[maybe_unused]] const __m512i gfm = _mm512_set1_epi64((long long)0x0000000001020408LL);
    if (kt_opt::proof_on()) { auto& c = kt_opt::counters(); c.a1_eligible_calls.fetch_add(1, std::memory_order_relaxed); if (a1) { if (RB == 1) c.a1_r1_calls.fetch_add(1, std::memory_order_relaxed); else if (RB == 2) c.a1_r2_calls.fetch_add(1, std::memory_order_relaxed); else if (RB == 4) c.a1_r4_calls.fetch_add(1, std::memory_order_relaxed); else c.a1_r8_calls.fetch_add(1, std::memory_order_relaxed); } else c.a1_fallback_not_enabled.fetch_add(1, std::memory_order_relaxed); }
"""
NEW_H = ""
OLD_L1 = "        __m512i b_lo0; if constexpr (A1) b_lo0 = _mm512_gf2p8affine_epi64_epi8(braw0, gfm, 0); else b_lo0 = _mm512_slli_epi32(_mm512_and_si512(lo, braw0), 4);"
NEW_L1 = "        __m512i b_lo0; if constexpr (A1) b_lo0 = _mm512_gf2p8affine_epi64_epi8(braw0, _mm512_set1_epi64((long long)0x0000000001020408LL), 0); else b_lo0 = _mm512_slli_epi32(_mm512_and_si512(lo, braw0), 4);"
OLD_L2 = "        __m512i b_lo1; if constexpr (A1) b_lo1 = _mm512_gf2p8affine_epi64_epi8(braw1, gfm, 0); else b_lo1 = _mm512_slli_epi32(_mm512_and_si512(lo, braw1), 4);"
NEW_L2 = "        __m512i b_lo1; if constexpr (A1) b_lo1 = _mm512_gf2p8affine_epi64_epi8(braw1, _mm512_set1_epi64((long long)0x0000000001020408LL), 0); else b_lo1 = _mm512_slli_epi32(_mm512_and_si512(lo, braw1), 4);"
OLD_D = """    int mb = 0;
    if (kt_opt::a1_enabled()) {   // IDE_076 A1-a: GFNI 언팩 인스턴스 (호출당 정적 bool 1회 읽기)"""
NEW_D = """    int mb = 0;
    if (kt_opt::proof_on()) {   // IDE_076 feature proof: RB 블록 수 계수 (호출당 1회; hot 루프 밖)
      auto& c = kt_opt::counters(); const bool on = kt_opt::a1_enabled(); int r = m_block_end;
      const unsigned long long n8 = r / 8, n4 = (r % 8) / 4, n2 = (r % 4) / 2, n1 = r % 2;
      c.a1_eligible_calls.fetch_add(n8 + n4 + n2 + n1, std::memory_order_relaxed);
      if (on) { c.a1_r8_calls.fetch_add(n8, std::memory_order_relaxed); c.a1_r4_calls.fetch_add(n4, std::memory_order_relaxed); c.a1_r2_calls.fetch_add(n2, std::memory_order_relaxed); c.a1_r1_calls.fetch_add(n1, std::memory_order_relaxed); } else c.a1_fallback_not_enabled.fetch_add(n8 + n4 + n2 + n1, std::memory_order_relaxed);
    }
    if (kt_opt::a1_enabled()) {   // IDE_076 A1-a: GFNI 언팩 인스턴스 (호출당 정적 bool 1회 읽기)"""


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def apply():
    b = f"{BK}/amx_kernels.hpp.pre_a1hoist"
    if not os.path.exists(b): shutil.copy(KH, b)
    s = open(KH).read()
    if "feature proof: RB 블록 수 계수" in s: print("already"); return
    for o in (OLD_H, OLD_L1, OLD_L2, OLD_D): assert s.count(o) == 1, o[:70]
    s = s.replace(OLD_H, NEW_H).replace(OLD_L1, NEW_L1).replace(OLD_L2, NEW_L2).replace(OLD_D, NEW_D); open(KH, "w").write(s); print("applied", sha(KH))


def revert():
    b = f"{BK}/amx_kernels.hpp.pre_a1hoist"
    if os.path.exists(b): shutil.copy(b, KH); print("reverted", sha(KH))


def status(): print("kernels", sha(KH), "hoisted" if "feature proof: RB 블록 수 계수" in open(KH).read() else "not")


if __name__ == "__main__": {"apply": apply, "revert": revert, "status": status}[sys.argv[1]]()
