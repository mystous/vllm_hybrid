#!/usr/bin/env python3
"""IDE_076 C03/C06 — kt-kernel A1 패치 (컨테이너 sgl-kt 안에서 실행: apply | revert | status).
내용:
 (1) cpu_backend/kt_opt.h (신규): KT_OPT_A1_ENABLE / KT_OPT_A2_ENABLE 파서 — 명시 0/1 만 허용, 미지정 0, 잘못된 값은 std::runtime_error (MOE 생성 시 호출 → 부팅 실패).
     feature proof 카운터(atomic; KT_OPT_PROOF=1 일 때만 증가, 성능 OFF 경로에는 원자 연산 없음): a1_eligible_calls, a1_r1_calls, a1_r2_calls, a1_fallback_calls_by_reason, a1_policy 문자열.
 (2) operators/amx/la/amx_kernels.hpp: avx_rb_rows<RB,SPLIT> 에 A1 정책 분기 추가.
     A1-a: lo 니블 언팩 (and+slli 2 op) → GFNI vgf2p8affineqb 1 op (행렬 0x0000000001020408, 256 바이트 전수 검산 bit 동일). hi 경로·A 브로드캐스트·누산·apply_scale 불변 → 정수 결과 bit 동일.
     적용 범위: avx_kernel_rb 가 호출하는 모든 RB 블록(R=1/2/4/8). R=1/2 전용 조정(A1-b) 은 별도 hunk 로 뒤에 추가.
 (3) ext_bindings.cpp: kt_opt 상태·카운터를 Python 으로 노출 (kt_kernel_ext.kt_opt_status()).
백업: /sgl-workspace/ide076_backup/<file>.pre_a1"""
import sys, os, shutil, hashlib
ROOT = "/sgl-workspace/ktransformers/kt-kernel"; BK = "/sgl-workspace/ide076_backup"
KH = f"{ROOT}/operators/amx/la/amx_kernels.hpp"; EB = f"{ROOT}/ext_bindings.cpp"; OPT = f"{ROOT}/cpu_backend/kt_opt.h"

KT_OPT_H = r'''// IDE_076 — A1/A2 기능 플래그·feature proof (PLAN.md §4.1~4.2). 명시 0/1 만 허용.
#pragma once
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
namespace kt_opt {
inline int parse01(const char* name) {
  const char* v = std::getenv(name);
  if (v == nullptr) return 0;
  if (std::strcmp(v, "0") == 0) return 0;
  if (std::strcmp(v, "1") == 0) return 1;
  throw std::runtime_error(std::string("kt_opt: invalid value for ") + name + " = '" + v + "' (only 0 or 1 allowed)");
}
inline bool a1_enabled() { static const bool v = parse01("KT_OPT_A1_ENABLE") == 1; return v; }
inline bool a2_enabled() { static const bool v = parse01("KT_OPT_A2_ENABLE") == 1; return v; }
inline bool proof_on() { static const bool v = parse01("KT_OPT_PROOF") == 1; return v; }
inline void validate_at_init() { (void)a1_enabled(); (void)a2_enabled(); (void)proof_on(); }
struct Counters {
  std::atomic<unsigned long long> a1_eligible_calls{0}, a1_r1_calls{0}, a1_r2_calls{0}, a1_r4_calls{0}, a1_r8_calls{0}, a1_fallback_not_enabled{0};
  std::atomic<unsigned long long> a2_eligible_jobs{0}, a2_bundled_jobs{0}, a2_fallback_jobs{0};
};
inline Counters& counters() { static Counters c; return c; }
inline const char* a1_policy() { return "a1-v1: GFNI lo-nibble unpack (0x0000000001020408) in avx_rb_rows<RB,SPLIT>, all RB; integer accumulation order unchanged"; }
}  // namespace kt_opt
'''

RB_OLD = r'''#pragma GCC unroll 16
      for (int k_i = 0; k_i < 16; k_i++) {
        const __m512i braw0 = _mm512_loadu_si512((const void*)(b512 + k_i));
        const __m512i braw1 = _mm512_loadu_si512((const void*)(b512 + 16 + k_i));
        const __m512i b_lo0 = _mm512_slli_epi32(_mm512_and_si512(lo, braw0), 4);
        const __m512i b_hi0 = _mm512_and_si512(hi, braw0);
        const __m512i b_lo1 = _mm512_slli_epi32(_mm512_and_si512(lo, braw1), 4);
        const __m512i b_hi1 = _mm512_and_si512(hi, braw1);
        const int j = k_i % SPLIT;'''
RB_NEW = r'''#pragma GCC unroll 16
      for (int k_i = 0; k_i < 16; k_i++) {
        const __m512i braw0 = _mm512_loadu_si512((const void*)(b512 + k_i));
        const __m512i braw1 = _mm512_loadu_si512((const void*)(b512 + 16 + k_i));
        // IDE_076 A1-a: lo 니블 언팩 (x & 0x0F) << 4 를 GFNI 1 op 로 (bit 동일; 행렬 0x0000000001020408 전수 검산). hi 경로 불변.
        const __m512i b_lo0 = a1 ? _mm512_gf2p8affine_epi64_epi8(braw0, gfm, 0) : _mm512_slli_epi32(_mm512_and_si512(lo, braw0), 4);
        const __m512i b_hi0 = _mm512_and_si512(hi, braw0);
        const __m512i b_lo1 = a1 ? _mm512_gf2p8affine_epi64_epi8(braw1, gfm, 0) : _mm512_slli_epi32(_mm512_and_si512(lo, braw1), 4);
        const __m512i b_hi1 = _mm512_and_si512(hi, braw1);
        const int j = k_i % SPLIT;'''
HEAD_OLD = r'''    const __m512i lo = K::lo_mask(), hi = K::hi_mask();
    const int pf = avx_pf_dist();
    for (int k_begin = 0; k_begin < K::K_BLOCK && k_block_begin + k_begin < k; k_begin += K::BufferB::B_K_STEP) {
      const int32_t* a32_lo = (const int32_t*)ba->get_submat(m, k, m_begin, k_block_begin + k_begin) + mb * 16;'''
HEAD_NEW = r'''    const __m512i lo = K::lo_mask(), hi = K::hi_mask();
    const int pf = avx_pf_dist();
    const bool a1 = kt_opt::a1_enabled();   // IDE_076 A1 정책 (정적 1회 파싱; OFF 이면 기존 경로 그대로)
    const __m512i gfm = _mm512_set1_epi64((long long)0x0000000001020408LL);
    if (kt_opt::proof_on()) { auto& c = kt_opt::counters(); c.a1_eligible_calls.fetch_add(1, std::memory_order_relaxed); if (a1) { if (RB == 1) c.a1_r1_calls.fetch_add(1, std::memory_order_relaxed); else if (RB == 2) c.a1_r2_calls.fetch_add(1, std::memory_order_relaxed); else if (RB == 4) c.a1_r4_calls.fetch_add(1, std::memory_order_relaxed); else c.a1_r8_calls.fetch_add(1, std::memory_order_relaxed); } else c.a1_fallback_not_enabled.fetch_add(1, std::memory_order_relaxed); }
    for (int k_begin = 0; k_begin < K::K_BLOCK && k_block_begin + k_begin < k; k_begin += K::BufferB::B_K_STEP) {
      const int32_t* a32_lo = (const int32_t*)ba->get_submat(m, k, m_begin, k_block_begin + k_begin) + mb * 16;'''
INC_OLD = r'''#include "la/amx.hpp"'''   # amx_kernels.hpp 가 아니라 include 위치 확인용 (실제 include 는 아래에서 파일 첫 include 뒤에 삽입)
BIND_OLD = r'''  py::class_<WorkerPool>(m, "WorkerPool").def(py::init<int>());'''
BIND_NEW = r'''  m.def("kt_opt_status", []() {   // IDE_076 feature proof
    auto& c = kt_opt::counters(); py::dict d;
    d["a1_requested"] = std::getenv("KT_OPT_A1_ENABLE") ? std::string(std::getenv("KT_OPT_A1_ENABLE")) : std::string("unset");
    d["a1_effective"] = kt_opt::a1_enabled(); d["a2_effective"] = kt_opt::a2_enabled(); d["proof_on"] = kt_opt::proof_on(); d["a1_policy"] = kt_opt::a1_policy();
    d["a1_eligible_calls"] = c.a1_eligible_calls.load(); d["a1_r1_calls"] = c.a1_r1_calls.load(); d["a1_r2_calls"] = c.a1_r2_calls.load(); d["a1_r4_calls"] = c.a1_r4_calls.load(); d["a1_r8_calls"] = c.a1_r8_calls.load(); d["a1_fallback_not_enabled"] = c.a1_fallback_not_enabled.load();
    d["a2_eligible_jobs"] = c.a2_eligible_jobs.load(); d["a2_bundled_jobs"] = c.a2_bundled_jobs.load(); d["a2_fallback_jobs"] = c.a2_fallback_jobs.load();
    return d; });
  m.def("kt_opt_validate", []() { kt_opt::validate_at_init(); return true; });
  py::class_<WorkerPool>(m, "WorkerPool").def(py::init<int>());'''


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def apply():
    os.makedirs(BK, exist_ok=True)
    for f in (KH, EB):
        b = f"{BK}/{os.path.basename(f)}.pre_a1"
        if not os.path.exists(b): shutil.copy(f, b)
    s = open(KH).read()
    if "IDE_076 A1-a" in s: print("kernels already patched"); return
    assert s.count(RB_OLD) == 1, s.count(RB_OLD); assert s.count(HEAD_OLD) == 1, s.count(HEAD_OLD)
    s = s.replace(HEAD_OLD, HEAD_NEW).replace(RB_OLD, RB_NEW)
    # include: 파일의 첫 #include 앞에 kt_opt.h 추가
    i = s.index("#include"); s = s[:i] + '#include "../../../cpu_backend/kt_opt.h"   // IDE_076\n' + s[i:]
    open(KH, "w").write(s)
    open(OPT, "w").write(KT_OPT_H)
    e = open(EB).read()
    if "kt_opt_status" not in e:
        assert e.count(BIND_OLD) == 1; e = e.replace(BIND_OLD, BIND_NEW)
        i = e.index("#include"); e = e[:i] + '#include "cpu_backend/kt_opt.h"   // IDE_076\n' + e[i:]
        open(EB, "w").write(e)
    print("applied", "kernels", sha(KH), "bindings", sha(EB), "kt_opt.h", sha(OPT))


def revert():
    for f in (KH, EB):
        b = f"{BK}/{os.path.basename(f)}.pre_a1"
        if os.path.exists(b): shutil.copy(b, f)
    if os.path.exists(OPT): os.remove(OPT)
    print("reverted", sha(KH), sha(EB))


def status():
    s = open(KH).read(); print("kernels", sha(KH), "A1-a" if "IDE_076 A1-a" in s else "unpatched", "| kt_opt.h", os.path.exists(OPT), "| bindings", "kt_opt_status" in open(EB).read())


if __name__ == "__main__": {"apply": apply, "revert": revert, "status": status}[sys.argv[1]]()
