#!/usr/bin/env python3
"""IDE_045: expert 가중치 버퍼 (gate/up/down BufferB) 를 2MB 정렬 + madvise(MADV_HUGEPAGE) 로 할당 (THP defrag=madvise 환경에서 huge page 보장)."""
P="/sgl-workspace/ktransformers/kt-kernel/operators/amx/moe_base.hpp"; s=open(P).read()
if "MADV_HUGEPAGE" in s: print("already"); raise SystemExit
import re
n=s.count("std::aligned_alloc(64,"); assert n>=3, n
s=s.replace("std::aligned_alloc(64,","kt_hp_alloc(")
helper='''#include <sys/mman.h>
static inline void* kt_hp_alloc(size_t sz) {
  static bool on = std::getenv("KT_HUGEPAGE") != nullptr;
  if (!on) return std::aligned_alloc(64, sz);
  size_t al = 2u << 20; size_t rsz = (sz + al - 1) / al * al;
  void* p = std::aligned_alloc(al, rsz);
  if (p) madvise(p, rsz, MADV_HUGEPAGE);
  return p;
}
'''
# 첫 #include 뒤에 삽입
i=s.index("#include"); j=s.index("\n",i)+1
s=s[:j]+helper+s[j:]
if "#include <cstdlib>" not in s: s="#include <cstdlib>\n"+s
open(P,"w").write(s); print("hugepage patch applied (%d sites)"%n)
