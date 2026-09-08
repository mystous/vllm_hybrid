#!/usr/bin/env python3
"""AMX MoE forward_prefill phase 분해 (FORWARD_TIME_PROFILE) 를 항상 컴파일하되, 출력은 런타임 env KT_PHASE_PROF=1 일 때 64회에 1회만."""
P="/sgl-workspace/ktransformers/kt-kernel/operators/amx/moe_base.hpp"; s=open(P).read()
if "KT_PHASE_PROF" in s: print("already"); raise SystemExit
old='''    printf(
        "Profiling Results (numa[%d]): activated_expert: %d, prepare: %ld us, cpy_input: %ld us, q_input: %ld us, "'''
new='''    static bool _pp_on = std::getenv("KT_PHASE_PROF") != nullptr; static thread_local unsigned _pp_cnt = 0;
    if (_pp_on && ((++_pp_cnt) & 63) == 0) printf(
        "Profiling Results (numa[%d]): activated_expert: %d, prepare: %ld us, cpy_input: %ld us, q_input: %ld us, "'''
assert s.count(old)==1; s=s.replace(old,new)
s = "#define FORWARD_TIME_PROFILE 1\n#include <cstdlib>\n" + s
open(P,"w").write(s); print("phase prof patch applied")
