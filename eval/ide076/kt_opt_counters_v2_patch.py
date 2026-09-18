#!/usr/bin/env python3
"""IDE_076 — proof 카운터를 스레드별(thread_local, 등록 목록 합산) 로 변경 (공유 atomic 경합 제거). apply | revert | status (컨테이너 안).
전제: kt_opt_a1_hoist_patch 적용 상태 (카운터 블록이 avx_kernel_rb 에 있음)."""
import sys, os, shutil, hashlib
ROOT = "/sgl-workspace/ktransformers/kt-kernel"; BK = "/sgl-workspace/ide076_backup"; OPT = f"{ROOT}/cpu_backend/kt_opt.h"; KH = f"{ROOT}/operators/amx/la/amx_kernels.hpp"; EB = f"{ROOT}/ext_bindings.cpp"
OLD_C = """struct Counters {
  std::atomic<unsigned long long> a1_eligible_calls{0}, a1_r1_calls{0}, a1_r2_calls{0}, a1_r4_calls{0}, a1_r8_calls{0}, a1_fallback_not_enabled{0};
  std::atomic<unsigned long long> a2_eligible_jobs{0}, a2_bundled_jobs{0}, a2_fallback_jobs{0};
};
inline Counters& counters() { static Counters c; return c; }"""
NEW_C = """struct Counters {   // 스레드별 (경합 없음); 합산은 sum_counters()
  unsigned long long a1_eligible_calls = 0, a1_r1_calls = 0, a1_r2_calls = 0, a1_r4_calls = 0, a1_r8_calls = 0, a1_fallback_not_enabled = 0;
  unsigned long long a2_eligible_jobs = 0, a2_bundled_jobs = 0, a2_fallback_jobs = 0;
};
struct CounterRegistry { std::mutex m; std::vector<Counters*> all; };
inline CounterRegistry& registry() { static CounterRegistry r; return r; }
inline Counters& counters() { static thread_local Counters* c = [] { Counters* p = new Counters(); auto& r = registry(); std::lock_guard<std::mutex> g(r.m); r.all.push_back(p); return p; }(); return *c; }
inline Counters sum_counters() { Counters s; auto& r = registry(); std::lock_guard<std::mutex> g(r.m); for (auto* c : r.all) { s.a1_eligible_calls += c->a1_eligible_calls; s.a1_r1_calls += c->a1_r1_calls; s.a1_r2_calls += c->a1_r2_calls; s.a1_r4_calls += c->a1_r4_calls; s.a1_r8_calls += c->a1_r8_calls; s.a1_fallback_not_enabled += c->a1_fallback_not_enabled; s.a2_eligible_jobs += c->a2_eligible_jobs; s.a2_bundled_jobs += c->a2_bundled_jobs; s.a2_fallback_jobs += c->a2_fallback_jobs; } return s; }"""
OLD_K = """      c.a1_eligible_calls.fetch_add(n8 + n4 + n2 + n1, std::memory_order_relaxed);
      if (on) { c.a1_r8_calls.fetch_add(n8, std::memory_order_relaxed); c.a1_r4_calls.fetch_add(n4, std::memory_order_relaxed); c.a1_r2_calls.fetch_add(n2, std::memory_order_relaxed); c.a1_r1_calls.fetch_add(n1, std::memory_order_relaxed); } else c.a1_fallback_not_enabled.fetch_add(n8 + n4 + n2 + n1, std::memory_order_relaxed);"""
NEW_K = """      c.a1_eligible_calls += n8 + n4 + n2 + n1;
      if (on) { c.a1_r8_calls += n8; c.a1_r4_calls += n4; c.a1_r2_calls += n2; c.a1_r1_calls += n1; } else c.a1_fallback_not_enabled += n8 + n4 + n2 + n1;"""
OLD_B = "    auto& c = kt_opt::counters(); py::dict d;"
NEW_B = "    kt_opt::Counters c = kt_opt::sum_counters(); py::dict d;"
OLD_B2 = 'd["a1_eligible_calls"] = c.a1_eligible_calls.load(); d["a1_r1_calls"] = c.a1_r1_calls.load(); d["a1_r2_calls"] = c.a1_r2_calls.load(); d["a1_r4_calls"] = c.a1_r4_calls.load(); d["a1_r8_calls"] = c.a1_r8_calls.load(); d["a1_fallback_not_enabled"] = c.a1_fallback_not_enabled.load();'
NEW_B2 = 'd["a1_eligible_calls"] = c.a1_eligible_calls; d["a1_r1_calls"] = c.a1_r1_calls; d["a1_r2_calls"] = c.a1_r2_calls; d["a1_r4_calls"] = c.a1_r4_calls; d["a1_r8_calls"] = c.a1_r8_calls; d["a1_fallback_not_enabled"] = c.a1_fallback_not_enabled;'
OLD_B3 = 'd["a2_eligible_jobs"] = c.a2_eligible_jobs.load(); d["a2_bundled_jobs"] = c.a2_bundled_jobs.load(); d["a2_fallback_jobs"] = c.a2_fallback_jobs.load();'
NEW_B3 = 'd["a2_eligible_jobs"] = c.a2_eligible_jobs; d["a2_bundled_jobs"] = c.a2_bundled_jobs; d["a2_fallback_jobs"] = c.a2_fallback_jobs;'


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def apply():
    for f in (OPT, KH, EB):
        b = f"{BK}/{os.path.basename(f)}.pre_cnt2"
        if not os.path.exists(b): shutil.copy(f, b)
    o = open(OPT).read()
    if "CounterRegistry" in o: print("already"); return
    assert o.count(OLD_C) == 1, "OLD_C"; o = o.replace(OLD_C, NEW_C)
    if "#include <mutex>" not in o: o = o.replace("#include <atomic>", "#include <atomic>\n#include <mutex>\n#include <vector>", 1)
    k = open(KH).read(); assert k.count(OLD_K) == 1, "OLD_K"
    e = open(EB).read(); assert e.count(OLD_B) == 1 and e.count(OLD_B2) == 1 and e.count(OLD_B3) == 1, "OLD_B*"
    open(OPT, "w").write(o); open(KH, "w").write(k.replace(OLD_K, NEW_K)); open(EB, "w").write(e.replace(OLD_B, NEW_B).replace(OLD_B2, NEW_B2).replace(OLD_B3, NEW_B3))
    print("applied", sha(OPT), sha(KH), sha(EB))


def revert():
    for f in (OPT, KH, EB):
        b = f"{BK}/{os.path.basename(f)}.pre_cnt2"
        if os.path.exists(b): shutil.copy(b, f)
    print("reverted")


def status(): print("kt_opt.h", sha(OPT), "cnt2" if "CounterRegistry" in open(OPT).read() else "cnt1")


if __name__ == "__main__": {"apply": apply, "revert": revert, "status": status}[sys.argv[1]]()
