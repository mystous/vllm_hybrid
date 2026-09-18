#!/usr/bin/env python3
"""IDE_076 C08 (Q-A2-1) — worker pool dispatch/장벽 tail 계측 (proof 모드 전용). 컨테이너 sgl-kt 안에서 apply | revert | status.
cpu_backend/worker_pool.cpp 의 InNumaPool::do_work_stealing_job / process_tasks / wait 에 kt_opt::proof_on() 일 때만 동작하는 계측을 넣는다:
  - dispatch 당: t0(async 시작 직전) → t1(모든 워커 notify 완료) → t2(부모 process_tasks 종료) → t3(wait 종료); 워커별 process_tasks 종료 시각.
  - 누적(atomic, pool 단위): n_dispatch, sum_notify_ns(t1−t0), sum_wall_ns(t3−t0), sum_tail_ns(t3 − 워커 종료 시각의 중앙값), sum_parent_ns(t2−t0), max_tail_ns, sum_tasks.
  OFF(proof 미설정) 경로: 정적 bool 분기 1회만 추가, task 수·순서·notify 방식 불변.
kt_opt_status() 에 pool 계측 합계를 노출 (ext_bindings 의 기존 kt_opt_status 람다에 필드 추가)."""
import sys, os, shutil, hashlib
ROOT = "/sgl-workspace/ktransformers/kt-kernel"; BK = "/sgl-workspace/ide076_backup"
WP = f"{ROOT}/cpu_backend/worker_pool.cpp"; WH = f"{ROOT}/cpu_backend/worker_pool.h"; OPT = f"{ROOT}/cpu_backend/kt_opt.h"; EB = f"{ROOT}/ext_bindings.cpp"

OPT_ADD_OLD = "struct Counters {"
OPT_ADD_NEW = """struct PoolProbe {   // IDE_076 Q-A2-1: dispatch/장벽 계측 (proof 모드 전용)
  std::atomic<unsigned long long> n_dispatch{0}, sum_tasks{0}, sum_notify_ns{0}, sum_wall_ns{0}, sum_parent_ns{0}, sum_tail_ns{0}, max_tail_ns{0}, sum_worker_span_ns{0};
};
inline PoolProbe& pool_probe() { static PoolProbe p; return p; }
struct Counters {"""

WP_INC_OLD = '#include "worker_pool.h"'
WP_INC_NEW = '#include "worker_pool.h"\n#include "kt_opt.h"   // IDE_076 Q-A2-1\n#include <algorithm>\n#include <chrono>\n#include <vector>'

# ThreadState 에 종료 시각(ns) 필드 추가
WH_OLD = "struct alignas(64) ThreadState {"
WH_NEW = "struct alignas(64) ThreadState {\n  std::atomic<long long> end_ns{0};   // IDE_076 Q-A2-1: process_tasks 종료 시각 (proof 모드)"

DJ_OLD = """void InNumaPool::do_work_stealing_job(int task_num, std::function<void(int)> init_func,
                                      std::function<void(int)> compute_func, std::function<void(int)> finalize_func) {
  do_work_stealing_job_async(task_num, init_func, compute_func, finalize_func);
  wait();
}"""
DJ_NEW = """static inline long long kt_now_ns_() { return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
void InNumaPool::do_work_stealing_job(int task_num, std::function<void(int)> init_func,
                                      std::function<void(int)> compute_func, std::function<void(int)> finalize_func) {
  if (!kt_opt::proof_on()) { do_work_stealing_job_async(task_num, init_func, compute_func, finalize_func); wait(); return; }
  // IDE_076 Q-A2-1 계측 경로 (proof 모드): 워커 종료 시각을 모아 장벽 tail 을 계산
  const long long t0 = kt_now_ns_();
  for (int i = 0; i < std::min(restricted_worker_count, task_num); i++) thread_state_[i].end_ns.store(0, std::memory_order_relaxed);
  do_work_stealing_job_async(task_num, init_func, compute_func, finalize_func);   // 안에서 부모가 process_tasks(0) 까지 수행
  const long long t2 = kt_now_ns_();
  wait();
  const long long t3 = kt_now_ns_();
  std::vector<long long> ends; ends.reserve(worker_count);
  for (int i = 0; i < worker_count; i++) { long long e = thread_state_[i].end_ns.load(std::memory_order_relaxed); if (e) ends.push_back(e); }
  auto& p = kt_opt::pool_probe(); p.n_dispatch.fetch_add(1, std::memory_order_relaxed); p.sum_tasks.fetch_add((unsigned long long)task_num, std::memory_order_relaxed);
  p.sum_wall_ns.fetch_add((unsigned long long)(t3 - t0), std::memory_order_relaxed); p.sum_parent_ns.fetch_add((unsigned long long)(t2 - t0), std::memory_order_relaxed);
  if (!ends.empty()) { std::sort(ends.begin(), ends.end()); long long med = ends[ends.size() / 2]; long long tail = t3 - med; if (tail < 0) tail = 0; p.sum_tail_ns.fetch_add((unsigned long long)tail, std::memory_order_relaxed); unsigned long long cur = p.max_tail_ns.load(); while ((unsigned long long)tail > cur && !p.max_tail_ns.compare_exchange_weak(cur, (unsigned long long)tail)) {} p.sum_worker_span_ns.fetch_add((unsigned long long)(ends.back() - ends.front()), std::memory_order_relaxed); }
}"""

PT_OLD = """  s.status.store(ThreadStatus::WAITING, std::memory_order_release);
#ifdef PROFILE_BALANCE"""
PT_NEW = """  if (kt_opt::proof_on()) s.end_ns.store(kt_now_ns_(), std::memory_order_relaxed);   // IDE_076 Q-A2-1
  s.status.store(ThreadStatus::WAITING, std::memory_order_release);
#ifdef PROFILE_BALANCE"""

BIND_OLD = '    d["a2_eligible_jobs"] = c.a2_eligible_jobs.load();'
BIND_NEW = '''    { auto& p = kt_opt::pool_probe(); d["pool_n_dispatch"] = p.n_dispatch.load(); d["pool_sum_tasks"] = p.sum_tasks.load(); d["pool_sum_wall_ns"] = p.sum_wall_ns.load(); d["pool_sum_parent_ns"] = p.sum_parent_ns.load(); d["pool_sum_tail_ns"] = p.sum_tail_ns.load(); d["pool_max_tail_ns"] = p.max_tail_ns.load(); d["pool_sum_worker_span_ns"] = p.sum_worker_span_ns.load(); }
    d["a2_eligible_jobs"] = c.a2_eligible_jobs.load();'''


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def apply():
    os.makedirs(BK, exist_ok=True)
    for f in (WP, WH, OPT, EB):
        b = f"{BK}/{os.path.basename(f)}.pre_a2probe"
        if not os.path.exists(b): shutil.copy(f, b)
    s = open(WP).read()
    if "IDE_076 Q-A2-1" in s: print("already"); return
    assert s.count(WP_INC_OLD) == 1 and s.count(DJ_OLD) == 1 and s.count(PT_OLD) == 1, (s.count(WP_INC_OLD), s.count(DJ_OLD), s.count(PT_OLD))
    # kt_now_ns_ 는 process_tasks 보다 앞에 정의되어야 함 → include 뒤에 전방 정의
    s = s.replace(WP_INC_OLD, WP_INC_NEW + "\nstatic inline long long kt_now_ns_();")
    s = s.replace(DJ_OLD, DJ_NEW).replace(PT_OLD, PT_NEW); open(WP, "w").write(s)
    h = open(WH).read(); assert h.count(WH_OLD) == 1; h = h.replace(WH_OLD, WH_NEW)
    if "#include <atomic>" not in h: h = h.replace("#pragma once", "#pragma once\n#include <atomic>", 1) if "#pragma once" in h else "#include <atomic>\n" + h
    open(WH, "w").write(h)
    o = open(OPT).read(); assert o.count(OPT_ADD_OLD) == 1; open(OPT, "w").write(o.replace(OPT_ADD_OLD, OPT_ADD_NEW))
    e = open(EB).read(); assert e.count(BIND_OLD) == 1; open(EB, "w").write(e.replace(BIND_OLD, BIND_NEW))
    print("applied", sha(WP), sha(WH), sha(OPT), sha(EB))


def revert():
    for f in (WP, WH, OPT, EB):
        b = f"{BK}/{os.path.basename(f)}.pre_a2probe"
        if os.path.exists(b): shutil.copy(b, f)
    print("reverted")


def status(): print("worker_pool.cpp", sha(WP), "probe" if "IDE_076 Q-A2-1" in open(WP).read() else "unpatched")


if __name__ == "__main__": {"apply": apply, "revert": revert, "status": status}[sys.argv[1]]()
