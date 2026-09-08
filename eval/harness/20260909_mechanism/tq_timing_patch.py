#!/usr/bin/env python3
"""TaskQueue 작업별 계측 (KT_TQ_TIMING=1): 워커가 각 task 의 실행시간(exec_us) 과 직전 작업 종료→이 작업 시작 대기(wait_us) 를 기록,
2048 작업마다 stderr 로 요약 출력 ([kt-tq] exec p50/p90/p99 + 구간 히스토그램, wait p50/p90/p99, busy 비율)."""
P="/sgl-workspace/ktransformers/kt-kernel/cpu_backend/task_queue.cpp"; s=open(P).read()
if "kt-tq" in s: print("already"); raise SystemExit
old="""      if (next->task) {
        try {
          next->task();
        } catch (...) {"""
new="""      if (next->task) {
        try {
          if (tq_on) tq_before_();
          next->task();
          if (tq_on) tq_after_();
        } catch (...) {"""
assert s.count(old)==1; s=s.replace(old,new)
helper = r'''
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <vector>
static bool tq_on = std::getenv("KT_TQ_TIMING") != nullptr;
static thread_local std::chrono::steady_clock::time_point tq_t0, tq_prev_end;
static thread_local bool tq_have_prev = false;
static thread_local std::vector<unsigned> tq_exec, tq_wait;
static void tq_before_() {
  tq_t0 = std::chrono::steady_clock::now();
  if (tq_have_prev) tq_wait.push_back((unsigned)std::chrono::duration_cast<std::chrono::microseconds>(tq_t0 - tq_prev_end).count());
}
static unsigned tq_pct(std::vector<unsigned>& v, double p) { if (v.empty()) return 0; size_t i = (size_t)(p * (v.size() - 1)); std::nth_element(v.begin(), v.begin() + i, v.end()); return v[i]; }
static void tq_after_() {
  auto t1 = std::chrono::steady_clock::now();
  unsigned us = (unsigned)std::chrono::duration_cast<std::chrono::microseconds>(t1 - tq_t0).count();
  tq_exec.push_back(us); tq_prev_end = t1; tq_have_prev = true;
  if (tq_exec.size() >= 2048) {
    static const unsigned edges[] = {5, 30, 100, 300, 600, 1000, 2000, 4000};
    unsigned hist[9] = {0}; double sum_e = 0, sum_w = 0;
    for (unsigned e : tq_exec) { int b = 0; while (b < 8 && e >= edges[b]) ++b; hist[b]++; sum_e += e; }
    for (unsigned w : tq_wait) sum_w += w;
    fprintf(stderr, "[kt-tq] n=%zu exec_us p50=%u p90=%u p99=%u mean=%.0f | hist(<5,<30,<100,<300,<600,<1k,<2k,<4k,>=4k)=%u,%u,%u,%u,%u,%u,%u,%u,%u | wait_us p50=%u p90=%u p99=%u mean=%.0f | busy=%.2f\n",
            tq_exec.size(), tq_pct(tq_exec, .5), tq_pct(tq_exec, .9), tq_pct(tq_exec, .99), sum_e / tq_exec.size(),
            hist[0], hist[1], hist[2], hist[3], hist[4], hist[5], hist[6], hist[7], hist[8],
            tq_pct(tq_wait, .5), tq_pct(tq_wait, .9), tq_pct(tq_wait, .99), tq_wait.empty() ? 0.0 : sum_w / tq_wait.size(),
            sum_e / (sum_e + sum_w + 1e-9));
    fflush(stderr); tq_exec.clear(); tq_wait.clear();
  }
}
'''
s=s.replace("#include <thread>\n","#include <thread>\n"+helper,1)
open(P,"w").write(s); print("tq timing patch applied")
