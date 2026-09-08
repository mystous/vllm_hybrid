#!/usr/bin/env python3
"""moe-tp.hpp forward(): do_numa_job 과 merge_results 시간을 KT_PHASE_PROF=1 일 때 64회당 1회 출력 ([kt-wrap])."""
P="/sgl-workspace/ktransformers/kt-kernel/operators/moe-tp.hpp"; s=open(P).read()
if "kt-wrap" in s: print("already"); raise SystemExit
old="""    auto pool = config.pool;
    pool->dispense_backend()->do_numa_job([this, pool, qlen, k, expert_ids, input, weights](int numa_id) {
      tps[numa_id]->forward(qlen, k, expert_ids, weights, input, this->local_output_numa[numa_id]);
    });

    merge_results(qlen, output, incremental);"""
new="""    auto pool = config.pool;
    static bool _wp_on = std::getenv("KT_PHASE_PROF") != nullptr; static thread_local unsigned _wp_cnt = 0;
    auto _wp_t0 = std::chrono::steady_clock::now();
    pool->dispense_backend()->do_numa_job([this, pool, qlen, k, expert_ids, input, weights](int numa_id) {
      tps[numa_id]->forward(qlen, k, expert_ids, weights, input, this->local_output_numa[numa_id]);
    });
    auto _wp_t1 = std::chrono::steady_clock::now();
    merge_results(qlen, output, incremental);
    if (_wp_on && ((++_wp_cnt) & 63) == 0) {
      auto _wp_t2 = std::chrono::steady_clock::now();
      printf("[kt-wrap] qlen: %d, numa_job: %ld us, merge: %ld us, incremental: %d\\n", qlen,
             (long)std::chrono::duration_cast<std::chrono::microseconds>(_wp_t1 - _wp_t0).count(),
             (long)std::chrono::duration_cast<std::chrono::microseconds>(_wp_t2 - _wp_t1).count(), (int)incremental);
    }"""
assert s.count(old)==1, s.count(old); s=s.replace(old,new)
if "#include <chrono>" not in s: s="#include <chrono>\n#include <cstdlib>\n#include <cstdio>\n"+s
open(P,"w").write(s); print("wrapper timing patch applied")
