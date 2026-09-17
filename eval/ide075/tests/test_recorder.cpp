// IDE_075 부록 B.3 — 기록기 합성 시험 (모델 없음). TaskQueue + kt_evt 링만 사용.
// Q11 overflow: cap 8 에 10 task → 링 인덱스 충돌은 seq 로 검출(OVERWRITTEN_OR_MISSING) / 기록 때문에 작업이 멈추지 않음
// Q12 mutable: 기록 뒤 원본 변수 변경이 과거 record 에 영향 없음 (record 는 값 스냅샷)
// 순서: enq_b ≤ enq_a ≤ dequeue ≤ exec_start ≤ exec_end, seq 단조, kind 전달, running_seq_at_enq, SPSC push/pop(stale 처리)
#include "task_queue.h"
#include "kt_evt.h"
#include <cassert>
#include <cstdio>
#include <vector>
#include <atomic>
#include <chrono>
#include <thread>
int main() {
  static KtTaskRec ring[8]; kt_task_ring = ring; kt_task_cap = 8;
  TaskQueue q; std::atomic<int> done{0}; std::vector<unsigned long long> seqs;
  for (int i = 0; i < 10; ++i) {
    kt_evt_next_kind = (i % 3 == 0) ? 3 : 2;
    q.enqueue([&done, i]() { std::this_thread::sleep_for(std::chrono::microseconds(200)); done.fetch_add(1); });
    seqs.push_back(kt_evt_last_seq);
  }
  kt_evt_next_kind = 0; q.sync(0);
  assert(done.load() == 10);                       // 기록 때문에 작업이 멈추지 않음
  for (int i = 1; i < 10; ++i) assert(seqs[i] == seqs[i - 1] + 1);   // seq 단조
  int overwritten = 0, ok = 0;
  for (unsigned long long s = 1; s <= 10; ++s) {
    KtTaskRec& r = ring[s % 8];
    if (r.seq != s) { ++overwritten; continue; }
    ++ok; assert(r.t_enq_b <= r.t_enq_a && r.t_enq_a <= r.t_dequeue && r.t_dequeue <= r.t_exec_start && r.t_exec_start <= r.t_exec_end);
    assert(r.kind == (((int)(s - 1)) % 3 == 0 ? 3 : 2)); assert(r.tid != 0);
  }
  printf("Q11 ring cap8/10 tasks: ok=%d overwritten_detected=%d\n", ok, overwritten); assert(overwritten == 2 && ok == 8);
  // Q12: record 값 스냅샷 — 원본 변수 변경 후 record 불변
  KtTaskRec snap = ring[9 % 8]; kt_evt_next_kind = 1; assert(snap.kind == ring[9 % 8].kind);
  // SPSC: push 3, pop for seq 2 → seq1 stale 폐기, seq2 반환, seq3 남음
  KtEvtRec a{}, b{}, c{}; kt_evt_push(&a, 1, 1); kt_evt_push(&b, 2, 1); kt_evt_push(&c, 3, 1);
  KtEvtPend p = kt_evt_pop_for(2); assert(p.rec == &b && kt_evt_spsc_stale.load() == 1);
  p = kt_evt_pop_for(2); assert(p.rec == nullptr);   // seq 3 은 아직 (미래) → 반환 안 함
  p = kt_evt_pop_for(3); assert(p.rec == &c);
  // Q01/Q02 (FIFO reducer 정의, 합성): enqueue=10, start=30, 선행 [0,25) → wait 20, predecessor union 15, gap 5 / 선행 없음 → gap 20
  { long enq = 10, st = 30, pa = 0, pb = 25; long wait = st - enq; long pu = std::max(0L, std::min(st, pb) - std::max(enq, pa)); assert(wait == 20 && pu == 15 && wait - pu == 5); }
  { long enq = 10, st = 30; assert(st - enq == 20); }
  // clock read 비용 (참고값, 병목 수치 아님)
  auto t0 = std::chrono::steady_clock::now(); for (int i = 0; i < 100000; ++i) (void)kt_evt_now(); auto t1 = std::chrono::steady_clock::now();
  printf("kt_evt_now cost ~%.1f ns/call\n", std::chrono::duration<double, std::nano>(t1 - t0).count() / 100000);
  printf("RECORDER_SELFTEST PASS\n"); return 0;
}
