# PROBE_MAP_V2 — IDE_075 기록기 v2 (실제 적용 소스 기준; 컨테이너 `sgl-kt`, 패치 `eval/ide075/kt_evt_patch_v2.py`, diff `evidence/kt_evt_patch_v2.diff`)

| probe | 파일::함수 | 찍는 스레드 | clock | 비용 | 포함/제외 | graph replay 동작 | OFF 우회 |
|---|---|---|---|---|---|---|---|
| task_seq, kind, pending_at_enq, running_seq_at_enq, t_enq_b/t_enq_a | `cpu_backend/task_queue.cpp::TaskQueue::enqueue` | enqueue 호출 스레드 (poller / Python) | CLOCK_REALTIME ns | clock 2회(≈27 ns 씩) + 링 write | 가시화 = `prev->next.store` 전후 bracket (단일 시점 아님) | 그래프와 무관 (host task) | `kt_task_ring==nullptr` 이면 seq 카운터·Node.seq 만 (상시) |
| t_dequeue, t_exec_start, t_exec_end, tid | `task_queue.cpp::TaskQueue::worker` | task worker | 동일 | clock 3회 | exec 는 std::function 호출 전후 (wrapper 포함) | — | 동일 |
| t_go, qlen, n_cold_ids | `cpu_backend/cpuinfer.h::CPUInfer::poll_loop_` go 관측 직후 | poller | 동일 | clock 1회 + ids 스캔(qlen×k) | go 플래그 = GPU memop 노드 실행 뒤 host 관측 | 층·graph 별 slot 고정, epoch 증가 | `evt_==nullptr` |
| t_imm/done/def_enq_b/a, *_task_seq | `poll_loop_` 각 enqueue 전후 | poller | 동일 | clock 2회/enqueue | task_seq 는 enqueue 가 설정한 thread_local | — | rec==nullptr |
| t_done_store_b/a | `poll_loop_` 가 enqueue 하는 done-setter 람다 (기존 업무 task) | task worker | 동일 | clock 2회 | `__atomic_store_n(done,1,RELEASE)` 전후 bracket; memory order 불변 | GPU `cuStreamWaitValue32` 가 관측 | rec==nullptr |
| t_fwd_entry/exit, t_imm_fwd_entry/exit | `operators/moe-tp.hpp::TP_MOE_Common::forward` 진입/반환 (SPSC pop 이 현재 task_seq 와 일치할 때만) | task worker | 동일 | clock 2회 + SPSC pop | 작업 본체 (fork-join + merge 포함) | — | pend.rec==nullptr |
| t_numa_start/end[s] | 같은 함수의 `do_numa_job` 람다 | NUMA 서브풀 dispatch 스레드 | 동일 | clock 2회/서브풀 | 서브풀 forward 호출 전후 | — | 동일 |
| stage_us[s][9], activated_expert, max_local_num, sum_rows, qlen_seen | `operators/amx/moe_base.hpp::forward_prefill` (qlen>1) / `forward_decode` | NUMA 서브풀 스레드 (`kt_rec_` 멤버 경유) | 기존 high_resolution_clock µs (상대) | 기존 타이머 값 복사 (추가 계산 0) | prefill 단계 9개 / decode 7개 (prepare·cpy 없음) | — | `kt_rec_==nullptr` |
| n_amx/n_avx[s] | `operators/amx/moe.hpp::do_gate_up_gemm`/`do_down_gemm` (ith==0) | 서브풀 워커 | — | atomic add 3/expert | 실제 분기 (qlen>KT_AMX_MIN_QLEN(80) or m≥KT_AMX_MIN_ROWS) | — | 동일 |
| slot→(layer, rows, capturing, ns) | site-packages `kt_kernel/experts_base.py::submit_forward` arm/rearm 직후, go 전 | TP0 스케줄러 Python | time.time_ns | 파일 append (캡처 시 1회/graph·층, eager 는 호출마다) | effective_from = ns (go 는 그 뒤) | 캡처 슬롯 영구 | env 미설정 시 없음 |
| flusher | `cpuinfer.h::evt_flusher_` (200 ms) | 전용 스레드 (affinity 미고정) | — | observer 비용 (CPU 시간 미측정) | 완료 순 기록, 5 s 미완료 partial | — | 미생성 |

GPU-side (변경 없음): kineto 트레이스의 graph id/node id + memcpy 패턴 (IDE_074 probe_map.md B). wait memop 은 트레이스에 없음 → HtoD 시작 = 해제 상한.
검증: 합성 시험 `tests/test_recorder.cpp` (Q11 overflow 검출, 순서, SPSC stale, Q12 스냅샷), 세션별 task_count_valid(패킷당 task = imm+done+def, 기록 전용 0), lifecycle(fwd 미기록 ≤ 0.1 %: S2 5/24,490, S4 2/24,490, S5 1/24,490).

## 확장 단계 추가 (v3 기록기·클라이언트·FOCUS)

| probe | 파일::함수 | 찍는 스레드 | clock | 비용 | 비고 |
|---|---|---|---|---|---|
| rec 포인터 (v3) | `task_queue.h::Node.rec` — enqueue 시 Node 가 현재 레코드 포인터를 보유, worker 가 실행 전 thread_local 에 복원 | enqueue 스레드 / task worker | — | 포인터 1개 복사 | v2 의 SPSC pop 경쟁(레코드 누락) 제거. fwd 레코드 누락 0 |
| 스레드 이름 | `cpuinfer.h` poller `kt-cf-poll`, flusher `kt-evt-flush`, `task_queue.cpp` worker `kt-task-worker` (pthread_setname_np) | 각 스레드 시작 시 1회 | — | 0 | TID 역할 분류·perf comm 식별용. NUMA 워커는 기존 이름 `numa_<n>_t_<i>` |
| expert 단위 rows 표본 | `moe_base.hpp::forward_prefill` 끝, `(slot+epoch)%16==0` 인 작업만: expert id·rows 쌍 ≤64 개를 ring(65,536) 에 기록 | NUMA 서브풀 dispatch 스레드 | — | 표본 작업당 expert_num 스캔 1회 | `.experts` 파일 (flusher 가 기록). 모집단 확대 없음 |
| 클라이언트 barrier·anchor (vllmw) | `eval/ide075/vllm_bench_wrapper.py` — `vllm.benchmarks.serve.benchmark` 진입 시 `<ctl>.ready` 생성 → `<ctl>.start` 대기 → `<ctl>.anchor.json` (realtime_ns, perf_counter, monotonic_ns) | 클라이언트 (vllm-h100 컨테이너) | CLOCK_REALTIME + perf_counter | 0 (요청 코드 무수정) | 요청별 `start_times`/`ttfts`/`itls` (vllm 저장) → anchor 로 realtime 변환. 동등성: OFF_Y 0.3 % 이내 |
| perf sched (S14/S19/S25) | `perf record -e sched:sched_switch -e sched:sched_wakeup -p <scheduler pids>` | 커널 tracepoint | perf(sched_clock≈monotonic) → clock_anchors | 손실 6~10 % | **-p 는 대상 스레드 switch-out 만 기록** → off-CPU 길이 NOT_RECOVERABLE |
| perf sched syswide (S28) | `IDE075_FOCUS_SYSWIDE=1`: `perf record -a -k CLOCK_MONOTONIC -m 4096 -e sched:sched_switch` | 커널 tracepoint | CLOCK_MONOTONIC → clock_anchors 직접 | 시스템 전체 | switch-in 포함 → 워커 off-CPU 구간 산출 가능 |
| GPU 프로파일 (S9) | 프로파일 id 불일치로 metrics.json trace_files 가 비어 있던 것을 실제 파일 `S9_CORR_1789629816-TP-*` 로 복구 | — | kineto | — | 트레이스는 16:23~16:29 전체, 세션 창으로 절단 |
