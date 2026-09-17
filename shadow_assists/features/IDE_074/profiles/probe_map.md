# probe_map — IDE_074 (M3). 실제 파일::함수::행, 타임스탬프 위치, 완료의 의미, clock domain, capture/replay 동작

경로는 컨테이너 `sgl-kt` 기준. 행 번호는 패치 전 원본(코드 지도 `code_map_survey.md`, 2026-09-17) 기준이며 패치 후 오프셋이 생긴 파일은 `+n` 표기. Python 실행본은 site-packages (`/usr/local/lib/python3.12/dist-packages/kt_kernel/experts_base.py`), C++ 는 `/sgl-workspace/ktransformers/kt-kernel/`.

## A. CPU-side 이벤트 (신규, `KT_EVT=<csv>` 로만 활성; `cpu_backend/kt_evt.h`, `clock_gettime(CLOCK_REALTIME)` ns)
| 이벤트 | 위치 | 찍는 스레드 | 의미 | 비고 |
|---|---|---|---|---|
| t_go | `cpu_backend/cpuinfer.h::CPUInfer::poll_loop_` — `go_host_[slot]==1` 관측 직후 (원본 210-211 뒤) | poller (busy-poll 전용 스레드) | GPU 스트림의 `cuStreamWriteValue32(go_dev_+slot)` 노드가 실행된 뒤 host 가 이를 본 시각 = 층 L 의 DtoH 4건 완료 이후 (스트림 순서) | 층·graph 별 slot 고정, epoch = slot 별 관측 횟수 |
| t_imm_enq | 같은 함수, `p.imm_fn(p.imm_args)` 직후 (원본 214) | poller | immediate 작업 enqueue 완료 (OPT4 에선 imm 없음) | |
| t_done_enq / t_done_set | `task_queue_->enqueue([d,rec]{ store done=1; t_done_set=now })` (원본 218) | poller / task worker | enqueue 시각 / **done 플래그가 실제로 1 이 된 시각 (GPU `cuStreamWaitValue32` 가 해제될 조건)** | 단일 FIFO 이므로 done_set(L) 은 deferred(L−1) 완료 직후 |
| t_def_enq / t_def_start / t_def_end | `p.def_fn()` 전후에 stamp 태스크 enqueue (원본 219) | poller / task worker | deferred 작업 enqueue / 실제 실행 시작 / 종료 | stamp 태스크 2개/층 추가 (ON 에만) |
| t_numa{0,1}_start/end | `operators/moe-tp.hpp::TP_MOE_Common::forward` 의 `do_numa_job` 람다 안 (원본 217-219) | NUMA 서브풀 워커 | 서브풀별 expert 계산 시작/종료 | `kt_evt_cur`(thread_local, task worker 에서 캡처) |
| qlen, n_cold_ids | poll_loop_ 에서 `FwdArgsView` 의 qlen·expert_ids 스캔 | poller | 캡처 rows 와 cold(=!gpu_mask) id 수 (deferred ids 기준) | ids 는 DtoH 완료본 (pinned) |
| slot→layer 맵 | `experts_base.py::submit_forward` arm/rearm 직후 (`KT_EVT.map`: slot,layer_idx,rows,capturing,ns) | TP0 스케줄러 Python | 캡처 시 1회/graph·층, eager(prefill) 시 호출마다 | |

## B. GPU-side 이벤트 (kineto Chrome trace, `/start_profile activities ["CPU","GPU"] start_step=1`; host CLOCK_REALTIME = baseTimeNanoseconds/1e3 + ts)
| 이벤트 | 식별 방법 | 의미 | 비고 |
|---|---|---|---|
| replay 경계 | 같은 `graph id` 에서 `graph node id` 감소 | CUDA graph replay 1회 = decode step 1회 | eager(prefill) 는 graph id 0 |
| step/phase | `gpu_user_annotation` `step[DECODE bs=..]` / `step[EXTEND bs=.. toks=..]` (scheduler 가 붙임) | 실제 scheduler phase·batch | |
| 층 L | replay 내 HtoD(Pinned→Device, rows×12288 B) 의 순번 (0..61) | `experts_base.py::copy_forward_output_to_device:950` 의 `output_gpu[slot].copy_` | HEURISTIC_LAYER_ID (패턴 검증: 층당 DtoH 4·HtoD 1·fused_moe 2) |
| t_dtoh_first_start / t_dtoh_last_end | 층 세그먼트의 DtoH 4건 (hidden 786,432 B·ids 4,096 B×2·weights 2,048 B @rows 64) | `experts_base.py::_prepare_forward_cpu_buffers:650-654` | CPU 입력 준비 완료 = last_end |
| (go 노드) | 트레이스에 없음 (memop 은 CUPTI 커널이 아님) | `cpuinfer.h::go_on_stream:195` | t_go(CPU) ≥ t_dtoh_last_end 로 순서 검증 |
| t_hot_first_start / t_prev_op_end(=hot_ready) | `fused_moe_kernel` 첫 시작 / HtoD 직전 커널(`moe_sum_reduce_kernel`) 종료 | `kt_ep_wrapper.py::apply:418` → triton fused_moe | hot 기여분 준비 완료 |
| (wait 노드) | 트레이스에 없음 | `cpuinfer.h::wait_done_on_stream:201-202` (`cuStreamWaitValue32 GEQ 1` + reset write) | 해제 시각의 상한 = HtoD 시작 |
| t_htod_start / t_htod_end | HtoD 노드 | cold 출력 `output_cpu[slot]` → GPU | cold 를 GPU 가 소비 가능 = htod_end |
| t_combine_start/end | HtoD 직후 elementwise 커널 | `kt_ep_wrapper.py::apply:429` `output + cpu_output` | 결합 |
| all-reduce | `allreduce_fusion_kernel_oneshot_lamport` (층당 2개) | `qwen3_moe.py:346` (MoE 뒤), attention 뒤 | rank 1~3 은 여기서 rank 0 대기 (순번 귀속 미확정) |

## C. CUDA graph capture / replay
- 디코드(bs ∈ capture_bs): 위 GPU 노드 전부가 그래프에 캡처되고 replay 마다 재실행. Python 은 replay 당 `execute→load_batch→graph.replay()` 만 실행 (층별 0회). `arm_packet` 은 캡처 시 1회(슬롯 영구), `go/wait` memop 도 그래프 노드.
- prefill(eager): `rearm_packet` 으로 층당 슬롯 1개 재사용, 호출마다 맵 행 기록.
- `ensure_cf_`/`evt_ensure_` 의 `cudaHostAlloc`·스레드 생성은 캡처 전 워밍업(eager 2회)에서 완료.

## D. clock
- CPU 이벤트: CLOCK_REALTIME(ns). GPU 트레이스: kineto host 시계(CUPTI 가 GPU 타임스탬프를 host 로 변환). 정렬 오차는 별도 측정하지 않음 → 순서 검사(`t_go ≥ t_dtoh_last_end`, `t_htod_start ≥ t_done_set`) 위반 0 을 유효 조건으로 사용. 위반 시 UNMEASURED_CLOCK_ALIGNMENT.
- 벤치 클라이언트(vllm-h100 컨테이너) 시계: 같은 host → 같은 CLOCK_REALTIME.

## E. 기존 타이머 (변경 없음, P2 에서만 stdout)
- `[kt-tq]` (`cpu_backend/task_queue.cpp:23-48`, KT_TQ_TIMING, 2048 태스크마다 exec/wait 분포, steady_clock µs) — stamp 태스크(ON)가 태스크 수에 포함됨을 주의.
- `Profiling Results (numa[i])` prefill 경로 (`operators/amx/moe_base.hpp:500-506`, KT_PHASE_PROF, 64회 1회): `cpy_input` = pinned→expert 별 연속 배열 gather(CPU RAM 내부), `weight` = top-k 가중합 리덕션. DtoH·DRAM weight fetch 가 아님.
- decode 경로 printf (`moe_base.hpp:711`) 는 IDE_074 에서 KT_PHASE_PROF + 64회 게이트로 변경 (CONFIG_DIFF).
- `[kt-wrap]` (`operators/moe-tp.hpp:222-227`): numa_job(fork-join)·merge µs, 64회 1회.

## F. 미확정·부재
- GPU-side wait 해제 정확 시각(memop 은 트레이스 미기록) → HtoD 시작으로 대체 (상한).
- `Packet`→layer 의 C++ 측 매핑 없음 → Python 맵 파일로 연결. job id/epoch 는 C++ 에 없어 slot_epoch 카운터를 신설.
- attention/router/collective 의 층별 시간 분해는 커널 이름 수준으로만 가능 (본 캠페인 미집계).
