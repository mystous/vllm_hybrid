# PLAN_RESOLVED — IDE_074 (2026-09-17 착수, 지시서 `PLAN.md` M0~M9 의 실행 확정본)

지시서의 제안 항목(이벤트 스키마·세션 배분·순서)을 실제 코드·환경 확인(코드 지도 `profiles/code_map_survey.md`) 후 아래처럼 확정한다. 확정 근거는 `WORK_LOG.md`.

## 1. 대상 구성 (변경 없음)
IDE_073 Q-OPT4 launch (`evidence/environment.json: reference_launch_cmd`) 그대로. env `KT_CALLBACK_FREE=1 KT_CF_SKIP_EMPTY_IMM=1 KT_GPU_EXPERTS_PER_LAYER=/models/kt/ide070/layer_budget_5952.json KT_AVX_RB=0`(=ON). 층별 예산은 부팅 시 `eval/ide070/patch_per_layer_experts.sh` 로 sglang `kt_ep_wrapper.py` 에 적용·종료 후 되돌림 (코드 지도 §12-1 의 "없음" 은 되돌린 상태를 본 것).

## 2. 코드 확인으로 확정된 실행 구조 (측정 설계의 전제)
- deferred=8, top-k 8 → `protected_k=0` → immediate 작업 없음(`KT_CF_SKIP_EMPTY_IMM` 로 생략), **모든 cold expert 는 deferred**. 층 L 의 deferred 작업(입력 = 층 L hidden)은 `output_cpu[(L+1)%2]` 에 쓰고 층 L+1 의 `sync_forward` 가 이를 HtoD 해 더한다. 즉 층 L 의 결합에 들어가는 cold 기여는 층 L−1 의 CPU 작업 결과이며, GPU 는 층 L 에서 `done[slot_L]` 을 기다린다. 단일 FIFO task queue 이므로 `done[slot_L]` 은 deferred(L−1) 완료 직후 set 된다.
- 결합 = hot(GPU, rank 0~3) + cold(rank 0 만) 의 elementwise add (`kt_ep_wrapper.apply:429`) → MoE 뒤 TP all-reduce. shared expert 없음 → other 입력은 잔차(attention 뒤 all-reduce 결과)뿐이며 결합 시점엔 항상 준비되어 있음 (`t_other_ready ≤ t_hot_ready`).
- 디코드는 CUDA graph replay: 층별 Python 실행 없음. GPU-side 층 이벤트는 kineto 트레이스의 graph id/graph node id 와 memcpy 패턴(층당 DtoH 4·HtoD 1·fused_moe 2)으로 분절한다 (검증: graph 2 첫 replay HtoD 62/DtoH 248/fused_moe 124). 이 층 식별은 **패턴 휴리스틱** 으로 표기한다.
- GPU 가 done 을 기다리는 구간은 같은 stream 에서 hot MoE 마지막 커널(`moe_sum_reduce_kernel`) 종료 → cold 출력 HtoD 시작 사이의 유휴로 나타난다. bs=2 graph 에서 이 값 p50 10.8 µs 가 memop+memcpy 발사 지연의 하한이다.

## 3. 이벤트와 식별자 (확정)
CPU-side (`cpu_backend/kt_evt.h`, `KT_EVT=<csv>` 로만 활성, CLOCK_REALTIME ns): slot, epoch(슬롯별 go 관측 횟수), qlen, n_cold_ids, imm/def 존재·skip, t_go(poller 가 go 플래그 관측), t_imm_enq, t_done_enq, t_def_enq, t_def_start/t_def_end(task worker 에서 deferred 작업 전후 stamp), t_numa{0,1}_start/end(서브풀 fork-join), t_done_set(done-setter 실행 = GPU 가 볼 수 있는 완료 게시).
slot→(layer_idx, rows, capturing) 맵: `KT_EVT.map` (arm/rearm 시 Python 이 기록).
GPU-side (kineto, host CLOCK_REALTIME 정렬): 층별 t_dtoh_first_start/last_end, t_hot_first_start, t_prev_op_end(=hot 마지막 커널 종료), t_htod_start/end, t_combine_start/end, all-reduce. step 은 `step[DECODE bs=..]`/`step[EXTEND ..]` annotation 으로 실제 scheduler phase 를 쓴다.
연결: (layer, replay k of graph g) ↔ (slot(layer,g), epoch k). 검증: `t_go ≥ t_dtoh_last_end` 가 전 표본에서 성립해야 시계 정렬 유효 (아니면 UNMEASURED_CLOCK_ALIGNMENT).
정의: `t_hot_ready = t_prev_op_end`, `t_cold_ready = max(t_done_set, GPU 가 wait 노드에 도달한 시각) → 관측 가능한 대리값은 t_htod_start` (HtoD 는 wait 직후 첫 노드). `cold_ready_lateness = max(0, t_done_set − t_hot_ready)` (CPU done 이 hot 종료보다 늦은 만큼), `exposed_wait = t_htod_start − t_hot_ready − floor(≈11 µs)`.

## 4. 세션 배분 (24 세션·부팅 12 회 상한 내)
| 부팅 | 모드 | 세션 | 계측 |
|---|---|---|---|
| B1 | OFF | B(MAIN_SHORT C64 n256)×3 + P0(PROBE128)×3, 순서 B,P0,B,P0,B,P0 | 새 계측 전부 OFF (KT_EVT unset, profiler 없음). 공통 경량 모니터(nvidia-smi/ps 시계열)만 |
| B2 | P1 | PROBE128×3 | KT_EVT + torch profiler CPU+GPU (`/start_profile activities ["CPU","GPU"], start_step=1` → 다음 forward 부터) |
| B3 | P2 | PROBE128×3 | KT_EVT + KT_TQ_TIMING + KT_PHASE_PROF + perf stat(-p 스케줄러 4 프로세스) + pcm-memory, torch profiler 없음 |
| 조건부 | F1/F2/G/R | ≤ 12 | 지시서 §9.2. G 는 M8 게이트 통과 시 |
KT_EVT 는 부팅 시 getenv 로 고정되므로 OFF/ON 을 한 부팅에서 전환할 수 없다 → 부팅 간 비교임을 한계로 기록. 새 kt_kernel 바이너리(이벤트 코드 컴파일 포함, env 미설정 시 비활성)와 decode printf 게이트 변경은 OFF/ON 양쪽에 공통 (CONFIG_DIFF.md).

## 5. 시간창
프로파일러: `start_step=1` 로 arm → 첫 forward 에서 시작, bench 종료·drain 확인 후 `/stop_profile`. 요청 창 = [서버 로그의 첫 요청 수신, bench 마지막 응답 수신]. 수집기별 actual_start/end·교집합·coverage 를 `profiles/observer_windows.csv` 에 저장. 앞 준비 구간을 상수로 빼지 않는다.

## 6. 유효성 상태명 (신규 정의)
INVALID_NO_GPU_ACTIVITY, INVALID_WINDOW, UNMEASURED_CLOCK_ALIGNMENT, PARTIAL_DEPENDENCY_MEASUREMENT, INVALID_AGGREGATION, INVALID_ARTIFACT, HEURISTIC_LAYER_ID(memcpy/graph 패턴), BLOCKED_NORMAL_OUTPUT(GLM).
