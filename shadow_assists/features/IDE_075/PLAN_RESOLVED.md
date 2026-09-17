# PLAN_RESOLVED — IDE_075 (2026-09-17, 지시서 `PLAN.md` SHA 8a3225f4…3971 의 실행 확정본)

## 1. 잔여 예산 (A00, IDE_074 원장 `eval/results/IDE_074_20260917/state/execution_events.jsonl` 재계산)
세션 17/24 · 부팅 7/12 · 품질 문항 4/80 → 잔여 **세션 7 · 부팅 5 · 품질 76**. IDE_075 원장은 별도 파일이지만 예산 계수는 IDE_074 잔여를 상한으로 한다 (`budget_reconciliation.json`). smoke(부팅당 4 요청)·warmup(C16·32 요청)은 IDE_074 와 같이 세션·품질 예산에 계수하지 않고 `smoke_request_attempts`/`warmup_request_attempts` 로 별도 기록.

## 2. 고정 구성
IDE_074 와 동일 (launch argv, env KT_CALLBACK_FREE=1 KT_CF_SKIP_EMPTY_IMM=1 KT_GPU_EXPERTS_PER_LAYER=layer_budget_5952.json KT_AVX_RB=0, 층별 예산 부팅 시 패치, non-KT pinning). 바이너리: **v2 기록기 .so (SHA 는 evidence/environment.json)** 를 OFF/ON 공통 사용. IDE_074 의 v1 .so(adf49e48…)와 원본(e29357f7…)은 `/sgl-workspace/ide074_backup/` 보존.

## 3. 기존 자료 정정 (A01/A02) — 확정
- PCM: `tsparse.py` (T01~T07, W01~W06 PASS). 시간대 Asia/Seoul (수집 호스트 로컬), 표본 구간 [ts−1 s, ts) `ASSUMED_END_TIMESTAMP` (근거: pcm-memory 는 delay 마다 출력; 첫 표본이 collector 시작 +2.7 s 로 초기화 포함 → 첫 표본 경계 불확실 기록). 창 = forward_envelope(kt_evt 첫 go ~ 마지막 done/def_end). legacy 값은 `legacy_collector_window_mean` 으로 병기 (`WINDOW_RECONSTRUCTION.md`).
- 시간창: `legacy_windows.csv` 의 7 창. client_request_window 는 요청별 timestamp 부재로 ESTIMATED_WINDOW. perf 누적값은 NOT_RECOVERABLE.
- v2 지표: `dependency_v2.py` (cold_pub_delta / cold_pub_lateness / gpu_pre_h2d_gap / h2d_duration / gpu_post_h2d_gap / cold_gpu_ready_lateness / combine_schedule_gap / go_to_enqueue / enqueue_to_start / deferred_service_span / predecessor_exec_union / queue_gap_unattributed / producer_end_to_pub / overlap_budget / gpu_idle_inside_gap). legacy `cold_ready_lateness_us` = CPU 게시 지연 (`legacy_cold_ready_lateness_us`), legacy `exposed_wait_us` = `gpu_pre_h2d_gap_us` (floor 미적용). `METRIC_DEFINITIONS_V2.yaml`.
- cohort: no_cold_consumed(L=0) / cold_path_empty(prev n_cold_ids=0 또는 skip) / cold_present_nonempty / unknown_path_or_mapping. late 비율 분모 = cold_present_nonempty 만.
- clock: 0/1/2/5 µs 민감도 전부 보고, 5 µs 이내 음수는 CLOCK_INDETERMINATE 집단(제외가 아니라 별도 집단). IDE_074 P1_r3 의 1,659 행은 재분류.
- 매핑: (slot,epoch)↔(layer,replay) 는 계수 일치 + map_at_go(slot 맵 effective_from) + 순서검사 → `VALIDATED_HEURISTIC_MAPPING`. 계수 불일치 (graph,layer) 는 전체 unresolved (zip 이동 금지).

## 4. 기록기 v2 (A03) — 확정
`eval/ide075/kt_evt_patch_v2.py`. 기록 전용 task 0개. TaskQueue: seq·kind·pending·running_seq·enqueue bracket·dequeue·exec start/end (링 1<<20). poller: go·imm/done/def enqueue bracket·task_seq·done store bracket(기존 done-setter 안). 작업 본체: `TP_MOE_Common::forward` 진입/반환·NUMA start/end·merge (poller→worker SPSC, 현재 task_seq 와 대조). 단계 µs·activated_expert·max/sum rows·AMX/AVX 분기 수. 플러셔 200 ms, 완료 순 기록, 5 s 미완료는 partial 기록, 누락/SPSC full/stale 계수. 합성 시험 `tests/test_recorder.cpp`.
관측 비용: 레코드 1개/패킷 (kt_evt_now ≈ 수십 ns, 아래 시험값), flusher 스레드 1개 (affinity 미고정 → observer 비용으로 기록).

## 5. 세션 배분 (A11) — 확정, 잔여 예산 내
| 부팅 | 모드 | 세션 | 계측 |
|---|---|---|---|
| B_OFF_OPEN | OFF | S1 PROBE128 | 공통 경량 모니터만 (v2 바이너리, KT_EVT unset) |
| B_CORE | CORE → CORR → CORR → RESOURCE | S2, S3, S4, S5 (PROBE128) | KT_EVT 상시(CORE). CORR = + torch profiler(start_step=1) 세션 내 on/off. RESOURCE = + perf stat -I 1000 (스케줄러 4 PID, 스레드 상속) + pcm-memory 1 s, profiler 없음 |
| B_OFF_CLOSE | OFF | S6 PROBE128 | S1 과 동일 |
| B_RESERVE | 조건부 | S7 (최대 1) | §14.3 규칙 |
합계 세션 6(+1) ≤ 7, 부팅 3(+1) ≤ 5. 같은 부팅 내 모드 전환은 세션 경계(요청 drain 뒤)에서만, 그래프 재캡처 없음 (profiler·perf·pcm 은 외부/런타임 토글).

## 6. 간섭 기준 (§15.3, 실행 전 등록)
같은 PROBE128 의 OFF(S1, S6 원값 각각·평균) 대비: 처리량 차이 절댓값 ≤ 3 % 및 TTFT/TPOT/E2EL p95 차이 ≤ 5 % 이면 `DESCRIPTIVE_LOW_DISTORTION`; 벗어나면 `DIAGNOSTIC_ONLY`; 정상성/ID/window/clock 검증 실패면 `UNUSABLE_FOR_TARGET`. OFF 2회의 자체 차이가 3 % 를 넘으면 `INCONCLUSIVE_BOOT_VARIANCE`. 단발 CORE/RESOURCE 에 SD/CI 없음.

## 7. 시간창 (A04)
vllm bench serve 클라이언트는 barrier 를 지원하지 않음(프로세스 기동 즉시 전송). 따라서 collector 는 클라이언트 기동 전 arm 하고, 창은 서버측 이벤트로 사후 절단: forward_envelope = kt_evt 첫 go ~ 마지막 done/def_end (CORE 이상), 트레이스 step annotation (CORR). window_events.jsonl 에 client_process_begin/end, collector_arm/actual_first_sample, first_go, last_deferred_end, drain_confirmed, collector_stop 을 ns 로 저장. 요청별 timestamp 는 없음 → client_request_window ESTIMATED.

## 8. GLM (G00)
Qwen 트랙과 분리. 이번 예산 내 실행 없음. 기존 게이트 원자료 재검토만 (`GLM_NORMAL_OUTPUT_BLOCKED`).

## 9. 확장 단계 (사용자 승인 2026-09-17 16:05: "세션 제약 없음, 미완 항목 전부 마무리")
상한 밖 실행은 원장 `phase=extended` 로 별도 계수. 바이너리 v3 (경쟁 조건 수정·스레드 이름·expert 표본). 클라이언트: barrier 클라이언트(`probe_client.py`, vllm bench 와 같은 요청 의미, 요청별 ns timestamp) — OFF_OPEN2/OFF_CLOSE2 에서 vllm bench serve 와 나란히 1회씩 실행해 동등성 기록.
| 부팅 | 세션 | 목적 |
|---|---|---|
| OFF_OPEN2 | S8 OFF(vllm), S8b OFF(probe) | 앞 대조 + 클라이언트 동등성 |
| CORE2 (v3, KT_EVT) | S9/S10 CORR(probe), S11 RESOURCE(+TID CPU 시간), S12 C1 CORR, S13 LONG CORR(EXTEND 분절), S14 FOCUS(perf sched_switch/wakeup, tail off-CPU 귀속) | 잔여 공백 |
| OFF_CLOSE2 | S15 OFF(vllm), S15b OFF(probe) | 뒤 대조 |
| GLM | 별도 (G00 §16.2 재검토 + 진단 부팅) | 정상 출력 원인 |
