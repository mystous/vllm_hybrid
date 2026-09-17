# READINESS — IDE_075 (2026-09-17 16:00 KST 기본 단계; 19:10 확장 단계 추가). 후보별 준비 상태와 허용 주장 범위

근거 세션: S3_CORR·S4_CORR (CORE+profiler, 같은 부팅 CORE_153555), S5_RESOURCE, OFF S1/S6. 원값: `eval/results/IDE_075_20260917/qwen/OPT4/CORE_153555/*/v2/`. 지표 정의 `METRIC_DEFINITIONS_V2.yaml`. 아래 수치는 bs=63 디코드 graph, cohort cold_present_nonempty (S3 n=15,598 / S4 n=15,596) 의 p50 이며 반복은 2 세션(같은 부팅)뿐이다.

## 공통 관측 (두 세션 일치)
- Cold 게시 − Hot 종료(cold_pub_delta) 321.0 / 315.0 µs, 늦은 비율 0.900 / 0.899 (임계 5 µs: 0.897 / 0.896). Hot 종료→HtoD 시작 간격 334.3 / 328.0 µs 안에 다른 GPU 작업 0 (간격 전체가 TP0 idle).
- 생산 작업의 enqueue→시작 663.5 / 649.8 µs 중 선행 task 실행 점유 662.8 / 648.8 (거의 전부 직전 층 deferred 작업; done-setter ≈0), 미분리 0.75 / 0.75 (p95 4.2), dequeue→exec 0.0. **FIFO 전달·dispatch 지연은 관측되지 않음.**
- 생산 작업 서비스 span 911.5 / 902.3 µs (numa0 ≈ numa1, 완료 시차 p50 24~41). 층별 p50 490~1,377 µs. 서비스 span 은 서브풀 unique cold expert 수와 r=0.925 (≈65 µs/expert + 166), assignments 수와 r=0.79 (S3 층별 표 `producer_layer_costs.csv`).
- 단계(numa0, 합 기준): up_gate 55 %, down 34 %, 나머지(prepare·cpy_input·q_input·act·q_down·weight) 11 %. **모든 expert 가 AVX `vec_mul` 경로** (n_amx=0; qlen 64 ≤ KT_AMX_MIN_QLEN 80, expert 당 rows 1~5 < KT_AMX_MIN_ROWS 기본).
- 게시→HtoD 시작 13~14 µs, HtoD 32 µs, HtoD→combine 4~9 µs. 게시·전송·결합 경로는 지연 구성에서 작음.
- tail: 서비스 >2.5 ms 10/15,598 (S3), 그중 8건은 NUMA 완료 시차 >1 ms (한 서브풀의 down 단계가 5 ms 급). 원인(스케줄링/페이지/원격 접근) 미분리.
- 부하 창 자원(S5, 20.9 s): 스케줄러 4 프로세스 task-clock 102.7 CPU 상당, IPC 1.30, host DRAM read 232.9 GB/s · write 28.3 GB/s (완전 포함 표본 19/21). 포화 판정 없음 (비교 기준 부재).
- 계측 간섭: CORE 1.006, CORR 0.993 / 0.997 (OFF 평균 대비 처리량), p95 비율 0.955~1.021 → DESCRIPTIVE_LOW_DISTORTION. RESOURCE 1.11 → DIAGNOSTIC_ONLY (부팅·시간 변동과 분리 불가). OFF 자체 변동 0.9 %.
- 검증: 매핑 불일치 0, clock 순서 위반 0 (5 µs 이내 CLOCK_INDETERMINATE 0.7~0.8 %), 항등식 residual 0, 기록 전용 task 0, fwd 미기록 ≤ 5/24,490.

## 후보별 상태

### 후보 A: Cold expert 작업의 서비스 시간 (C64 디코드, up_gate/down, AVX vec_mul 경로)
- 상태: **READY_WITH_LIMITED_SCOPE**
- 대상: Qwen OPT4, step[DECODE bs=63], 생산층 전부 (unique cold expert 수 5~18/서브풀), AVX vec_mul 분기, rows 1~5.
- 직접 관측: 서비스 p50 902~912 µs (층별 490~1,377), 단계 up_gate 55 %/down 34 %, 게시 지연 p50 315~321 µs, 늦은 비율 0.90; S3/S4 두 세션, 각 15.6k 표본.
- 실제 지연 연결: 생산 작업 종료 → done 게시 +0.5 µs → GPU HtoD +13 µs → combine +4~9 µs. 게시 지연의 구성 = (직전 층 작업 대기 650~663) + (서비스 902~912) − (예산 1,248~1,270).
- 큐/계산 분리: 선행 점유 649~663 (직전 deferred), 직접 서비스 902~912, 미분리 0.75.
- 계측 제한: 같은 부팅 2 세션, OFF 는 별도 부팅 2회, clock 일정 offset 미검출, expert 단위 rows 미수집(작업 단위 합계만).
- 변경할 단일 변수(다음 단계 후보, 미실행): expert 당 GEMM 커널 경로 선택 규칙 (예: 작은 m 에 대한 vec_mul 대체·KT_AMX_MIN_ROWS) 또는 vec_mul 자체의 up_gate/down 구현. 과거 IDE_030 에서 `KT_AMX_MIN_ROWS=3` 은 qlen>80 체제에서만 +8 %, C32/C64 변화 없음 → 같은 변수의 재시험은 그 기록과 대조해야 함.
- 보존 조건: top-k 8, deferred 8, hotmap·예산, 정밀도 AMXINT4, HBM/KV.
- 다음 검증: 새 MAIN_SHORT 3회 대조 + 후보 3회, PROBE128 CORR 로 같은 서비스 span·게시 지연 구간이 줄었는지, C1/LONG 회귀, smoke 동일성.
- 금지할 주장: memory-bound/DRAM 포화 확정, 서비스 감소량 = tok/s 개선율.

### 후보 B: 관측 비용 기반 Hot 예산 재배치 (층별 unique cold expert 수가 큰 층)
- 상태: **READY_WITH_LIMITED_SCOPE**
- 대상: 생산층 10, 13, 20, 21, 42, 53, 55 등 unique cold ≥ 14/서브풀·서비스 ≥ 1.1 ms 인 층 (S3 표), 반대로 31, 46, 32 등 cold 가 적어 게시가 hot 보다 빠른 층.
- 직접 관측: 층별 서비스·unique experts·assignments·게시 지연·늦은 비율 (producer_layer_costs.csv, 2 세션).
- 제한: 관측 비용은 한계 이득이 아님 (expert 하나 이동의 효과 미확정); HBM 여유 캡처 시 4.08 GB (IDE_074 capture 로그) → 슬롯 합 5,952 고정 재분배만 가능; hotmap 은 logical→physical 재배치와 함께 바뀌어야 함.
- 변경할 단일 변수: layer_budget_5952 의 층간 재분배 (합 고정). 다음 검증은 held-out PROBE128 CORR 로 층별 게시 지연 변화.
- 금지할 주장: 재배치 후 층별 지연이 비례해 준다는 예측.

### FIFO / dispatch / 신호 경로 개선 — **BLOCKED (근거 없음)**
미분리 gap p50 0.75 µs, dequeue→exec 0.0, enqueue bracket 0.6 µs, 게시→HtoD 13 µs. 대기는 직전 deferred 계산 점유가 전부.

### 입력/결과 복사 묶음화 — **선택 안 함**
HtoD 32 µs, DtoH 4건은 hot 이전에 완료 (go − dtoh_end p50 5 µs). 노출 구간에서 복사 비중 작음.

### GPU Hot 커널 — **선택 안 함**
Hot 이 늦은 표본 10 %; hot 커널 합 p50 324 µs 는 예산 안에서 완료.

### NUMA/worker 배치 — **BLOCKED (스케줄링 원인 기각; 잔여 원인 미분리)**
tail 10/15.6k (S3), 31/23,808 (S28), 8건 서브풀 시차 >1 ms (down 단계 5 ms). 확장 FOCUS2 (S28, 시스템 전체 sched_switch, 손실 0): tail 창 안 워커 off-CPU p50 0 · p90 0 · 최대 284 µs, >1 ms 0 건 → **워커 선점/off-CPU 는 tail 의 원인이 아님**. 워커 선점은 초당 1.26 회/스레드 (kworker·migration), 길이 수십~수백 µs. 남는 후보(원격 NUMA 접근·페이지·캐시)는 미측정 → 변경 근거 없음.

### TP 통신 — **CONDITIONAL_DIAGNOSTIC_ONLY** (확장 단계에서 위치 귀속: post-MoE all-reduce 에서 rank 0 이 항상 마지막 도착, rank 1~3 커널 437 µs vs rank 0 12.5 µs, 도착 시차 p50 577 µs; post-attention 은 4 µs. 이는 rank 0 의 CPU 대기(cold 게시)가 all-reduce 도착 시차로 나타나는 것이며 통신 자체의 지연 근거는 아님). prefill/chunk 정책 — **CONDITIONAL_DIAGNOSTIC_ONLY** (EXTEND 층 연결 S3/S22 682/682, S13 1,178, S27 1,240: 스텝 토큰 ≤2.6k 와 7~8k 청크에서는 CPU 게시가 hot 보다 1.3~3.1 ms 빠름(늦음 0~1 %), 그러나 **4.0~5.1k 토큰 스텝에서는 CPU 게시가 0.1~1.1 ms 늦음** (늦음 비율 0.49~0.77; S27 6 그룹·S13 5 그룹·S3 5,065 tok 0.28), service 6.6~7.0 ms vs 예산 8.2~9.9 ms, 전 expert AMX 경로. 원인(스텝 구성·큐 대기·AMX 고정비) 미분리, 스텝별 n=61). GLM — **정상 출력 회복 (확장 단계 G00, TSK_060)**; 성능 후보 판단은 별도 (아래 확장 절).

## 확장 단계 추가 관측 (2026-09-17 16:00~, 사용자 지시 "완료 못 한 것 전부"; 원장 phase=extended)

근거 세션 (대표값은 **vllm 래퍼(vllmw) 클라이언트 + 기록기 v3** 인 CORE4_174930/S22·S23): bs=63 cold_present_nonempty p50 — cold_pub_delta 392.0 / 328.3 µs, enqueue→start 670.5 / 667.5 (선행 실행 669.8 / 666.0, 미분리 0.75 / 1.25), service 921.8 / 915.8, pre_h2d gap 337.3 / 341.3 (창 경계 zip 어긋남 그룹을 순서 검사로 재동기화한 최종값; S22 의 pub_delta 는 재동기화 전 346 → 392). **CORE_153555 S3/S4 (vllm bench, v2) 의 321/315 · 663/650 · 911/902 · 334/328 과 일치** → 공통 관측·후보 A·B 의 수치는 클라이언트 래퍼·기록기 v3·부팅에 걸쳐 재현됨 (부팅 3개, 세션 4개).

- 클라이언트 동등성: vllmw 는 vllm bench 와 0.3 % 이내 (OFF_Y 705.1/703.1/705.7 tok/s). probe 클라이언트(aiohttp) 는 핀 여부와 무관하게 비동등 (410~510 tok/s, TPOT 1.5~1.8 배) 이고 **CPU 서비스 시간 자체를 늘림** (S9/S10/S17/S18: pub_delta 848~972, service 1,427~1,556 µs). probe 로 얻은 CORE2/CORE3 값은 '클라이언트 간섭 하 관측' 으로만 인용.
- C1 (bs=1, S26 vllmw / S12 probe): cold_pub_delta −62.8 / −59.3 µs (CPU 게시가 GPU 소비보다 먼저), 늦은 비율 0.018, enq→start 3.75 µs, service 134 / 149 µs. **bs=1 에서는 cold 경로가 GPU 를 기다리게 하지 않음** → 후보 A 의 대상은 bs 가 큰 디코드(수십 이상) 로 한정.
- LONG 디코드 (S27 vllmw, bs=8 그래프 26, n 23,693): cold_pub_delta −180.8 µs (CPU 먼저), service 245.8 µs, enq→start 3.75 µs. S13(probe) 은 +23.8 / 472.8 µs → probe 간섭. **bs=8 에서도 cold 경로는 GPU 를 기다리게 하지 않음.**
- LONG prefill (S27/S13 EXTEND 스텝): 위 prefill/chunk 항목 — 4~5k 토큰 스텝에서 CPU 늦음 0.1~1.1 ms.
- TID 역할 (RESOURCE/FOCUS, /proc stat 1 s): KT NUMA worker 96 스레드 94.6~94.9 CPU 상당(busy-poll 포함), 스케줄러 본체/자식 60 스레드 3.8~3.9, poller 1.0, task worker 0.9, **기록기 flusher 0.0** (관측 비용 무시 가능), python/tokenizer 0.1.
- expert 단위 rows 표본 (v3, slot 별 1/16, CORE2/CORE3 각 7,948 decode 표본): 서브풀 unique expert p50 17~18, **expert 별 rows 1~2 인 비율 0.677** → 후보 A 의 "작은 m 의 vec_mul" 대상이 expert 의 2/3 임을 표본으로 확인.
- tail off-CPU (FOCUS S14/S19/S25, perf -p): 워커 switch-out 은 워커당 초당 0.9~1.4 회, 상태 R(선점) 87~91 % / S 9~13 %, tail 창 안 switch-out 78~663 건 — 그러나 `perf record -p` 는 switch-in 을 기록하지 않아 **off-CPU 길이 NOT_RECOVERABLE**; perf 손실 6~41 %. 시스템 전체 기록(FOCUS2 S28, `-a -k CLOCK_MONOTONIC`) 결과는 FULL_REPORT §8b 갱신분 참조. NUMA/worker 배치 후보 상태는 그 결과에 따라 갱신.
- 기록기 v3 vs v2: fwd 레코드 누락 0 (v2 는 SPSC 경쟁으로 일부 누락), 값 불변 (위 대표값 일치).
- 계측 간섭 (CORE4, OFF vllm 평균 대비): S22 0.963 (DIAGNOSTIC_ONLY 경계), S23 0.973, S24 0.978 → 3 % 안팎; p95 비율 1.00~1.04.

### GLM-4.7-FP8 BASIC4 — **정상 출력 회복, 성능 후보 없음 (관측만)**
- 근본 원인 2건: (1) kt-kernel `moe_base.hpp` if-constexpr dangling-else (IDE_046-b/051 도입) 로 FP8/BF16 CPU 경로의 입력 gather·양자화가 컴파일에서 제거 → 출력 0 (커널 테스트 100 % 오차 → 수정 후 0.5829 %). (2) `routed_scaling_factor` 2.5 가 GPU 기여에만 적용 + decode 이중 적용 (`glm_rsf_patch.py`). D6 게이트 4/4 (rsf 수정 포함). **전 GPU(IDE_073 G-GPU8) greedy 텍스트와 대조: D6 는 4문항 중 3문항 완전 일치, 1문항 129자 공통 후 분기; rsf 미수정(D7) 은 4/4 정답이나 첫 토큰부터 분기** → 두 수정 모두 필요 (G00_code_review §8-5).
- 처리량 (D6 G_OFF, 참고): 42.8 tok/s, TPOT p50 1,086 ms (C=64, PROBE128) — Qwen OPT4 의 1/16. FP8 per-channel CPU 경로의 비용 구조는 Qwen 과 다르므로 후보 A·B 를 GLM 에 그대로 적용할 근거 없음. D8(G_OFF2/G_CORR2/G_OFF3, glm47 토크나이저) 44.0~44.2 tok/s, D9(callback-free, G_CORR3/G_OFF4) 44.0/43.9 → callback-free 전달 경로의 처리량 효과 없음. GLM CPU↔GPU 의존성 값은 **NOT_ANALYZED** (GPU 층 분절 휴리스틱이 Qwen 그래프 전용; 데이터 보존, WORK_LOG 20:25). callback-free 경로는 GLM greedy 시퀀스를 바꿈 (전 GPU 기준 일치 1/4 vs 3/4; 정답 유지, 분포 비교 미실시).

## 전환 게이트 (§17.3)
- [x] 고정 구성 보존 (CONFIG_DIFF.md) · [x] PCM/창/지표/분모 정정 · [x] producer→consumer 연결 (VALIDATED_HEURISTIC_MAPPING, 불일치 0) · [x] 지연의 큐/서비스/전송 귀속 · [x] 실제 분기(AVX)·작업 크기 확인 (expert 단위 rows 는 미수집) · [x] 기록 전용 task 0·동기 순서 불변 · [x] observer 영향 공개 (부팅 간 한계) · [x] 정상성: smoke 4/4 완료, lifecycle 미기록 ≤0.02 %, 새 오류 없음 · [x] 근거↔변경 표 (OPTIMIZATION_HANDOFF.md)
종합: **READY_WITH_LIMITED_SCOPE** (후보 A·B). 확장 단계 후 한계 갱신: 대표값은 3 부팅·4 세션(vllm/vllmw) 에서 재현; 일정 clock offset 은 anchor 실측으로 ≤ +3.5 µs; DRAM 포화 판정은 여전히 없음; NUMA tail 의 스케줄링 원인은 FOCUS2 로 기각(잔여 원인 미분리); TP 는 CPU 대기의 반영으로 귀속; prefill 은 ≤2.6k·8k 청크에서 CPU 가 빠르나 4~5k 스텝에서는 0.1~1.1 ms 늦음 (조건부). 후보 A 의 대상은 bs 큰 디코드로 한정 (bs=1 은 CPU 가 먼저).
