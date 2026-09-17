# READINESS — IDE_075 (2026-09-17 16:00 KST). 후보별 준비 상태와 허용 주장 범위

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

### NUMA/worker 배치 — **CONDITIONAL_DIAGNOSTIC_ONLY**
tail 10/15.6k, 8건 서브풀 시차 >1 ms (down 단계 5 ms). 원인 미분리 (off-CPU/원격 접근 미측정). 예산 소진으로 FOCUS 미실행.

### TP 통신 — **미측정 (조건부)**. prefill/chunk 정책 — **BLOCKED (EXTEND 분절 없음)**. GLM — **GLM_NORMAL_OUTPUT_BLOCKED**.

## 전환 게이트 (§17.3)
- [x] 고정 구성 보존 (CONFIG_DIFF.md) · [x] PCM/창/지표/분모 정정 · [x] producer→consumer 연결 (VALIDATED_HEURISTIC_MAPPING, 불일치 0) · [x] 지연의 큐/서비스/전송 귀속 · [x] 실제 분기(AVX)·작업 크기 확인 (expert 단위 rows 는 미수집) · [x] 기록 전용 task 0·동기 순서 불변 · [x] observer 영향 공개 (부팅 간 한계) · [x] 정상성: smoke 4/4 완료, lifecycle 미기록 ≤0.02 %, 새 오류 없음 · [x] 근거↔변경 표 (OPTIMIZATION_HANDOFF.md)
종합: **READY_WITH_LIMITED_SCOPE** (후보 A·B). 한계: 2 세션·1 부팅, 일정 clock offset 미검출, DRAM 포화·NUMA 원인·TP·prefill 미확정.
