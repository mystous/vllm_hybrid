# CPU MoE Hot/Cold 부분 상주 병목 측정 작업 지시서

> 작성일: 2026-09-17 (Asia/Seoul)  
> 문서 상태: **작업 계획 — 구현·계측·서버 실험 미실행**  
> 주 대상: Qwen3-Coder-480B OPT4  
> 조건부 대상: GLM-4.7 native FP8 BASIC4  
> 기준 저장소: `mystous/vllm_hybrid`  
> 기준 커밋: `67d24dd84ed5681e97851c018af01d3a2d917543`

## 1. 작업 목적과 범위

**현재 Qwen OPT4 구성을 유지한 상태에서, CPU Cold 경로·GPU Hot 경로·전송·작업 전달·결합·TP 통신 중 실제 진행을 늦추는 구간을 직접 측정한다.** CPU 계산 시간이나 GPU 커널 시간의 크기만으로 병목을 판정하지 않는다. 동일한 작업의 입력 준비, 실행, 결과 준비, 결과 소비를 연결하는 것이 핵심이다. [H §1·8~10]

실행 구조는 **동일 물리 서버**의 H100 80GB 4장과 CPU 2소켓이다. Hot expert는 GPU HBM에 상주하고 GPU가 계산한다. Cold expert는 Host Memory에 상주하고 CPU가 계산한다. Attention·Router·KV cache는 GPU에 유지한다. 서로 다른 서버 간 오프로딩이나 CPU 가중치를 GPU로 옮겨 계산하는 구조로 바꾸어 해석하지 않는다. [H §0.1·4.3]

이번 작업은 병목 측정에 한정한다. CPU 워커 수·Hot 배치·deferred 수·커널 backend·정밀도·chunk 크기를 바꾸는 성능 최적화, 전문가 전량 CPU 비교군 추가, GLM INT4 전체 변환은 포함하지 않는다. 기존 callback-free·skip-empty·RB·층별 배치는 그대로 둔다. **계측 활성화/비활성화만을 비교하며, 계측 변경과 성능 변경을 같은 셀에 넣지 않는다.**

### 1.1 측정이 끝나면 답해야 할 질문

| 질문 | 필요한 직접 근거 |
|---|---|
| 결합 시점에 Hot과 Cold 중 어느 결과가 더 늦게 준비되는가? | 같은 layer·실행 step·결합 단위의 Hot ready, Cold ready, other ready, combine start |
| Cold가 늦다면 입력 전송, CPU 작업 대기, CPU 계산, 결과 반환 중 어디서 늦어지는가? | DtoH→CPU 제출/시작/완료→done 전달→HtoD 완료의 연결 |
| Hot 또는 다른 GPU 작업이 늦다면 어떤 연산·rank·phase인가? | Attention·Router·Hot MoE·결합·collective의 실제 GPU 구간과 의존 관계 |
| 지연이 특정 layer·expert·NUMA 서브풀·rank·배치 형태에 집중되는가? | layer/step/job/expert ID, expert rows, 실제 scheduler phase, rank/NUMA 정보 |
| 계측 결과를 실제 무계측 실행에도 적용할 수 있는가? | 동일 manifest의 OFF/ON 반복 비교, 계측 창·이벤트 누락·시계 정렬 검증 |

답을 얻지 못한 항목은 미측정 상태와 부족한 이벤트를 남긴다. 측정이 불완전한 상태에서 하나의 병목으로 결론을 강제하지 않는다.

### 1.2 근거와 신규 설계의 구분

이 문서는 첨부 인계서 [H], 결과 검토서 [R], ZIP 내 IDE_073 원 보고서 [O]를 바탕으로 작성했다. 기존 결함은 **첨부 문서에 기록된 검토 결과**이며, 이번 문서 작성 과정에서 원격 저장소 코드나 서버의 대형 trace를 다시 검증한 것은 아니다.

이벤트 필드의 구체화, 합성 단위시험, 유효성 검사표, 측정 전용 세션 배분은 **구현 제안**이다. 현재 코드에 해당 이벤트나 기능이 이미 있다는 뜻이 아니다. 실제 함수·라인·지원되는 수집 방식은 작업 0~3에서 확인하고 기록한다.

## 2. 전체 작업 순서

| 작업 ID | 할 일 | 선행 조건 | 완료 산출물 |
|---|---|---|---|
| M0 | 원자료·실행 환경·Qwen OPT4 고정 구성 확인 | 없음 | `SOURCE_INDEX.md`, `environment.json`, `effective_config.json` |
| M1 | GPU union/gap 집계 수정 및 기존 자료 재집계 | 원 trace 또는 단위시험 입력 확보 | 단위시험, `corrected_metrics.csv`, 수정 전후 차이 |
| M2 | 부하와 profiler의 측정 시간창 정렬 | 실행 하네스 위치 확인 | 시작 barrier, `observer_windows.csv`, 수집기 종료 상태 |
| M3 | CPU↔GPU 작업 상관관계와 완료 이벤트 추가 | 실제 실행 코드·graph 경로 확인 | `probe_map.md`, 이벤트 스키마, correlation 검증 |
| M4 | CPU·GPU·전송·NUMA·통신 구간별 수집 구현 | M2·M3 | 단계 이벤트, 작업·전송·자원 시계열 |
| M5 | 계측 자체의 정상성 검증 | M1~M4 | smoke 결과, clock·누락·의존 관계 검사 |
| M6 | 무계측 기준 및 계측 OFF/ON 반복 측정 | M5 통과 | 반복별 성능, 계측 간섭, 유효성 표 |
| M7 | layer·phase·rank별 지연과 의존 관계 집계 | 유효한 M6 자료 | `dependency_metrics.csv`, `BOTTLENECK_EVIDENCE.md` |
| M8 | GLM native FP8 조건부 측정 | 정상 출력·종료 확인 | 별도 GLM 결과 또는 차단 사유 |
| M9 | 원시 자료·실패·미측정 항목까지 보존 | 성공 여부와 무관 | 최종 MD, 원자료 색인·해시·전달 기록 |

**M1~M5가 통과하기 전에 본 측정을 시작하지 않는다.** 기존 trace에 없는 이벤트는 재집계로 복원하지 않는다. 원 trace 접근이 막히면 M1의 기존 자료 재집계만 차단 처리하고, 합성 단위시험과 독립적인 계측 구현은 계속할 수 있다.

## 3. M0 — 기준 환경과 입력을 고정한다

### 3.1 먼저 확인할 원자료와 코드

| 구분 | 기존 경로 | 확인할 내용 |
|---|---|---|
| 결과·상태 | `shadow_assists/features/IDE_073/{FULL_REPORT.md,COMPLETION_STATUS.md,CONFIG_DIFF.md,BASELINE_PROVENANCE.md,PLAN_RESOLVED.md}` | 최초 실패와 최종 성공 시도 구분, 실제 실행 설정 |
| 측정기 | `eval/ide073/probes.py` | profiler 시작/종료, 대상 PID/TID, 프로브 모드 |
| 분석기 | `eval/ide073/analyze_probes.py` | `analyze_trace()`, busy/gap·phase 분류 |
| 실행 하네스 | `eval/ide073/run_campaign.py`, `eval/ide071/{common.py,run_cell.py}` | warmup·cache flush·벤치 실행·실패 처리 |
| 입력 정의 | `eval/ide071/configs/workloads.json` | `M073_QWEN_*`, `M073_GLM_*` |
| 실제 원장 | `eval/results/IDE_073_20260917/state/` | `RUN_MANIFEST.json`, `RUN_STATE.json`, `execution_events.jsonl` |
| 유효 GPU 프로브 | `eval/results/IDE_073_20260917/qwen-PROBES/bootA_retry2/D2/` | `metrics.json`과 manifest가 가리키는 TP0~3 trace |
| CPU 단계 | `eval/results/IDE_073_20260917/qwen-PROBES/bootB/D1/cpu_jobs.csv` | 원 타이머 필드·표본 방식 |
| 무계측·CPU 프로브 | `eval/results/IDE_073_20260917/qwen-PROBES/bootA_retry/` | D0·D3의 실제 부하 및 수집기 시각 |
| 실제 적용 증빙 | `shadow_assists/features/IDE_073/config/` | effective config, feature paths, affinity |

경로의 존재를 먼저 확인한다. 대형 trace와 `perf.data`는 서버 보존 자료이며 첨부 ZIP에 전체가 들어 있지 않다. `state/`는 feature 디렉터리가 아니라 result 디렉터리를 기준으로 확인한다. [H §2]

- [ ] 새 campaign ID·boot ID·attempt ID와 고유 작업 브랜치 또는 worktree를 만든다.
- [ ] `git status`, 현재 HEAD, 기준 커밋, 실제 Python import 경로, 로드한 KT 확장 `.so` SHA256, 로컬 patch diff, 컨테이너 image digest를 저장한다.
- [ ] 저장소 checkout과 `/sgl-workspace/sglang`의 실제 실행 파일이 별개임을 확인한다. 기존 사용자 변경을 reset/clean/stash로 지우지 않는다.
- [ ] GPU UUID·TP rank·PCIe/NUMA 매핑, CPU 물리코어/SMT, TID affinity, CPU·GPU clock/power 정책, 메모리·디스크 여유를 읽어 저장한다.
- [ ] 모델 다운로드·변환·타 벤치·다른 heavy profiler와 본 측정을 겹치지 않게 한다. 타인 작업을 종료하지 말고 자원이 겹치면 충돌 상태를 기록한다.
- [ ] 실제 환경이 원 기록과 다르면 `CONFIG_DIFF.md`에 차이를 남긴다. 이전 값으로 현재 환경을 채우지 않는다.

### 3.2 Qwen OPT4 고정값

다음은 인계서의 복원 기준이다. 요청 설정과 실제 적용값을 둘 다 저장한다. [H §3~5·13.2]

| 항목 | 고정 기준 |
|---|---|
| GPU 모델 | `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8` |
| revision | `003f183a92fbe5b9a8325aaa8b2ae797c91dd90f` |
| CPU weight / backend | `/models/kt/qwen3-480b-int4` / `AMXINT4` |
| GPU / TP | GPU 0~3 / TP4 |
| 모델 구성 | 62층, routed expert 160/층, 논리 top-k8 유지 |
| Hot 선정 | `/models/kt/ide070/hotmap_v2.json` |
| 층별 예산 | `/models/kt/ide070/layer_budget_5952.json`, 62층 합계 5,952 |
| CPU 배치 | 96 workers / NUMA pool 2 / non-KT pinning 유지 |
| 작업 전달 | callback-free ON, skip-empty ON, deferred8, RB ON |
| KV / graph | KV dtype auto, capacity 40,960, decode graph 최대 batch 64, prefill graph disabled |
| 실행 한도 | context 32,768 / max-running 64 / chunked-prefill 8,192 |
| 기타 | memory fraction 0.95 / attention triton / 기존 dynamic dispatch 설정 유지 |

```bash
# 기존 OPT4의 설정값. 계측 ON/OFF에서도 이 값은 변경하지 않는다.
KT_CALLBACK_FREE=1
KT_CF_SKIP_EMPTY_IMM=1
KT_GPU_EXPERTS_PER_LAYER=/models/kt/ide070/layer_budget_5952.json
KT_AVX_RB=0
```

**이 빌드에서 `KT_AVX_RB=0`은 ON이다.** 계측 OFF를 만들기 위해 위 최적화 환경변수를 unset하지 않는다. OFF는 새 계측 기능을 끈다는 뜻이다. [H §0.1·4.2]

- [ ] Hotmap·layer budget 전체 SHA256, 62개 실제 layer별 값, logical↔physical expert map과 CPU/GPU mask를 저장한다.
- [ ] CF·skip·RB의 실제 분기 및 CPU96·NUMA2·non-KT affinity 적용을 확인한다.
- [ ] graph capture 여부와 실제 replay 여부를 구분해 기록한다.
- [ ] 합계 슬롯 수 외에 실제 HBM·Host RSS도 저장한다. 슬롯 수를 바이트 사용량과 동일시하지 않는다.

### 3.3 워크로드와 실행 프로토콜

| 용도 | 워크로드 | 동시성 | 요청 수 | 출력 토큰 | seed |
|---|---|---:|---:|---:|---:|
| 기준 성능 | `M073_QWEN_MAIN_SHORT` | 64 | 256 | 128 | 20260916 |
| 직접 프로브 | `M073_QWEN_PROBE128` | 64 | 128 | 128 | 20260916 |
| 조건부 저동시성 | `M073_QWEN_LOW_CONCURRENCY` | 1 | 16 | 128 | 20260917 |
| 조건부 긴 입력 | `M073_QWEN_LONGER_PREFILL` | 8 | 32 | 128 | 20260918 |

MAIN은 입력 약512토큰, LONG은 약4,096토큰의 기존 custom JSONL이다. PROBE128은 기존 manifest에서 선택된 요청·순서를 복원한다. 새 generator로 비슷한 입력을 만드는 것은 동일 조건 재현이 아니다. [H §5.1]

- [ ] text/token ID manifest, tokenizer revision, 파일 해시, 요청 순서를 고정한다.
- [ ] 기존 `bench_cmd.sh`에서 `/v1/completions`, raw prompt, request-rate inf, temperature0, top_p1, 성능용 ignore_eos를 복원한다.
- [ ] Qwen MAIN 완료 시 input/output 합계 **131,786/32,768**을 대조한다. PROBE128 합계는 해당 manifest에서 확인하며 MAIN 입력 합계를 단순히 절반으로 나누지 않는다.
- [ ] 부팅당 smoke·warmup은 각1회로 제한하고, 기존 실제 warmup인 C16·32요청·출력128을 유지한다. 변경이 필요하면 모든 대조군에 동일 적용하고 기록한다.
- [ ] 세션 전 engine cache flush와 3초 대기를 동일 적용한다. OS page cache까지 비우는 조작으로 해석하지 않는다.
- [ ] warmup 실패 시 본 측정을 시작하지 않는다. 복구 후 새 boot ID를 부여한다.

IDE_073의 714.95 tok/s는 과거 관측값이다. 새 대조군 대신 성능 분모로 사용하지 않는다. IDE_072의 844.0 tok/s와도 입력·프로토콜을 확인하지 않고 직접 비교하지 않는다. [H §1.2·5.2]

## 4. M1 — 기존 GPU 시간 집계부터 수정한다

### 4.1 현재 결함과 조치

| 첨부에 기록된 결함 | 수정할 내용 | 완료 기준 |
|---|---|---|
| `gpu_busy_us`가 stream별 duration 합계 | 원 합계와 interval union을 분리 | 각 GPU에서 busy+idle=측정창, busy fraction 0~1 |
| 요청 전 준비 구간이 전체 trace에 포함 | 실제 요청 구간·전체 trace 구간을 각각 집계 | 시작/끝의 원 이벤트와 clock domain이 저장됨 |
| inter-MoE 마지막 gap 누락 | 모든 GPU stream 구간을 clip한 뒤 여집합 계산 | 작업 없는 gap·마지막 tail·경계 침범 시험 통과 |
| MoE를 이름 정규식만으로 식별 | 기존 결과는 휴리스틱으로 표시, 신규는 layer/step ID 사용 | 커널 순서만으로 layer 번호를 생성하지 않음 |
| annotation과 내부 커널 중복 가능 | 실제 operation과 annotation을 분리 | annotation을 busy/계산 합계에 더하지 않음 |

근거: [H §9.1~9.2], [R §4.3].

### 4.2 집계 정의

장치 `g`의 측정창을 `W=[start,end)`로 두고, **실제 GPU operation만** W에 clip한 후 집계한다. GPU별로 먼저 계산하며 TP0~3의 구간을 하나로 합쳐 단일 GPU 활용률을 만들지 않는다.

```text
gpu_op_duration_sum_us     = Σ duration(clipped operation)
gpu_busy_union_us          = |union(kernel, memcpy, memset intervals)|
gpu_idle_union_us          = |W| - gpu_busy_union_us
gpu_busy_fraction          = gpu_busy_union_us / |W|
compute_union_us           = |union(kernel intervals)|
memcpy_union_us            = |union(memcpy intervals)|
compute_memcpy_overlap_us  = |compute_union ∩ memcpy_union|
```

`gpu_op_duration_sum_us`는 stream 중첩 때문에 창 길이보다 커질 수 있다. `gpu_busy_union_us`는 창 길이보다 커지면 안 된다. memset 등 별도 작업이 있으면 compute·memcpy의 합만으로 전체 busy를 재구성하지 않는다. GPU 작업이 존재한다는 의미의 busy와 연산 효율·하드웨어 포화는 구분한다.

인접한 두 **검증된 MoE 경계** 사이 `G=[prev_end,next_start)`의 유휴시간은 `|G|-|union(all GPU ops clipped to G)|`로 계산한다. G 이전에 시작해 내부로 걸쳐 들어온 작업도 포함한다. MoE 경계 자체가 휴리스틱이면 결과 이름과 validity에 그 한계를 남긴다. 이 유휴시간을 곧바로 Cold 대기라고 부르지 않는다.

### 4.3 합성 단위시험 — 실험 관측값이 아닌 검증용 입력

아래 시간 단위는 모두 임의의 µs이며, 분석기 테스트용으로 새로 정의한 값이다.

| 시험 | 입력 | 기대 결과 |
|---|---|---|
| 중첩·annotation | W=[0,10), kernels=[1,5),[3,7), copy=[6,9), annotation=[0,10) | op sum=11, compute union=6, copy union=3, overlap=1, busy=8, idle=2 |
| 연속 커널 | W=[0,6), kernels=[1,3),[3,5) | busy=4, 커널 사이 gap=0 |
| 중간 작업 없음 | MoE 사이 G=[2,8), 내부 GPU 작업 없음 | gap=6 |
| 마지막 tail | G=[2,8), 내부 작업=[3,5) | gap=1+3=4 |
| 경계 밖에서 진입 | G=[2,8), 작업=[0,3),[7,10) | G 내부 busy=2, gap=4 |
| 측정창 clip | W=[0,10), 작업=[-2,2),[8,12) | busy=4, idle=6 |
| 포함된 interval | W=[0,10), 작업=[1,9),[2,3) | busy=8, 최종 병합 end=9 |
| 장치 분리 | W=[0,10), GPU0 작업=[0,10), GPU1 작업 없음 | GPU0 fraction=1, GPU1 fraction=0; 장치별 결과 유지 |

- [ ] 분석기 수정에 위 단위시험을 포함한다.
- [ ] 각 장치별 단위·clock domain·event category를 검증한다.
- [ ] trace 전체 창과 실제 요청 창의 결과를 모두 저장한다. 모든 실행에서 앞16.4초를 상수로 빼지 않는다.
- [ ] 기존 D2 유효 trace를 재집계하고 원 보고값·정정값·정정 사유를 나란히 저장한다.
- [ ] 기존 자료에 없는 layer·step·done 이벤트를 추정해 채우지 않는다.

## 5. M2 — 요청 부하와 계측 시간창을 일치시킨다

기존 D3는 `sleep15 → perf record20초 → perf report → perf stat20초`와 별도 `sleep60 → PCM22초` 구조였다. 실제 부하 약23.86초에 대해 PCM이 부하 종료 뒤 시작했다. 해당 PCM 값이나 `2.082 CPUs utilized`를 정상 모델 부하의 지표로 재사용하지 않는다. [H §9.3], [R §4.3]

### 5.1 하네스 변경

- [ ] 클라이언트의 tokenizer·JSONL·의존 패키지 준비와 수집기 준비를 요청 전송 전에 완료한다.
- [ ] 수집기를 시작 또는 arm하고, 실제 수집 준비 완료를 확인한 뒤 공통 시작 barrier를 해제한다.
- [ ] 각 수집기의 실제 시작 시각과 첫 요청 전송·서버 수신 시각을 별도로 저장한다. barrier만 있다고 완전 동시 시작으로 가정하지 않는다.
- [ ] 마지막 응답 이후 해당 요청들의 GPU 완료를 확인하고 수집기를 종료한다. 필요한 동기화는 세션 경계에서 수행하며, drain 비용을 벤치 요청 처리시간에 임의로 더하지 않는다.
- [ ] 수집기는 이번 실행에서 생성한 PID만 종료하고 `wait`로 출력 flush·exit code를 확인한다. 광역 `pkill`을 사용하지 않는다.
- [ ] timeout·서버 오류에서도 `finally`에 해당하는 종료 경로로 원 로그와 시각을 보존한다.

### 5.2 시간창 저장 규격 — 신규 필드 제안

```text
run_id, boot_id, attempt, collector_id, collector_pid, target_scope
clock_domain, actual_start, actual_end
first_request_send, first_server_receive, last_response_receive
last_correlated_gpu_complete
request_window_start, request_window_end
intersection_duration_us, request_coverage_fraction
sample_interval_us, exit_code, validity, invalid_reason
```

`request_coverage_fraction = |collector_window ∩ request_window| / |request_window|`로 계산한다. wall clock은 timezone을 포함하고 elapsed는 monotonic 차이로 저장한다. 시계 변환을 검증하지 않은 client·host·GPU timestamp를 직접 빼지 않는다.

주 비교의 요청 구간과 전체 수집 구간을 분리한다. 누락된 요청 구간이 있으면 해당 수집기는 전체 부하 집계에서 제외하고, 공통 교집합의 부분 측정만 별도로 보고한다. 시작/끝에 걸친 PCM 표본은 `boundary_partial`로 표시하고, 표본 평균을 임의로 잘라 정확한 부분구간 대역폭인 것처럼 만들지 않는다.

## 6. M3 — CPU↔GPU 의존 관계를 추적할 식별자를 추가한다

이벤트 이름은 신규 인터페이스 제안이다. `probe_map.md`에 **실제 파일::함수::라인, timestamp를 찍는 위치, 완료의 의미, clock domain, 캡처/리플레이 동작**을 연결한다. 구현 위치가 확인되지 않은 항목은 미확정으로 남긴다. [H §10]

### 6.1 공통 식별자와 데이터 관계

```text
run_id, boot_id, attempt, model, profile, manifest_sha256
batch_id, step_id, scheduler_phase_raw, phase_normalized
layer_id, tp_rank, gpu_uuid, numa_id
job_id, parent_job_id, numa_subjob_id
ring_slot, slot_epoch, graph_replay_id, combine_fragment_id
producer_step_id, consumer_step_id
expert_id, expert_rows, hot_or_cold, immediate_or_deferred
clock_domain, timestamp, event_type, event_seq
bytes, precision, path_present, sampling_probability
validity, missing_reason
```

- [ ] 요청과 batch는 별도 매핑으로 저장한다. 하나의 batch에 한 request ID만 붙이지 않는다. `request_id ↔ batch_id/step_id ↔ token 위치`를 연결한다.
- [ ] CPU parent job과 두 NUMA subjob을 구분해, 동일 작업의 병렬 완료를 재구성한다.
- [ ] expert ID는 logical/physical 중 어느 것인지 표시한다. 라우팅 선택 건수, 서로 다른 expert 수, expert rows를 구분한다.
- [ ] deferred 작업은 실제 producer→consumer 관계를 저장한다. 같은 step 번호라고 가정해 합치지 않는다.
- [ ] ring slot 번호만으로 작업을 연결하지 않고 `slot_epoch`와 함께 사용한다.
- [ ] 조각별 결합이면 `combine_fragment_id`를 사용한다. 전체 layer가 한 번에 결합된다고 강제하지 않는다.
- [ ] 누락·buffer overflow·표본 선택/탈락 수를 계수한다. 표본에 없는 작업을 실행되지 않은 작업으로 처리하지 않는다.

### 6.2 반드시 연결할 경계 이벤트

| 구간 | 필요한 이벤트 | 측정 의미 |
|---|---|---|
| 입력·스케줄링 | `request_received`, `scheduled`, queue depth | 사용자 TTFT 중 요청 대기 구간 |
| Attention | GPU begin/end, 실제 phase·batch token 수 | prefill/decode별 Attention 구간 |
| 라우팅 | `route_ready`, Hot/Cold assignment | expert 선택이 끝나 작업을 분배할 수 있는 시점 |
| CPU 입력 | DtoH enqueue/start/end, payload·bytes | 제출 지연·실제 복사·입력 준비 구분 |
| CPU 작업 큐 | `job_submitted`, `cpu_input_ready`, `job_started` | 제출 후 대기와 입력 미준비 구분 |
| Cold 계산 | 단계별 begin/end, `numa_job_done` | 서브풀별 계산·임계 완료 |
| 완료 전달 | `done_published`, `done_observed` | 생산자의 완료와 소비자의 인지 구분 |
| GPU Cold 준비 | HtoD start/end, `cold_ready` | Cold 결과를 GPU 결합에서 실제 소비할 수 있는 시점 |
| GPU Hot 준비 | Hot GPU begin/end, `hot_ready` | Hot 기여분 준비 완료 |
| 결합 | `other_ready`, `combine_begin/end` | 입력 미준비와 준비 후 스케줄 지연 구분 |
| TP 통신 | collective start/end, collective ID·bytes·rank | rank 간 통신/동기화 구간 |
| 출력 | sampled token, response chunk, finish/error | 요청 지표와 실행 step의 연결 |

위 표는 체크할 경계 목록이지 모든 backend의 호출 순서가 동일하다는 뜻이 아니다. 실제 코드의 부분순서와 소비 조건을 기준으로 `probe_map.md`를 작성한다.

### 6.3 CUDA graph와 clock alignment

- [ ] graph capture 때의 Python range를 replay별 이벤트로 간주하지 않는다. 각 replay와 실제 GPU node/activity·ring epoch를 연결할 수 있는지 확인한다.
- [ ] 안전한 GPU event 또는 driver activity 등 설치 환경에서 지원되는 방식으로 완료 시점을 수집한다. 지원 여부를 확인하지 않은 API·환경변수를 작동한다고 보고하지 않는다.
- [ ] CPU·GPU 및 GPU 간 clock domain과 정렬 방법·오차 범위를 저장한다. CPU timestamp와 GPU duration을 바로 연결하지 않는다.
- [ ] 각 단계에 `device synchronize`를 삽입하지 않는다. 기존 event wait·stream 의존을 바꾸지 않고, 완료 이벤트는 안전한 지점에서 수집한다.
- [ ] 이벤트 기록의 동기 출력·무제한 버퍼 증가를 피하고, 기록 buffer의 용량·누락 카운터·flush 방식을 남긴다.
- [ ] clock 정렬이 없으면 `UNMEASURED_CLOCK_ALIGNMENT`, 필수 의존 이벤트가 없으면 `PARTIAL_DEPENDENCY_MEASUREMENT`로 저장한다.

## 7. M4 — 구간별 측정 항목을 구현한다

### 7.1 CPU Cold 경로

- [ ] 기존 `prepare/cpy_input/q_input/up_gate/act/q_down/down/weight/total` 타이머의 구현 범위를 확인하고 단위·포함관계를 저장한다.
- [ ] `weight`를 이름만 보고 DRAM weight fetch 시간으로, `cpy_input`을 GPU DtoH 시간으로 해석하지 않는다.
- [ ] job 제출→시작, 시작→완료, 완료→신호 게시를 구분한다. 제출→시작에는 입력 준비 대기가 포함될 수 있으므로 전부 CPU 스케줄러 경합으로 설명하지 않는다.
- [ ] 동일 parent job에서 NUMA0/1 시작·종료와 `max(numa_job_done)`을 계산한다. 서로 다른 표본의 중앙값을 합치지 않는다.
- [ ] expert별 rows, active expert 수, 실제 AVX/AMX 분기, worker TID·NUMA pool을 연결한다. 설정값만으로 커널 경로를 판정하지 않는다.
- [ ] immediate 비어 있음과 deferred 작업 존재 여부를 함께 기록한다. 빈 immediate라고 전체 Cold 작업이 없다고 판단하지 않는다.
- [ ] deferred produced/consumed/stale 및 중복 소비·미소비 검사를 epoch 단위로 정의한다. 세션 시작/종료 시 이미 in-flight인 작업도 따로 계수해 수량 불일치를 무조건 오류로 처리하지 않는다.

`qlen≤80`을 decode의 확정 라벨로 사용하지 않는다. scheduler의 실제 phase를 보존하고, phase·batch token 수·expert rows별로 집계한다. 고정 stride64만 반복하지 않고 layer/step 층화 또는 seed 고정 표본을 사용하며 선택확률을 남긴다. [H §8.2·10.3]

### 7.2 GPU Hot·Attention·Router·결합

- [ ] TP0뿐 아니라 TP0~3의 실제 GPU operation을 수집한다. 특정 rank가 빠지면 해당 범위를 명시한다.
- [ ] full kernel name·backend·stream·device·start/end·graph replay를 저장한다. 화면용 축약 이름과 원 이름을 분리한다.
- [ ] Attention, Router, Hot MoE, combine, TP collective를 실제 코드/trace correlation으로 구분한다.
- [ ] `hot_ready`는 해당 결합에 필요한 Hot 계산이 모두 완료된 시점으로 잡는다. 첫 커널 또는 마지막으로 나열된 커널 하나만 선택하지 않는다.
- [ ] graph/eager 실행과 실제 batch 형태를 저장한다. capture 로그만으로 모든 step이 graph를 사용했다고 판단하지 않는다.
- [ ] 작업이 없는 GPU interval과 특정 의존 입력이 늦은 interval을 별도로 집계한다. 다른 요청의 작업이 겹치면 전역 idle로 계산하지 않는다.

### 7.3 전송·완료 신호

- [ ] 각 DtoH/HtoD의 enqueue/start/end, direction, 실제 bytes, payload 종류, source/destination device, job ID를 수집한다.
- [ ] payload가 hidden state, expert IDs, routing weights, CPU 결과, 완료 신호 중 무엇인지 코드와 대조한다. 식별 불가능하면 `unknown`으로 남긴다.
- [ ] Host buffer의 pinned 여부와 ring/epoch 수명을 가능한 범위에서 기록한다.
- [ ] `CPU done`과 `Cold GPU ready` 사이를 완료 신호 게시·인지·HtoD로 나눠 기록한다. 구현 순서에 맞게 의존 관계를 보존한다.
- [ ] 전송 count·bytes·duration sum·union을 따로 집계한다. 호출 수가 많다는 이유만으로 중복 전송이라고 결론 내리지 않는다.
- [ ] 지연된 경로에 연결된 복사와 다른 요청에 가려진 복사를 구분한다. 복사 duration 합계를 그대로 회수 가능한 지연으로 쓰지 않는다.

기존 TP0의 DtoH 약98,853회, HtoD 약24,716회라는 관측은 전송을 조사할 근거이지 payload·중복·병목의 직접 증명은 아니다. [H §8.3], [R §4.2]

### 7.4 CPU·DRAM·NUMA 및 TP 통신

- [ ] scheduler 프로세스뿐 아니라 모든 KT worker와 관련 child PID/TID가 perf 대상에 포함됐는지 목록으로 검증한다.
- [ ] KT/non-KT별 CPU 사용시간·affinity, 소켓별 메모리 read/write, GPU HBM 사용량·clock·power·사용률을 실제 부하 창과 함께 기록한다.
- [ ] perf 이벤트명·지원 여부·권한·측정 범위와 가능한 경우 `time_enabled/time_running`, 누락·multiplex 정보를 저장한다.
- [ ] perf record/stat/PCM의 자원 또는 counter 충돌을 확인한다. 함께 수집할 수 없으면 같은 manifest의 별도 세션으로 분리하고 교집합 부하 창을 각각 맞춘다.
- [ ] host 전체 PCM 값을 KT만의 대역폭으로 귀속하지 않는다. 배경 부하와 counter 범위를 함께 남긴다.
- [ ] remote NUMA 접근·메모리 지연 등을 직접 수집하지 못하면 미수집으로 표시한다. 낮은 IPC나 특정 대역폭 한 값만으로 원인을 확정하지 않는다.
- [ ] TP collective는 collective ID·rank별 도착·시작·종료를 연결한다. rank 간 시계 정렬이 없으면 도착 시차를 계산하지 않는다.
- [ ] collective kernel duration에는 동기화가 섞일 수 있으므로 이를 순수 네트워크 전송시간이라고 단정하지 않는다.

DRAM 포화를 판정할 동시 부하 자료나 비교 기준이 없으면 **대역폭 관측값까지만 보고하고 포화 판정은 보류**한다. 새 메모리 대역폭 sweep은 본 작업에 자동으로 추가하지 않는다.

## 8. M5 — 본 측정 전 계측 정상성을 확인한다

| 검증 | 통과 조건 | 실패 시 처리 |
|---|---|---|
| GPU activity | 지정 GPU/rank의 실제 kernel event가 존재하고, 실행한 copy 경로의 이벤트가 식별됨 | `INVALID_NO_GPU_ACTIVITY` 또는 해당 copy 경로 부분 미측정 |
| 요청 창 | 실제 요청 start/end와 수집기 창·교집합이 확인됨 | `INVALID_WINDOW` 또는 교집합 범위만 부분 측정 |
| correlation | 분석 대상 job에 layer/step/rank/epoch/결합 단위가 연결됨 | 해당 job 제외 및 누락 개수 보고 |
| clock | 차감 대상 이벤트가 같은 domain이거나 검증된 변환이 있음 | `UNMEASURED_CLOCK_ALIGNMENT` |
| ready 의존 | 필요한 모든 입력의 준비 완료와 combine begin이 연결됨 | `PARTIAL_DEPENDENCY_MEASUREMENT` |
| 집계 | union 범위·busy+idle 일치·단위시험 통과 | `INVALID_AGGREGATION` |
| 원시 보존 | trace·로그 parse 가능, flush·exit 상태 확인 | `INVALID_ARTIFACT` |
| 실행 정상성 | smoke/warmup 성공, 새 NaN·illegal access·hang 미발생 | 본 측정 중단, 오류 원문·요청·epoch 보존 |

위 상태명 중 일부는 이 작업을 위한 신규 제안이다. 구현 시 상태 스키마에 정의하고, 기존 상태를 덮어쓰지 않는다.

- [ ] 해당 SGLang 기록의 profiler key는 `activities:["CPU","GPU"]`다. `CUDA`를 넣어 GPU event 0건이 된 과거 시도를 반복하지 않도록 실제 설치본과 smoke 결과를 확인한다. [H §9.4]
- [ ] 계측 검증은 부팅당 smoke 안에 포함한다. 별도 요청 부하를 실행하면 조건부/재시도 세션으로 계수한다.
- [ ] OFF/ON의 동일 smoke 입력·출력 token IDs·finish reason·오류를 보존한다. 차이가 있으면 계측이 실행 의미를 바꿨는지 먼저 조사한다.
- [ ] 단순 logits 전체 덤프·동기 출력 등 무거운 정확성 진단을 성능 프로브에 섞지 않는다.
- [ ] 이벤트 누락이 일부이면 유효한 관측까지 지우지 않되, 불완전한 자료로 전체 경로를 재구성하지 않는다.

## 9. M6 — 측정 세션과 계측 간섭을 관리한다

### 9.1 계측 모드

| 모드 | 목적 | 수집 구성 |
|---|---|---|
| BASE | 주 성능 기준 재현 | MAIN_SHORT, 새 상세 계측 OFF; 공통 경량 모니터 조건 고정 |
| P0 | 프로브용 OFF 대조 | PROBE128, P1/P2와 같은 manifest, 새 상세 계측 OFF |
| P1 | CPU↔GPU 의존 관계 | PROBE128, CPU 작업 이벤트 + GPU activity + ready/결합 correlation; heavy CPU profiler OFF |
| P2 | CPU·메모리 진단 | PROBE128, 최소 CPU 작업 이벤트 + perf stat/PCM 등 지원 수집기; heavy GPU trace OFF |
| P3 | 조건부 CPU symbol 진단 | PROBE128, perf record 중심. 동시 counter 충돌·간섭으로 분리가 필요한 경우만 |

공통 경량 모니터, cache flush, warmup, 모델·가중치·graph 설정은 같게 유지한다. 모드별로 바뀌는 계측기·샘플링·buffer만 `observer_config.json`에 기록한다. **P1과 P2가 별도 실행이면 job 단위 인과 timeline을 서로 이어 붙이지 않는다.** P2는 동일 워크로드의 자원 관측이며 P1의 동일 순간 실측은 아니다.

### 9.2 세션 배분 — 병목 측정용 신규 제안

인계서의 최소 측정 구성은 MAIN3 + 관측2 + OFF1의 **6세션**이다. 아래는 OFF/ON 단발 비율에 의존하지 않기 위해 P0/P1/P2를 각3회로 늘린 측정 전용 배분안이다. 원 인계서의 최적화 후보 검증 단계를 실행한다는 뜻은 아니며, 원 상한인 **24세션·부팅12회** 안에서 재배분한다. 실행 전에 `PLAN_RESOLVED.md`에 확정한다. [H §12]

| 셀 | 내용 | 최대 세션 | 실행 조건 |
|---|---|---:|---|
| B | BASE MAIN_SHORT 3회 | 3 | 필수 |
| P0 | PROBE128 OFF 3회 | 3 | 필수 |
| P1 | PROBE128 의존 관계 계측 3회 | 3 | 필수 |
| P2 | PROBE128 CPU·메모리 계측 3회 | 3 | 필수; 일부 counter 차단 시 가능한 범위 명시 |
| F1 | C1 OFF/ON 각1회 + LONG OFF/ON 각1회 | 4 | phase·batch 의존성 확인이 필요한 경우 |
| F2 | perf record 또는 분리 계측 | 2 | counter 충돌·symbol 진단 필요 시 |
| G | GLM native FP8 짧은 부하 OFF/ON 각1회 | 2 | M8 정상성 게이트 통과 시 |
| R | 원인을 좁힌 재시도 | 4 | 실패 원인·변경 내용이 있는 경우만 |
| **합계** | 필수12 + 조건부8 + 재시도4 | **24** | 남은 예산을 소진하려고 실행하지 않음 |

각 attempt는 한 항목에만 계수한다. 서버 부팅 시도도 실패를 포함해 상한12회에 계수한다. 미실행은 0 tok/s의 유효 반복으로 넣지 않는다. 진행 중인 다른 캠페인과 합쳐 실행한다면 잔여 예산부터 확인하고 별도 예산이 자동 부여됐다고 가정하지 않는다.

### 9.3 실행 순서와 간섭 계산

가능한 경우 같은 부팅에서 모드를 전환하고, 시간 경과에 따른 편향이 한 모드에만 몰리지 않도록 예를 들어 다음 순서를 사용한다. **이 순서는 신규 제안**이며 실행 가능성을 확인한 후 고정한다.

```text
주 기준: BASE → BASE → BASE
프로브 묶음 1: P0 → P1 → P2
프로브 묶음 2: P2 → P0 → P1
프로브 묶음 3: P1 → P2 → P0
```

계측 모드 변경에 재부팅·재캡처가 필요하면 boot ID와 소스/설정 차이를 남긴다. 동일 부팅 비교가 아니면 그 영향을 분리할 수 없는 한계로 보고한다.

```text
throughput_ratio  = throughput_ON / throughput_OFF
elapsed_ratio     = request_duration_ON / request_duration_OFF
latency_ratio     = latency_ON / latency_OFF
```

- [ ] P1/P2는 MAIN256이 아니라 같은 PROBE128의 P0와 비교한다.
- [ ] 반복 원값, 평균, 표본 SD(ddof=1), 실행 순서·부팅 관계를 보존한다. p95는 반복별 p95 평균과 pooled p95를 구분한다.
- [ ] 처리량·TTFT/TPOT뿐 아니라 batch/phase 분포와 이벤트 누락률도 비교한다.
- [ ] 계측 ON이 더 빠른 단발 결과를 성능 향상으로 해석하지 않는다.
- [ ] 첨부에는 허용 간섭률의 확정 기준이 없다. 임의의 5%·10%를 기존 기준이라고 쓰지 말고 실행 전 허용 기준을 별도로 정하거나 비율과 한계만 보고한다.
- [ ] 계측 간섭이 크거나 실행 분포를 바꾸면 샘플링을 낮추거나 수집기를 분리한다. 해당 변경은 새 observer config·attempt로 남기고 재시도 상한을 지킨다.

F1·F2·G의 단발 비교는 진단 자료이며 반복 대표 성능으로 확대하지 않는다.

## 10. M7 — 직접 측정값으로 지연을 분해한다

### 10.1 Hot/Cold 준비 완료의 지연

같은 결합 단위·실제 producer/consumer 관계·정렬된 clock domain에서 계산한다. 아래 세 입력이 모두 존재하고 준비 완료 이벤트가 유효한 경우에만 식을 적용한다. [H §10.3]

```text
t_hot_ready   = Hot 기여분을 GPU 결합에서 소비할 수 있는 시각
t_cold_ready  = Cold 기여분을 GPU 결합에서 소비할 수 있는 시각
                (필요한 H2D와 완료 신호 포함)
t_other_ready = 잔차 등 다른 결합 입력 준비 완료 시각

cold_ready_lateness = max(0, t_cold_ready - max(t_hot_ready, t_other_ready))
hot_ready_lateness  = max(0, t_hot_ready  - max(t_cold_ready, t_other_ready))
combine_schedule_gap = max(0, t_combine_start
                             - max(t_hot_ready, t_cold_ready, t_other_ready))
```

입력이 실제로 없는 경우 `path_present=false`, timestamp는 null로 둔다. 없는 경로를 0시각으로 만들어 비교하지 않는다. 존재하는 입력 집합으로 결합 준비 완료는 계산할 수 있지만 Hot/Cold 비교가 성립하지 않으면 해당 lateness는 N/A로 보고한다.

`combine_start < required_input_ready` 같은 불가능한 순서는 먼저 clock·graph mapping·완료 정의 오류로 검사한다. max(0)로 잘못된 순서를 숨기지 않는다. `cold_ready_lateness`가 크다고 곧바로 CPU 계산이 원인이라고 보지 말고 다음 절의 Cold 경로를 연결한다.

### 10.2 Cold 내부와 병렬 서브풀

같은 작업의 실제 timestamp로 다음을 저장한다.

```text
submit_to_start     = t_job_started - t_job_submitted
cpu_compute_span    = t_job_done - t_job_started
publish_delay       = t_done_published - t_job_done
cpu_to_gpu_ready    = t_cold_ready - t_cpu_parent_done
numa_parent_done    = max(같은 parent의 필수 subjob 완료 시각)
numa_completion_skew= max(필수 subjob 완료) - min(필수 subjob 완료)
```

각 식의 필수 이벤트가 없거나 clock 정렬이 없으면 계산하지 않는다. `submit_to_start`는 입력 미준비·buffer 의존이 포함될 수 있고, `cpu_to_gpu_ready`는 신호·전송·GPU 소비 가능 조건을 포함한다. 둘을 곧바로 순수 큐 대기·순수 PCIe 시간으로 이름 붙이지 않는다.

CPU 실행·전송·Hot 계산의 overlap은 실제 interval 교집합으로 계산한다. 두 NUMA 서브풀의 합계 worker time, interval union, parent 임계 완료 span은 서로 다른 지표로 저장한다. stage별 p50들을 더해100% breakdown을 만들지 않는다.

### 10.3 집계 단위와 보고할 분포

- [ ] `model/profile/phase/layer/tp_rank`별 유효 표본 수, 누락 수, 샘플 선택확률을 기록한다.
- [ ] Hot/Cold 준비 완료 차이, combine gap, CPU queue/compute, NUMA skew, 전송 count/bytes/time의 p50/p95/p99와 반복별 값을 저장한다.
- [ ] expert별 rows·Hot/Cold 라우팅을 기록하되 TP shard나 여러 표본을 논리 라우팅 건수로 중복 합산하지 않는다.
- [ ] Cold·Hot·other 중 최종 준비 입력의 관측 건수와 비율을 유효한 의존 측정 표본을 분모로 계산한다. 표본 비율을 전체 실행 비율로 확대하지 않는다.
- [ ] phase·layer별 Hot/Cold 지연과 실제 GPU idle이 겹친 구간을 따로 저장한다.
- [ ] 지연이 큰 개별 사례는 원 요청/step/job ID·원 trace interval을 함께 제공한다. 대표 사례만 남기고 나머지 원자료를 버리지 않는다.
- [ ] TP rank 시차와 collective 구간을 분리해, 특정 rank가 늦게 도착한 관측과 통신 자체의 관측을 구분한다.

### 10.4 최종 해석의 제한

| 관측 | 보고 가능한 내용 | 추가 근거 없이 금지할 결론 |
|---|---|---|
| Cold ready가 Hot/other보다 늦음 | 해당 표본의 Cold 경로 준비 지연 | CPU 연산만 최적화하면 동일 시간만큼 빨라짐 |
| CPU 완료는 이르지만 Cold ready가 늦음 | 완료 이후 신호/반환/소비 준비 구간 지연 | H2D 또는 PCIe가 단독 원인 |
| Hot ready가 늦음 | 해당 표본의 Hot 경로 준비 지연 | GPU MoE kernel 자체가 항상 전체 병목 |
| 모든 입력 ready 뒤 combine이 늦음 | 준비 후 결합 시작 gap | GPU가 그 시간 내내 유휴 |
| 두 NUMA 완료 시점이 다름 | 동일 parent의 서브풀 완료 불균형 | 원격 메모리 접근이 원인 |
| GPU busy/CPU 사용률/DRAM 값이 높음 | 지정 창·범위에서 관측된 자원값 | 포화·최적화 회수 가능 시간 확정 |

ready-time 차이는 현재 실행의 관측 지표다. 여러 layer·step의 lateness 합계를 E2E 지연이나 성능 개선 상한으로 사용하지 않는다. 반복 간 일관성·계측 간섭·표본 coverage까지 확인해 **확인된 지연 구간, 원인이 미분리된 구간, 측정 불가 구간**을 나누어 기록한다.

## 11. M8 — GLM은 정상성 확인 후 별도로 측정한다

GLM BASIC4는 native `FP8_PERCHANNEL`, CPU100·NUMA pool2, uniform80, deferred2 경로에서 성능 요청을 완료했지만 품질 시험은 시간 초과·절단이 있었다. OPT4는 per-channel FP8→INT4 변환 실패로 미실행이다. Qwen OPT4의 계측/최적화 경로가 그대로 지원된다고 가정하지 않는다. [H §4·6·11.1], [R §3·5]

- [ ] 기존 native FP8 BASIC4의 effective config와 요청 JSON부터 복원한다.
- [ ] 기존 품질 문항 중4문항을 먼저 사용해 thinking·EOS·stop·max_tokens·종료 상태·오류를 확인한다. 성능용 ignore_eos와 혼합하지 않는다.
- [ ] timeout·절단·비정상 종료가 반복되면 장시간 병목 프로브를 진행하지 않고 `BLOCKED_NORMAL_OUTPUT`와 원 요청·오류를 남긴다.
- [ ] 정상성 게이트 통과 후 G 예산의 짧은 고정 부하 OFF/ON을 수행한다. 단발 진단이라는 범위를 유지한다.
- [ ] dense층0~2, routed층3~91의89개 층, shared expert를 구분한다. Qwen62층·5,952슬롯 기준을 복사하지 않는다.
- [ ] native FP8에서 실제 지원되는 CPU 완료·전송·결합 경계에만 계측한다. AMXINT4의 RB·CF가 적용된다고 가정하지 않는다.
- [ ] INT4 converter 수정, 가짜 `weight_block_size` 추가, 전체 변환, 신규 Hotmap 생성은 별도 형식/최적화 과제로 남긴다.

정상성 확인·재시도 문항은 인계서의 전체 품질 예산 최대80문항 시도 안에서 계수한다. 이번 측정 작업이 별도의80문항 예산을 추가하는 것은 아니다. 제한된 정답률이나 동일 출력 확인을 범용 정확도·무손실 증명으로 쓰지 않는다.

## 12. M9 — 결과와 미측정 항목까지 보존한다

### 12.1 결과 디렉터리 제안

```text
shadow_assists/features/<NEW_ID>/
  PLAN_RESOLVED.md
  SOURCE_INDEX.md
  CONFIG_DIFF.md
  COMPLETION_STATUS.md
  FULL_REPORT.md
  BOTTLENECK_EVIDENCE.md
  WORK_LOG.md
  FULL_RAW_DATA.md
  PROGRESS.md
  PROGRESS_EVENTS.jsonl
  PUBLISH_RECEIPT.md
  ARTIFACT_INDEX.csv
  SHA256SUMS.txt
  raw_md/part-*.md
  profiles/
    probe_map.md
    corrected_metrics.csv
    observer_windows.csv
    observer_overhead.csv
    dependency_metrics.csv
    missing_coverage.csv
    validation_results.json

eval/results/<NEW_CAMPAIGN>/
  state/
    RUN_MANIFEST.json
    RUN_STATE.json
    execution_events.jsonl
  <model>/<profile>/<boot>/<attempt>/<workload>/<rep>/
    requested_config.json
    effective_config.json
    feature_proofs.json
    observer_config.json
    environment.json
    launch_cmd.sh
    bench_cmd.sh
    timestamps.json
    metrics.json
    request_results.jsonl
    request_batch_map.jsonl
    token_timing.jsonl
    stage_events.jsonl
    stage_intervals.csv
    expert_jobs.csv
    deferred_events.csv
    transfer_events.csv
    cpu_timeseries.csv
    gpu_timeseries.csv
    memory_timeseries.csv
    stdout.log
    stderr.log
    error.json
    failing_request.json
```

신규 경로의 `state/`도 result 디렉터리에 둔다. 대형 trace·`perf.data`는 승인된 보존 위치에 두고 정확한 경로·크기·SHA256·접근 방법을 색인에 남긴다. 수집하지 못한 파일을 빈 파일로 만들어 수집 완료처럼 표시하지 않는다.

### 12.2 필수 결과 내용

| 파일 | 필수 내용 |
|---|---|
| `FULL_REPORT.md` | 환경·설정·모든 시도·반복 원값·유효성·누락·오류·원자료 경로. 성능 우수성·원인 추정·개선 권고 제외 |
| `BOTTLENECK_EVIDENCE.md` | 측정 질문별 이벤트 근거, observed ready 지연, 표본 수·coverage, 반복성, 확인/미분리/미측정 구분 |
| `WORK_LOG.md` | 계측 수정의 이유, 조사 중 가설, 선택·보류 근거. 원시 결과와 분리 |
| `observer_overhead.csv` | manifest·boot·mode·rep, OFF/ON 원값·비율·실행 순서·간섭 한계 |
| `missing_coverage.csv` | 요청→batch→layer→job→ready 연결의 대상 수·성공 수·누락 사유 |
| `COMPLETION_STATUS.md` | 최초 상태가 아니라 최종 attempt를 반영한 작업별 상태 |

- [ ] 성공/실패 요청 수, 실제 input/output token 합계, duration, 처리량, TTFT/TPOT/ITL/E2EL, 통계 정의를 저장한다.
- [ ] 실패·무효·미실행을 유효한 0성능 반복으로 평균내지 않는다.
- [ ] stdout/stderr·전체 kernel name·오류 prompt와 token IDs를 보존한다. tail/head만 영구 저장하지 않는다.
- [ ] source 수정은 최소 diff로 남기고, 계측 OFF 복원 방법을 기록한다.
- [ ] 인계서의 30분 보고 요건은 실제 실행기/보고 채널이 지원되는 경우에만 수행한다. 저장은 `SAVED`, 실제 전달 확인은 `DELIVERED`, 전달 불가는 `SAVED_NOT_DELIVERED` 등으로 구분한다. 이 문서 작성으로 정기 보고가 생성된 것은 아니다.
- [ ] 지정 원격에 게시할 때에는 현재 권한·브랜치를 확인하고, 비밀정보를 제외한 명시적 파일만 stage한다. force push를 사용하지 않는다.
- [ ] push exit code·remote SHA·원격 결과 파일 해시를 확인한다. 게시 실패와 로컬 보존 성공을 구분한다.
- [ ] MD·작은 원자료 묶음의 실제 존재 경로와 다운로드를 전달한다. 대형 원자료는 접근 방법을 별도 전달한다.

## 13. 최종 완료 체크리스트

아래가 충족되면 측정 작업을 완료로 닫는다. 병목이 하나로 확정되지 않아도 차단·미측정 이유를 빠짐없이 남겼다면 작업 상태를 정직하게 종료할 수 있다.

- [ ] Qwen OPT4 고정 구성·모델/manifest/실행 코드 해시를 확인했다.
- [ ] GPU duration sum/union/gap 분석기를 수정하고 합성 단위시험을 통과했다.
- [ ] 실제 요청과 수집기 시간창을 정렬하고 장치별 유효 구간을 저장했다.
- [ ] layer·step·rank·NUMA·job·ring epoch·graph replay 상관관계를 검증했다.
- [ ] Hot ready·Cold ready·other ready·combine start를 연결했거나, 연결 실패 항목과 이유를 명시했다.
- [ ] CPU 큐/연산/완료 전달, 전송 count/bytes/time, GPU Hot/Attention/Router/TP 통신을 구분했다.
- [ ] 같은 PROBE128의 OFF/ON을 비교하고 반복성·계측 간섭·누락률을 보고했다.
- [ ] 실제 scheduler phase를 사용했고, 서로 다른 percentile·rank·세션을 임의 합산하지 않았다.
- [ ] GLM 조건부 작업은 정상성 게이트 결과에 따라 수행 또는 차단 처리했다.
- [ ] 각 항목의 최종 상태, 원시 자료, 해시, 보고·게시·전달 상태를 보존했다.

**최종적으로 남길 것은 ‘CPU가 느리다’는 설명이 아니라, ‘어떤 phase·layer·rank·작업에서 어떤 입력 준비가 얼마만큼 늦었고, 그 판단에 필요한 이벤트가 얼마나 수집됐는가’에 대한 재검증 가능한 자료다.**

## 부록 A. 근거 문서와 식별자

이 문서에서 `[H §n]`, `[R §n]`, `[O §n]`는 아래 첨부 문서의 절 번호를 가리킨다. 외부 최신 문헌이나 현재 저장소 상태를 추가 조사한 결과는 포함하지 않는다.

| ID | 첨부 파일 | 주 참조 범위 |
|---|---|---|
| H | `CPU_MoE_optimization_session_handoff-2.md` | §2 원자료, §3~5 고정 구성·프로토콜, §8~10 계측 결함·설계, §12~16 예산·저장·보고 |
| R | `IDE073_results_review-2.md` | §3 GLM 차단, §4 계측 제약, §5 품질·상태, §6 후속 우선 작업 |
| O | `IDE073_optimization_handoff_bundle.zip`의 `evidence/IDE073_FULL_REPORT_original.md` | §5 프로브 원값, §6~8 누락·자원·간섭 기록 |

```text
H SHA256
67f1a818f1d87a9ea8fd84f9d527e5ee012b72ae4109542fd4ddd1d0ca4ee15c

R SHA256
57c8791e7fe74c7767acc72cfea4d0745dcb1c7587cbc6b51a8e81010c462464

O SHA256
4ba6aa473be578ea4ec5600263633ea96629d841cfdeaac40b6dd9c63472f0e5
```

개별 첨부 H·R은 ZIP의 대응 MD와 바이트 단위로 동일함을 확인했다. 원 보고서의 수치는 원 관측으로 보존하되, H·R에서 지적한 GPU union/gap·시간창 문제를 해결하기 전에는 기존 활용률·Cold 대기 상한·PCM 값을 검증된 병목 근거로 사용하지 않는다. [H §8~9], [R §4], [O §5~8]
