# IDE_074 후속 추가 측정 작업 지시서
## 성능 개선 착수 전: 측정 오류 정정, Cold 실행 지연 분해, 최적화 판단 근거 확보

> 작성일: 2026-09-17 (Asia/Seoul)  
> 문서 상태: **실행 계획 — 이 문서 작성 과정에서 신규 추론·계측·성능 변경을 실행하지 않음**  
> 주 대상: Qwen3-Coder-480B OPT4 / 같은 물리 서버의 H100 80GB 4장 + CPU 2소켓  
> 근거 캠페인: `IDE_074_20260917`  
> 분석 자료의 게시 기준 커밋: `87dee8e55b2ba0bcaa3a9c630e375a51a565cf68`  
> 이전 설정 인계 기준 커밋: `67d24dd84ed5681e97851c018af01d3a2d917543`  
> 실행 순서: **기존 자료 정정 → 최소 추가 측정 → 범위별 증거 판정 → 별도 성능 개선 작업**

---

## 0. 다음 실행자에게 전달할 지시

이 문서는 IDE_074를 처음부터 다시 실행하라는 지시가 아니다. 이미 확보한 자료를 재사용하고, 성능 개선 후보를 고르는 데 필요한 공백만 보완한다. **CPU가 오래 실행된다는 사실과, CPU 때문에 GPU가 실제로 기다렸다는 사실을 구분하고, 그 기다림이 어디에서 발생했는지 생산자 작업 기준으로 분해하는 것이 목적**이다.

먼저 `FULL_REPORT-5.md`, `IDE074_bottleneck_analysis.md`, 이 문서의 고정 조건과 잔여 예산을 읽는다. 과거 코드 결함은 아래 근거 목록의 고정 소스 또는 현재 실제 실행 소스에서 다시 확인한다. 이미 수정된 결함을 다시 수정하거나, 현재 코드를 확인하지 않고 과거 행 번호에 패치를 적용하지 않는다.

**이번 단계에서 하지 않을 일:** Hotmap·층별 예산 변경, CPU 워커 수·affinity sweep, deferred 수 변경, AMX/AVX threshold 변경, GPU backend 교체, 정밀도 변경, chunk 정책 변경, GLM INT4 변환, 신규 모델 추가. 이런 조작은 계측 변경이 아니라 다음 성능 개선 단계의 실험이다.

**이번 단계가 끝나면 남아야 할 답:**

1. 큰 decode 배치에서 실제로 지연을 발생시키는 생산층·Cold job·CPU 단계는 무엇인가?
2. `go→start` 중 선행 FIFO 작업 때문에 기다린 시간과 큐/스레드 전달 자체의 지연은 각각 얼마인가?
3. CPU 완료 이전, 완료 신호, HtoD, GPU 결합 중 실제 소비를 늦추는 구간은 어디인가?
4. 그 결과가 계측 태스크 삽입·clock 오차·잘못된 시간창·잘못된 layer/epoch 연결의 산물은 아닌가?
5. 다음 단계에서 어떤 변경을 시험할 근거가 생겼고, 어떤 변경은 아직 근거가 부족한가?

모든 자원 병목을 끝없이 조사할 필요는 없다. **선정할 최적화 후보의 원인 경로를 검증하는 데 충분한 증거가 확보되면 측정을 종료**한다. 예를 들어 Cold 커널 경로를 선택할 근거가 충분하다면 미측정인 통신 대역폭 전체를 추가로 조사하느라 그 결정을 지연시키지 않는다. 반대로 DRAM 포화를 근거로 변경하려면 별도의 포화 근거가 필요하다.

### 0.1 문서의 사실·제안 구분

| 표기 | 의미 |
|---|---|
| `[R074]` | IDE_074 첨부 원 보고서의 관측값·실행 기록 |
| `[A074]` | 직전 분석 MD의 재계산, 당시 고정 코드·원장 검토 결과, 해석 |
| `[P074]` | 이전 병목 측정 작업 지시서. 계획과 실제 실행의 차이는 보존 |
| `[H073]` | 이전 환경·모델·구성·실험 제약 인계서 |
| `[C-*]` | `[A074]`에서 확인한 게시 커밋의 코드 또는 원장 식별자 |
| **신규 요구 / 제안** | 이 문서에서 구체화한 이벤트·분석식·검증 기준·세션 배분. 이미 구현·측정됐다는 의미가 아님 |

이 문서는 위 첨부와 대화에서 확인된 코드 검토를 작업 지시로 구체화했다. 새로운 외부 기술 조사나 원 서버 접근 결과를 섞지 않았다. 고정 소스 링크를 제시한 것은 현재 저장소 HEAD나 서버 상태가 동일하다는 뜻이 아니다. 신규 파일명·필드명·명령 계약은 실제 구현자가 지원 여부를 확인하고 `PLAN_RESOLVED.md`에 확정한다.

### 0.2 탐색 목차

- 1~3장: 현재 근거, 고정 조건, 추가 작업 우선순위
- 4~5장: 기존 자료 재집계와 표준 지표 정의
- 6~8장: 저간섭 계측, 실제 시간창, producer/consumer 식별과 clock
- 9~12장: FIFO, CPU expert 단계, 자원 카운터, GPU·전송·통신·prefill
- 13~15장: 정상성 검사, 제한된 실행 배분, 통계·유효성 판정
- 16~19장: GLM 분리, 성능 개선 전환 기준, 결과 저장, 최종 체크리스트
- 부록 A~D: 스키마, 분석기 시험, 실행 순서·코드 점검 위치, 근거와 해시

---

## 1. 이미 확인한 결과와 아직 남은 공백

### 1.1 재측정 없이 계승할 관측

아래는 역사적 관측이다. 새 실험 결과로 바꾸어 표시하거나, 신규 개선율의 분모로 자동 사용하지 않는다.

| 항목 | 기존 관측 | 다음 단계에서의 용도 |
|---|---|---|
| Qwen MAIN_SHORT OFF | 743.02 / 747.53 / 768.93 tok/s, 평균 753.16 ± 13.84 | IDE_074 기준 기록. 한 부팅 3회 |
| PROBE128 OFF | 720.80 / 780.72 / 732.61 tok/s, 평균 744.71 | 같은 PROBE 계측 비교의 과거 대조 |
| P1 본 반복 | 696.39 / 705.62 / 690.76 tok/s, 평균 697.59 | 과거 OFF 평균 대비 −6.33%. 부팅 차이 포함 |
| P1+SMOKE | 평균 693.10 tok/s | 별도 smoke 부팅까지 포함한 4회 값. 본 반복과 분리 |
| P2 | 평균 710.79 tok/s | 과거 OFF 평균 대비 약 −4.55%. P1과 같은 KT 이벤트 패치 |
| 큰 decode graph | CPU 게시 지연 p50 302.2~328.5µs | Cold 완료 이전 경로의 조사 근거 |
| 큰 decode graph | Hot 종료→HtoD 시작 p50 315.2~341.2µs, p95 812.0~849.5µs | GPU 내부 timestamp로 확인된 진행 간격 |
| 큰 decode graph | go→deferred 시작 p50 650.5~677.2µs, 실행 span p50 898.2~923.0µs | FIFO와 실제 서비스 시간을 분리할 필요 |
| 큰 decode graph | CPU 게시가 Hot보다 늦은 표본 88.4~89.6% | 현재 분모에는 layer0 등이 포함. 정정 후 다시 보고 |
| 작은 decode graph / C1 | 위 GPU 간격 p50 약 10~11µs, CPU 게시 지연 p50 0 | 모든 배치가 같은 병목이 아니라는 근거 |
| NUMA 완료 시차 | 큰 graph p50 약 30~33µs, 일부 ms급 tail | 중앙부와 tail을 구분해 조사 |
| LONG dependency 표 | `DECODE bs=8` | prefill/EXTEND 분석의 대체 자료가 아님 |
| GLM | `BLOCKED_NORMAL_OUTPUT` | Qwen 측정과 분리. 장시간 성능 측정의 선행 조건 미충족 |

근거: `[R074]` 1~8장, `[A074]` 1~9장. 작은 그래프의 10~11µs를 모든 graph에서 빼는 상수 보정은 하지 않는다. P1과 P2의 유사한 값도 같은 계측 패치를 공유하므로 독립적인 무계측 검증으로 취급하지 않는다.

### 1.2 반드시 정정하거나 검증해야 하는 문제

| ID | 남은 문제 | 영향 | 현재 상태의 취급 |
|---|---|---|---|
| D01 | PCM 시각의 소수초를 파싱하지 못하고 실패 표본을 창 내 표본에 포함 | 부하 평균 대역폭 오류 | 요청 창 평균으로 재사용 금지 |
| D02 | `run_bench()` 호출 전후를 실제 부하 창으로 사용 | 준비 약 13초 및 drain 등이 자원 집계에 섞임 | 자원 활용·포화 판정 보류 |
| D03 | perf 전체 누적값만 있고 실제 부하 창별 delta가 없음 | 추론 중 KT 워커 사용량 복원 제한 | 복원 불가면 새 측정 |
| D04 | CPU done 게시를 `cold_ready`로 명명 | GPU 결과 준비와 CPU 완료 지연 혼동 | 기존 필드 보존 + v2 이름·정의 분리 |
| D05 | `submit_to_start`가 `def_start−go` | 실제 enqueue/큐 대기/선행 계산 미분리 | FIFO의 실제 task provenance 추가 |
| D06 | eager slot map의 마지막 값으로 과거 이벤트 분류 | P2 `rows=1` 등 phase·크기 오류 가능 | 시간·epoch 기준 복원, 불가면 UNKNOWN |
| D07 | 순서대로 zip한 graph/slot 연결, 휴리스틱 layer | 누락 한 건으로 후속 매핑 오염 가능 | 명시적 ID 또는 다중 불변식 검증 |
| D08 | layer0·Cold 없는 행을 Cold-later 분모에 포함 | 지연 비율의 의미 불명확 | 경로·소비 상태별 분모 분리 |
| D09 | 5µs 이내 순서 위반을 검증 없이 jitter로 허용 | 미세 지연 해석 과신 | 오차 구간·민감도 분석 추가 |
| D10 | start/end stamp 태스크 2개를 FIFO에 삽입 | 관측하려는 대기열 자체 변경 | 태스크 삽입 없는 기록 설계 우선 |
| D11 | expert별 rows·실제 분기와 지연 job 연결 부족 | 다음 커널/Hot 배치 변경 근거 부족 | 표본 작업의 상세 단계 추가 |
| D12 | 생산층 L−1·소비층 L 관계의 경계/수명 검증 부족 | 잘못된 층 최적화, stale/중복 소비 미검출 | 첫·마지막 층/step·graph 전환 검증 |

D01~D10의 근거는 `[A074]` 5~8장과 `[C-DEP]`, `[C-RENDER]`, `[C-HARNESS]`, `[C-PROBE]`다. D11~D12는 현재 자료가 부족한 검증 항목이며, 이미 특정 오동작이 발생했다고 단정한 것이 아니다.

### 1.3 이번 측정에서 결론을 강제하지 않을 사항

현재 자료만으로 CPU 연산 포화, DRAM 포화, PCIe 포화, Linux 스케줄러 경합, NUMA 원격 접근 중 하나를 단독 원인으로 확정하지 않는다. 시간 합계나 p50들을 더해 E2E 지연 비중·최대 개선율을 만들지 않는다. 최종 판정이 “CPU 특정 단계가 소비 지연과 연결됨, DRAM 세부 원인은 미분리”여도 해당 단계 최적화를 시험할 근거가 될 수 있다.

---

## 2. 고정 구성과 재현 조건

### 2.1 성능 관련 설정은 그대로 유지

| 구분 | 고정값 / 확인 사항 |
|---|---|
| 구조 | 동일 서버 CPU 계산 + GPU 계산. CPU 가중치의 GPU 전송 실행으로 바꾸지 않음 |
| 모델 | `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8` |
| revision | `003f183a92fbe5b9a8325aaa8b2ae797c91dd90f` |
| CPU 가중치·backend | `/models/kt/qwen3-480b-int4`, `AMXINT4` |
| GPU·TP | GPU 0~3, TP4. UUID·실제 장치 대응 별도 기록 |
| Hotmap | `/models/kt/ide070/hotmap_v2.json`, 전체 해시 검증 |
| 층별 예산 | `/models/kt/ide070/layer_budget_5952.json`, 62층 합계 5,952 |
| CPU | 96 workers, NUMA pool 2, 기존 non-KT pinning |
| 실행 의미 | callback-free, skip-empty, deferred8, RB 모두 기존 상태 유지 |
| 모델 선택 | logical top-k8 고정. 성능을 위해 선택 expert 수를 줄이지 않음 |
| KV | dtype auto, capacity 40,960 |
| graph | decode max batch 64, prefill graph disabled |
| 기타 | context 32,768, max-running 64, chunk 8,192, mem fraction 0.95, attention triton |
| 시스템 | CPU turbo/governor/SMT, GPU clock/power 설정 변경 금지. 실제 값 기록 |

```bash
# 기존 최적화 설정이다. 아래를 지우는 것이 계측 OFF가 아니다.
KT_CALLBACK_FREE=1
KT_CF_SKIP_EMPTY_IMM=1
KT_GPU_EXPERTS_PER_LAYER=/models/kt/ide070/layer_budget_5952.json
KT_AVX_RB=0  # 이 빌드에서는 환경변수 존재로 RB가 활성화됨
```

근거: `[H073]` 3~5장, `[P074]` 3장. 현재 import 경로·소스 diff·확장 바이너리 해시를 다시 확인한다. IDE_074의 기록된 재빌드 `.so` SHA256은 `adf49e48ec0510fafb55f14e72ee348938c7410653af18b77529d9124d1b52be`다. 새 계측 패치로 바뀌면 그 해시를 새로 저장하며, 기존 해시로 표기하지 않는다. `[R074]` 1장.

**중복 증빙 주의:** IDE_074 runtime proof의 `per_layer 행 248 / 합 23808`을 논리 expert 슬롯 23,808개로 해석하지 않는다. 4개 rank별 기록인지 확인하고 rank별 62층·합 5,952를 검증한다. 반복 출력 행의 합과 모델의 논리 배치 예산을 분리한다.

### 2.2 입력과 정상 실행 절차

| 용도 | 워크로드 | C | 요청 수 | input/output 합계 | 기본 사용 |
|---|---|---:|---:|---|---|
| 성능 기준 | `M073_QWEN_MAIN_SHORT` | 64 | 256 | 131,786 / 32,768 | 기존 기록 보존. 다음 성능 개선 단계의 새 대조군에 사용 |
| 추가 측정 주 대상 | `M073_QWEN_PROBE128` | 64 | 128 | 65,886 / 16,384 | 이번 제한된 추가 실행 |
| 저동시성 | `M073_QWEN_LOW_CONCURRENCY` | 1 | 16 | 8,234 / 2,048 | 기존 자료 우선. 조건부 |
| 긴 입력 | `M073_QWEN_LONGER_PREFILL` | 8 | 32 | 131,147 / 4,096 | 기존 EXTEND 재분석 우선. 조건부 |

위 합계는 기존 완료 관측이다. 새 실행에서는 실제 토큰 합계를 검증한다. 비슷한 길이의 새 generator 입력으로 바꾸지 않는다. 요청 순서·tokenizer revision·text/token ID manifest와 해시를 저장한다. PROBE128의 입력 합계는 MAIN의 단순 절반이 아니다. `[R074]` 2장, `[H073]` 5장.

기존 성능 요청 설정(`/v1/completions`, raw prompt, request-rate inf, temperature0, top_p1, ignore_eos, output128)을 `bench_cmd.sh`에서 복원한다. 품질 요청에 ignore_eos를 적용하지 않는다. 부팅당 smoke 1회, warmup 1회(C16·32요청·출력128), 세션 전 engine cache flush와 3초 대기를 동일 적용한다. 새 barrier는 클라이언트 준비를 분리할 뿐 요청 스케줄이나 생성 설정을 바꾸지 않아야 한다.

### 2.3 실행 환경과 보존

- 새 결과 namespace를 정하되 `IDE_075` 등 사용 여부를 확인하지 않은 ID를 확정된 기존 캠페인처럼 쓰지 않는다.
- 기준 커밋, 현재 HEAD, dirty diff, 실제 컨테이너 소스와 `.so`, 모델 파일, Hotmap, 입력 manifest의 전체 해시를 저장한다.
- 기존 파일·결과는 덮어쓰지 않는다. 정정 결과는 `v2` 또는 별도 하위 디렉터리에 쓴다.
- 타 작업·다운로드·변환과 겹치지 않게 실행한다. 타인의 프로세스를 종료하거나 BIOS·권한을 우회하지 않는다.
- OFF와 ON에 **같은 새 계측 바이너리**를 사용한다. 기존 바이너리 OFF와 새 바이너리 ON만 비교해 순수 observer overhead라고 부르지 않는다.
- 공통 경량 모니터도 개수·주기·affinity를 고정한다. 모니터 스레드를 배치하느라 모델 워커의 기존 affinity를 바꾸지 않는다.

---

## 3. 추가 작업 목록과 우선순위

### 3.1 작업 분류

| 작업 ID | 할 일 | 기존 자료로 가능한 부분 | 새 수집이 필요한 부분 | 우선순위 |
|---|---|---|---|---|
| A00 | 근거·코드·예산 고정 | manifest·원장·해시 확인 | 현재 실행 환경 증빙 | 필수 |
| A01 | PCM·시간창·GPU union·분모 정정 | 원 CSV/trace/metrics 재집계 | 정확한 부하 이벤트가 없으면 신규 창 수집 | 필수 |
| A02 | 표준 ready·전송 지표 생성 | 기존 타임라인의 start/end | 없는 other/consume 경계만 보완 | 필수 |
| A03 | 저간섭 기록기 | 기존 패치 코드 검토 | 실제 task entry/exit·buffered event | 필수 |
| A04 | request barrier·clock·TID 범위 | 기존 collector 메타데이터 검토 | 준비 완료/요청/카운터 창·clock anchors | 필수 |
| A05 | producer/consumer·FIFO 분해 | 기존 L−1/L 사건 연결 재분석 | job ID, task seq, enqueue/dequeue, dependency | 필수 |
| A06 | expert rows·CPU 단계·NUMA 연결 | 기존 layer별 지연 분포 | 실제 expert 분기·단계·소켓 parent span | 필수 표본 |
| A07 | 실제 부하의 CPU·DRAM 카운터 | PCM 재집계 가능 범위 | workload-window perf delta·TID coverage | 필수 기본값, 세부 PMU는 조건부 |
| A08 | GPU 전송·결합·rank 정렬 | 기존 TP0~3 trace 재분석 | 부족한 correlation/collective 위치 | 전송은 필수, 통신 심화는 조건부 |
| A09 | prefill·저동시성·tail | 기존 F1·P2 eager 자료 재분석 | phase 복원 실패·선정 후보에 필요할 때 | 조건부 |
| A10 | 정상성·버퍼 수명 검증 | 기존 smoke/hash·오류 기록 | epoch lifecycle counters·표본 정상성 | 필수 |
| A11 | OFF/CORE/CORR/RESOURCE 비교 | 과거 간섭 기록 참고 | 잔여 예산 안의 최소 비교 | 필수 |
| A12 | 판정·산출물·최적화 인계 | 모든 결과 통합 | 없음 | 필수 |
| G00 | GLM 정상 출력 원인 조사 | 기존 gate/request/출력 재검토 | Qwen과 독립된 조건부 정상성 실행 | 별도 트랙 |

**‘필수’는 데이터를 반드시 만들어 채우라는 뜻이 아니다.** 지원되지 않거나 원자료가 없으면 해당 측정은 차단 사유를 남긴다. 다만 그 측정에 의존하는 성능 개선 후보는 선택하지 않는다. 다른 원인 경로의 유효 자료까지 버릴 필요는 없다.

### 3.2 먼저 하는 일과 나중에 하는 일

```text
A00 원자료·환경·예산
 └─ A01/A02 기존 자료 정정 ──> 복원 가능한 질문을 먼저 종료
     └─ A03/A04 저간섭 기록·barrier·clock 단위시험
         └─ A05/A06/A08 작업·expert·GPU 연결
             └─ A10 정상성 smoke
                 └─ A11 제한된 실행 + A07 부하 카운터
                     └─ 필요할 때만 A09/G00
                         └─ A12 범위별 READY/BLOCKED 판정
```

기존 GPU union/gap 단위시험이 통과하고 소스가 바뀌지 않았다면 다시 구현하지 않는다. 이번의 추가 초점은 **시간창 정정·작업 연결·측정 간섭**이다. 모든 기능을 동시에 켜는 profiler 한 세션으로 대체하지 않는다.

---

## 4. A00~A01 — 기존 자료의 재집계부터 완료한다

### 4.1 원자료 확보 목록

다음 경로는 IDE_074 기록에서 확인한 위치다. 실제 존재·권한·해시를 먼저 검사한다.

```text
shadow_assists/features/IDE_074/
  FULL_REPORT.md, PLAN_RESOLVED.md, CONFIG_DIFF.md, WORK_LOG.md
  ARTIFACT_INDEX.csv, SHA256SUMS.txt, PUBLISH_RECEIPT.md
  profiles/probe_map.md
  profiles/observer_windows.csv, observer_overhead.csv
  profiles/corrected_metrics.csv

eval/results/IDE_074_20260917/
  state/execution_events.jsonl
  state/RUN_STATE.json
  qwen/OPT4/B1_131834/{B_r1,B_r2,B_r3,P0_r1,P0_r2,P0_r3}/
  qwen/OPT4/B2_132633/{P1_r1,P1_r2,P1_r3}/
  qwen/OPT4/B3_133528/{P2_r1,P2_r2,P2_r3}/
  qwen/OPT4/F1_OFF_134055/{F1_C1_OFF,F1_LONG_OFF}/
  qwen/OPT4/F1_ON_134513/{F1_C1_P1,F1_LONG_P1}/

서버에 보존된 원자료의 호스트 경로:
  /home/mystous/.cache/huggingface/kt/ide074/<boot>/kt_evt.csv
  /home/mystous/.cache/huggingface/kt/ide074/<boot>/kt_evt.csv.map
  /home/mystous/.cache/huggingface/kt/ide074/<boot>/profiles/*.trace.json.gz
```

세션별 `metrics.json`, `dependency_metrics.csv.gz`, `gpu_layer_timeline_TP0.csv.gz`, `requests.jsonl`, `perf_stat.txt`, `pcm_memory.csv`, CPU/GPU 시계열을 읽는다. 큰 F1 파일은 원격 Git에 없고 서버에만 있을 수 있다. 첨부 보고서만으로 원시 사건을 복원했다고 표시하지 않는다.

`artifact_availability.csv`에 `path, bytes, sha256, accessible, parsed, reason`을 기록한다. 해시가 다르면 즉시 삭제하지 말고 source variant를 구분한다. 서버 대형 trace를 가져오기 전에 용량·보존 정책을 확인한다.

### 4.2 PCM 시각 파싱과 경계 표본 처리

**현재 검토된 결함:** `13:37:59.527` 형식에 초 단위 포맷을 사용하고 파싱 실패를 `None`으로 바꾼 뒤 창 내 표본에 포함했다. `[A074]` 6.1장, `[C-RENDER]`, `[C-PCM]`.

신규 분석기는 다음 계약을 지킨다.

- 소수초를 보존하고 명시적인 `clock_domain`·시간대를 받는다. PCM의 로컬 시각, 서버 로그의 UTC, 클라이언트 ISO 시각을 문자열 그대로 비교하지 않는다.
- 세션 메타데이터에서 PCM 시간대를 확정한다. 장비 위치가 한국이라는 이유만으로 로그 시간대를 자동 추정하지 않는다.
- 파싱 실패는 `INVALID_TIMESTAMP`로 제외하고 원행·이유·개수를 남긴다. 실패를 0시각·현재 시각으로 대체하지 않는다.
- 표본 timestamp가 구간의 시작·끝·순간값 중 무엇을 의미하는지 도구 출력/소스에서 확인한다. 확인 전에는 임의로 1초 구간을 붙이지 않는다.
- 평균 대역폭 표본이면 해당 표본 구간 `I_i=[a_i,b_i)`와 분석창 `W`를 함께 저장한다.
- 전 구간이 W 안에 있는 표본, 경계와 겹치는 표본, W 밖 표본, 해석 불가 표본을 별도로 계수한다.
- 기본 보고는 완전 포함 표본의 시간 가중 평균이다. 경계 표본을 비례 배분한 추정치는 **구간 내 속도 균일 가정**을 표시한 보조값으로만 제공한다.
- 표본 간 gap·중복·역순·시간 변경을 검출한다. 겹친 표본 구간을 단순 합산해 coverage를 부풀리지 않는다.
- `Read`, `Write`, `DRAMRead`, `PMMRead`, 소켓별 값, 시스템 합계가 어떤 메모리 범위를 뜻하는지 헤더로 매핑한다. 시스템 합계와 소켓 합계를 다시 더하지 않는다.

```text
W_sample_full = union(I_i where I_i ⊆ W and sample_valid)
full_sample_coverage = |W_sample_full| / |W|
mean_full = Σ(value_i × duration_i) / Σ(duration_i)

경계 비례 추정(별도 표시):
mean_overlap_est = Σ(value_i × |I_i ∩ W|) / Σ|I_i ∩ W|
가정 = each value_i represents a constant rate over its sample interval
```

두 번째 식을 정확한 부하 중 측정값이라고 쓰지 않는다. 표본 간격이 1초라면 약 0.9ms Cold job 하나의 DRAM 트래픽을 그 표본으로 분리할 수 없다.

**산출물:** `pcm_samples_v2.csv`, `pcm_window_summary_v2.csv`, `timestamp_parse_errors.jsonl`, `window_reconstruction.md`. 기존 155~162GB/s 수준의 평균은 `legacy_collector_window_mean`으로 남기고, 정정된 값과 차이·사유를 병기한다. 유효한 부하 창을 복원하지 못하면 정정 평균은 null이다.

### 4.3 실제 시간창 복원

다음 창을 별개의 이름으로 유지한다.

| 창 | 시작·종료 | 의미 |
|---|---|---|
| `client_process_window` | 벤치 호출 시작~프로세스 종료 | imports/tokenizer/클라이언트 준비 포함 가능 |
| `client_request_window` | 첫 실제 요청 전송~마지막 응답 완료 | 요청 집합의 end-to-end 관측 범위 |
| `server_service_window` | 첫 요청 수신~해당 집합의 마지막 서버 처리 완료 | 서버측 요청 처리 범위 |
| `forward_envelope` | 첫 관련 forward~마지막 관련 forward 완료 | 연산 실행의 외곽 범위. 내부 idle 포함 가능 |
| `phase_intervals` | scheduler가 기록한 DECODE/EXTEND 등의 각 구간 | phase별 측정용. 단일 연속 창이 아닐 수 있음 |
| `collector_window` | 수집기가 실제 데이터를 기록한 시작~끝 | arm 호출 시각과 구분 |
| `drain_window` | 마지막 응답 이후 관련 비동기 작업·기록 종료까지 | 성능 duration과 분리 |

우선순위는 요청별 timestamp/벤치 타이밍 이벤트 → 관련 scheduler/forward 이벤트 → 초 해상도 서버 로그다. 로그에 초만 있으면 정확한 시각이 아니라 해당 초의 범위라는 한계를 저장한다. `t_end−benchmark_duration`으로 역산한 시작은 코드에서 두 시각의 측정 기준이 같다는 증거가 있을 때만 사용한다. 그렇지 않으면 `ESTIMATED_WINDOW`다.

P2_r1의 역사적 기록은 호출 36.155초, benchmark 23.192초, perf 약 40.295초다. 차이가 있다는 사실을 그대로 보존하며 모든 세션에서 13초를 빼는 보정은 금지한다. `[A074]` 6.2장, `[C-MET]`.

### 4.4 복원할 수 없는 값의 처리

- perf가 전체 누적값 하나만 제공했다면, 총 `task-clock`·instructions·cache-misses를 부하시간 비율로 곱해 정확한 구간값을 만들지 않는다.
- request별 timestamp가 없으면 원래 client window와 확인 가능한 forward window까지만 제공한다.
- 기존 trace에 없는 `job_id`·enqueue·buffer epoch를 표의 행 순서로 생성해 실측 ID라고 부르지 않는다.
- eager slot 변경의 timestamp/epoch가 부족하면 기존 `rows=1` 분류를 `UNKNOWN_EAGER_ROWS`로 바꾼다. 정확한 크기는 새 수집 대상이다.
- 데이터가 없는 0과 실제 측정된 0을 구분한다. 경로 없음은 N/A, 계측 누락은 null+reason이다.

### 4.5 기존 GPU와 layer 자료 재집계

기존 union/gap 알고리즘은 회귀시험을 통과시킨 뒤 재사용한다. 실제 GPU operation만 창에 clip하고 장치별 `duration_sum`, `busy_union`, `idle_union`, copy/compute overlap을 보존한다. annotation과 내부 커널을 합산하지 않는다.

새로 할 일은 다음이다.

1. `trace_full`, 잘못 붙였던 `request_window`, 복원한 실제 창을 명확히 구분한다.
2. 소비층별 대기와 생산층별 CPU 비용을 같은 행으로 연결한다.
3. layer0, nonempty cold, empty cold, graph padding, 매핑 불명 행을 구분한다.
4. P1 본 반복과 SMOKE를 분리한다. 반복 단위와 boot 관계를 저장한다.
5. 간격들의 union과 실제 GPU idle의 교집합을 계산하되 회수 가능한 성능 상한이라고 하지 않는다.
6. 기존 graph 내 실제 `bs`·capture rows·CPU qlen을 다른 필드로 유지한다.

**A01 종료 기준:** 새 서버 실행 없이 답할 수 있는 질문마다 정정 결과 또는 복원 불가 이유가 존재하고, 새 실행에서 추가해야 하는 이벤트 목록이 확정돼 있다.

---

## 5. A02 — 지표 이름과 식을 고정한다

### 5.1 기본 timestamp와 경로 구분

소비자 `c`가 실제로 소비하는 생산자 작업을 `p=producer(c)`로 정의한다. L−1이라는 규칙은 기록된 현재 경로의 후보 매핑이며, 경계/전환에서는 실제 ID 연결을 우선한다.

```text
CPU:
  t_go(p)             poller의 입력 준비/go 관측
  t_enqueue(p)        CPU 작업이 큐에 실제 공개되는 시점 또는 그 bracket
  t_dequeue(p)        worker가 해당 작업을 꺼낸 시점
  t_task_start(p)     실제 deferred 작업 함수 시작
  t_task_end(p)       결과가 CPU buffer에 준비된 작업 함수 종료
  t_numa_start/end(p,s) 소켓/서브풀 s의 실제 계산 범위
  t_pub(c)            c가 기다리는 done 신호 공개 시점 또는 bracket

GPU:
  t_d2h_start/end(p,k) 입력 payload k의 실제 복사 범위
  t_hot_ready(c)      결합에 필요한 Hot 결과의 GPU 준비 완료
  t_h2d_start/end(c)  c가 사용할 Cold 출력의 실제 복사 범위
  t_other_ready(c)    필요한 나머지 입력의 준비 완료
  t_combine_start/end(c) 실제 결합 범위
```

`task_end`, `NUMA parent done`, `done public`, `HtoD 완료`, `combine 완료`는 같은 사건이 아니다. 함수 반환이 실제 결과 준비와 같지 않은 backend이면 더 정확한 완료 이벤트를 찾아 정의한다.

### 5.2 canonical v2 지표

| v2 필드 | 식 | 해석 |
|---|---|---|
| `cold_pub_delta_us` | `t_pub(c)−t_hot_ready(c)` | CPU 게시가 Hot보다 빠른지/늦은지. signed 원값 |
| `cold_pub_lateness_us` | `max(0,cold_pub_delta)` | CPU 게시의 양의 지연. 검증 후에만 생성 |
| `gpu_pre_h2d_gap_us` | `t_h2d_start(c)−t_hot_ready(c)` | Hot 완료 후 복사 시작까지의 원 간격 |
| `h2d_duration_us` | `t_h2d_end(c)−t_h2d_start(c)` | GPU activity의 실제 복사 duration |
| `gpu_post_h2d_gap_us` | `t_combine_start(c)−t_h2d_end(c)` | 복사 완료 후 결합 시작까지 |
| `cold_gpu_ready_us` | `t_h2d_end(c)` | 이 복사가 유일한 준비 조건일 때의 Cold GPU ready |
| `cold_gpu_ready_lateness_us` | `max(0,cold_gpu_ready−max(hot_ready,other_ready))` | GPU 결과 준비 차이. CPU 원인으로 자동 귀속 금지 |
| `hot_gpu_ready_lateness_us` | `max(0,hot_ready−max(cold_gpu_ready,other_ready))` | Hot ready의 차이. 존재 경로에서만 |
| `combine_schedule_gap_us` | `combine_start−max(all required ready)` | 필요한 입력이 준비된 뒤의 시작 간격 |
| `go_to_enqueue_us` | `enqueue(p)−go(p)` | poller/입력 처리에서 큐 공개까지 |
| `enqueue_to_start_us` | `task_start(p)−enqueue(p)` | 공개 이후 대기. 선행작업·OS 등은 별도 분해 |
| `deferred_service_span_us` | `task_end(p)−task_start(p)` | parent 작업 span. 순수 행렬 계산만은 아님 |
| `producer_end_to_pub_us` | `pub(c)−task_end(p)` | 계산 완료와 소비측 done 게시 사이 |
| `producer_end_to_gpu_ready_us` | `cold_gpu_ready(c)−task_end(p)` | CPU 실제 결과 준비 이후 GPU 준비까지 |
| `pub_to_h2d_start_us` | `h2d_start(c)−pub(c)` | GPU의 선행 Hot 진행 시간도 포함 가능 |
| `overlap_budget_us` | `hot_ready(c)−go(p)` | CPU가 Hot 준비 전까지 가진 관측 budget |

복사가 Hot 뒤 같은 stream에서 수행되면 CPU가 먼저 끝나도 `cold_gpu_ready > hot_ready`일 수 있다. 따라서 `cold_gpu_ready_lateness`만으로 CPU-bound라고 분류하지 않는다. `cold_pub_delta`, 전송, stream 순서를 함께 본다.

### 5.3 옛 지표와의 호환

`metric_definitions_v2.yaml`에 기존 이름·기존 식·신규 이름·신규 식·버전을 기록한다. 예를 들어 기존 `cold_ready_lateness_us`는 `legacy_cold_ready_lateness_us`로 보존하고 CPU 게시 지연이었음을 명시한다. 기존 `exposed_wait_us`는 floor를 빼지 않은 원 간격이었다. 이 문서에서는 `gpu_pre_h2d_gap_us`를 사용한다.

10~11µs를 전 배치 공통 floor로 빼지 않는다. 기준 간격을 추정하는 별도 진단을 하더라도 같은 graph·payload·stream 경로에서 얻은 조건부 분포를 원값 옆에 제시하고, 정확한 순수 CPU wait로 재명명하지 않는다.

### 5.4 동일 사건의 항등식과 오차 검사

같은 producer/consumer에서 다음을 계산한다.

```text
Q = task_start(p) − go(p)
C = task_end(p) − task_start(p)
D = pub(c) − task_end(p)
B = hot_ready(c) − go(p)

signed_pub_lateness = Q + C + D − B
residual = (pub(c) − hot_ready(c)) − (Q + C + D − B)
```

동일 timestamp를 사용했는데 residual이 단위/반올림 허용 범위를 넘으면 ID·단위·구현 오류다. 식이 맞는다는 사실만으로 매핑의 진실성이 입증되는 것은 아니므로 별도 ID 검증도 필요하다. 각 항목 p50을 더해 이 관계를 재구성하지 않는다.

### 5.5 분모와 표본 코호트

최소 다음 집단을 분리한다.

- `cold_present_nonempty`: 실제 Cold 기여를 소비했고 유효 assignment가 있음.
- `cold_path_empty`: 작업/신호 경로는 있지만 유효 Cold assignment는 0.
- `no_cold_consumed`: 설계상 Cold를 소비하지 않는 경계. layer0 여부는 코드로 확인.
- `padding_only_or_mixed`: capture padding의 포함 여부를 추적.
- `unknown_path_or_mapping`: 누락·불명 매핑. 다른 집단에 넣지 않음.

지연 비율은 `late_count / valid_eligible_count`로 보고하고 분자·분모·clock-indeterminate 수를 함께 기록한다. 표본 수 비율, step 시간 중 간격 비중, GPU idle 비중은 다른 지표다. 정확히 연결한 이벤트가 없는 행을 분모만 채우는 데 사용하지 않는다.

---

## 6. A03 — 관측 대상의 대기열을 바꾸지 않는 기록기

### 6.1 기존 방식의 제한

IDE_074 계측은 deferred 앞뒤 stamp 태스크를 FIFO에 추가했다. 시작·종료 시각을 얻었지만 원래의 task 수와 큐 부하도 달라졌다. `[A074]` 2.2장, `[C-PROBE]`. 새로운 계측은 **실제 작업 함수 내부의 entry/exit에서 기록**하는 방향을 우선한다. 단순히 stamp 태스크 개수만 줄이고 순수 원래 스케줄이라고 설명하지 않는다.

### 6.2 구현 요구

| 영역 | 요구 사항 | 검증 증빙 |
|---|---|---|
| 작업 수 | 모델 작업 큐에 기록 전용 task 추가 금지 | OFF/ON의 업무 task 종류·개수 비교 |
| 기록 경로 | 사전 할당한 bounded buffer 사용 | buffer 크기·record 크기·할당 위치 |
| 기록 비용 | hot path에서 파일 I/O·stdout·JSON 직렬화·동적 메모리 할당 금지 | 소스 diff와 code map |
| 경합 | 전 워커가 같은 로그 mutex를 잡는 구조 지양 | thread-local/per-thread 구조와 병합 규칙 |
| 수명 | mutable packet/slot 포인터만 저장하지 말고 epoch·ID·필수 메타데이터를 스냅샷 | slot 재사용 시험 |
| NUMA | 기록 메모리의 배치 때문에 기존 model buffer 정책을 바꾸지 않음 | 로그 buffer 정책과 RSS 변화 |
| 버퍼 초과 | overwrite/drop 여부를 명시하고 누락 수 기록 | attempted/written/dropped 계수 |
| flush | 본 부하 밖의 안전 지점에서 수행 | 시작·종료·flush 구간 분리 |
| 종료 | 작업·로그 producer 종료 확인 후 병합·저장 | 종료코드·파일 해시·parse 검사 |
| OFF 경로 | 비활성 시의 분기·추가 state도 기록 | 새 바이너리 OFF 대조 |

전용 수집 스레드가 필요하면 그 CPU 시간·affinity·메모리를 observer 비용으로 남긴다. 기존 non-KT 스레드의 배치를 임의 변경하지 않는다. 새 C++ atomics, cache-line 공유, record ownership이 오히려 false sharing이나 race를 만드는지 검토한다.

### 6.3 태스크 경계에서 직접 기록할 항목

기존 `TaskQueue`의 실제 dequeue/dispatch 지점과 deferred 함수 경계에 아래 이벤트를 넣는다. **이름은 신규 제안이며 실제 타입과 구현 위치를 확정해야 한다.**

```text
TASK_ENQUEUE_BEGIN / TASK_VISIBLE / TASK_ENQUEUE_END
TASK_DEQUEUE
TASK_EXEC_BEGIN / TASK_EXEC_END
DEFERRED_COMPUTE_BEGIN / DEFERRED_COMPUTE_END
NUMA_SUBJOB_BEGIN / NUMA_SUBJOB_END
DONE_STORE_BEFORE / DONE_STORE_AFTER
RECORD_DROP_COUNT / SESSION_LOG_FLUSH
```

`TASK_VISIBLE`이 lock-free queue 내부의 명확한 linearization point를 의미한다면 그 지점을 설명한다. 정확한 단일 timestamp를 찍을 수 없으면 enqueue 전후 bracket을 사용하고 측정 오차에 포함한다. 함수 진입/반환을 실제 공개 순간으로 오인하지 않는다.

기존 `done` store 후에만 timestamp를 찍으면 GPU가 이미 플래그를 관측한 뒤 CPU timestamp가 찍힐 수 있다. 이 경우 `HtoD start < 기록된 done time`이 곧 clock 오류는 아니다. store 전후 bracket과 memory-order 계약을 남기고, 계측을 위해 기존 memory order를 변경하지 않는다.

### 6.4 기록 모드의 분리

| 모드 | 내용 | 목적 |
|---|---|---|
| `OFF` | 새 상세 기록 비활성, 공통 경량 모니터만 | 새 바이너리 기준 대조 |
| `CORE` | 모든 필요한 job의 최소 ID·queue/start/end·done·lifecycle | FIFO·ID·저간섭 확인 |
| `CORR` | CORE + 제한된 GPU activity/ready/step·rank 연결 | CPU↔GPU 소비 지연 확인 |
| `RESOURCE` | CORE + 계층화 CPU 단계 표본 + perf/PCM | 실행 단계와 부하 자원 진단 |
| `FOCUS` | 조건부로 특정 phase/tail/symbol 하나만 심화 | 원인 공백 한 가지를 닫기 |

이 모드들은 기존 P1/P2와 같은 이름이 아니다. `observer_config.json`에 구체적인 차이를 저장한다. 계측 모드 선택은 기존 Hot/Cold 실행 경로 선택과 독립이어야 한다.

같은 부팅 내 전환은 모든 관련 요청·deferred 작업이 drain된 뒤에만 한다. 기존 API나 안전한 host-side 게이트로 전환할 수 없으면 별도 부팅을 사용한다. 전환 편의를 위해 CUDA graph의 실행 의미·동기 순서를 변경하지 않는다.

### 6.5 샘플링 원칙

- CORE의 기본 수량·ID·drop 카운터는 가능한 한 전수 유지한다. 상세 expert/단계 기록만 표본화한다.
- 상세 표본의 선택 단위는 가급적 **완결된 parent job 또는 replay**다. start는 있고 end는 없는 선택을 만들지 않는다.
- 고정 stride64만 사용하지 않는다. 층·phase·배치 크기별 층화와 고정 seed hash 선택을 사용한다.
- `sample_selected`, `sample_probability`, `sample_reason`, `sampling_seed`, 전체 eligible 수를 저장한다.
- graph별/단계별 선택확률이 다르면 raw 표본 분포와 가중 추정 분포를 구분한다. 표본 percentile을 곧바로 모집단 percentile이라고 하지 않는다.
- 지연이 큰 사건만 남기는 tail-trigger 자료와 균등/층화 표본을 분리한다. tail-trigger 표본으로 전체 평균·p95를 계산하지 않는다.
- GPU trace는 지원되는 범위에서 완결된 여러 replay와 필요한 경계 여유를 수집한다. 첫 forward만 측정해 steady decode를 대표시키지 않는다.
- 준비·초기 decode·중간 큰 batch·drain의 작은 batch 중 무엇이 포함됐는지 기록한다. 세션 전체 trace가 필요 없으면 짧게 수집하되 부분 coverage를 숨기지 않는다.

### 6.6 기록기의 단위·부하 없는 검사

실제 모델 부팅 전 합성 queue 작업으로 다음을 확인한다. 합성 결과는 모델 성능으로 보고하지 않는다.

- 작업 순서·종류·결과가 OFF와 CORE에서 동일한가?
- producer가 slot을 재사용해도 과거 로그가 변경되지 않는가?
- buffer overflow 때 drop이 정확히 기록되고, 기록기 때문에 원래 작업이 멈추지 않는가?
- 여러 NUMA worker의 parent ID가 섞이지 않는가?
- start/end pair와 thread ID가 일관적인가?
- clock 읽기·record 쓰기 비용을 따로 측정해 단위·표본 수를 보존했는가?
- 계측 자체의 나노초/마이크로초 비용을 모델의 병목 수치로 제시하지 않았는가?

---

## 7. A04 — 실제 요청 부하, collector 창, 시계를 정렬한다

### 7.1 준비 완료 barrier

기존처럼 collector 시작 직후 별도 Python 벤치 프로세스를 실행해 imports·tokenizer 준비가 측정창에 섞이게 하지 않는다. 다음 상태 기계를 구현한다.

```text
SERVER_READY → WARMUP_DONE → ENGINE_CACHE_FLUSHED
CLIENT_READY (입력/토크나이저/연결 준비 완료, benchmark 요청 미전송)
COLLECTORS_READY (지원 이벤트·권한·기록 버퍼 확인)
ARMED → START_SIGNAL → REQUESTS_SENT
LAST_RESPONSE → RELEVANT_WORK_DRAINED → COLLECTORS_STOPPED → LOGS_FLUSHED
```

`CLIENT_READY` 전에 성능 요청을 보내지 않는다. 준비 과정의 HTTP healthcheck는 본 요청 집합과 다른 ID로 기록한다. 첫 요청 전송·첫 서버 수신·첫 forward는 서로 다른 사건이며 각각 저장한다.

barrier 자체는 host-side orchestration이다. 레이어마다 CUDA 동기화를 추가하는 방식으로 구현하지 않는다. collector가 시간 구간 gate를 지원하지 않으면 전체 기록을 남기고 정확한 경계 이벤트로 사후 자른다. 그 지원 여부는 실제 설치 도구에서 검증한다.

### 7.2 필수 시간 기록

`window_events.jsonl`의 최소 필드는 다음과 같다.

```text
campaign_id, boot_id, session_id, event_name,
clock_domain, timestamp_ns, wall_time_iso8601,
process_id, thread_id, request_id(optional), step_id(optional),
source, precision_ns, uncertainty_ns, validity
```

필수 사건:

```text
client_init_begin/end
client_ready
collector_arm / collector_actual_first_sample
start_signal_send/receive
first_request_sent / first_request_received
first_forward_begin
last_response_received
last_related_forward_end
last_related_deferred_end
last_related_gpu_consume_end
drain_confirmed
collector_actual_last_sample / collector_stop
log_flush_begin/end
```

모든 사건을 하나의 프로세스에서 찍었다고 가정하지 않는다. 실제 원본 clock과 변환 방법을 보존한다. `nvidia-smi utilization=0`을 두 번 확인하는 것은 보조 신호일 뿐, 모든 비동기 job·GPU 소비 완료를 입증하는 유일한 방법으로 사용하지 않는다.

### 7.3 collector별 coverage

```text
intersection = W_requested ∩ W_collected_valid
coverage = |intersection| / |W_requested|
```

이 식의 `W_collected_valid`는 샘플이 실제로 유효한 interval들의 union이다. collector의 시작·종료 프로세스 시각만 넓다고 coverage 100%가 아니다.

각 collector에 대해 다음을 저장한다.

| 필드 | 의미 |
|---|---|
| `requested_window_id` | 어떤 부하/phase 창을 설명하려는가 |
| `actual_start/end` | 데이터가 실제로 유효한 범위 |
| `coverage_fraction` | 유효 교집합 비율 |
| `boundary_uncertainty` | 시작·종료 불확실성 |
| `sample_interval` | 자원 표본의 시간 해상도 |
| `missing_intervals` | 유실·실패 구간 |
| `target_scope` | PID/TID, socket, device, host 전체 등 |
| `clock_alignment_id` | 적용한 clock 변환의 식별자 |

### 7.4 clock 정렬과 정밀도

CPU 단계의 duration은 가능한 한 동일한 monotonic domain에서 계산한다. 기록된 GPU activity는 해당 수집기가 변환한 host domain의 시각일 수 있으므로 `baseTimeNanoseconds`와 event `ts`의 단위·정렬 의미를 코드로 확인한다.

**신규 요구:**

1. 원 timestamp는 정수 단위를 사용한다. 큰 epoch timestamp를 float로 변환한 뒤 차감해 미세 정밀도를 잃지 않는다. 소수 단위 원자료는 decimal 또는 정수 변환 계약을 정한다.
2. CPU monotonic↔realtime anchor를 세션 전후와 필요 구간에 기록한다. clock jump/drift가 있으면 단일 offset으로 덮지 않는다.
3. GPU↔host, GPU↔GPU 변환은 지원되는 clock correlation 자료를 활용한다. 없는 동기화 정보를 생성하지 않는다.
4. 검증을 위한 host-device 왕복/anchor는 부하 밖에서 수행한다. 부하 중 장치 동기화는 사용하지 않는다.
5. offset·drift 추정, anchor 수, 잔차 분포, 측정 해상도, timestamp 기록 비용을 저장한다.
6. 순서 검사 통과만으로 clock 오차가 0이라고 하지 않는다. 일정 offset이 있어도 순서 검사는 통과할 수 있다.
7. cross-clock 지연이 불확실하면 GPU 내부 `hot→HtoD`와 CPU 내부 queue/service 값은 각자의 유효 범위에서 계속 보고한다.

### 7.5 오차 구간을 이용한 판정

이벤트 시각을 `t ∈ [t_low,t_high]`로 표현할 수 있으면 차이의 범위는 다음과 같다.

```text
Δ = A − B
Δ_low  = A_low  − B_high
Δ_high = A_high − B_low

Δ_low > 0  : A가 늦었다는 방향이 오차 범위 밖에서 확인됨
Δ_high < 0 : A가 먼저였다는 방향이 오차 범위 밖에서 확인됨
그 외      : CLOCK_INDETERMINATE
```

통계적 잔차에서 추정한 오차 범위는 보장된 절대 경계와 구분한다. 경험적 범위만 있으면 그 추정 방법을 표시한다.

기존 −5µs 허용 정책을 그대로 통과 기준으로 계승하지 않는다. 0/1/2/5µs 등 사전 등록한 민감도 시나리오에서 late 비율·분포가 얼마나 달라지는지 보고할 수 있지만, 임계값을 바꿔 통과 결과만 선택하지 않는다. 기존 P1_r3 1,659건은 별도로 재분류한다.

---

## 8. A05-1 — producer/consumer, slot epoch, graph replay를 확정한다

### 8.1 기본 식별자

```text
campaign_id / boot_id / session_id / attempt_id
host_id / process_id / thread_id / tp_rank / gpu_uuid
request_id / request_token_position / scheduler_step_id
scheduler_phase_raw / phase_normalized
actual_batch_size / capture_rows / cpu_qlen
layer_id / producer_layer_id / consumer_layer_id
job_id / parent_job_id / subjob_id / task_seq
graph_id / graph_replay_id / capture_generation
ring_slot / slot_epoch / buffer_id / buffer_generation
expert_id_logical / expert_id_physical
path_present / cold_nonempty / capture_padding_present
```

scope가 다른 ID를 한 필드에 넣지 않는다. `graph_id`가 같다는 것만으로 같은 replay가 아니며, slot 번호가 같다는 것만으로 같은 작업도 아니다. `request_id`는 batch에 여러 개 연결될 수 있다.

### 8.2 producer→consumer 연결

기록된 deferred8 구조는 일반적인 동층 결합을 가정하지 않는다. 현재 경로의 기본 관계는 생산층 L−1의 deferred 결과를 소비층 L이 사용하는 것으로 설명돼 있다. `[A074]` 3.2장, `[C-PROBE]`, `[C-DEP]`.

새 로그에는 가능하면 다음을 직접 저장한다.

```text
producer_job_id
producer_layer_id / producer_step_id
output_buffer_id / output_buffer_generation
consumer_layer_id / consumer_step_id
consumer_combine_id
done_task_seq
consumer_h2d_id / consumer_gpu_activity_id
mapping_method / mapping_validity
```

직접 ID를 GPU activity에 삽입할 수 없으면 다음 다중 근거를 함께 확인한다.

- graph replay 경계와 slot별 실제 go 관측 횟수
- layer별 packet/capture map과 graph node mapping
- bytes·payload·dtype·stream 순서
- producer 완료→done→HtoD→combine의 부분순서
- 이벤트 수와 epoch 연속성
- sampled buffer generation 또는 별도 정상성 검사

다중 근거로 연결해도 명시적 ID가 아니라면 `VALIDATED_HEURISTIC_MAPPING`처럼 한계를 남긴다. 서로 개수가 같다는 이유만으로 정렬된 두 배열을 zip해 확정 연결하지 않는다. 중간 누락이 발생한 지점 이후는 명확한 anchor로 재동기화할 때까지 unresolved 처리한다.

### 8.3 graph capture와 실제 replay

Python의 layer 함수는 capture 때만 실행될 수 있다. replay별 시각을 capture 시각으로 복제하지 않는다. 실제 replay dispatch에서 host metadata를 남기고, GPU activity의 graph/replay 범위와 연결한다. 해당 정보가 없는 CPU/CUDA API를 이미 지원하는 것처럼 이름만 추가하지 않는다.

다음 전환을 정상성 검사에 포함한다.

| 전환 | 확인 사항 |
|---|---|
| eager prefill → graph decode | 이전 eager slot/map이 graph 작업에 잘못 붙지 않는가 |
| 큰 capture rows → 작은 rows | padding과 실제 유효 row 수가 분리되는가 |
| graph A → graph B → graph A | graph별 epoch/replay 연속성이 유지되는가 |
| 같은 graph 반복 | slot epoch가 덮어쓰기 없이 증가하는가 |
| 요청 일부 종료 후 batch compaction | row→request/token 대응이 갱신되는가 |
| session 경계 | 이전 세션 in-flight 기록이 새 세션으로 잘못 집계되지 않는가 |

### 8.4 eager slot map 정정

`map[slot]=마지막 rows` 방식은 폐기한다. 다음 중 실제 구현 가능한 방법을 선택한다.

- job enqueue 때 `layer, phase, qlen, valid_rows, capture_rows, epoch`를 immutable record로 저장.
- 또는 `map_version/effective_from/effective_to/slot_epoch`로 유효 구간을 가진 mapping table 사용.

CPU 이벤트와 map 업데이트가 다른 스레드에서 일어나면 단순 timestamp 근접 검색보다 공통 generation/epoch를 우선한다. map 업데이트가 실행 후 기록됐을 수 있는지 점검한다. 애매한 매핑은 가장 가까운 행으로 자동 보정하지 않는다.

### 8.5 경계와 exactly-once 검사

다음은 검사할 항목이지 현재 오동작이 확인됐다는 뜻이 아니다.

- 첫 층의 Cold 입력이 실제로 없는지, 초기화된 중립 buffer인지, 이전 step의 결과인지 코드로 구분.
- 마지막 층에서 생산한 deferred 결과가 어디에서 소비되는지 또는 설계상 미소비인지 확인.
- step/요청 종료 후 남는 deferred 결과의 flush/discard 규칙 확인.
- CPU의 결과 쓰기, done 공개, HtoD 읽기, buffer 재사용 순서 확인.
- 중복 consume, stale generation, producer 없는 consume, overwrite-before-read 검출.
- graph 전환과 empty-cold 경로에서도 동일 규칙 확인.

수량 보존은 경계 재고를 포함한다.

```text
inflight_start + produced
  = consumed + explicitly_discarded_by_verified_contract + inflight_end
```

위 수량식만으로 correctness가 입증되지는 않는다. 같은 수의 다른 작업이 소비될 수 있으므로 ID별 매칭도 검사한다. `HtoD 완료`만으로 모델 결합에서 exactly-once 소비됐다고 하지 않는다. 실제 combine 연결이 없으면 `CONSUMPTION_PARTIALLY_OBSERVED`다. 현재 deferral 의미와 표준 모델의 수치적 동등성은 별도의 정확성 문제로 남긴다.

---

## 9. A05-2 — FIFO 대기와 CPU 서비스 시간을 분리한다

### 9.1 답해야 하는 구체적인 질문

현재 `go→start≈0.65~0.68ms` 중 얼마가 다음에 해당하는가?

| 분류 | 필요한 직접 근거 |
|---|---|
| poller 처리 / enqueue 지연 | go 관측·enqueue 공개·poller TID |
| 앞선 deferred 계산의 잔여 시간 | 실행 중이던 task ID와 start/end |
| 큐에 이미 들어온 다른 업무 task | enqueue sequence, task kind, 앞선 task 범위 |
| done-setter/제어 task | task kind, task seq, 실행 duration |
| 입력/버퍼/명시적 선행 조건 대기 | dependency-ready 이벤트와 이유 |
| worker dequeue/dispatch 공백 | 큐에 runnable task가 있는데 worker가 시작하지 않은 범위 |
| OS deschedule 또는 runnable 대기 | 해당 TID의 스케줄링 이벤트 또는 thread CPU-time 대조 |
| 관측기 비용 | 기록 전용 task 없음의 증빙, logger 비용·drop |
| 미분리 | 위 근거가 없는 구간은 `unattributed`로 보존 |

이미 다른 deferred 계산을 실행 중이라서 기다린 시간은 OS 스레드 깨우기 비용으로 부르지 않는다. 측정 전부터 “큐 최적화가 답”으로 정하지 않는다.

### 9.2 전수 최소 queue 이벤트

실제 TaskQueue의 유형과 직렬 소비자 수를 먼저 확인한다. 기록된 “단일 FIFO”가 parent 수준인지 NUMA 하위 pool까지 포함하는지 구분한다.

```text
queue_id
queue_kind
task_seq
task_kind = deferred / immediate / done_setter / control / other
job_id
producer_job_id (done_setter가 어떤 결과를 공개하는지)
enqueue_before_ns / enqueue_visible_ns(optional) / enqueue_after_ns
dequeue_ns
exec_start_ns / exec_end_ns
queue_depth_at_enqueue / queue_depth_at_dequeue
running_task_seq_at_enqueue
predecessor_task_seq
input_ready_ns / dependency_ready_ns(optional)
worker_tid / worker_cpu_id / numa_pool_id
```

queue depth를 세기 위해 전체 큐를 순회하거나 추가 mutex를 잡아 작업을 지연시키지 않는다. 실제 효율적인 counter가 없다면 depth는 선택 항목으로 두고 enqueue/dequeue sequence에서 사후 복원한다.

### 9.3 서로 배타적인 queue-window 분해

각 job p의 대기창을 `Wq=[enqueue_visible,task_start)`로 둔다. 같은 queue의 선행 task 실행 interval을 이 창에 clip한다.

```text
queue_wait_wall = |Wq|
predecessor_exec_union = |union(known predecessor execution intervals ∩ Wq)|
queue_gap_unattributed = queue_wait_wall − predecessor_exec_union
```

`queue_gap_unattributed`를 전부 OS scheduler overhead라고 부르지 않는다. 입력 준비, lock 경합, 관측되지 않은 제어 작업, worker dispatch, clock/bracket 오차가 포함될 수 있다.

세부 분류는 겹침을 없앤 다음 합계한다. 선행 deferred 작업의 NUMA child span과 parent span을 동시에 더하지 않는다. 스레드가 선행 task 안에서 deschedule됐다면 그 시간은 **선행 서비스 span의 일부**다. queue 관점의 선행 실행 지연과 CPU 실행 관점의 OS 지연을 각각 보고하되 한 E2E 합계에 중복 더하지 않는다.

### 9.4 최장 선행 작업의 영향과 전파

지연 소비 사건 c마다 다음을 저장한다.

- 직접 생산자 p의 queue wait·service·publish·HtoD.
- p가 enqueue될 때 실행 중인 task와 이미 대기 중인 task 목록/범위.
- 그중 c의 게시 지연과 시간적으로 연결되는 선행 작업.
- 지연 전파가 시작된 layer/step과 회복된 지점.
- 동일 생산층의 반복별 분포와 큰-batch/작은-batch 차이.

`fifo_dependency_edges.csv`에 task 순서/producer-consumer edge를 저장한다. 단순한 `layer=L−1` 표 하나로 모든 step 경계를 연결하지 않는다.

### 9.5 산출물과 종료 기준

**산출물:** `cpu_tasks.csv.gz`, `fifo_dependency_edges.csv.gz`, `queue_wait_breakdown.csv.gz`, `producer_consumer_metrics_v2.csv.gz`, `fifo_casebook.md`.

`fifo_casebook.md`에는 큰-batch 전형 사건, p95 부근, 최대 tail, 작은-batch 숨겨진 계산, empty-cold를 각각 최소 한 건 포함한다. ID·원 timestamp·부분순서·어떤 구간이 미분리인지 제시한다. 사례는 전체 통계의 대체가 아니다.

**종료 기준:** 대표 지연 생산자들에 대해 “선행 업무 task 점유”, “직접 service span”, “전달/dispatch 공백”, “미분리”의 크기를 같은 사건 단위로 구분할 수 있다. 선행 작업 대기를 근거 없이 독립적으로 제거 가능한 비용으로 바꾸지 않는다.

---

## 10. A06 — 지연 job의 expert rows·실제 CPU 분기·단계를 측정한다

### 10.1 라우팅 수량의 정의

기존 `n_cold_ids`라는 이름만으로 distinct expert 수라고 판단하지 않는다. 실제 코드가 세는 단위를 확인한다.

| 필드 | 정의 |
|---|---|
| `n_valid_token_rows` | padding을 제외한 실제 입력 row 수 |
| `capture_rows` | graph에 캡처된 buffer row 용량 |
| `logical_top_k` | 모델의 논리 선택 수, 현재 8 |
| `n_hot_assignments` | 유효 token-expert 선택 중 GPU에서 처리하는 선택 건수 |
| `n_cold_assignments` | 유효 token-expert 선택 중 CPU에서 처리하는 선택 건수 |
| `n_unique_cold_experts` | 해당 job에서 실제 처리한 서로 다른 logical Cold expert 수 |
| `expert_rows[e]` | logical expert e에 모인 유효 row 수 |
| `expert_rows_padded[e]` | 실제 커널에 전달된 padding 포함 row 수 |
| `n_skipped_or_masked` | sentinel·padding·마스크 등으로 계산되지 않은 선택 |
| `logical_to_physical_map_id` | Hotmap/압축/재배치로 바뀐 expert ID의 변환표 |

라우팅의 top-k 보존 검사는 모델별 유효 선택 계약에 맞게 한다. TP4의 같은 logical assignment를 4배로 더하지 않는다. 같은 Cold expert를 두 NUMA pool이 분할 처리하면 expert 개수와 작업량을 중복 계수하지 않는다.

### 10.2 실제 커널 분기

설정이 `AMXINT4`라는 이유만으로 모든 expert가 AMX로 계산된다고 쓰지 않는다. 표본 job의 실제 분기 지점에 다음 정보를 기록한다.

```text
job_id, expert_logical_id, expert_physical_id,
numa_pool_id, row_count_valid, row_count_kernel,
input_dtype, weight_format, output_dtype,
kernel_path_actual, rb_path_actual,
dispatch_condition_values, dispatch_site,
worker_count_effective, sharding_or_partition_id
```

`kernel_path_actual`은 코드에서 실행한 분기/함수명으로 채운다. 이름을 확정하지 못하면 raw symbol과 `UNKNOWN_BRANCH`를 저장한다. 이번 단계에서 threshold 값을 바꾸어 AMX를 강제하지 않는다.

### 10.3 CPU 단계 경계

아래 명칭은 기존 기록을 계승하되 실제 함수 경계를 확인한다. 계층·포함 관계를 `stage_dictionary.csv`에 저장한다.

| 단계 후보 | 확인할 내용 | 잘못된 해석 방지 |
|---|---|---|
| prepare / routing gather | expert별 입력 구성·인덱스 처리 | CPU 전체 prefill/decode 시간과 구분 |
| cpy_input | CPU pinned 입력→expert 연속 배열 등 실제 범위 | GPU DtoH가 아님 |
| q_input | 입력 양자화 | 전체 INT4 가중치 변환과 구분 |
| up_gate | 실제 up/gate 계산 | parent 전체 service와 구분 |
| activation | 활성화/elementwise | 다른 커널에 fused됐으면 중복 타이머 금지 |
| q_down | down 입력 준비/양자화 | 없는 경로는 N/A |
| down | 실제 down projection | NUMA별 병렬 구간 구분 |
| weight / weighted reduce | top-k 가중합 등 실제 연산 | 이름만 보고 DRAM weight fetch로 부르지 않음 |
| NUMA fork/join | subjob 배포·두 pool 완료 대기 | 각 socket 계산시간 합계가 아님 |
| cross-pool merge | 부분 결과 통합 | CPU DRAM 전송과 자동 동일시 금지 |
| task wrapper overhead | 작업 함수 전후의 잔여 부분 | 알려진 child span과 중복 합산 금지 |

근거: `[A074]` 3·7장, `[C-PROBE]`, `[H073]` 8장. 존재하지 않는 별도 단계 타이머를 임의로 만드는 대신 실제 fused 범위를 하나로 기록한다.

### 10.4 병렬 시간의 집계

같은 parent p에 대해 다음을 모두 구분한다.

```text
parent_service_span = task_end(p) − task_start(p)
numa_critical_done  = max(required numa_end)
numa_completion_skew = max(required numa_end) − min(required numa_end)
numa_busy_union = |union(required numa execution intervals)|
numa_duration_sum = Σ(required numa duration)
worker_cpu_time_sum = Σ(measured worker CPU-time deltas)
```

두 NUMA pool의 단계 p50을 더하거나, parent span에서 서로 다른 표본 단계 중앙값을 빼서 overhead를 만들지 않는다. 단계별 exclusive duration을 만들려면 같은 parent의 중첩 interval 구조로 계산한다. 그렇지 않으면 inclusive span·sum으로 표시한다.

### 10.5 표본 설계

우선 큰 decode 배치의 지연 job을 설명할 수 있는 층화 표본을 수집한다. 다음 층화 변수를 기록한다.

- producer layer, scheduler phase, actual batch/capture rows
- `n_cold_assignments`, unique Cold experts, expert별 rows 분포
- nonempty/empty Cold, 실제 AMX/AVX/RB 분기
- 정상 중앙부, queue가 큰 경우, service가 큰 경우, NUMA tail

층별 최소 표본 수는 원자료의 step 수·기록 비용에 맞춰 실행 전 등록한다. 표본이 부족하면 `INSUFFICIENT_SAMPLES`로 두며, 표본 수를 맞추기 위해 예산 밖 모델 요청을 몰래 추가하지 않는다. CORE 기록에서 tail이 검출되면 이미 보존된 in-memory ring 범위를 사후 추출하는 방식을 우선한다. tail을 얻기 위해 매번 새로운 전체 세션을 실행하지 않는다.

### 10.6 다음 성능 개선에 넘길 비용표

`producer_layer_costs.csv`와 `expert_cost_samples.csv.gz`에는 최소 다음을 넣는다.

| 범주 | 필드 |
|---|---|
| 식별 | manifest, boot, rep, phase, producer layer, consumer layer, job/expert ID |
| 규모 | valid rows, padded rows, assignments, unique experts, actual branch |
| 시간 | queue wait, service, 단계 span, NUMA skew, 게시 지연, HtoD, 소비 간격 |
| 노출 | CPU가 실제로 늦었는지, 해당 간격의 GPU idle 교집합 |
| 신뢰도 | 표본 확률, mapping/clock validity, missing, observer mode |
| 배치 정보 | 현재 GPU 상주 여부, 현재 층별 예산, actual HBM 관련 증빙 |

이 표는 **관측 비용표**다. 같은 job 안의 shared setup·병렬 연산·가중합 비용을 expert마다 중복 배분하지 않는다. expert 하나를 옮겼을 때의 정확한 한계 이득은 관측 표만으로 확정할 수 없다. 다음 단계의 후보 선정과 별도 검증 입력이다.

---

## 11. A07 — 실제 부하 중 CPU·DRAM·NUMA 자원을 측정한다

### 11.1 PID/TID 대상 범위부터 증명

기존 `perf -p`가 스케줄러 4개 PID를 받았다는 사실만으로 모든 KT worker를 포함했다거나, 반대로 전부 제외했다고 단정하지 않는다. 실제 설치 도구의 attach/inherit 의미와 프로세스·스레드 관계를 확인한다.

부하 전·중·후 다음 표를 저장한다.

```text
pid, ppid, tid, thread_name, role,
role_evidence, start_time, exit_time,
affinity_list, current_cpu, physical_core, smt_sibling,
numa_pool, perf_included, inclusion_method, inclusion_window
```

역할은 `KT_parent`, `KT_NUMA_worker`, `poller`, `scheduler`, `tokenizer`, `HTTP`, `observer`, `unknown` 등으로 구분하되 이름 문자열만으로 확정하지 않는다. 생성 코드·기존 affinity·실행 stack 등 실제 근거를 남긴다. 새로 생성되거나 사라진 thread도 기록한다.

**필수 결과:** `thread_role_map.csv`, `perf_target_coverage.csv`, 측정 범위 설정 원문. 동일 TID를 process 전체 집계와 thread별 집계에 중복 더하지 않는다.

### 11.2 CPU 기본 카운터

가능한 기본 카운터는 다음과 같다. 정확한 이벤트명·지원 여부·권한은 현장에서 확인한다. 아래 목록 자체가 설치된 명령의 성공 보장은 아니다.

| 카운터/지표 | 목적 | 한계 |
|---|---|---|
| task-clock 또는 thread CPU time | 벽시계 대비 실제 CPU 실행량 | busy-poll도 CPU 사용량에 포함 |
| cycles / instructions | 실제 부하의 IPC 등 보조 관측 | 낮은 IPC만으로 memory-bound 확정 불가 |
| context-switches | 작업 중 deschedule 가능성의 보조 단서 | 전역 값만으로 어느 job 원인인지 모름 |
| cpu-migrations | 실제 worker 배치 이동 확인 | 고정 worker와 non-KT를 분리해야 함 |
| cache-misses | 캐시 접근 관련 보조 관측 | 이벤트 의미·level·scope 확인 필요 |
| cpu_id·frequency/clock 자료 | NUMA tail과 코어 상태의 동시 기록 | 설정 클럭과 순간 유효 클럭 구분 |
| PMU running/enabled, lost | 카운터 유효성 | multiplex·유실 보정의 한계 기록 |

핵심은 요청 구간의 delta다. 다음 중 하나를 구현한다.

1. collector의 enable/disable 제어를 barrier에 연결해 정확한 창을 계수.
2. 시작/종료 counter snapshot의 차이를 사용하고 경계 오차를 기록.
3. 검증된 interval 출력으로 실제 창 내부의 구간을 집계.

전체 수집 구간의 누적값 하나를 실제 부하 duration으로 나누지 않는다. interval 경계와 요청창이 겹치면 PCM과 같은 완전 포함·경계 구분 원칙을 적용한다.

### 11.3 CPU 사용량의 분모

```text
cpu_equivalents_in_window = Σ(thread CPU time within W) / |W|
```

이 값은 논리 CPU의 시간 사용량에 해당하는 집계이지 “행렬 연산에 유효하게 사용한 물리코어 수”가 아니다. 96 worker는 96개의 thread이고, 실제 affinity와 SMT 관계를 확인하지 않으면 물리코어 개수로 바꿀 수 없다.

다음을 별도로 보고한다.

- KT worker CPU-time, poller CPU-time, scheduler/non-KT CPU-time, observer CPU-time.
- 전체 eligible TID 수와 실제 측정 TID 수.
- affinity의 고유 물리코어 수·논리 CPU 수·SMT 공유 상태.
- runnable이지만 실행되지 못한 구간을 직접 측정했는지 여부.

IPC를 역할별로 계산할 때 분자·분모는 같은 이벤트·같은 창·같은 TID 집합이어야 한다. 개별 thread IPC의 단순 평균을 전체 instructions/cycles와 혼동하지 않는다.

### 11.4 메모리 대역폭과 NUMA 기본값

실제 부하 창에 대해 PCM 또는 검증된 대응 counter로 소켓별 read/write와 host 합계를 측정한다. 도구가 출력하는 채널 수와 실제 DIMM이 채워진 채널 수를 혼동하지 않는다. DIMM/주파수/소켓 토폴로지가 확인되지 않으면 이론 대역폭을 만들지 않는다.

필수 산출:

```text
socket_id, sample_start/end, read_MBps, write_MBps,
unit_definition, counter_scope, full_or_boundary,
request_window_id, overlap_duration,
background_process_summary, valid, reason
```

worker buffer·가중치·입력 pinned memory의 NUMA placement는 가능한 비침습적 방법으로 확인한다. process RSS와 가중치 크기, 페이지 residency와 실제 remote access traffic은 다른 값이다. `/proc`/NUMA 통계가 있다고 remote traffic을 직접 측정한 것으로 표시하지 않는다.

### 11.5 microarchitecture 심화는 조건부

A06의 단계·branch·rows에서 후보가 좁혀졌는데도 원인을 더 구분해야 할 때만 추가한다.

| 추가 측정 | 실행 조건 | 얻을 수 있는 판단 |
|---|---|---|
| 특정 worker의 off-CPU/runnable 추적 | 큰 queue gap이나 NUMA tail이 실제 지연과 연결 | 스케줄링이 해당 사례에 개입했는지 |
| symbols/call-stack 표본 | service span의 큰 부분이 단계 타이머로 설명되지 않음 | 실제 CPU 코드 위치 |
| 지원되는 memory-stall/remote counter | memory/NUMA 개선 후보를 검증해야 함 | 지정 scope의 memory 관련 직접 근거 |
| 제한된 메모리 진단 workload | DRAM 포화 주장을 위해 비교 기준이 반드시 필요 | 해당 접근 패턴 조건의 비교치 |

별도 메모리 진단은 새 workload다. CPU 자원을 사용하며 비용·시도 수를 원장에 기록하고 조건부 예산에서 배정해야 한다. STREAM류 값 하나를 그대로 MoE의 달성 가능 bandwidth 상한으로 적용하지 않는다. 측정 목적에 맞는 접근 패턴·working set·NUMA·thread 조건의 차이를 명시한다.

**현재 기본 계획에는 이런 진단 workload를 자동 추가하지 않는다.** 필요한 권한·PMU 지원이 없으면 `UNMEASURED_MEMORY_STALL` 등으로 닫고, 확인된 CPU 단계 경로만 다음 단계로 전달한다.

### 11.6 수집기 간 충돌과 job 연결 한계

GPU trace, perf record, perf stat, PCM을 모두 동시에 켜지 않는다. 특히 같은 PMU를 소유하거나 multiplex하는 도구는 충돌/측정률을 확인한다. 충돌로 invalid하면 해당 결과를 버리는 대신 상태·원문을 남기고 지원되는 부분을 분리한다.

별도 CORR·RESOURCE 실행의 시간축이나 job 번호를 하나로 합치지 않는다. 동일 manifest에서 비슷한 분포가 보였다는 보조 검증은 가능하지만, “CORR의 이 job에서 RESOURCE의 이 counter가 증가했다”는 인과 연결은 할 수 없다.

---

## 12. A08~A09 — GPU 전송·결합·TP·prefill·tail의 남은 공백

### 12.1 전송 payload와 실제 bytes

기존 기록의 층당 DtoH 4회/HtoD 1회라는 패턴은 출발점이다. 다음을 전송별로 확정한다.

```text
transfer_id, job_id, producer_job_id, consumer_combine_id,
direction, payload_kind, bytes, dtype, shape,
source_buffer_id/generation, destination_buffer_id/generation,
source_device, destination_device,
host_pinned, stream_id, graph_node_id, replay_id,
start/end, correlation_id, validity
```

payload는 hidden state, expert IDs, routing weights, CPU 결과 등 실제 코드의 의미로 구분한다. 확인 불가면 `unknown`이다. 같은 층에 copy가 여러 개 있다는 이유만으로 중복이라고 결론 내리지 않는다.

bytes가 capture rows에 비례한다면 유효 row와 padding을 모두 저장한다. 같은 memcpy가 trace에 여러 annotation으로 나타나도 실제 전송은 한 번만 계수한다. count, bytes sum, duration sum, interval union을 모두 보존한다.

### 12.2 완료 게시·GPU 대기·결합

GPU wait memop가 activity에 직접 나타나지 않을 수 있다. 이런 경우 다음처럼 보고한다.

- `done` 공개: CPU의 store bracket.
- wait 해제: 직접 이벤트가 없으면 정확 시각 미측정.
- HtoD 시작: wait 뒤의 첫 검증된 GPU 작업이라면 해제의 대리 경계.
- HtoD 완료: 해당 Cold 결과의 준비 조건.
- combine 시작/끝: 실제 결합 커널의 준비와 실행.

`pub_to_h2d_start`를 순수 신호 전파 시간으로 이름 붙이지 않는다. CPU가 먼저 끝난 경우 GPU Hot 실행이 끝날 때까지의 시간도 포함된다. 작은 batch에서 이 값이 크더라도 노출 병목일 수 없다.

### 12.3 실제 GPU idle과 의존 간격의 분리

소비자 c의 `G_c=[hot_ready,h2d_start)`를 계산하고, TP0의 실제 operation union을 U라고 둔다.

```text
consumer_gap = |G_c|
other_gpu_ops_inside_gap = |G_c ∩ U|
gpu_idle_inside_gap = |G_c| − |G_c ∩ U|
```

동일 장치·동일 trace·같은 시간창에서 계산한다. 다른 stream이나 다른 요청의 작업이 있으면 소비자는 기다려도 GPU 전체는 idle이 아닐 수 있다. 여러 c의 구간이 겹치면 별도 합계와 union을 구분한다.

이 값은 관측된 idle이다. 해당 의존 관계를 없앴을 때 줄어드는 E2E 시간과 같다고 가정하지 않는다. 나머지 병목·배치 스케줄·자원 경합이 바뀔 수 있다.

### 12.4 Attention·Router·Hot·결합 위치 확정

가능한 기존 graph node/커널 위치로 다음 표를 만든다.

```text
step/phase → layer → attention → router → hot_moe
           → cold return → combine → post_moe_collective
```

위 화살표는 예시다. 실제 모델의 residual·fused collective 순서에 맞게 수정한다. Attention 뒤 collective와 MoE 뒤 collective를 분리하고, fused된 범위는 한 범위로 남긴다. 같은 커널 이름이라도 위치가 다르면 다른 연산일 수 있다.

Hot ready는 필요한 모든 Hot 커널의 완료를 의미한다. 단순 파일상의 마지막 커널이나 가장 마지막에 파싱된 이벤트 하나를 고르지 않는다. 다른 입력이 Hot보다 먼저 준비된다는 전제가 코드로 성립하면 그 증거를 저장하고, 성립하지 않는 경로는 `other_ready`를 직접 수집한다.

### 12.5 TP collective의 조건부 심화

기존 TP1~3 all-reduce duration 약 445µs는 위치 귀속이 미확정이다. `[R074]` 7장. 통신을 다음 최적화 후보로 선택하려는 경우에만 아래를 필수로 완성한다.

| 항목 | 필수 연결 |
|---|---|
| collective 위치 | Attention 뒤 / MoE 뒤 / 그 외 |
| logical collective ID | step, layer, collective ordinal, communicator 또는 동등 ID |
| rank별 dependency ready | collective 입력이 준비된 시점 |
| rank별 kernel start/end | activity 실제 실행 범위 |
| bytes·dtype·algorithm | 실제 적용값 또는 미확인 표시 |
| clock | rank/device 간 정렬 방법·오차 |

rank별 ready/start 분산은 관측할 수 있지만, 모든 rank가 도착한 뒤부터 종료까지를 무조건 순수 링크 전송시간이라고 하지 않는다. 프로토콜·내부 동기화·진행 방식이 섞일 수 있다. 통신 포화를 주장하려면 추가 링크/프로토콜 근거가 필요하다.

TP0 Cold 완료가 늦은 시점과 다른 rank의 collective 대기가 맞물리는지 확인되면, “TP0의 선행 계산 지연과 collective 대기가 연계된다”고 보고한다. 이것만으로 NVLink가 빠르다/느리다의 판정을 내리지 않는다.

### 12.6 Prefill/EXTEND

기존 LONG 결과의 dependency table은 decode다. 먼저 기존 trace에서 실제 `EXTEND` annotation과 eager 작업을 분리하고 A05의 시간별 slot map으로 연결한다.

필수 구분:

```text
request_arrival → scheduler_admission → prefill_chunks
               → first_decode_or_sample → first_response_token
```

chunk별 input token 수, cached token 수, 신규 요청 수, 이미 running인 decode 요청 수를 기록한다. 실제 `scheduler_phase_raw`를 보존하고 normalize 규칙을 명시한다. qlen 임계치만으로 prefill/decode를 나누지 않는다.

추가 LONG 실행은 다음 중 하나에 해당할 때만 예비 셀에 배정한다.

- 기존 eager map으로 prefill 작업의 크기·생산자를 복원할 수 없음.
- 다음 후보가 prefill 공용 CPU 커널을 바꾸어 LONG 영향 검증이 필수임.
- TTFT가 큰 원인을 queue와 계산으로 나눠야 후보를 선택할 수 있음.

이번 최소 추가 계획에서 LONG 반복 대표 성능까지 확보하려고 하지 않는다. 본 최적화 검증 단계에서 같은 manifest의 OFF/후보 비교로 다룬다.

### 12.7 C1·작은 배치·tail

작은 batch에서 Cold가 대부분 숨겨진다는 기존 관측을 보존한다. 추가 C1은 다음 경우에만 수행한다.

- 새 기록기나 timer가 작은 job 시간과 비슷한 크기의 교란을 만들 가능성이 큼.
- 후보가 empty-cold·done 신호·small-row 경로를 변경할 예정.
- 기존 ms급 NUMA tail의 원인이 작은 batch의 p99 문제와 연결됨.

tail 사건은 job·TID·CPU·NUMA·queue predecessor·page fault/스케줄링 자료 등 실제로 수집한 것만 연결한다. 최대값 하나로 전체 병목을 설명하지 않으며, outlier를 이유 없이 삭제하지 않는다. 빈도·전체 시간 기여·요청 p95/p99와의 관련성을 따로 보여준다.

---

## 13. A10 — 정상성·계측 유효성을 먼저 검증한다

### 13.1 모델 부하 전 검사

A01~A08의 parser·ID 연결·queue 분석기는 합성 자료로 검사한다. 실제 GPU/CPU 실행을 재배치하지 않고 기록만 추가했는지 소스 diff를 검토한다. 상세 시험은 부록 B에 정의한다.

### 13.2 부팅당 smoke에 포함할 검사

기존 동일 4개 smoke 요청을 사용한다. 신규 정상성 요청을 추가하면 별도 시도로 원장에 기록한다. 다음을 저장한다.

- 입력 text와 token IDs, request options, seed, template/tokenizer 식별자.
- 생성 text 전체, output token IDs(제공되는 경우), finish_reason, truncation, 오류.
- OFF/CORE/CORR/RESOURCE 간 결과 대응표.
- NaN/Inf·illegal memory access·hang·stale generation·중복 소비 카운터.
- graph/eager 전환, empty-cold, 첫/마지막 layer/step이 검사됐는지 coverage.

텍스트 SHA만 같으면 `TEXT_MATCH_ON_SMOKE`다. token ID가 같으면 그 사실을 별도로 기록한다. 전체 logits를 본 성능 세션에서 덤프하지 않는다. 필요한 수치 검증은 대표 tensor·expert 또는 기존 smoke 범위의 별도 진단으로 수행한다.

### 13.3 이상 발생 시

동일 요청에서 output mismatch·NaN·stale/중복 소비가 생기면 성능 본 측정을 시작하지 않는다. 실패 요청·token IDs·epoch·buffer generation·전체 오류를 보존하고, 계측이 원래 실행 의미를 바꿨는지 우선 점검한다.

무조건 exact match만을 유일한 수치 기준으로 강제하지 않는다. 원래 backend의 비결정성이나 허용 수치 차이를 확인해야 할 수 있다. 하지만 이번은 계측-only 변경이므로 결과 차이를 단순히 “양자화 오차”로 설명하고 통과시키지 않는다. 허용 기준과 근거가 없으면 정상성 게이트는 미통과다.

현재 deferred8 기준의 수치 의미가 표준 모델과 동일하다고 새로 입증한 것은 아니다. 정상성 검증은 **동일 기준 구성에서 계측이 추가한 변화**와 buffer 계약을 검증하는 범위로 적는다.

### 13.4 유효성은 항목별로 분리

`valid=true` 하나로 모든 데이터를 승인하지 않는다.

```text
execution_valid
config_valid
workload_valid
window_valid
clock_valid_by_pair
mapping_valid
producer_consumer_valid
counter_scope_valid
counter_running_valid
sampling_valid
artifact_valid
observer_distortion_status
```

예를 들어 perf 창이 invalid여도 benchmark 처리량이 자동 invalid는 아니다. CPU↔GPU clock 정렬이 불명이어도 GPU 내부 간격은 유효할 수 있다. 반대로 요청이 HTTP 200으로 완료됐다는 이유만으로 모델 정상성·측정 정확성이 모두 통과한 것은 아니다.

---

## 14. A11 — 실행 배분과 잔여 예산

### 14.1 자동으로 신규 24세션을 부여하지 않는다

IDE_074 보고 원장은 **세션 17/24, 부팅 7/12, 품질 4/80**을 기록했다. 보고 시점 기준 산술 잔여는 **세션 7회, 부팅 5회, 기록된 품질 항목 76회**다. `[R074]` 9장.

실제 실행 시작 전 최신 원장을 읽어 이후 실행이 추가됐는지 확인한다. smoke/품질 요청의 기존 계수 범위도 확인한다. 품질 4라는 기록만으로 기존 모든 smoke가 80문항 예산 밖이었다고 임의 해석하지 않는다. 미계수 시도가 있으면 먼저 원장을 정정한다.

이 문서는 **기존 잔여분 안에서 마칠 최소 추가 측정안**을 기본으로 한다. 독립된 새 캠페인 예산이 명시적으로 확정된 경우에만 다른 배분을 사용하며 기존 기록과 따로 집계한다. 이번 파일 생성은 서버 실행·게시·신규 예산 사용을 이미 수행했다는 뜻이 아니다.

### 14.2 권장 최소 실행안: 본 측정 6회 + 조건부/재시도 1회

다음은 신규 배분 제안이다. A01의 재집계로 충분한 질문은 생략하고 그 이유를 남길 수 있다.

| 셀 | 부팅 그룹 | 모드 | workload | 목적 | 신규 세션 수 |
|---|---|---|---|---|---:|
| S1 | B_OFF_OPEN | OFF | PROBE128 | 새 동일 바이너리의 시작 대조 | 1 |
| S2 | B_CORE | CORE | PROBE128 | 기록기만의 영향·전수 FIFO 구조 | 1 |
| S3 | B_CORE | CORR | PROBE128 | CPU↔GPU 연결의 본 표본 1 | 1 |
| S4 | B_CORE | CORR | PROBE128 | 같은 조건 반복 표본 2 | 1 |
| S5 | B_RESOURCE | RESOURCE | PROBE128 | 정확한 부하 perf/PCM + CPU 단계 표본 | 1 |
| S6 | B_OFF_CLOSE | OFF | PROBE128 | 시간 경과/부팅 차이를 포함한 종료 대조 | 1 |
| S7 | 필요 시 B_RESERVE | 한 목적만 선택 | 원 workload 또는 조건부 C1/LONG | 유일한 재시도/추가 공백 보완 | 최대 1 |
| **합계** | 기본 4부팅 + 예비 최대 1 | | | **기존 잔여 예산 안** | **최대 7** |

S2~S4는 CORE 기록이 켜진 같은 부팅에서 GPU profiler만 안전하게 켜고 끌 수 있는 경우의 배치다. 그렇지 않으면 부팅 배분을 재계산한다. 부팅 5회 상한을 넘기는 일정을 조용히 추가하지 않는다.

`RESOURCE`의 단계 기록을 안전한 세션 경계에서 켤 수 있다면 B_CORE와 합쳐 부팅을 줄일 수 있다. 반대로 계측 설정 때문에 graph 재캡처가 발생하면 변경 내용을 기록하고 별도 boot/capture generation으로 취급한다. 모드 전환을 위해 기존 모델 task 순서를 변경하지 않는다.

### 14.3 S7 선택 규칙

S7은 남아서 쓰는 셀이 아니다. A01~S6에서 남은 공백 중 **선택하려는 성능 개선 후보를 가로막는 가장 중요한 한 가지**에만 배정한다.

| 조건 | S7 우선 사용 |
|---|---|
| CORE/CORR의 간섭이 커서 현상이 계측 때문인지 구분 불가 | 기록 밀도를 낮춘 동일 PROBE 재검증 |
| 필수 collector나 correlation 실패, 원인을 수정함 | 동일 PROBE 재시도 |
| 후보가 prefill CPU 경로와 공용이고 기존 eager 자료 복원 불가 | LONG의 실제 EXTEND 표본 |
| 후보가 작은-row/empty 신호 경로를 바꿈 | C1 또는 해당 작은 배치 경로 검증 |
| CPU 단계가 원인으로 연결됐지만 위치가 불명 | 제한된 symbol/off-CPU FOCUS |
| 기본 질문이 이미 해결됨 | 실행하지 않음 |

한 번의 S7을 여러 독립 workload 요청 세션으로 쪼개 실행하고 1회로 계수하지 않는다. 추가 collector 때문에 한 세션을 다시 실행하면 새 attempt다. 같은 오류를 아무 수정 없이 반복하지 않는다.

### 14.4 왜 MAIN3와 GLM을 기본 7회 안에 넣지 않는가

이번 목적은 성능 개선 결과를 발표하는 것이 아니라 다음 변경의 근거를 확보하는 것이다. 이미 존재하는 MAIN3는 역사적 기준으로 보존한다. 새 계측 binary의 상세 OFF/ON 비교는 같은 PROBE128로 수행한다. **다음 성능 개선 단계에서는 MAIN_SHORT를 새로 반복 측정해 그 단계의 분모를 확정**해야 하며, 이번 PROBE 처리량을 MAIN 결과처럼 사용하지 않는다.

GLM은 정상 출력 차단이 아직 해소되지 않았다. Qwen의 필수 측정 예산을 정상성 선행 작업이 없는 GLM 성능 시험에 쓰지 않는다. GLM 정상성 조사까지 실행하려면 목적·질문·예산을 명시해 S7을 재배정하거나 별도 트랙 예산을 확정한다. Qwen 측정 완료의 선행 조건으로 GLM 해결을 강제하지 않는다.

### 14.5 실행량 원장

다음 항목을 별도로 기록한다.

```text
server_boot_attempts
benchmark_session_attempts
smoke_request_attempts
warmup_request_attempts
quality_question_attempts
synthetic_test_runs
standalone_diagnostic_workload_attempts
collector_retries_without_model_requests
```

부팅/bench 실패는 사용량에 포함한다. 단위시험을 bench 세션으로 세지는 않지만, 모델 요청을 사용한 검증을 합성 단위시험으로 숨기지 않는다. 수집기만 다시 파싱한 작업은 새 추론 세션이 아니다.

### 14.6 중단과 계속 진행

- 환경·권한·모델 가중치가 없으면 해당 실행을 차단하고 가능한 offline 분석을 완료한다.
- warmup 실패 후 본 벤치를 강행하지 않는다.
- buffer stale·NaN·결과 mismatch가 새로 발생하면 그 모드의 상세 성능 해석을 보류한다.
- 필수 correlation이 끊기면 연결되지 않은 값을 임계 경로로 계산하지 않는다.
- 예산 소진 시 `BLOCKED_BY_BUDGET`와 부족한 증거를 남긴다. 무리하게 조건을 낮춰 READY로 바꾸지 않는다.
- 이미 충분히 답한 항목은 닫고 남은 독립 항목·보존 작업은 계속한다.

---

## 15. 통계, 계측 간섭, 증거의 신뢰도

### 15.1 최소 실행안의 한계를 사전에 명시

기본안은 OFF 2회, CORR 2회, CORE·RESOURCE 각 1회다. 따라서 모든 모드의 안정적인 반복 분포나 인과적 observer overhead를 추정할 수 있는 설계는 아니다. **유한 예산의 후속 진단**이며, IDE_074의 본 반복 3회 및 GPU 내부 관측과 함께 방향성을 평가한다.

새 바이너리/계측 방식이 바뀌면 과거와 신규 값을 한 모집단으로 합쳐 반복 수를 늘리지 않는다. CORE·RESOURCE 단발 값에는 SD/CI를 만들지 않는다. S3/S4의 수만 건 layer 이벤트도 독립적인 수만 회 실험이 아니라 같은 두 세션의 반복 사건이다.

### 15.2 필수 비교

동일 PROBE128의 같은 token 합계·옵션을 확인한 뒤 계산한다.

```text
throughput_ratio = observed_ON_throughput / reference_OFF_throughput
request_duration_ratio = ON_request_duration / OFF_request_duration
latency_ratio = ON_latency_metric / OFF_latency_metric
```

OFF 평균 대비 비율과 함께 **각 OFF 원값 대비 비율 범위**도 제공한다. OFF_OPEN/OFF_CLOSE 자체 차이, boot ID, 시간 순서, clock/power·온도·background workload 차이를 보여준다. 감소율을 개별 queue·compute duration에 일괄 적용해 보정하지 않는다.

간섭이 처리량에는 작아도 batch 분포·tail을 바꿀 수 있으므로 아래를 비교한다.

- actual batch/capture rows/phase별 step 수와 실행 시간.
- empty-cold 및 nonempty-cold 비율, route assignment 분포.
- TTFT/TPOT/ITL/E2EL의 반복별 p50/p95/p99.
- CPU task 수·queue depth 분포, logger task 삽입 여부.
- event drop·mapping coverage·clock-indeterminate 비율.

### 15.3 신규 간섭 기준 제안

첨부에는 확정된 수치 기준이 없었다. 다음은 **이번 실행안을 위한 제안값**이며 최초 모델 부하 전에 `PLAN_RESOLVED.md`에 등록한다. 기준을 정한 뒤 결과를 보고 통과하도록 바꾸지 않는다.

| 분류 | 제안된 기준 / 의미 |
|---|---|
| 목표 `DESCRIPTIVE_LOW_DISTORTION` | 같은 workload OFF 비교에서 처리량 차이 절댓값 약 3% 이내, 주요 p95 차이 약 5% 이내, 실행 phase/batch 분포의 설명되지 않은 변화 없음 |
| `DIAGNOSTIC_ONLY` | 목표 범위를 벗어나거나 반복/부팅 차이 때문에 범위를 분리할 수 없음. 현상 위치의 보조 자료로 사용 |
| `UNUSABLE_FOR_TARGET` | 기록기가 task/동기 의미를 바꾸거나 정상성·ID·window·필수 clock 검증 실패 |

위 3%·5%는 통계적 등가성의 증명이 아니다. 과거 OFF 변동이 컸으므로 평균 비율만 기준 안에 든다고 통과를 확정하지 않는다. 최소 계획에서 CI를 뒷받침할 반복이 부족하면 status를 `DESCRIPTIVE` 또는 `INCONCLUSIVE_BOOT_VARIANCE`로 남긴다.

엄밀한 정량 등가성 주장이 필요하면 충분한 독립 반복·부팅 균형과 사전 정의한 등가성 구간을 갖춘 별도 설계가 필요하다. 그것이 이번 7회 안에 자동 포함된 것은 아니다.

### 15.4 clock/mapping coverage 기준

- 분석에 포함한 각 행은 필요한 start/end·producer/consumer ID가 모두 유효해야 한다. 누락이 있는 행은 해당 지표에서 제외한다.
- 모집단 대표성을 주장하려면 layer/phase/배치별 누락·표본 선택이 편향되지 않았는지 확인한다.
- 계층별 coverage가 부족하면 그 범위는 미측정으로 남긴다. 전체 평균 coverage 하나로 특정 층 누락을 숨기지 않는다.
- 명확히 잘못된 clock 순서를 max(0)로 덮지 않는다. ambiguous는 별도 집단이다.
- 목표층의 유효 표본이 없으면 그 층에 대한 Hot 예산 변경의 근거로 사용하지 않는다.

### 15.5 통계 표기

반복별 원값, 평균, 중앙값, 표본 SD(ddof=1), min/max, 표본 수를 남긴다. 모든 요청을 합친 pooled p95와 반복별 p95의 평균을 구분한다. 동일 job의 구간 합은 가능하지만 서로 다른 분포의 p50 합은 사용하지 않는다.

boot/session/step의 군집 구조를 보존한다. bootstrap 등을 사용하면 resampling 단위를 저장하고, 연속 layer 행을 독립으로 재표본화해 과도하게 좁은 CI를 만들지 않는다. 극소수 boot로 얻은 CI는 탐색적 범위라는 한계를 남긴다.

### 15.6 관측 상관관계와 원인 분해

다음은 가능한 분석이다.

- same-job service·queue·budget·pub lateness의 joint distribution.
- producer layer 및 실제 kernel branch/rows를 조건으로 한 분포.
- CPU가 늦은 사례와 먼저 끝난 사례의 비교.
- 직접 선행 task와 소비 지연의 의존 edge 분석.
- mapping/clock 의심 행 포함/제외에 따른 민감도.

상관계수나 회귀에서 유의한 feature가 나왔다는 이유만으로 그 feature를 바꾸면 같은 비율로 성능이 좋아진다고 하지 않는다. 측정으로 후보를 좁히고, 실제 개선 효과는 다음 단계의 단일변경 대조로 검증한다.

---

## 16. G00 — GLM은 정상성 트랙으로 분리한다

### 16.1 계승할 상태

IDE_074의 GLM gate는 `BLOCKED_NORMAL_OUTPUT`다. 직전 분석에서 확인한 4개 원장 항목은 HTTP 200, `finish_reason=length`, 절단 상태이며 약 94.85~94.95초가 걸렸다. 이번 직접 종료 사유를 HTTP timeout으로 바꾸어 쓰지 않는다. `[A074]` 9장, `[C-GLM]`.

Qwen의 큰 decode Cold 지연 분석 결과를 GLM의 native FP8 실행에 그대로 적용하지 않는다. Qwen의 INT4/RB/callback-free가 GLM backend에서 지원·작동하는지도 별도 검증 대상이다.

### 16.2 실행 전 기존 자료로 확인

1. gate의 실제 입력·token IDs·template·tokenizer revision·요청 JSON을 확보한다.
2. thinking/EOS/stop/max_tokens와 성능용 ignore_eos의 혼입 여부를 확인한다.
3. 검증된 비교 경로의 같은 입력 결과가 있는지 확인한다. 다른 질문의 정상 출력은 대조가 아니다.
4. native FP8 scale·dtype·axis·loader 계약과 합산 경로를 기존 로그/코드에서 점검한다.
5. 최근 변경·가중치 해시·오류 로그를 정리한다. 특정 원인을 사전에 단정하지 않는다.

### 16.3 추가 정상성 측정이 필요한 경우

별도 예산 또는 명시적 재배정이 있을 때만 대표 tensor/expert 또는 짧은 동일 요청의 비교를 수행한다. 진단 목적·비교값·오차 기준·문항 시도 수를 등록한다. 순서는 정상 출력/수치 경로 확인 → 짧은 부하 → 병목 측정이다.

INT4 전체 변환, 가짜 `weight_block_size` 추가, timeout 연장만으로 통과 처리, 수치 오류를 품질 저하로만 보고 장시간 benchmark 강행은 금지한다.

GLM 미완료는 Qwen의 `READY_FOR_TARGETED_OPTIMIZATION` 상태와 독립이다. 최종 파일에는 `GLM_NORMAL_OUTPUT_BLOCKED`와 현재 부족한 자료를 명시한다.

---

## 17. A12 — 성능 개선 단계로 넘어가는 기준

### 17.1 전체 완료와 후보별 준비 상태는 다르다

모든 하위 하드웨어 현상을 완전히 설명하는 것이 목표가 아니다. **계측 오류 때문에 잘못된 후보를 고르지 않을 정도의 원인 경로와 정상성이 확인됐는지**를 판단한다.

다음 두 결과를 별도로 작성한다.

- `MEASUREMENT_COMPLETION_STATUS`: 요청한 측정 항목의 완료/부분/차단 현황.
- `OPTIMIZATION_READINESS`: 다음에 시험할 후보별 준비 상태와 허용 주장 범위.

최종 상태의 예시는 다음과 같다. 상태명은 신규 스키마 제안이다.

| 상태 | 의미 |
|---|---|
| `READY_FOR_TARGETED_OPTIMIZATION` | 한정된 후보의 원인 경로·정상성·측정 신뢰도가 충분함 |
| `READY_WITH_LIMITED_SCOPE` | 예: C64 Cold 단계 후보만 준비됨. DRAM 포화/통신/GLM은 미확정 |
| `CONDITIONAL_DIAGNOSTIC_ONLY` | 현상은 보이나 간섭·반복·clock 한계로 후보 원인을 확정하기 부족 |
| `BLOCKED_BY_MEASUREMENT_VALIDITY` | 필수 ID·window·normality 검증 실패 |
| `BLOCKED_BY_ACCESS_OR_BUDGET` | 필요한 원자료/실행 권한/잔여 예산 부족 |

단순히 파일이 생성됐다는 이유로 READY를 주지 않는다. 반대로 관련 없는 미측정 항목 때문에 이미 증거가 확보된 독립 후보를 무조건 차단하지 않는다.

### 17.2 후보별 필수 근거

| 다음 후보 | 지금 확보해야 하는 최소 근거 | 없어도 되는 근거 / 제한 |
|---|---|---|
| Cold 커널/작업 크기 기반 서비스 시간 개선 | 지연 producer job의 실제 branch·expert rows·단계 span, queue와 service 분리, 결과 소비 연결, 정상성 | DRAM 세부 PMU 전체가 없어도 가능. 단 memory-bound 주장은 금지 |
| 비용 기반 Hot 예산 재배치 | producer layer·logical expert 단위 관측 비용, 유효 assignment 수, Hot ready·Cold 노출, ID 매핑, 현재 예산/HBM | 관측 비용은 한계 이득의 확정치가 아님. 다음 단계 held-out 검증 필요 |
| FIFO/dispatch/신호 경로 개선 | 선행 계산 점유를 제외하고도 실제 노출되는 전달/dispatch 지연, task 종류·ID, buffer 계약 | 선행 CPU 계산 때문에 생긴 대기를 queue 자료구조 비용으로 볼 수 없음 |
| 입력/결과 복사 묶음화 | payload별 bytes/count/duration, 실제 노출 전송 구간, buffer 수명, 신호 계약 | 호출 수가 많다는 이유만으로 선택하지 않음 |
| GPU Hot 커널 개선 | Hot 또는 해당 GPU 경로가 실제 준비 병목인 표본, 실제 backend·shape, CPU와의 결합 관계 | Cold가 대부분 늦은 큰 batch 표만으로 Hot 최적화를 우선하지 않음 |
| NUMA/worker 배치 변경 | 같은 parent의 지연 socket/TID·실제 배치·원격/스케줄링 근거 | 단순 completion skew와 낮은 IPC만으로 선택하지 않음 |
| TP 통신 개선 | collective 위치·ID·rank ready/start/end, clock·bytes, 선행 계산과 구분 | collective kernel duration만으로 링크 포화 주장 금지 |
| prefill/chunk 정책 변경 | 실제 EXTEND 및 decode 간섭·요청 queue·TTFT 연결 | LONG의 decode 표만으로 선택하지 않음 |

### 17.3 공통 전환 게이트

아래를 모두 만족하거나, 해당 후보와 무관한 항목의 제외 이유가 명시돼야 한다.

- [ ] Qwen 모델·가중치·Hotmap·5,952슬롯·CPU96·NUMA2·deferred8·KV·graph 설정이 보존됐다.
- [ ] 이전 PCM/시간창/지표/분모 문제를 정정했거나 그 값에 의존하지 않는 범위를 명시했다.
- [ ] 목표 phase/배치에서 producer→consumer가 유효하게 연결됐다.
- [ ] 목표 지연이 queue 선행 작업·실제 service·전달/전송 중 어디에 속하는지 근거가 있다.
- [ ] 선택할 CPU 단계 또는 expert/배치의 실제 분기·작업 크기를 확인했다.
- [ ] 계측이 모델 task를 추가/삭제하거나 동기 순서를 바꾸지 않았다.
- [ ] observer 영향과 boot/반복 한계가 공개됐으며, 현상이 그 영향만으로 설명되는지 검토했다.
- [ ] 동일 기준 구성의 정상성·buffer/epoch 검사에 해결되지 않은 새 오류가 없다.
- [ ] 증거와 다음 변경을 연결한 표가 있고, 기대 성능 향상 수치를 측정 결과처럼 쓰지 않았다.

`DIAGNOSTIC_ONLY` 자료만으로 절대 마이크로초 절감량·전체 tok/s 개선율을 보장하지 않는다. 그래도 기존 여러 실행의 GPU 내부 간격과 새 CPU 내부 service/ID 자료가 일관되게 특정 단계에 연결된다면 **범위를 제한한 단일변경 검증 후보**로 제시할 수 있다. 그 제한을 `READINESS.md`에 명시한다.

### 17.4 다음 단계용 인계 패키지

`OPTIMIZATION_HANDOFF.md`에 아래를 작성한다. 이번 문서 단계에서는 성능 변경을 실행하지 않는다.

```text
1. 재현 기준
   모델·manifest·런타임·소스·바이너리·환경 해시
   다음 성능 비교에 사용할 MAIN_SHORT 프로토콜
   계측을 끄는 방법과 실제 OFF 증빙

2. 확인된 지연 경로
   phase/batch/producer layer/expert·job
   queue/service/pub/copy/combine의 joint evidence
   대표 원자료 ID와 전체 분포

3. 최대 두 후보
   변경할 파일/함수 후보와 변경 변수 1개
   해결하려는 직접 관측
   실행 의미·수치·HBM·KV·작업량 보존 조건
   알려진 위험·기존 실패와의 차이

4. 다음 검증 기준
   새 MAIN3 대조와 같은 조건의 후보 반복
   C1/LONG 회귀 확인
   성능 외 정확성·완료율·NaN/stale·안정성
   같은 관측 구간이 줄었는지 확인할 저간섭 계측

5. 제외한 후보와 이유
   근거 부족 / 기존 실패 / 범위 초과 / 자원 제약
```

다음 성능 개선 예산은 별도로 확정한다. 이번 잔여 7세션을 모두 사용한 뒤 자동으로 최적화 실험을 이어 실행하지 않는다. 이 문서에서 후보를 기록하는 것과 실제 코드·배치·커널을 변경하는 것은 별개다.

### 17.5 측정 종료 규칙

최대 두 후보의 근거가 확보되면 남은 조건부 측정을 실행하지 않고 종료할 수 있다. 목표가 C64의 Cold 서비스 시간 개선인데 unrelated TP 링크 카운터·GLM 변환·전수 워커 sweep까지 완료할 필요는 없다.

목표 후보를 고를 증거가 부족하면 부족한 이벤트·필요한 추가 셀·제약을 정확히 적는다. “더 측정해야 한다”라는 포괄적인 결론 대신 **어떤 결정이 어떤 미측정 때문에 막혔는지**를 남긴다.

---

## 18. 결과 저장 규격과 분석 재현

### 18.1 신규 경로 제안

아래 이름은 산출물 계약이며 지금 해당 서버 파일이 만들어졌다는 뜻이 아니다.

```text
shadow_assists/features/<FOLLOWUP_ID>/
  PLAN_RESOLVED.md
  SOURCE_INDEX.md
  CONFIG_DIFF.md
  PROBE_MAP_V2.md
  METRIC_DEFINITIONS_V2.yaml
  WINDOW_RECONSTRUCTION.md
  VALIDATION_REPORT.md
  FULL_REPORT.md
  BOTTLENECK_EVIDENCE.md
  MEASUREMENT_COMPLETION_STATUS.md
  READINESS.md
  OPTIMIZATION_HANDOFF.md
  WORK_LOG.md
  FULL_RAW_DATA.md
  ARTIFACT_INDEX.csv
  SHA256SUMS.txt
  PUBLISH_RECEIPT.md                  # 실제 게시한 경우만
  raw_md/part-*.md
  profiles/
    legacy_vs_corrected.csv
    observer_overhead_v2.csv
    missing_coverage_v2.csv
    clock_alignment_summary.csv
    producer_layer_costs.csv
    fifo_casebook.md

eval/results/<FOLLOWUP_CAMPAIGN>/
  state/
    RUN_MANIFEST.json
    RUN_STATE.json
    execution_events.jsonl
    budget_reconciliation.json
  offline/
    input_inventory.csv
    parser_test_results.json
    legacy_windows.csv
    pcm_samples_v2.csv
    pcm_window_summary_v2.csv
    timestamp_parse_errors.jsonl
    dependency_metrics_v2.csv.gz
    mapping_validation_v2.csv
  qwen/OPT4/<boot>/<session>/<attempt>/
    requested_config.json
    effective_config.json
    feature_proofs.json
    observer_config.json
    environment.json
    launch_cmd.sh
    bench_cmd.sh
    window_events.jsonl
    request_results.jsonl
    request_batch_map.jsonl
    token_timing.jsonl
    thread_role_map.csv
    perf_target_coverage.csv
    cpu_tasks.csv.gz
    cpu_stage_events.csv.gz
    expert_cost_samples.csv.gz
    fifo_dependency_edges.csv.gz
    queue_wait_breakdown.csv.gz
    producer_consumer_edges.csv.gz
    deferred_lifecycle.csv.gz
    transfer_events.csv.gz
    gpu_layer_timeline_TP0.csv.gz
    gpu_activity_TP1.csv.gz
    gpu_activity_TP2.csv.gz
    gpu_activity_TP3.csv.gz
    producer_consumer_metrics_v2.csv.gz
    cpu_counter_intervals.csv
    pcm_memory.raw.csv
    pcm_samples_v2.csv
    clock_anchors.jsonl
    validation_results.json
    metrics.json
    stdout.log
    stderr.log
    error.json                        # 실제 오류가 있는 경우
```

모든 파일을 반드시 빈 파일로 생성할 필요는 없다. 미수집은 manifest의 `NOT_COLLECTED`와 이유로 나타낸다. 원시 `.trace.json.gz`, `perf.data`, C++ binary event log는 승인된 대용량 경로에 보존한다.

### 18.2 파일별 내용

| 파일 | 포함 내용 |
|---|---|
| `FULL_REPORT.md` | 환경·설정·모든 시도·반복 원값·통계 정의·유효성·오류·자료 경로. 원인 추정·개선 권고와 분리 |
| `BOTTLENECK_EVIDENCE.md` | 질문별 실측·기계적 계산·표본 수·한계·원자료 링크 |
| `READINESS.md` | 후보별 READY/제한/차단, 각 판단의 이벤트 근거 |
| `OPTIMIZATION_HANDOFF.md` | 다음 단계의 최대 두 후보·고정 조건·비교 프로토콜·미확정 사항 |
| `WORK_LOG.md` | 가설·결정·코드 변경 이유·실패 복구. 원시 결과와 분리 |
| `VALIDATION_REPORT.md` | parser/단위/clock/ID/window/normality/observer 검증 |
| `SOURCE_INDEX.md` | 어떤 주장·식·코드·수치가 어떤 자료에서 왔는지 |
| `ARTIFACT_INDEX.csv` | 실제 경로·크기·해시·자료 종류·접근 방법·게시 여부 |

### 18.3 유효성 상태 사전

최소 다음 상태를 정의한다. 기존 상태를 새 상태로 일괄 덮어쓰지 않고 변환 이력을 남긴다.

```text
OK
NOT_APPLICABLE_NO_PATH
NOT_COLLECTED
INSUFFICIENT_SAMPLES
INVALID_TIMESTAMP
INVALID_WINDOW
ESTIMATED_WINDOW
UNMEASURED_CLOCK_ALIGNMENT
CLOCK_INDETERMINATE
CLOCK_ORDER_VIOLATION
UNMATCHED_PRODUCER
UNMATCHED_CONSUMER
UNKNOWN_EAGER_ROWS
VALIDATED_HEURISTIC_MAPPING
PARTIAL_DEPENDENCY_MEASUREMENT
CONSUMPTION_PARTIALLY_OBSERVED
INVALID_COUNTER_SCOPE
COUNTER_MULTIPLEXED_OR_LOW_RUNNING_RATIO
DROPPED_EVENT
INVALID_ARTIFACT
NORMALITY_MISMATCH
BLOCKED_NORMAL_OUTPUT
BLOCKED_BY_ACCESS
BLOCKED_BY_BUDGET
```

상태는 여러 개가 함께 존재할 수 있다. `OK`와 중대한 invalid를 동시에 남기지 않는다. 어느 metric이 어느 상태에 영향을 받는지 `metric_validity`로 연결한다.

### 18.4 재현 가능한 분석기 계약

실제 스크립트 이름·CLI는 구현 시 확정한다. 최소 기능을 독립시킨다.

| 분석 모듈 | 입력 | 출력 | 모델 재실행 |
|---|---|---|---|
| timestamp/window parser | raw timestamp·request·collector events | 정규화 창·오류 | 없음 |
| PCM reducer | raw CSV·sample 의미·창 | 소켓/host 표본·요약 | 없음 |
| GPU interval reducer | 실제 activity·창 | device별 union/gap | 없음 |
| event linker | queue·map·replay·lifecycle | producer/consumer edges | 없음 |
| FIFO reducer | task seq·entry/exit·edges | 선행 점유·미분리 gap | 없음 |
| stage/expert reducer | CPU 단계·rows·branch | producer/expert 비용표 | 없음 |
| validator | 모든 원자료·스키마 | 항목별 validity·coverage | 없음 |
| report renderer | 원값·정정값·판정 | MD·CSV | 없음 |

보고서 숫자를 수동 복사한 표만 남기지 않는다. schema version·분석기 commit/hash·실행 argv·입력 파일 해시가 같으면 재생성할 수 있어야 한다. 계산 실패를 null로 숨기기보다 오류 메시지와 영향을 받는 지표를 기록한다.

### 18.5 저장·게시·전달

기존 인계의 보존 원칙을 따른다. 새 측정 산출물의 로컬 저장 성공, 원격 게시 성공, 사용자 전달 성공은 별개의 상태다.

- 파일 parser·참조·해시 검사를 완료한다.
- credential·개인정보·무관한 파일을 제외한다. 대형 자료는 위치·해시·접근 수단을 명시한다.
- 원격 게시가 실행 범위에 포함되고 권한이 확인된 경우에만 지정 branch로 명시적 파일을 stage/push한다. force push·일괄 destructive cleanup을 하지 않는다.
- 게시한 commit SHA·remote SHA·원격 파일 크기/해시를 확인한다. data commit과 게시 확인용 metadata commit을 구분한다.
- 게시 실패여도 로컬 보존·MD 전달을 완료하고 실패를 표시한다.
- 이전 문서의 30분 진행 보고는 실제 실행기와 전달 수단이 지원되는 경우에만 수행한다. 파일 생성과 전달을 구분하고, 지원하지 않는 알림·백그라운드 실행을 약속하지 않는다.

---

## 19. 최종 작업 체크리스트

### 19.1 기존 자료 정정

- [ ] 최신 실행 원장으로 실제 잔여 세션·부팅·품질 시도 수를 확인했다.
- [ ] 원 보고서·직전 분석·이전 계획·고정 코드의 해시와 현재 변형을 구분했다.
- [ ] PCM 소수초·시간대·파싱 실패·경계 표본 처리를 수정하고 단위시험을 통과했다.
- [ ] client 호출창·실제 요청창·server/forward/phase·collector/drain 창을 분리했다.
- [ ] 복원 불가한 perf 구간값을 비례 환산하지 않았다.
- [ ] GPU sum/union/gap을 실제 장치·유효 창 기준으로 계산했다.
- [ ] legacy 지표와 v2 지표의 대응 및 floor 미적용 사실을 기록했다.
- [ ] layer0·empty path·padding·불명 매핑의 분모를 분리했다.
- [ ] eager slot map의 마지막 값 덮어쓰기 문제를 해결하거나 UNKNOWN으로 남겼다.

### 19.2 추가 계측

- [ ] 기록 전용 task를 모델 FIFO에 넣지 않는 계측을 구현했다.
- [ ] 입력/준비/barrier/요청/종료 이벤트를 실제로 저장했다.
- [ ] worker PID/TID 범위와 PMU counter scope를 검증했다.
- [ ] timestamp 단위·clock 변환·오차·store bracket을 저장했다.
- [ ] producer/consumer/job/task/slot/epoch/graph/replay를 연결했다.
- [ ] queue wait에서 선행 업무 실행 점유와 미분리 공백을 구분했다.
- [ ] 지연 job의 actual expert rows·AMX/AVX/RB 분기·CPU 단계 표본을 확보했다.
- [ ] NUMA child와 parent의 병렬 span·합계·CPU-time을 구분했다.
- [ ] HtoD start/end·Hot ready·other ready·combine을 필요한 범위에서 연결했다.
- [ ] 버퍼 수명·첫/마지막 층·step/graph 전환 정상성을 확인했다.
- [ ] 전수/표본·drop·선택확률·mapping/clock coverage를 보고했다.

### 19.3 실행·통계·다음 단계

- [ ] OFF/CORE/CORR/RESOURCE를 같은 입력·같은 성능 설정으로 비교했다.
- [ ] 같은 새 바이너리 OFF를 사용하고 boot/order 한계를 공개했다.
- [ ] 기본 6세션과 최대 1예비의 사용량을 기록했고 자동 확장하지 않았다.
- [ ] CORE/RESOURCE 단발과 CORR 2회의 통계 한계를 표시했다.
- [ ] source 사실·신규 측정·기계적 계산·가설을 분리했다.
- [ ] DRAM·PCIe·통신 포화를 미검증 값으로 단정하지 않았다.
- [ ] GLM 정상성 차단을 Qwen 완료와 별도로 처리했다.
- [ ] 후보별 READINESS와 선택/제외 근거를 작성했다.
- [ ] 최대 두 후보의 다음 검증 계약을 작성했으나 이번에 성능 변경을 실행하지 않았다.
- [ ] 실패·미수집·차단도 최종 상태와 원자료 참조를 남겼다.
- [ ] 최종 MD·원자료 색인·해시·전달 상태를 확인했다.

**완료 시 남길 핵심 문장:** “어떤 배치·생산층·작업에서, 선행 FIFO 작업과 직접 CPU 실행 중 무엇이 결과를 늦췄으며, 그 결과를 GPU가 소비할 때 실제로 얼마의 간격이 남았는지 확인했다. 그 근거가 있는 변경만 다음 성능 개선 단계에서 시험한다.” 수치·근거를 확보하지 못한 부분에는 이 문장을 적용하지 않는다.

---

## 부록 A. 최소 이벤트·지표 스키마

다음은 신규 구현 계약이다. 모든 필드를 hot path에 매번 중복 저장하라는 뜻이 아니다. 세션/부팅 메타데이터와 사전(dictionary)을 분리해 record 크기를 줄일 수 있다. 다만 join key·단위·lifecycle이 손실돼서는 안 된다.

### A.1 `cpu_tasks.csv.gz`

```text
schema_version,campaign_id,boot_id,session_id,attempt_id,
queue_id,task_seq,task_kind,job_id,parent_job_id,
producer_job_id,predecessor_task_seq,worker_tid,numa_pool_id,
layer_id,producer_layer_id,scheduler_step_id,phase_raw,
graph_id,graph_replay_id,capture_generation,
ring_slot,slot_epoch,buffer_id,buffer_generation,
actual_batch_size,capture_rows,cpu_qlen,valid_rows,
t_go_ns,t_enqueue_before_ns,t_visible_ns,t_enqueue_after_ns,
t_dequeue_ns,t_exec_start_ns,t_exec_end_ns,
clock_domain,clock_alignment_id,
path_present,cold_nonempty,sample_probability,validity,missing_reason
```

### A.2 `producer_consumer_edges.csv.gz`

```text
producer_job_id,producer_step_id,producer_layer_id,
consumer_combine_id,consumer_step_id,consumer_layer_id,
output_buffer_id,output_generation,done_task_seq,
t_done_before_ns,t_done_after_ns,
gpu_h2d_activity_id,gpu_combine_activity_id,
producer_observed,consumer_observed,
mapping_method,mapping_evidence_ids,mapping_validity,
source_clock_domain,target_clock_domain,clock_alignment_id,
clock_error_model_id,edge_validity,missing_reason
```

### A.3 `expert_cost_samples.csv.gz`

```text
job_id,producer_layer_id,phase_raw,
expert_id_logical,expert_id_physical,expert_map_id,
numa_pool_id,partition_id,valid_rows,kernel_rows,
input_dtype,weight_format,output_dtype,
kernel_branch_actual,rb_branch_actual,dispatch_site,
prepare_span_ns,gather_span_ns,q_input_span_ns,
up_gate_span_ns,activation_span_ns,q_down_span_ns,
down_span_ns,weighted_reduce_span_ns,merge_span_ns,
inclusive_or_exclusive,stage_dictionary_version,
sample_probability,sample_reason,clock_domain,validity
```

단계별 시간이 fused 또는 미수집이면 null과 이유를 저장한다. 0을 넣어 합계가 맞도록 보정하지 않는다.

### A.4 `producer_consumer_metrics_v2.csv.gz`

```text
metric_schema_version,session_id,boot_id,observer_mode,
producer_job_id,consumer_combine_id,producer_layer,consumer_layer,
phase_raw,actual_batch_size,capture_rows,path_cohort,
go_to_enqueue_us,enqueue_to_start_us,
predecessor_exec_union_us,queue_gap_unattributed_us,
deferred_service_span_us,numa_completion_skew_us,
producer_end_to_pub_us,cold_pub_delta_us,
cold_pub_delta_low_us,cold_pub_delta_high_us,
gpu_pre_h2d_gap_us,h2d_duration_us,gpu_post_h2d_gap_us,
cold_gpu_ready_lateness_us,combine_schedule_gap_us,
gpu_idle_inside_gap_us,overlap_budget_us,
n_cold_assignments,n_unique_cold_experts,
clock_validity,mapping_validity,sampling_probability,
metric_validity,excluded_reason
```

### A.5 `READINESS.md`의 후보 항목 템플릿

```markdown
### 후보 <ID>: <변경 후보 이름>
- 상태: READY_WITH_LIMITED_SCOPE / CONDITIONAL / BLOCKED
- 대상: 모델·phase·배치·producer layer/branch
- 직접 관측: 숫자·단위·rep·표본 수·원자료 경로
- 실제 지연 연결: producer → done → GPU return → combine
- 큐/계산 분리: 선행 점유·직접 service·미분리 gap
- 계측 제한: OFF 비교·clock·mapping·sampling·boot
- 변경할 단일 변수: 아직 실행하지 않은 구체적인 후보
- 보존 조건: 정확성·deferred 의미·top-k·정밀도·HBM/KV
- 다음 검증: MAIN 대조/후보 반복, C1/LONG, 정상성
- 추가로 필요한 측정: 없으면 없음, 있으면 이벤트와 이유
- 금지할 주장: 예를 들어 memory-bound 확정·예상 개선율 실측 취급
```

---

## 부록 B. 분석기·기록기 단위시험 계약

아래 값은 **신규 합성 테스트 입력**이며 서버 실측이 아니다. 기대값을 자동 검사하고 결과를 JSON으로 보존한다.

### B.1 Timestamp·시간창

| 시험 | 입력/조건 | 기대 |
|---|---|---|
| T01 소수초 | `2026-09-17 13:37:59.527`, 명시적 Asia/Seoul | 527ms 보존 |
| T02 UTC 대응 | 위 시각과 `2026-09-17T04:37:59.527+00:00` | 같은 instant |
| T03 초 단위 | `13:37:59` | 소수초 0. 원 정밀도는 1초로 별도 기록 가능 |
| T04 잘못된 날짜 | 9월 31일 | 파싱 실패, inside에 포함 금지 |
| T05 시간대 없음 | metadata에도 없음 | 추정하지 않고 UNRESOLVED_TIMEZONE |
| T06 잘못된 문자열 | 빈 값·문자 혼입 | 원행·실패 이유 보존 |
| T07 epoch 정밀도 | 큰 ns 정수 두 값의 차이 200ns | float 왕복 없이 200ns 유지 |
| T08 clock jump | anchor에서 offset 불연속 | 단일 offset 변환 금지 |
| W01 완전 포함 | W=[10,20), sample=[11,12) | full=1, overlap=1 |
| W02 경계 | W=[10,20), sample=[9.5,10.5) | boundary, overlap=0.5 |
| W03 창 밖 | sample=[20,21) | overlap=0, inside 제외 |
| W04 interval 의미 불명 | timestamp만 있고 도구 계약 불명 | 구간 평균으로 정확 집계하지 않음 |
| W05 중복 sample | 동일 interval 두 번 | coverage 중복 계수 금지 |
| W06 준비 구간 | collector=[0,30), 실제 부하=[10,20) | collector 전체 평균과 부하 평균 분리 |

### B.2 GPU interval·ready 정의

| 시험 | 입력 | 기대 |
|---|---|---|
| G01 sum/union | W=[0,10), kernels=[1,5),[3,7), copy=[6,9) | sum=11, busy=8, idle=2 |
| G02 annotation | 위에 annotation=[0,10) 추가 | 실제 busy 불변 |
| G03 tail gap | G=[2,8), operation=[3,5) | idle=4 |
| G04 장치 분리 | GPU0 busy, GPU1 idle | 장치별 값 유지 |
| G05 겹친 consumer gap | 같은 GPU에서 두 gap 중첩 | raw sum과 union 분리 |
| R01 CPU 늦음 | hot=100, pub=130, h2d_start=135, end=145, combine=148 | pub delta=30, pre-gap=35, copy=10, post-gap=3 |
| R02 CPU 먼저 | pub=60, hot=100, h2d_start=110, end=115 | pub delta=−40, pre-gap=10, pub→start=50. 50을 신호 비용으로 확정 금지 |
| R03 다른 입력 늦음 | hot=100, cold=110, other=120, combine=123 | combine schedule gap=3 |
| R04 경로 없음 | cold_present=false | Cold-later 분모 제외, 0시각 생성 금지 |
| R05 오차 구간 | delta=2, error range=±5 | CLOCK_INDETERMINATE |
| R06 불가능 순서 | h2d_end가 combine_start 뒤, 동일 clock·ID 확정 | 해당 지표 invalid, max(0)로 은폐 금지 |

### B.3 FIFO·ID·수명

| 시험 | 입력/조건 | 기대 |
|---|---|---|
| Q01 선행 점유 | enqueue=10, start=30, 선행 task=[0,25) | wait=20, predecessor union=15, gap=5 |
| Q02 idle queue gap | enqueue=10, start=30, 선행 task 없음 | wait=20, gap=20, 원인 OS로 자동 확정 금지 |
| Q03 child 중복 | parent=[0,10), child 두 개가 parent 안에서 병렬 | parent/child span 중복 합계 금지 |
| Q04 done 의미 | done(L)가 deferred(L−1)에 연결 | 같은 층 CPU job으로 잘못 join하면 검출 |
| Q05 event 누락 | 중간 epoch 1개 유실 | 이후 모든 행을 zip해 한 칸 이동시키지 않음 |
| Q06 slot 재사용 | 같은 slot, 다른 epoch/layer/rows | 과거 사건의 map 불변 |
| Q07 graph 왕복 | graph A→B→A | 각 replay와 generation 유지 |
| Q08 padding | capture64, actual63 | 두 row 수 별도 기록 |
| Q09 lifecycle | inflight_start=1, produced=4, consumed=4, inflight_end=1 | 수량식 통과, ID 검사는 별도 |
| Q10 stale consume | 수량은 같지만 잘못된 generation 소비 | 정상성 실패 |
| Q11 recorder overflow | 10기록 시도, 용량8, overwrite 금지 | written8/dropped2 |
| Q12 mutable pointer | 기록 뒤 packet 메타데이터 수정 | 과거 record 값 변경 없음 |

### B.4 통계·예산·보고

| 시험 | 조건 | 기대 |
|---|---|---|
| S01 P1 smoke 분리 | 본 반복 3개 + 별도 smoke 1개 | 본3/전체4 표 별도 |
| S02 단발 | RESOURCE 1회 | SD/CI null |
| S03 layer 표본 | 2세션 안의 수만 행 | 독립 반복 수는 2로 표시 |
| S04 분모 | layer0·empty·unmatched 섞임 | cohort별 분자/분모 별도 |
| S05 invalid 요청 | 요청0건/부팅실패 | 유효 0tok/s 반복에 넣지 않음 |
| S06 budget | 기존17 + 신규6 + 예비1 | 세션24. 추가 요청은 차단/별도 승인 필요 |
| S07 crash ledger | session_start 뒤 비정상 종료 | 시도 누락 없이 최종 실패 상태 복구 |
| S08 report refs | 없는 raw path 참조 | artifact validation 실패 |

### B.5 안전한 시각 파싱 예시

아래는 PCM의 날짜·시각을 **명시적 시간대와 정수 ns로 변환하는 신규 예시 코드**다. sample interval의 의미·경계 집계·clock correlation까지 해결하는 전체 분석기는 아니다. 시각 의미가 불명확하면 별도 검증이 필요하다.

```python
from __future__ import annotations

import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

TIME_RE = re.compile(
    r"^(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})"
    r"(?:\.(?P<frac>\d{1,9}))?$"
)
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def pcm_timestamp_ns(date_text: str, time_text: str, timezone_name: str) -> int:
    """PCM 현지 시각을 UTC epoch ns로 변환한다.

    timezone_name은 수집 metadata로 확인된 값이어야 한다.
    DST 중복/누락 현지 시각은 모호하므로 이 함수에서는 거부한다.
    파싱 실패는 호출자가 INVALID_TIMESTAMP로 기록해야 한다.
    """
    if not timezone_name:
        raise ValueError("UNRESOLVED_TIMEZONE")
    match = TIME_RE.fullmatch(time_text.strip())
    if match is None:
        raise ValueError(f"INVALID_TIME: {time_text!r}")
    try:
        tz = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"UNKNOWN_TIMEZONE: {timezone_name}") from exc

    day = datetime.strptime(date_text.strip(), "%Y-%m-%d")
    naive = day.replace(
        hour=int(match['h']), minute=int(match['m']), second=int(match['s'])
    )
    a = naive.replace(tzinfo=tz, fold=0)
    b = naive.replace(tzinfo=tz, fold=1)
    if a.utcoffset() != b.utcoffset():
        raise ValueError("AMBIGUOUS_OR_NONEXISTENT_LOCAL_TIME")
    utc = a.astimezone(timezone.utc)
    if utc.astimezone(tz).replace(tzinfo=None) != naive:
        raise ValueError("NONEXISTENT_LOCAL_TIME")

    delta = utc - EPOCH
    seconds = delta.days * 86400 + delta.seconds
    fraction_ns = int((match['frac'] or '').ljust(9, '0'))
    return seconds * 1_000_000_000 + fraction_ns


# 합성/형식 검증 예. 모델 성능 측정이 아니다.
x = pcm_timestamp_ns('2026-09-17', '13:37:59.527', 'Asia/Seoul')
y = pcm_timestamp_ns('2026-09-17', '04:37:59.527', 'UTC')
assert x == y
assert pcm_timestamp_ns('2026-09-17', '13:37:59.527000200', 'Asia/Seoul') - x == 200
```

이 함수로 timestamp를 읽었다고 요청창이 자동으로 정확해지는 것은 아니다. A01/A04의 실제 요청 경계, PCM 표본 구간 계약, timezone 증빙을 함께 사용한다. 잘못된 행을 예외 처리 후 valid 표본에 포함시키는 과거 결함을 반복하지 않는다.

---

## 부록 C. 실행 순서와 코드 점검 지도

### C.1 작업 실행 순서

```text
[서버 추론 없음]
1. 첨부와 원격/서버 source 식별자 확인, 현재 원장으로 잔여 예산 확정.
2. 기존 PCM/window/분모/metric/eager map 정정, legacy vs v2 표 생성.
3. 재집계로 답할 수 없는 질문만 gap register에 남김.
4. 실제 runtime source를 기준으로 PROBE_MAP_V2 작성.
5. 새 계측기를 구현하고 부록 B 합성/기록기 시험 실행.
6. 실행 manifest에 관측 모드·sampling·window·기준·세션 순서 등록.

[잔여 예산 안의 실제 요청]
7. 부팅별 smoke/warmup 정상성 통과.
8. S1 OFF → S2 CORE → S3/S4 CORR → S5 RESOURCE → S6 OFF.
9. 매 세션 뒤 parse/coverage/normality/collector scope 즉시 점검.
10. 후보를 막는 한 가지 공백이 남을 때만 S7 사용.

[서버 추론 없음]
11. source와 raw hash를 고정한 분석기로 전체 재집계.
12. 질문별 실측 근거, 한계, READY/BLOCKED와 후보 최대2개 정리.
13. 성능 개선 작업용 인계 MD와 원자료·해시·실제 전달 상태 저장.
```

### C.2 구현 위치 조사

다음은 기존 분석에서 확인한 파일·함수의 출발점이다. 새 계측 위치와 라인 번호는 현재 소스에서 다시 확정한다. 특히 컨테이너 설치 경로와 저장소 사본이 다를 수 있다.

| 기존 파일/함수 | 추가 점검/계측 |
|---|---|
| `eval/ide074/render_report.py::parse_pcm` | 소수초·시간대·파싱 실패·표본 구간·창 선택 |
| `eval/ide074/render_report.py::cpu_only_metrics` | slot 마지막 map 사용 폐기, epoch/phase join |
| `eval/ide074/dependency_metrics.py` | producer/consumer 직접 연결, canonical 식·분모·clock |
| `eval/ide074/harness.py::session` | client readiness/barrier, 수집 범위, 정확한 request/drain |
| `eval/ide074/harness.py::drain` | GPU 사용률 보조 확인 + job/consume 실제 완료 |
| `eval/ide074/recompute_d2.py` | old request-window 라벨과 실제 창 정정 |
| `eval/ide074/gpu_union.py` | 기존 회귀시험 재사용, 창 clip·device 분리 |
| `eval/ide074/gpu_layer_timeline.py` | layer/replay heuristic 검증·유실 재동기화 |
| `eval/ide074/kt_evt_patch.py` | 실제 적용 패치 조사. stamp task 삽입 제거 방향 검토 |
| KT `cpu_backend/cpuinfer.h::poll_loop_` | go/packet 관측, immutable metadata, done-store bracket |
| KT `cpu_backend/task_queue.cpp` 및 실제 queue 구현 | task seq·enqueue/dequeue·업무 task entry/exit |
| KT `operators/moe-tp.hpp::TP_MOE_Common::forward` | NUMA parent/subjob·merge span |
| KT `operators/amx/moe_base.hpp` 및 실제 dispatch 함수 | rows·AMX/AVX/RB 실제 분기·단계 boundary |
| KT `experts_base.py::submit_forward` 등 | slot/map generation·eager/capture 구분 |
| KT `experts_base.py::copy_forward_output_to_device` | output buffer/HtoD/producer 연결 |
| SGLang `kt_ep_wrapper.py::apply` | Hot·Cold 결합, input 준비 조건 |
| scheduler/model runner/graph replay 진입 | 실제 step·phase·row→request·replay metadata |

경로·함수 이름이 달라졌으면 현재 값을 사용하되 원 비교 지도와 차이를 저장한다. `PROBE_MAP_V2.md`에는 각 probe의 timestamp thread, clock, 비용, 포함/제외 범위, graph replay 동작, OFF 우회 경로까지 기록한다.

### C.3 분석기 CLI 계약 예시

아래는 **만들어야 할 인터페이스의 예**이며 현재 존재하는 실행 명령이 아니다.

```text
reconcile_sources --campaign <id> --reference-commit <sha>
reconstruct_windows --input <raw-dir> --output <v2-dir>
parse_pcm_v2 --raw <csv> --clock-metadata <json> --windows <jsonl>
link_jobs_v2 --cpu <events> --map <map> --gpu <activities> --clocks <anchors>
analyze_fifo_v2 --tasks <csv> --edges <csv>
analyze_experts_v2 --stages <csv> --jobs <csv> --branches <csv>
validate_campaign_v2 --manifest <json> --results <dir>
render_measurement_report --manifest <json> --validated-results <dir>
```

현재 프로젝트 하네스에 통합하든 별도 스크립트로 나누든 입력·출력·오류 계약을 동일하게 유지한다. 이 예시 문자열을 실제 지원되는 CLI라고 보고서에 복사하지 않는다.

---

## 부록 D. 근거, 출처 계승 범위, 파일 식별자

### D.1 현재 문서 작성에 사용한 로컬 파일

아래 SHA256은 이 문서 작성 시 실제 로컬 파일 바이트에서 계산했다.

| ID | 파일 | bytes | SHA256 |
|---|---|---:|---|
| `[R074]` | `FULL_REPORT-5.md` | 35,502 | `b1012bf9fba512cf08877e125d86f78ba7fdcc0fdb670a714c8bb6150e855665` |
| `[A074]` | `IDE074_bottleneck_analysis.md` | 28,730 | `d534ee215d084fa9285e407a43f9dcd4a1c36407ebe95518386551a481e5c4d6` |
| `[P074]` | `CPU_MoE_bottleneck_measurement_todo.md` | 47,640 | `374c8f6f60bb20717d3a63dbc74a6e8bf49469487fd39a80c187c2e181c520f7` |
| `[H073]` | `CPU_MoE_optimization_session_handoff-2.md` | 61,342 | `67f1a818f1d87a9ea8fd84f9d527e5ee012b72ae4109542fd4ddd1d0ca4ee15c` |
| `[R073]` | `IDE073_results_review-2.md` | 14,669 | `57c8791e7fe74c7767acc72cfea4d0745dcb1c7587cbc6b51a8e81010c462464` |

`FULL_REPORT-5.md`의 Git blob SHA-1은 `8585df9587312bbfcb1af225316a39c0ffc00262`다. `[A074]`는 이 값이 당시 게시 파일 metadata와 일치함을 기록한다. 이번 문서에서는 로컬 blob 계산을 확인했으며 원 서버의 trace 재집계나 신규 원격 상태 검증을 수행한 것은 아니다.

### D.2 고정 코드·원장 위치

아래 경로는 `[A074]`와 이전 대화의 코드 검토에 근거한다. 기준 repository와 commit은 다음과 같다.

```text
repository = mystous/vllm_hybrid
commit = 87dee8e55b2ba0bcaa3a9c630e375a51a565cf68
URL prefix = https://github.com/mystous/vllm_hybrid/blob/87dee8e55b2ba0bcaa3a9c630e375a51a565cf68/
```

| ID | 경로 | 참조 용도 |
|---|---|---|
| `[C-DEP]` | `eval/ide074/dependency_metrics.py` | legacy 지표 식, zip 연결, layer0 분모, clock 허용 |
| `[C-PROBE]` | `shadow_assists/features/IDE_074/profiles/probe_map.md` | 실제 probe 경계, deferred/FIFO·graph 구조, timer 의미 |
| `[C-RENDER]` | `eval/ide074/render_report.py` | PCM parser, 실제 호출창 사용, eager map 최종값 |
| `[C-HARNESS]` | `eval/ide074/harness.py` | collector·run_bench·drain·모드/부팅 관계 |
| `[C-PCM]` | `eval/results/IDE_074_20260917/qwen/OPT4/B3_133528/P2_r1/pcm_memory.csv` | 소수초가 포함된 실제 CSV 형식 |
| `[C-MET]` | `eval/results/IDE_074_20260917/qwen/OPT4/B3_133528/P2_r1/metrics.json` | 호출/부하/수집기/첫 prefill 시각 |
| `[C-D2]` | `eval/ide074/recompute_d2.py` | 기존 GPU request-window와 clock 한계 |
| `[C-GLM]` | `eval/results/IDE_074_20260917/glm/BASIC4/G_135723/gate_result.json` | 4문항 정상성 gate의 직접 종료/출력 기록 |

`PLAN_RESOLVED.md`의 floor 관련 정의는 직전 검토 당시 blob `e9de293c90d03f59dd772389c080ce214d548a47`로 기록됐다. 실행 시 실제 파일과 분석 코드의 정의를 함께 확인한다. 데이터 수집 당시 HEAD `82ad7bca44c2a46b9ac1e4bb2f90483b9743069e`와 게시 기준 commit을 혼동하지 않는다.

### D.3 주요 요구의 근거 대응

| 본 문서 내용 | 근거 | 신규 구체화 |
|---|---|---|
| 큰 decode에서 Cold 완료 지연 | `[R074]` 5·7장, `[A074]` 3장 | producer/FIFO·단계 원인 분리 |
| PCM/실제 창 정정 | `[A074]` 6장, `[C-RENDER]`, `[C-MET]` | strict parser·sample interval·barrier 계약 |
| 지표 명명 정정 | `[A074]` 5장, `[P074]` 10장 | canonical v2·오차 구간·cohort |
| stamp task 간섭 | `[A074]` 2.2장, `[C-PROBE]` | task 삽입 없는 bounded 기록기 |
| L−1 생산자와 L 소비자 | `[A074]` 3.2·8장 | job/lifecycle ID·경계 검사 |
| expert rows와 actual branch | `[P074]` 7장, `[H073]` 10~11장 | 단계·분기·표본·비용표 스키마 |
| prefill·TP 미확정 | `[R074]` 5~7장, `[A074]` 7장 | 후보 의존 조건부 gate |
| 잔여 7세션/5부팅 | `[R074]` 9장 | 본6+예비1 후속 배분 |
| 성능 개선과 측정 분리 | 사용자 요청, `[P074]` 1장 | 후보별 readiness·후속 인계 |

---

**문서 끝.** 이 파일은 추가 측정의 상세 실행 계획이다. 새 실측값·성능 개선 결과·서버 실행 성공·원격 게시 성공을 기록한 결과 보고서가 아니다.
