# EPOCH 추가 실험 및 논문화 실행계획

작성일: 2026-09-09  
문서 버전: 1.0  
기준 기록: `PERFORMANCE_RECORD_20260909_02.md`  
적용 시스템: Qwen3-Coder-480B-A35B-Instruct, H100 80GB ×4 + Xeon 8480+ ×2  
상태: **후속 실험 설계서. 이 문서에서 새로 제안한 실험은 아직 실행하지 않았다.**

---

## 0. 이번 실험의 목적과 범위

**EPOCH의 기존 성능 개선을 중심으로 논문을 만든다. 성공한 EPOCH를 기각하거나, 기각된 EPOCH-X 후보를 다시 구현하는 계획이 아니다.**

첨부 기록에서 EPOCH 실현형은 FP8 KV와 CUDA graph 버킷 축소를 이용해 실행 가능한 동시성을 확대했고, C64의 769 tok/s에서 C224의 1110.3 tok/s로 약 44% 높은 처리량을 기록했다. 동시에 읽기당 행 증가와 DDR bytes/token 감소가 보고됐다. 이 성과는 보존한다. 다만 서로 다른 동시성·지연·패치 구성이 포함된 결과이므로, 논문에서는 **최대 처리량 개선**, **개선의 원인**, **지연 제약 아래의 성능**, **품질 조건**을 각각 입증한다. [S1 §1–5]

핵심 연구 질문은 다음과 같다.

> **GPU 메모리를 가중치·KV·CUDA graph에 배분하는 방식을 바꾸어 실행 가능한 동시성을 높였을 때, CPU expert가 처리하는 유효 연산을 가중치 스트리밍 한 번당 얼마나 늘릴 수 있으며, 그 효과가 전체 서빙 처리량과 지연에 어떻게 나타나는가?**

후속 실험에서는 다음 네 주장을 구분한다.

| 주장 | 현재 근거 | 추가로 필요한 증거 |
|---|---|---|
| 동일 하드웨어에서 더 높은 최대 처리량을 달성했다 | 769 → 1110.3 tok/s라는 운영점 기록 | 구성별 최대 처리량 곡선, 반복 재현, 동일 품질 조건 |
| 메모리 확보가 동시성을 높여 CPU의 가중치 읽기 비용을 상각했다 | KV 37k → 127k, rows/read 증가, bytes/token 감소 | 메모리·동시성 제한 대조군과 실측 DDR 계측 |
| CPU가 실제 연산 주체로 GPU와 함께 시스템 성능에 기여했다 | CPU expert 연산과 CPU 스레드 축소에 따른 성능 저하 기록 | 유효 CPU 연산량, 결과 소비, 실제 중첩 및 노출 대기 시간 |
| 같은 지연 제약에서도 더 높은 서비스 용량을 제공한다 | 아직 충분한 비교 없음 | 같은 SLO에서 최대 지속 가능 도착률·출력 처리량 비교 |

마지막 주장이 모든 조건에서 성립해야 앞의 성과가 인정되는 것은 아니다. 엄격한 지연 조건에서는 개선이 없고 완화된 조건에서만 이득이 있다면, 그 적용 범위를 그대로 논문에 제시한다. 원래 목표였던 `throughput +15%, TPOT p99 악화 ≤5%`도 별도 목표로 유지한다.

### 0.1 출처와 제안의 구분

- **[S1]–[S3]**: 사용자가 제공한 실험 기록. 원시 로그·코드의 직접 검증 결과가 아니라 기록에 보고된 관측이다.
- **[W1]–[W4]**: 계측 방법과 비교 기준을 확인하기 위한 공식 문서·논문. 현재 온라인 문서를 과거 실행 버전의 동작으로 간주하지 않는다.
- **실험 조건·반복 수·성공 기준·수식·데이터 스키마**: 이번에 제안하는 설계다. 기존 실험 결과나 논문 채택을 보장하는 기준이 아니다.

문서에서 사용하는 `EXP-E00` 등의 ID는 **이 문서 안의 실험 ID**다. 기존 `IDE_***` 레지스트리에 자동 등록된 것으로 취급하지 않는다.

### 0.2 이번 범위에서 제외하는 것

EHES, SER, GRC의 재구현, EAGLE3 재시도, 다음 자기회귀 토큰을 기다리는 expert rendezvous, CPU partial attention 재개는 제외한다. 이들은 기존 기각 결과를 부록의 설계 경계로 사용한다. TriX와 JIT-EO도 이번 EPOCH 논문화의 필수 작업으로 추가하지 않는다. [S1 §6, §9; S3 §2]

---

## 1. 기존 성과를 고정하는 기준표

아래 값은 새 실험 목표치가 아니라 첨부 기록의 보존 대상이다.

| 항목 | 기록된 값 | 해석 시 주의점 |
|---|---|---|
| EPOCH 이전 기준 운영점 | C64, 769 tok/s, TPOT 약 61 ms | 정확한 commit·graph 목록·KV dtype은 원시 설정에서 복구 |
| 최종 B 최대 운영점 | C224, 1110.3 tok/s; 재현 1098.2 tok/s | graph 최대 224, chunk 4096, mixed off |
| 최종 B 지연 | TPOT 149.0 ms, TTFT 6.76 s | TPOT 집계 방식 및 TTFT percentile은 원시 JSON 확인 |
| 최종 A의 C64 | 796.6 tok/s, TPOT 57.6 ms, TTFT 2.97 s | **최종 B의 C64 결과가 아니다.** A/B를 섞어 기여를 계산하지 않음 |
| KV capacity | 약 37k → 127,309 tokens | CLI `max-total-tokens=143360`과 실제 pool 용량을 구분 |
| rows/read | 1.1 → 1.34; 별도 오라클 1.582 | 측정 운영점·분모·expert 집계 방식이 같다는 근거 없음 |
| mixed rows/read | 1.58 → 24.5 | prefill 행이 추가된 효과와 실제 중복 읽기 감소를 분리 |
| DDR bytes/token | 3.75 → 3.1 MB | 물리 DRAM byte인지 논리 proxy인지, 층·expert 집계 범위 확인 필요 |
| 최종 GSM100 | 97/100 | 출력 동일성 또는 비열등성의 확정 증거로 대체하지 않음 |
| 전체 캠페인 개선 | 56.3 → 1110.3 tok/s, 약 19.7배 | 약한 초기 구성·결함 수정·여러 최적화가 섞인 개발 이력 |

출처: [S1 §1–5, §8–9].

**재현한 새 값이 과거와 다르면 과거 값을 덮어쓰지 않는다.** `reported_historical`과 `remeasured`를 별도 열로 남기고, 조건 차이를 설명한다. 같은 이유로 기록의 `무손실`이라는 표현은 출처의 표현으로 보존하되, 새 실험에서는 수치 동등성과 과제 품질을 따로 판정한다.

---

## 2. 비교 설계: 무엇을 고정하고 무엇을 바꿀 것인가

### 2.1 EPOCH의 세 부분을 분리한다

| 구분 | 구성 요소 | 실험에서의 역할 |
|---|---|---|
| 메모리·동시성 설계 `E` | FP8 KV, graph 버킷 축소, 확보한 KV를 활용하는 동시성 | 핵심 인과관계 검증 대상 |
| 실행 효율 패치 `R` | expert별 AMX/AVX, AVX 레지스터 블로킹·prefetch, gather·양자화 융합, 핸드오프 등 | E와의 상호작용 및 기여를 분리 |
| 운영 정책 `O` | mixed forward, prefill delayer, chunk 크기 | 처리량·TTFT·TPOT의 운영점 비교 |

hot map, CPU/GPU 가중치 정밀도, deferral, 코어 배치가 함께 달라지면 어떤 효과를 측정하는지 알 수 없다. 해당 항목을 연구하는 실험이 아니라면 같은 값으로 유지한다.

### 2.2 기준선 이름을 통일한다

| ID | 정의 | 사용 목적 |
|---|---|---|
| `HIST-769` | 과거 769 tok/s를 낸 정확한 snapshot | 역사적 성과 재현. 일반적인 강한 기준선으로 대체 사용하지 않음 |
| `REF-COMMON` | 정확성 수정과 공통 R을 적용하되, E 이전 KV·graph 설정을 사용 | E의 메모리·동시성 효과를 분리하는 인과 대조군 |
| `KV-ONLY` | REF-COMMON에서 KV만 FP8로 변경 | FP8의 용량 효과와 실행 경로 효과 분리 |
| `GRAPH-ONLY` | REF-COMMON에서 graph 버킷 정책만 축소 | graph 메모리와 padding·실행 효과 분리 |
| `EPOCH-CORE` | 공통 R + FP8 KV + 축소 graph + 가용 동시성 확대, mixed off | 핵심 E 효과 |
| `EPOCH-B-REPRO` | 기록의 최종 B를 그대로 복원 | 1110.3/1098.2 tok/s 재현 |
| `EPOCH-OPS` | EPOCH-CORE에서 mixed on; delayer는 별도 변형 | 운영 지연 개선 평가 |
| `TUNED-EXISTING` | 정확성 결함을 고친 기존 경로에서 사용 가능한 기존 옵션을 동등한 탐색 예산으로 최적화 | 논문의 강한 외부/기존기법 기준선 |

`REF-COMMON`은 기전 분리용이지 자동으로 최강 기준선이 되는 것은 아니다. `TUNED-EXISTING`에 FP8 KV, graph 설정, mixed를 금지해서 약하게 만들지 않는다. 반대로 연구자가 새로 만든 패치를 기존 기능이라고 넣거나 빼지 않도록 commit별 출처표를 작성한다.

**기존 옵션 최적화가 EPOCH와 같은 성능을 재현한다면**, EPOCH의 측정된 성과는 유지하되, 그 구성 선택 방법·실행 메커니즘이 기존 방법과 무엇이 다른지 주장 범위를 조정한다. 이 경우 추가 알고리즘 검증은 EXP-E11에서 수행한다. 자동으로 EPOCH 전체를 실패로 분류하지 않는다.

### 2.3 정확성 모드를 별도 축으로 둔다

| 모드 | 정의 | 적용 |
|---|---|---|
| `D0` | deferral을 끄고 CPU 결과를 원래 소비 지점 전에 반영 | 정확성·기전 대조 및 핵심 셀 재확인 |
| `Dτ` | 기록의 τ=0.25 및 deferred 설정 그대로 | 기존 성과 재현과 운영 구성 평가 |

D0와 Dτ 사이의 설정 차이가 실제로 deferral 하나인지 로그로 검증한다. Dτ의 정확성은 EXP-E09에서 확인한다. Dτ가 비열등성을 통과하지 못하더라도 D0의 EPOCH 효과를 독립적으로 평가한다.

### 2.4 세 가지 비교를 모두 제시한다

1. **고정 동시성 비교:** 같은 C에서 latency·처리량·CPU 효율이 어떻게 달라지는가.
2. **각 구성의 최대 처리량 비교:** 메모리 제약 안에서 각자 최적인 C를 사용했을 때 처리량이 얼마나 달라지는가.
3. **동일 SLO 비교:** 같은 TTFT·TPOT 제약에서 지속 가능한 최대 도착률과 출력 처리량이 얼마나 달라지는가.

2번에서 C가 다른 것은 비교의 결함이 아니라 실험의 목적이다. 단, 지연과 메모리 사용을 같이 공개한다. 1번만으로 EPOCH를 평가하거나, 2번만으로 지연까지 개선됐다고 주장하지 않는다.

---

## 3. 공통 실행 규약

### 3.1 하드웨어·소프트웨어 고정

기본은 기록의 H100 80GB ×4, TP=4, Xeon 8480+ ×2, CPU 96 workers/2 pools, turbo off 2.0GHz다. CPU 모델·스레드 수는 기록에서 확인되지만, 실제 worker affinity·NUMA memory binding·GPU topology·전력 제한은 실행 시 덤프한다. [S1 서두, §8]

모든 비교에서 다음을 manifest에 남긴다.

| 분류 | 필수 기록 |
|---|---|
| 코드 | SGLang·kt-kernel·벤치 클라이언트 commit, dirty diff, 빌드 옵션, container digest |
| 모델 | GPU·CPU checkpoint revision과 파일 hash manifest, tokenizer·chat template hash, 변환 설정 |
| 메모리 | rank별 weight/KV/graph/workspace 수치, 실제 KV dtype·scale·pool token 수 |
| 실행 | 전체 환경변수 allowlist, 실제로 해석된 server args, graph capture 목록, fallback 횟수 |
| 자원 | CPU pool별 worker 목록, 비-worker pinning, NUMA locality, GPU UUID·topology, GPU/CPU clocks |
| 워크로드 | 입력 token ID hash, 도착열 hash, 출력 길이 정책, prefix 정책, seed, cache 초기 상태 |

`/tmp/hotmap_mixed_0.25.json`은 실험 폴더로 복사하고 hash를 남긴다. 서버를 다시 켤 때 해당 파일이 바뀌거나 사라지면 같은 구성으로 인정하지 않는다. 다른 GPU나 프로세스가 같은 CPU·DDR·PCIe를 사용한 구간도 기록한다.

### 3.2 동시성의 세 값을 구분

- `C_client`: 클라이언트가 동시에 outstanding으로 유지하는 요청 수.
- `C_admit`: 서버가 실행 중으로 승인한 요청 수 또는 그 상한.
- `B_decode(t)`: 해당 forward에서 실제 decode를 수행한 요청 수.

`C_client=224`가 모든 step의 decode 행 224개를 뜻하지 않는다. 모든 성능 곡선에 `B_decode`의 p50/p90와 실제 KV occupancy를 함께 저장한다.

### 3.3 워밍업·캐시·반복

탐색 셀은 서로 다른 도착·입력 순서 seed 3개로 우선 측정한다. 최종 논문 핵심 비교는 seed 5개 이상으로 확대하고, 같은 seed의 두 구성을 짝지어 비교한다. 제안 seed는 `11, 23, 42, 71, 101`이며 기존 42를 포함한다.

각 블록에서 구성 순서를 무작위화하고 AB/BA 순서를 균형 있게 배치한다. 핵심 비교는 적어도 3회의 독립 server restart에 분산한다. 같은 서버에서의 반복과 재시작 간 변동을 구분한다.

모델 로딩·JIT·graph capture를 제외한 상태에서 워밍업한다. prefix cache는 cold/warm을 따로 실험한다. **워밍업 프롬프트가 본 측정과 우연히 공통 prefix를 공유하지 않도록 한다.** 과거 결과를 재현할 때는 기존 cache 조건을 먼저 복원하고, 통제 실험은 별도 ID로 수행한다.

### 3.4 출력 길이와 요청 샘플

성능 기전 실험은 동일 tokenized 입력과 동일 목표 출력 길이를 사용한다. EOS를 무시하는 고정 길이 모드의 지원 여부와 실제 반환 길이를 확인한다. 자연 종료 모드는 별도 실서비스 평가에 사용하고 출력 길이 분포를 공개한다.

같은 프롬프트 수에서 더 짧게 답해 빨라진 결과를 실행 성능 개선으로 계산하지 않는다. 반대로 품질 평가에서는 EOS 무시를 사용하지 않는다. 성능·품질 요청의 decoding 설정은 각각 명시한다.

### 3.5 계측 오버헤드 통제

실험을 `PERF`와 `TRACE` 모드로 분리한다. PERF는 낮은 빈도의 자원 카운터와 요청 타이밍만 수집하고, TRACE는 일부 구간에서 상세 CPU/GPU 이벤트와 expert 통계를 수집한다.

같은 셀에서 tracing on/off를 비교해 처리량·TPOT 변화가 2%를 넘으면 상세 trace를 성능 수치의 근거로 쓰지 않는다. sampling 또는 메모리 ring buffer로 낮춘다. 층마다 파일 출력, hot path JSON 직렬화, 계측을 위한 강제 전역 GPU 동기화는 사용하지 않는다. 기존 이력의 로그 포화·핸드오프 병목을 되풀이하지 않기 위한 통제다. [S3 §2.1, §4]

---

## 4. 측정 지표의 정확한 정의

### 4.1 사용자 지연과 처리량

요청 i의 계획 도착 시각을 a_i, 실제 전송 시각을 s_i, 첫 출력 token 도착을 f_i, 마지막 출력을 l_i, 출력 token 수를 n_i라 한다.

```text
TTFT_user(i) = f_i - a_i
TTFT_send(i) = f_i - s_i
client_queue(i) = s_i - a_i
TPOT(i) = (l_i - f_i) / (n_i - 1), n_i >= 2
E2EL_user(i) = l_i - a_i
```

논문 서비스 지연은 TTFT_user를 기본으로 하고, TTFT_send도 병기한다. TPOT p99는 **요청별 TPOT의 p99**다. token별 ITL p99와 혼용하지 않는다. n_i=1은 TPOT에서 제외하되 그 수를 기록한다.

HTTP streaming 한 chunk에 여러 token이 묶이면 그 도착 시각만으로 개별 token ITL을 복원할 수 없다. server token timestamp를 계측하거나 `chunk gap`으로 표시한다. 재토큰화한 token 수를 동일한 시각으로 늘려 p99를 만들지 않는다.

```text
T_out(window) = 측정 시간창에 실제 방출된 출력 token 수 / 시간창 길이
T_job = 완성한 전체 job의 출력 token 수 / 첫 전송부터 마지막 완료까지 시간
```

T_out은 정상상태 처리량, T_job은 유한 burst job 처리량이다. 서로 다른 정의의 수치를 한 열에 넣지 않는다. 완료 요청의 모든 token을 완료 시각 하나에 몰아 넣어 정상상태 token 처리량을 계산하지 않는다.

### 4.2 SLO 지표

```text
pass(i) = 성공적으로 완료했고
          TTFT_user(i) <= L_F 이며
          TPOT(i) <= L_D 인 요청

G_req = 측정 도착 cohort 중 pass 요청 수 / 도착 측정창 길이
G_tok = 같은 cohort의 pass 요청 출력 token 합 / 도착 측정창 길이
```

G_req/G_tok은 보조 지표이며 기존 목표인 출력 처리량을 대체하지 않는다. 원시 출력 처리량, 실패율, TTFT/TPOT p50·p95·p99를 함께 공개한다. 도착 cohort의 지연·성공 여부는 배수 구간까지 추적하되, 배수 시간의 token을 정상상태 T_out 분자에 넣지 않는다.

### 4.3 expert 행과 읽기

일정 측정 구간의 CPU expert invocation을 j라 하고, 그 invocation에서 실제 계산한 유효 token-expert 행 수를 m_j라 한다.

```text
R_useful = Σ_j m_j
J = CPU expert invocation 수
rows_per_invocation = R_useful / J
```

이 값은 단순히 **invocation당 행 수**다. 커널이 가중치를 실제 DRAM에서 한 번만 읽었다고 확인하지 않았다면 `rows/read`로 이름 붙이지 않는다. padding 행, masked dummy row, 중복 실행, 폐기된 결과를 유효 행에 포함하지 않는다.

다음 세 지표를 분리한다.

| 지표 | 정의 | 한계 |
|---|---|---|
| 논리 weight byte proxy | invocation별 사용 가중치·scale·metadata 크기의 합 | cache hit와 타일 재읽기를 반영하지 않음 |
| 실측 DRAM read bytes | 양 소켓 memory-controller read counter의 적분 | expert 외 CPU·DMA traffic도 포함할 수 있음 |
| 유효 CPU 처리 효율 | 유효 expert rows / CPU 구간 시간, rows / 실측 DRAM bytes | phase·cold routing 비율을 같이 보고해야 함 |

`DDR bytes/output-token`, `DDR bytes/useful-expert-row`, `DDR bytes/input+output-token`은 서로 다른 값이다. 주 지표는 첫째와 둘째이며, 순수 decode·prefill·mixed를 따로 보고한다. 전체모델 기준인지 한 층·한 socket 기준인지 반드시 명시한다.

Intel PCM은 memory bandwidth와 NUMA 관련 계측 도구를 제공한다. 실제 실행 노드에서 지원하는 이벤트·권한·단위를 확인하고, 버전·counter 설정을 저장한다. CPU core의 load 카운터를 DRAM byte로 대신하지 않는다. [W3]

### 4.4 동시 실행과 노출 대기

`I_CPU`는 유효 CPU expert 계산 구간들의 시간상 합집합, `I_GPU`는 유효 GPU 계산 kernel 구간들의 합집합으로 정의한다. 폴링·spin·순수 대기 kernel은 유효 계산에서 제외한다.

```text
O_CPU = |I_CPU ∩ I_GPU| / |I_CPU|
W_exposed = 다음 필수 계산이 CPU 결과 미도착만을 이유로 진행하지 못한 시간
```

worker 96개의 CPU 시간을 합한 core-seconds와 wall time을 혼용하지 않는다. GPU 4개의 kernel duration 합도 wall time과 다르다. 중첩률이 높아졌다는 이유만으로 성능이 개선됐다고 판정하지 않는다. **유효 행 처리량 증가와 W_exposed 감소, 전체 처리량을 함께 확인한다.**

---

## 5. 실험 실행 순서와 산출물

| ID | 우선순위 | 실험 | 선행 조건 | 핵심 산출물 |
|---|---|---|---|---|
| EXP-E00 | P0 | 구성·지표·패치 출처 고정 | 없음 | manifest, patch ledger, metric dictionary |
| EXP-E01 | P0 | 기존 44% 개선 재현 | E00 | 과거/재측정 대조표 |
| EXP-E02 | P0 | KV × graph × capacity 인과 분리 | E00 | 메모리 분해, 2×2 표, capacity-cap 대조 |
| EXP-E03 | P0 | 유효 CPU 연산·DDR 읽기 기전 검증 | E02 | bytes/token, 행 분포, CPU 효율 곡선 |
| EXP-E04 | P1 | CPU/GPU 동시 실행·임계 경로 검증 | E00, E03 | 유효 연산 timeline, CPU 민감도 |
| EXP-E05 | P1 | 실행 패치와 E의 기여 분리 | E02 | E×R 상호작용 및 ablation |
| EXP-E06 | P0 | 각 구성의 최대 처리량·지연 곡선 | E01, E02 | throughput–latency frontier |
| EXP-E07 | P0 | 동일 SLO의 지속 서비스 용량 | E06 | arrival frontier, SLO별 용량 표 |
| EXP-E08 | P1 | mixed forward의 동일 작업량 검증 | E03 | separate/mixed DRAM·시간 대조 |
| EXP-E09 | P0 | 정확성·품질 검증 | E00; 성능 실험과 병행 | 수치 동등성·비열등성 표 |
| EXP-E10 | P1 | 길이·prefix·모델·CPU 자원 일반화 | E06, E09 | 적용 범위 및 반례 |
| EXP-E11 | P2 | EPOCH 구성 선택 규칙의 알고리즘화 | E02, E03, E06 | held-out 구성 선택 성능 |
| EXP-E12 | P1 | 장시간 안정성·부하 변화 | 핵심 구성·품질 확정 | 60분 이상 soak 결과 |

P0는 EPOCH의 기존 성과를 논문 주장으로 연결하기 위한 필수 작업이다. P1은 시스템 논문의 기전·일반화·운영 신뢰성을 보강한다. P2는 기존 기능의 수동 조합과 구별되는 선택 알고리즘을 주장할 때 수행한다.

---

## 6. EXP-E00 — 구성 복구와 측정 정의 고정

### 목적

769와 1110.3의 비교에서 실제로 바뀐 항목을 복구하고, 이후 실험의 결과 귀속을 가능하게 한다.

### 절차

1. HIST-769와 EPOCH-B-REPRO의 시작 명령·실제 resolved args·commit·patch·hot map을 복원한다. 기록에 없는 KV dtype, graph 목록, vLLM client 버전은 `unknown`으로 두고 원시 로그에서 확인한다. 추정값으로 채우지 않는다.
2. 패치를 `correctness fix / existing option / authored mechanism / measurement-only`로 분류한다. 절대-pin, expert mask, expert_location, NUMA pool 일치 등의 결함 수정은 비교 양쪽에 공통 적용한다. [S3 §2.3, §2.5]
3. 자료의 rows/read 1.34와 1.582, DDR 3.1 MB의 산출 코드와 분모를 확인한다. 399패스·62층·160 expert라는 오라클 입력 크기를 모델의 모든 구성에 일반화하지 않는다. [S1 §9]
4. tokenizer를 통해 실제 input/output token 수, `prefix 100`의 의미, prefix cache hit 비율을 확인한다.
5. `KT_*` 환경변수가 어떤 code path에서 읽히는지 확인한다. 환경변수가 설정돼 있어도 현재 빌드에서 사용되지 않으면 활성화로 기록하지 않는다.
6. 각 GPU rank의 memory allocation과 CPU worker placement를 덤프한다. graph padding용 dummy row가 CPU expert 큐에 들어가지 않는지 확인한다.

### 추가 확인 목록

| 확인 대상 | 필요한 증거 | 확인 전 처리 |
|---|---|---|
| 기존 KV dtype | resolved args와 cache tensor dtype | K0라는 기호로만 표기 |
| 기존 graph 목록 | capture log와 실제 graph key | G0라는 기호로만 표기 |
| TTFT·TPOT 집계 | 원시 JSON field 및 계산 코드 | 과거 수치에 p99를 덧붙이지 않음 |
| FP8 scale | 실제 scale 로드·적용 경로 | 기본값이나 calibration 유무를 추정하지 않음 |
| τ deferral 소비 지점 | dependency trace·코드 위치 | 출력 동일성 주장 보류 |
| hot-96 가중치 70GB | rank별 allocator 계측 | per-GPU인지 총합인지 임의 해석하지 않음 |

### 완료 기준과 실패 처리

두 재현 구성의 필수 필드가 완성되고 계측 정의가 고정되면 완료한다. 일부 역사 설정을 복구하지 못해도 이후 실험을 중단할 필요는 없다. HIST-769는 `historical-unrecovered`로 남기고, **현재 코드에서 만든 REF-COMMON을 새로운 통제 기준선으로 사용한다.** 새 대조군을 과거와 동일하다고 부르지는 않는다.

산출물: `manifest_schema.json`, `historical_config_diff.md`, `patch_ledger.csv`, `metric_dictionary.md`.

---

## 7. EXP-E01 — 기존 44% 개선의 재현

### 가설

기록의 두 운영점을 동일 노드에서 복원하면, 단발성 최고치가 아닌 반복 측정에서도 EPOCH 최대 구성의 처리량 우위가 관측된다. **TPOT가 같을 것이라는 가설은 포함하지 않는다.**

### 실험 셀

| 셀 | 구성 | C_client | 워크로드 | 목적 |
|---|---|---:|---|---|
| R1 | HIST-769 | 64 | 기존 sonnet 512/128, prefix 100, 3C 요청 | 과거 기준 재현 |
| R2 | EPOCH-B-REPRO | 224 | 동일 생성 규칙, 3C 요청 | 기존 최대 결과 재현 |
| R3 | EPOCH-B-REPRO | 64 | 동일 512/128 조건 | 최종 B의 고정 C 비교를 새로 확보 |
| R4 | REF-COMMON / EPOCH-CORE | 각 구성의 후보 C | 긴 포화 workload | 짧은 burst와 정상상태 차이 확인 |

R1/R2는 프롬프트 수가 C에 따라 다르는 **과거 프로토콜 재현**이다. 논문의 주 비교는 E06에서 같은 입력 pool·출력 길이와 같은 측정 규약으로 다시 수행한다.

### 측정 및 반복

seed 42로 historical sanity check 후, 5 seed 짝 비교를 수행한다. 처리량, TPOT의 mean/p50/p99, TTFT p50/p99, 실제 output token 수, cache hit, B_decode, OOM/retraction, 종료 시간까지 저장한다.

### 판정

과거 처리량 대비 ±10%를 벗어나면 환경·워크로드·계측 차이를 우선 조사한다. ±10%는 재현성 감사의 제안 임계값이지 합격을 맞추기 위한 튜닝 허용폭이 아니다. 실제 재현 속도비와 95% 신뢰구간을 보고하며, 44%에 맞도록 일부 반복을 제외하지 않는다.

1110.3보다 낮게 재현돼도 안정적인 우위가 있으면 그 값을 논문의 주 결과로 사용한다. 1110.3은 기록된 historical peak로 남긴다.

산출물: `reproduction_runs.csv`, `historical_vs_remeasured.md`, `repro_failures.md`.

---

## 8. EXP-E02 — KV·graph·동시성의 인과 분리

### 가설

EPOCH의 개선에는 두 경로가 있다.

```text
KV 정밀도 / graph 정책
    ├─ 실행 시간·padding·kernel 경로의 변화
    └─ 사용 가능한 KV capacity 증가
          └─ 실행 가능한 동시성 증가
                └─ expert당 유효 행 증가·streaming 비용 상각
                      └─ 시스템 처리량 증가
```

용량 경로를 끊는 대조군을 넣어 두 효과를 분리한다.

### 8.1 기본 2×2 실험

K0는 복구된 E 이전 KV dtype, F는 기록의 FP8 E5M2, G0는 복구된 graph 정책, GS는 축소 정책이다. hot-96, α=0.25, common R, chunk 4096, mixed off를 통일한다. Dτ와 D0 중 어느 모드인지 명시하고, 핵심 대조는 양 모드에서 재확인한다.

| 셀 | KV | graph | 목적 |
|---|---|---|---|
| A00 | K0 | G0 | REF-COMMON |
| A10 | F | G0 | KV-ONLY |
| A01 | K0 | GS | GRAPH-ONLY |
| A11 | F | GS | EPOCH-CORE |

각 셀에서 C32/C64를 우선 수행하고, 가능한 C128/160/192/224를 탐색한다. **OOM은 throughput=0이 아니라 infeasible로 표시**한다. C는 client cap, admission, 실제 B_decode를 모두 남긴다.

### 8.2 graph 최대 크기와 버킷 밀도를 별도로 비교

G0와 GS의 최대 batch size가 다르면 graph 목록 감소와 최대 batch 확대가 섞인다. 따라서 공통으로 들어가는 최대 graph 크기에서 dense/sparse 목록을 비교한다. 먼저 max=64, 이후 양쪽이 안전하게 들어가는 더 큰 max를 선택한다.

같은 max에서 실제 batch 크기를 32/33/48/63/64처럼 경계 안팎으로 바꾸고, padding rows, graph 실행 시간, CPU에 전달된 유효 행, memory footprint를 비교한다. 큰 max의 dense graph가 OOM이면 그 제약 자체를 남기고 없는 교차 셀을 추정하지 않는다.

### 8.3 동시성 제한 대조군: 가장 중요한 실험

| 셀 | EPOCH 메모리 구성 | 추가 제한 | 구분되는 효과 |
|---|---|---|---|
| C0 | A00 | 없음 | 기존 용량의 기준 |
| C1 | A11 | 서버 admission과 client C를 기준 C로 제한 | 실행 경로 개선만 남김 |
| C2 | A11 | 실제 할당 가능한 KV token 수를 기준 pool 수준으로 제한 | 추가 KV capacity의 활용 경로를 차단 |
| C3 | A11 | 인위적 제한 없음 | capacity + concurrency 전체 효과 |

C1과 C2는 서로 다른 실험이다. admission 제한과 KV 제한을 같은 것으로 취급하지 않는다. KV 제한은 실제 allocator·scheduler에 반영됐는지 확인한다. 확보한 공간을 아무 일도 하지 않는 tensor로 채워 cache/allocator에 별도 간섭을 만드는 방식은 주 대조군으로 사용하지 않는다.

C2의 graph 실행은 A11과 같아야 한다. FP8와 K0의 token당 byte가 다르므로 **동일 token capacity 대조**와 **동일 VRAM byte 대조**의 의미도 구분한다.

### 8.4 메모리 계측

rank별로 모델 로드 후, KV pool 설정 후, graph capture 전후, 워밍업 후, 실제 prefill peak 시점을 기록한다. `allocated`, `reserved`, GPU 전체 사용량을 나누고 공유 allocator 영역을 중복 합산하지 않는다.

단일 snapshot 차이를 graph의 순수 소유 메모리라고 단정하지 않는다. capture 전후 peak와 다른 버킷 정책의 통제 실행을 함께 사용한다. 안전한 가용 KV는 부팅 시 최대치가 아니라 해당 workload에서 workspace를 포함해 OOM 없이 유지되는 값으로 판정한다.

### 8.5 판정

A11이 A00보다 더 큰 실행 가능 동시성을 제공하고, C1/C2에서 줄었던 이득이 C3에서 다시 나타나면 capacity 경로를 지지한다. A11이 같은 C에서도 빨라지면 그 몫은 별도의 직접 실행 효과다. 두 효과는 상호작용할 수 있으므로 단순 퍼센트 합으로 44%를 분해하지 않는다.

row·DDR 기전은 E03에서 검증한다. capacity만 증가하고 유효 행이나 bytes가 변하지 않으면, batching·GPU 효율 등 다른 원인을 계측하고 설명을 바꾼다. 처리량 성과 자체를 삭제하지 않는다.

산출물: `memory_breakdown.csv`, `kv_graph_factorial.csv`, `capacity_cap_controls.csv`, Figure 1 메모리 분해, Figure 2 용량 제한 대조.

---

## 9. EXP-E03 — CPU 연산 효율과 가중치 스트리밍 기전 검증

### 가설

같은 hot map과 가중치를 사용할 때 EPOCH의 더 큰 실제 decode batch가 CPU expert invocation에 들어가는 유효 행을 늘리고, 유효 행 또는 출력 token당 DRAM 읽기량을 줄인다. CPU 사용률만 높아지는 것은 성공 기준이 아니다.

### 9.1 실험 셀

| 묶음 | 구성·조건 | 목적 |
|---|---|---|
| 실제 서빙 | A00/A11의 공통 C32/C64 및 각각의 최대 처리량 C | 같은 C 효과와 용량 확대 효과를 구분 |
| EPOCH 내부 sweep | C64/128/160/192/224 중 feasible 셀 | 실제 B_decode와 CPU 효율의 관계 |
| capacity 대조 | E02의 C1/C2/C3 | 메모리 확대 → CPU 연산 효율의 인과 경로 |
| 고정 routing replay | 같은 token-layer-expert 작업 집합을 작은 묶음/큰 묶음으로 재생 | routing 변화 없이 weight-read 상각 확인 |

routing replay는 **기전 분석용 오프라인 실행**이다. 미래 토큰을 미리 사용할 수 있는 온라인 알고리즘이나 SER의 부활로 설명하지 않는다. 실제 런타임의 자기회귀 의존성은 변경하지 않는다.

### 9.2 필요한 계측 필드

expert invocation별로 아래 정보를 ring buffer에 저장한다. 전량 상세 저장이 부담이면 layer/phase별 집계를 기본으로 하고 짧은 구간만 상세 수집한다.

```text
run_id, forward_id, phase, layer_id, expert_id,
logical_invocation_id, rank, socket, pool,
request_token_key_hash, useful_rows, prefill_rows, decode_rows,
padding_rows, discarded_rows, kernel_kind,
weight_payload_bytes, scale_metadata_bytes,
cpu_enqueue, cpu_start, cpu_done, result_consumed,
numa_local_bytes_if_available, trace_sample_weight
```

동일 expert의 CPU 처리가 소켓별 shard인지 복제인지 확인한다. 논리 invocation 수는 같은 작업의 shard를 중복 세지 않되, **물리 DRAM bytes는 실제로 읽은 모든 소켓의 양을 포함**한다. 일부 trace 표본만 수집했으면 물리 전체 카운터와 단순히 나누지 않는다. 동일 측정창의 전체 집계 또는 명시적 가중 추정을 사용한다.

### 9.3 반드시 함께 보고할 지표

| 지표 | 보고 단위 |
|---|---|
| 유효 expert rows/s | 순수 decode, prefill, mixed 각각 |
| invocation당 행 | 전체 ratio, p50/p90/p99, 1행·2행·3행 이상 비율 |
| cold routing 비율 | 전체 token-expert assignment 중 CPU 비중 |
| 논리 weight bytes | 전체와 output-token당·expert-row당 |
| 물리 DRAM read/write | 소켓별 및 양 소켓 합, GB/s와 누적 bytes |
| CPU 시간 | queue wait, useful compute, result-ready 이후 소비 대기 |
| 커널 선택 | AMX/AVX 호출 비율, 각 경로의 유효 행·시간 |
| 유효 결과 비율 | 실제 소비한 유효 행 / 계산한 유효 행 |

expert별 평균을 다시 단순 평균한 값과 전체 rows/전체 invocation 비율을 섞지 않는다. 크기가 다른 expert가 있다면 byte-weighted 지표도 제시한다.

### 9.4 DRAM 계측의 오염 통제

PERF run에서는 양 소켓 memory-controller read/write를 같은 주기로 수집한다. 1초 주기로 시작하고 시간 해상도가 부족한 원인 분석만 짧게 고해상도 측정한다. background process, 모델 로딩, 로그 flush, client가 같은 CPU를 사용하는 구간을 기록한다.

전체 서빙 카운터는 `system_dram_read_bytes`로 부른다. expert 외 traffic을 분리하지 못했으면 `expert_dram_bytes`라고 부르지 않는다. expert replay는 해당 계산 외 작업을 최소화한 별도 실행에서 측정해 두 결과를 연결한다. idle traffic 보정을 사용하더라도 raw 값과 보정값을 모두 저장한다.

### 9.5 수치 일관성 검사

```text
전체 DRAM read bytes ≈ Σ_socket ∫ read_bandwidth_socket(t) dt
전체 DRAM bytes/output-token = 전체 read bytes / 같은 시간창의 output token 수
```

예를 들어 **가상의** 전체 read bandwidth 400 GB/s와 처리량 1000 tok/s라면 전체 bytes/token은 약 400 MB다. 이는 기존 3.1 MB를 재계산한 값이 아니라, 층·expert·socket 단위를 빠뜨렸는지 찾기 위한 차원 검사 예시다. 기존 3.1 MB의 분모가 다르면 정확한 이름으로 바꾸고 원래 값은 기록에 남긴다.

보고된 3.75 → 3.1 MB를 같은 작업·같은 범위의 byte 감소라고 가정하면, 그 비용 항목만의 가속비는 `3.75/3.1 ≈ 1.21`이다. 이 비율만으로 전체 44%를 설명하지 않는다. 전체 실행 중 해당 항목의 비중을 f라고 할 때 단순 직렬 근사는 다음과 같다.

```text
speedup ≈ 1 / ((1-f) + f × 3.1/3.75)
```

실제 시스템에는 동시성·CPU/GPU 중첩·GPU 효율 변화가 있으므로 이 식은 **일관성 검사**에만 쓴다. 기록의 `decode 층 시간 88%`를 전체 서빙의 f로 대입하지 않는다. [S1 §7]

### 판정

주요 capacity 대조에서 `useful CPU rows/s`가 증가하고 `DRAM bytes/useful-row`가 줄며 처리량도 증가하면 CPU 연산 효율 개선의 기전을 지지한다. 제안 실용 목표는 bytes/useful-row 10% 이상 감소지만, 효과 크기와 신뢰구간을 그대로 보고한다. 이 목표 미달은 44% 처리량 성과의 기각이 아니라 **기여 설명 수정 사유**다.

산출물: `expert_histograms.csv`, `dram_counters.csv`, `streaming_efficiency.csv`, Figure 3 CPU 유효 행·DDR 효율 곡선.

---

## 10. EXP-E04 — CPU·GPU 동시 실행과 임계 경로 검증

### 목적

CPU가 단지 DRAM 저장 장치로 쓰인 것이 아니라 실제 expert 연산을 수행하고, 그 결과가 GPU 경로와 함께 유효하게 사용됨을 보여준다. 동시에 CPU 부하가 커져 GPU를 기다리게만 하는 구조인지 확인한다.

### 10.1 timeline 계측

A00/C64, A11/C64, A11/최적 C에서 대표 decode step과 prefill/mixed step을 채집한다. 다음 이벤트를 같은 logical forward/layer ID로 연결한다.

```text
GPU input/routing ready
CPU task enqueue → worker start → expert compute done
GPU independent useful kernels
CPU result transfer/availability
result consume/join
next dependent useful GPU kernel
```

CPU와 GPU timestamp를 바로 빼지 않는다. profiler의 clock correlation 또는 명시적인 시간축 보정으로 정렬한다. CPU의 작업 시간 합과 GPU kernel 시간 합을 더해 TPOT를 예측하지 않는다.

### 10.2 CPU 연산량 축소 대조

기본 2개 pool·가중치 배치·hot map을 유지하고 CPU workers를 `48 / 72 / 96`으로 바꾼다. 두 socket에 균형 있게 배치한다. 구성에 따라 기존 변환 NUMA layout과 threadpool 설정이 불일치하면 먼저 호환성을 검증하고, 자동으로 1 pool로 바꾸지 않는다.

공통 C64와 EPOCH 최적 C에서 측정한다. worker 축소로 DRAM bandwidth, rows/s, W_exposed, TPOT가 어떻게 변하는지 본다. GPU-only가 불가능한 480B에서 CPU를 완전히 제거한 셀을 0 tok/s 기준선으로 만들지는 않는다.

### 10.3 직렬화 진단 대조 — 선택 항목

같은 연산과 가중치를 사용하면서 CPU/GPU 유효 작업의 중첩만 제거하는 debug 모드를 만들 수 있을 때 한정한다. D0에서 dependency event를 추가해 같은 연산을 직렬화한다. graph를 끄거나 다른 kernel로 바꾸면 중첩 효과와 런타임 효과가 섞이므로 그 결과는 별도 진단으로 분류한다.

기존 graph 경로를 보존한 직렬화가 어렵다면 **억지로 구현하지 않는다.** 고정-input layer replay와 CPU worker 민감도, 실제 dependency timeline만으로 검증한다. 이 실험은 GRC 재구현이 아니다.

### 판정

유효 CPU 결과 소비율, O_CPU, W_exposed, 유효 CPU rows/s와 전체 처리량을 함께 제시한다. worker 축소 시 처리량이 줄어도 그것만으로 새로운 알고리즘의 우월성을 주장하지 않는다. **E02/E03의 효율 개선이 실제 임계 경로 단축 또는 더 높은 유효 작업 처리량으로 연결되는지**가 핵심이다.

산출물: `critical_path_events`, `cpu_scaling.csv`, Figure 4 동시 실행 timeline. 수치가 보여주지 않는 완전한 overlap을 도식에 그리지 않는다.

---

## 11. EXP-E05 — EPOCH와 실행 패치의 기여 분리

### 목적

44% 개선을 FP8·graph·커널 패치·운영 정책 중 어디까지 귀속할 수 있는지 확인한다. 누적 개선 퍼센트를 더하거나 곱해 각 기법의 독립 효과라고 주장하지 않는다.

### 11.1 먼저 작은 2×2 상호작용 실험

R0는 정확성 수정이 포함된 이전의 유효 실행 경로, R1은 최종 CPU 실행 패치 묶음이다. correctness fix는 R0에서도 제거하지 않는다.

| 셀 | E: KV/graph capacity 설계 | R: 실행 패치 |
|---|---|---|
| F00 | 이전 | R0 |
| F10 | EPOCH | R0 |
| F01 | 이전 | R1 |
| F11 | EPOCH | R1 |

공통 C64에서 먼저 수행하고, 이후 각 셀의 최대 처리량 구성도 평가한다. 어느 셀이 빌드·정확성 문제로 구현되지 않으면 `incompatible`로 남기며, 가상의 값을 넣어 interaction을 계산하지 않는다.

같은 C에서 처리량 Y의 interaction은 다음처럼 요약할 수 있다.

```text
I_ER = [Y(F11)-Y(F01)] - [Y(F10)-Y(F00)]
```

최대 처리량에서 동일 식을 사용할 경우 Y는 각 셀에서 최적화된 성능이므로, **구성 최적화까지 포함한 상호작용**이라고 설명한다. 고정 C의 순수 실행 효과와 혼동하지 않는다.

### 11.2 대표 셀에서만 개별 ablation

| 항목 | 대조 방법 | 반드시 통제할 것 |
|---|---|---|
| expert별 AMX/AVX 선택 | 최종 선택 규칙 vs 이전의 유효 규칙 | 가중치·routing·동일 kernel 구현 |
| AVX register blocking/prefetch | 원본 vs 패치 | CPU clock·placement·cache 조건 |
| gather·양자화 융합 | 동일 입력의 두 경로 | 양자화 수치 차이 별도 기록 |
| callback-free | 유효한 callback 경로와 비교 | 정확성·동일 task 수·graph 조건 |
| mixed forward | off/on | prefill 청크·요청·부하 동일; E08에서 기전 분석 |
| deferral | D0/Dτ | 다른 env·최대 deferred 수·hot map 고정 |

전체 조합 폭발을 피하기 위해 C64와 EPOCH 최적 C의 두 대표점에서 시작한다. E×R interaction이 크거나 효과의 부호가 달라진 항목만 추가 C를 측정한다.

### 판정

각 요소가 항상 좋아야 한다는 기준을 두지 않는다. 예를 들어 AMX 선택 패치가 큰 batch에서만 이득이면 EPOCH와의 상호작용으로 설명한다. 최종 주장에는 가장 강한 기존 기준선 대비 전체 개선과, 이 실험의 독립·상호작용 결과를 함께 사용한다.

산출물: `factorial_ER.csv`, `ablation_components.csv`, Figure 5 통제된 ablation. 개발 연표의 19.7배 그래프는 별도 부록에 둔다.

---

## 12. EXP-E06 — 최대 처리량과 지연 경계

### 가설

EPOCH는 기존 구성보다 큰 실제 동시성을 지원하여 같은 하드웨어의 최대 출력 처리량을 높인다. 일부 낮은 C에서 이득이 작거나 TPOT가 증가해도 최대 처리량 주장을 별도로 평가한다.

### 12.1 비교군과 C sweep

주 비교는 REF-COMMON, EPOCH-CORE, TUNED-EXISTING이다. EPOCH-OPS는 운영용 경계로 별도 표시한다. HIST-769는 참고점이다.

```text
C_client 후보: 32, 64, 96, 128, 160, 192, 224
추가 C256: 안전한 메모리 여유가 실측된 경우에만 경계 확인
```

현재 기록에서 EPOCH-B의 상한은 224다. C256을 실행 가능하다고 가정하지 않는다. 큰 C가 admission queue만 늘리고 B_decode를 늘리지 못하면 그 사실을 표시한다. [S1 §7]

### 12.2 두 종류의 workload

**유한 burst:** 동일한 입력 pool에서 같은 수의 요청을 준비해 각 C로 처리한다. 확인용 기본 요청 수는 2048개로 제안하되, 과거 3C 프로토콜은 E01에만 별도 보존한다. 첫 전송부터 전체 완료까지 T_job, 지연, fill/drain 구간을 보고한다.

**지속 포화:** 완료된 요청을 즉시 보충해 설정한 outstanding C를 유지한다. 워밍업 후 최소 600초 및 최소 2000개 완료를 탐색 기준으로 사용한다. 핵심 최적점과 인접점은 5회 반복한다. 정상상태 시간창의 T_out과 B_decode, backlog, KV occupancy를 사용한다.

두 종류의 요청 token ID·출력 길이·cache 정책을 고정한다. 새 결과의 protocol이 과거와 달라졌으므로 769/1110을 새 곡선에 동일 측정값처럼 끼워 넣지 않는다.

### 12.3 보고할 곡선

1. X=C_client, Y=출력 처리량. 실제 B_decode와 KV occupancy를 보조 패널/표에 표시.
2. X=TPOT p99, Y=출력 처리량. TTFT p99 조건을 구분해 여러 경계를 작성.
3. X=TTFT p99, Y=출력 처리량. mixed off/on을 분리.
4. X=실측 KV token capacity, Y=안정적 최대 처리량. E02 대조군 포함.

Pareto 지배 여부는 같은 모델·품질·자원·workload에서만 판정한다. 다른 품질 모드·다른 출력 길이의 점을 같은 경계에 넣지 않는다.

### 판정 기준

주 성공 기준은 EPOCH 최대 처리량 / 기존 최대 처리량의 짝 비교 신뢰구간 하한이 1보다 큰 것이다. 원래의 실용 목표인 +15%도 별도 표시한다. 평균 +15%지만 신뢰구간이 넓으면 `유망하나 불확실`로 기록한다.

같은 C에서만 개선이 작다는 이유로 실패로 판정하지 않는다. 반대로 EPOCH만 C를 최적화하고 기준선은 C64로 고정한 비교를 최종 최대 성능 주장으로 사용하지 않는다.

산출물: `saturation_frontier.csv`, `burst_job_results.csv`, Figure 6–7 처리량–지연 경계.

---

## 13. EXP-E07 — 동일 SLO의 최대 지속 서비스 용량

### 목적

"EPOCH가 좋은 운영점을 선택한 것"과 "같은 서비스 지연 조건에서 더 많은 요청을 처리하는 것"을 구분한다. 최대 처리량 성과는 E06에서 독립적으로 보존한다.

### 13.1 open-loop 도착열

새 실험의 λ는 **초당 요청 수**로 명시한다. 기존 rate 5/7/9의 정확한 단위와 클라이언트 옵션은 E00에서 별도로 확인한다.

초기 후보는 `3, 5, 7, 9, 11 req/s`다. 먼저 5/7/9를 탐색하고 포화 경계의 위·아래 값만 추가한다. 후보 사이의 경계는 0.5 req/s 간격으로 좁힌다. 모든 구성은 같은 입력·계획 도착 timestamp 파일을 재생한다.

vLLM의 request-rate와 max-concurrency를 함께 사용하면 실제 전송 도착률이 설정값보다 낮아질 수 있다. 따라서 open-loop 실험에서는 클라이언트의 숨은 대기열을 제거하거나, 계획 도착과 실제 전송을 모두 계측해 TTFT_user에 포함한다. CLI 값만 보고 실제 λ를 판단하지 않는다. [W1]

### 13.2 SLO를 결과 확인 전에 고정

다음은 제안 평가 grid이며 기존 운영 계약이 아니다.

| 축 | 제안 값 |
|---|---|
| TTFT p99 상한 | 1 / 2 / 5 / 10 s |
| TPOT p99 상한 | 50 / 100 / 200 / 300 ms |
| 오류·timeout·거절 비율 | 전체 계획 도착 대비 ≤0.1% |
| 요청 단위 공동 SLO 충족 비율 | ≥99% |

주 그림은 `TTFT 2s·TPOT 100ms`, `5s·200ms`, `10s·300ms` 세 조건으로 시작하고 전체 grid를 부록에 제시한다. 불가능한 셀은 `SLO-feasible point 없음`으로 표시한다. 기준선이나 EPOCH가 유리해 보이는 한 조건만 사후 선택하지 않는다.

원래 상대 목표도 유지한다.

```text
동일 workload와 비교 가능한 서비스 조건에서:
처리량 증분 >= 15%
TPOT p99 비율 <= 1.05
```

이 목표와 절대 SLO 평가는 서로 다른 표에 둔다. 낮은 고정 도착률에서는 두 시스템 처리량이 모두 `λ × 평균 출력 길이`로 결정될 수 있으므로, 거기서 처리량 차이가 작다고 포화 용량의 차이까지 부정하지 않는다.

### 13.3 지속 가능성의 판정

최대 λ는 순간적으로 높은 token/s가 나오는 지점이 아니라, 다음 조건을 만족하는 가장 높은 측정 도착률로 정한다.

```text
SLO 통과
오류·timeout·거절 비율 통과
후반부 backlog가 지속적으로 증가하지 않음
KV retraction·재계산이 폭증하지 않음
완료/실패와 도착량의 차이가 누적 발산하지 않음
```

backlog는 적어도 1초 단위로 저장한다. 마지막 측정 구간에서 backlog slope와 신뢰구간을 계산하고, 제안 기준 `slope의 상한 < 0.01×λ` 및 후반 TTFT의 지속 상승 부재를 함께 확인한다. 이 경험적 기준은 유한 측정의 안정성 판단이지 수학적인 무한시간 안정성 증명이 아니다.

### 13.4 표본과 timeout

탐색은 3 seed, 각 600초 이상으로 수행한다. 논문에서 p99·최대 λ를 주장하는 경계점은 **5개의 독립 도착열, 각 10,000개 이상 계획 도착**으로 확인하는 것을 기본으로 한다. 자원이 부족해 축소하면 표본 수와 p99 신뢰구간을 공개하고 정밀한 tail 보장을 주장하지 않는다.

워밍업과 측정 cohort를 분리한다. 모든 측정 도착 요청은 완료 또는 사전에 고정한 timeout까지 추적한다. 제안 timeout은 `max(120초, 2×[L_F+(O_max-1)×L_D])`이며 동일 workload의 비교군에 동일 적용한다. 긴 출력 조건에서는 이 식에 맞춰 늘린다.

실패·거절·timeout을 제거하고 성공 요청 p99만 계산하지 않는다. 실패율을 전체 계획 도착 분모로 계산하고, SLO 결과에서는 해당 요청을 실패로 처리한다. 지연 quantile은 성공 요청 조건부 값과 실패를 무한 지연으로 간주한 보수적 값을 구분해 저장한다.

### 13.5 결과표

```text
config, workload, quality_mode, slo_ttft_ms, slo_tpot_ms,
lambda_offered, lambda_sent, lambda_completed,
T_out, G_req, G_tok,
ttft_user_p50/p95/p99, tpot_p50/p95/p99,
fail_rate, backlog_slope, kv_retractions,
feasible, infeasible_reason, ci95
```

### 판정

엄격 SLO에서 이득이 없고 완화된 SLO에서만 용량이 늘면 "고처리량 서비스 영역에서 효과"로 쓴다. 어떤 SLO에서도 EPOCH가 유리하지 않더라도 E06의 batch/offline 최대 처리량 개선은 유지한다. 대신 온라인 동일-SLO 우위 주장은 하지 않는다.

산출물: `open_loop_runs.csv`, `slo_capacity.csv`, Figure 8 동일 SLO의 최대 λ·출력 처리량.

---

## 14. EXP-E08 — mixed forward의 동일 작업량·동일 expert 읽기 검증

### 목적

기록의 rows/read 1.58 → 24.5가 실제로 **같은 유효 작업의 중복 가중치 읽기를 줄이는가**를 확인한다. prefill 행이 분자에 추가된 것만으로 생기는 지표 상승을 분리한다. mixed 자체는 이미 기존 시스템의 기능이므로, 옵션 활성화만을 신규 알고리즘이라고 주장하지 않는다. [S1 §9; W2; W4]

### 14.1 layer/operator 수준의 정확한 대조

독립적인 prefill 요청 P와 decode 요청 D에서 같은 layer 입력 activation과 routing을 채집한다. 작업 집합·가중치·token-expert assignment는 고정한다.

| 셀 | 실행 | 비교 목적 |
|---|---|---|
| M0 | P expert 작업과 D expert 작업을 별도로 실행 | separate 기준 |
| M1 | 같은 expert의 P/D 행을 묶어 한 invocation으로 실행 | 실제 coalescing 효과 |
| M2 | M1과 같은 scheduler batch이지만 CPU expert 호출은 P/D로 분리 | batch 형태 효과와 CPU 읽기 공유 효과 분리 |
| M3 | 현재 mixed 구현 그대로 | 현재 코드가 M1인지, 추가 분리·재읽기가 있는지 확인 |

M2는 구현하기 쉬운 replay 수준에서 우선 수행한다. 실서빙에 새로운 복잡한 scheduler를 넣을 필요는 없다. 출력은 원래 row 순서로 되돌려 비교한다.

같은 expert 연산을 바로 연달아 재생하면 cache locality가 실제 separate 서빙과 다를 수 있다. 따라서 짧은 operator replay와 실제 layer 순서를 보존한 trace replay를 구분한다. 임의로 매번 cache를 flush해 coalescing 효과를 과장하지 않는다.

### 14.2 비교할 비용

```text
ΔB = B_separate(P,D) - B_mixed(P,D)
ΔT_CPU = T_CPU_separate(P,D) - T_CPU_mixed(P,D)
ΔW_exposed = W_separate(P,D) - W_mixed(P,D)
```

동일 P,D에서 nominal weight bytes, 실제 DRAM bytes, CPU wall time, 유효 GPU compute, 전체 step makespan을 측정한다. 별도 실행의 CPU 시간 두 개를 더한 값과 mixed wall time을 비교할 때, 기준선에서 원래 중첩 가능했던 시간을 제거하지 않도록 한다.

직관상 P와 D가 공통으로 사용하는 cold expert에서 읽기 공유가 가능하다. 하지만 실제 공유 이득은 cache residency와 kernel tiling에 달려 있으므로 `공통 expert weight 합 = 실측 절감 byte`라고 단정하지 않는다. 해당 합은 논리 proxy로만 사용한다.

### 14.3 행 수 조건

실제 trace에서 자주 나오는 `(prefill_rows, decode_rows)` 쌍을 먼저 사용한다. 보조 microbenchmark 후보는 prefill `0/16/64/256/1024`, decode `1/2/4/8`이다. 모두 실현 가능한 operator 입력 형상에 한정한다. 관측되지 않은 대형 형상에서만 좋아지는 결과를 실제 서빙 개선으로 일반화하지 않는다.

### 14.4 실제 서빙 검증

EPOCH-CORE와 EPOCH-OPS를 같은 chunk와 C64/160/224에서 비교한다. open-loop는 같은 λ=5/7/9의 도착열을 사용한다. delayer는 기본 off이며, mixed만의 효과를 확인한 뒤 별도 on 대조를 추가한다.

512/128 외에 긴 입력 4096/128과 긴 출력 512/1024에서 검사한다. 단계별 prefill token 수·decode token 수·공통 cold expert 수·cache hit·CPU bytes를 저장한다. 최종 지연은 E07 정의를 따른다.

### 판정

M1/M3에서 동일 작업량의 물리 byte와 CPU 시간이 줄면 CPU weight streaming 상각 기전이 강화된다. rows/invocation만 증가하고 물리 byte·시간이 줄지 않으면 혼합 배치의 이득을 그 지표로 설명하지 않는다.

TTFT 개선이 크지만 TPOT·최대 처리량이 나빠지면 기록처럼 운영상 trade-off로 보고한다. **혼합 실행으로 EPOCH 전체의 성공·실패를 결정하지 않는다.**

산출물: `mixed_matched_work.csv`, `mixed_serving.csv`, Figure 9 동일 P/D의 DRAM·시간 비교.

---

## 15. EXP-E09 — 수치 정확성, KV 품질, deferral 영향의 분리 검증

### 목적

기존 97/100 점수는 보존하면서, 논문에서 `bit 동일`, `수치적으로 근접`, `품질 비열등`을 구분한다. 재부팅 변동보다 작다는 이유만으로 출력 동등성이 입증됐다고 판단하지 않는다. [S1 §3]

### 15.1 세 층의 검증

| 층 | 검증 대상 | 기준 |
|---|---|---|
| 구조적 정확성 | expert 선택·CPU/GPU 중복·결과 소비·token dependency | 누락·중복·잘못된 join·NaN/Inf가 없어야 함 |
| 수치 동등성 | same-input kernel, graph padding, fusion, 캐시 경로 | 변경 성격에 맞는 exact 또는 수치 오차 분포 |
| 과제 품질 | FP8 KV, Dτ, mixed를 포함한 실제 생성 | 사전 정의한 비열등성 한계와 신뢰구간 |

### 15.2 구조적 검증

토큰별 routed expert ID·weight, CPU/GPU ownership, 결과 합산 횟수를 점검한다. CPU가 계산한 결과가 정확히 한 번 소비됐는지 확인한다. deferred 결과는 원래 필요한 소비 지점 전인지 후인지 기록한다.

Dτ가 동일 함수의 단순 실행 순서 변경이라면 dependency 관점에서 설명할 수 있어야 한다. 결과가 필요한 계산을 먼저 진행하고 나중에 보정하는 방식이면 근사 실행으로 분류해 품질 비용을 따로 평가한다. 코드가 첨부되지 않아 현재 문서에서는 어느 쪽이라고 단정하지 않는다.

### 15.3 같은 입력의 kernel·graph 검사

원본과 최적화 kernel에 같은 activation·INT4 payload·scale·routing을 넣는다. 행 수는 실제 분포에 더해 `1,2,3,4,7,8,15,16,17,31,32,33,63,64,65` 및 graph 경계 `127/128/129,159/160/161,191/192/193,223/224`를 검사한다.

동일 연산 순서를 유지한 bit-preserving 패치는 bit 동일성을 요구한다. 연산 순서·정밀도가 달라진 경로는 절대·상대 오차, p50/p99/max, argmax 변화율, NaN/Inf를 보고한다. 단일 임계값을 모든 tensor에 적용하지 말고 reference scale과 downstream 오차를 같이 본다.

자료의 `4개 형상에서 max abs diff 0`과 `양자화 0.016% ±1`을 확대 검증할 뿐, 이미 전 형상이 동일하다고 간주하지 않는다. [S1 §3]

### 15.4 teacher-forced incremental decode

다음 대조를 한 번에 하나씩 변경한다.

| 비교 | 고정 | 변경 |
|---|---|---|
| Q1 | 가중치·routing 입력·KV dtype·D0 | 이전 kernel vs 최적화 kernel |
| Q2 | KV·D0·동일 batch | dense vs sparse graph |
| Q3 | graph·D0·가중치 | K0 vs FP8 KV |
| Q4 | KV·graph·가중치 | D0 vs Dτ |
| Q5 | 같은 P/D 입력·결과 순서 | separate vs mixed |

KV 검증은 전체 텍스트를 한 번에 넣는 prefill-only teacher forcing만으로 끝내지 않는다. 같은 정답 token을 한 token씩 주입하는 **실제 cached incremental decode**에서 KV를 계속 읽고 갱신한다. 짧은 문맥과 4k/8k 문맥을 포함한다.

첫 divergence가 발생한 layer·expert·연산을 저장한다. gold-token logprob의 ΔNLL, PPL ratio, top-1 일치율, 출력 token 일치율을 구분한다. 전체 분포를 확보하지 못한 top-k 로그만으로 전체 KL divergence를 계산했다고 쓰지 않는다.

같은 구성의 반복과 구성 간 짝 비교를 별도로 수행한다. nondeterminism을 줄이는 진단 모드가 성능 경로와 다르면 품질 디버깅용으로 표시하고, 그 모드의 속도를 주 성능값으로 사용하지 않는다.

### 15.5 과제 수준 비열등성

기존 GSM8K 앞 100문항은 회귀 검사로 유지한다. 최종 주장은 그 100문항만으로 하지 않는다. tuning에 사용한 문항을 제외한 고정 평가 집합을 우선 정하고, 전체 benchmark 결과도 별도로 병기한다. 같은 문항의 재시작 반복을 서로 다른 문제 표본처럼 세지 않는다.

주력 모델이 Coder이므로 코드 생성·단위테스트 기반 평가 하나를 추가한다. 현재 첨부에는 해당 데이터셋·하네스가 없으므로 이름·버전·split·문제 수를 확정해 manifest에 등록한 뒤 수행한다. 생성 코드는 외부 네트워크·비밀정보 접근이 없는 격리 환경에서 자원 제한을 두고 실행한다.

제안 비열등성 한계는 주 과제 정확도 `-1.0 percentage point`다. 과제 특성상 다른 한계가 필요하면 결과를 보기 전에 근거와 함께 고정한다. 기준선 대비 차이의 단측 95% 신뢰구간 하한이 이 한계 이상인지 판단한다. 표본이 작아 구간이 넓으면 **불확정**으로 남기고, 결과를 본 뒤 한계를 완화하지 않는다.

### 실패 시 조치

- 구조적 오류: 해당 비교의 성능값을 보류하고 수정 후 양쪽 재측정.
- Dτ만 품질 하락: D0에서 EPOCH 효과를 주 주장으로 전환하고 Dτ는 품질–성능 변형으로 분리.
- FP8 KV만 품질 하락: 같은 FP8 품질 모드끼리 graph·구성 선택 효과를 평가하거나, 통과하는 KV 설정으로 재측정.
- bit 동일성 미달이지만 과제 비열등성 통과: `출력 동일` 대신 실제 입증 범위로 기술.

산출물: `correctness_cases.csv`, `incremental_decode_diff.csv`, `quality_paired_results.csv`, Figure/Table 10 품질 조건별 성능.

---

## 16. EXP-E10 — 워크로드·모델·자원에 대한 일반화

### 목적

sonnet 512/128 한 조건의 튜닝 결과에 그치지 않고 EPOCH가 이기는 영역과 이기지 못하는 영역을 보여준다. 모든 모델에서 CPU/GPU hybrid가 GPU-only보다 빨라야 한다는 가설은 세우지 않는다.

### 16.1 워크로드 행렬

| ID | 입력/목표 출력 tokens | 텍스트 성격 | 주 검증 대상 |
|---|---:|---|---|
| W0 | 512/128 | 기존 sonnet | 기존 성과와 연결 |
| W1 | 512/128 | 코드 설명·코드 생성 입력 | 길이는 같고 routing 분포만 다른 조건 |
| W2 | 512/128 | 일반 질의·대화 입력 | 특정 도메인 hot set 과적합 여부 |
| W3 | 4096/128 | 긴 문서 또는 코드 입력 | KV capacity와 prefill 비용 |
| W4 | 512/1024 | 긴 생성 | decode streaming 상각의 지속 효과 |
| W5 | 4096/512 | 긴 입력과 중간 출력 | 더 강한 KV 제약 |
| W6 | W0/W3/W4를 50/25/25% 혼합 | 고정 도착열 | 길이 변동과 batch 불균일 |

텍스트는 공개·사용 허가된 corpus 또는 사용자 보유 평가셋에서 고르고 dataset revision·샘플 ID를 고정한다. 실제 토큰 수를 맞추기 위해 무작위 token ID만 넣는 부하는 기전 microbenchmark로 분리한다. 의미 있는 텍스트와 routing 분포가 다를 수 있으므로 주 일반화 결과를 대신하지 않는다.

prefix는 W0에서 `공유 없음 / 기록의 prefix 조건 / 높은 공유 조건`을 구분한다. 높은 공유 조건의 제안값은 입력 token의 50%이며, 공통 prefix pool의 수와 cache hit를 같이 기록한다. 같은 prefix를 반복해서 얻은 메모리 이득을 일반적인 KV 용량 효과로 설명하지 않는다.

### 16.2 긴 문맥에서 동시성을 재탐색

C224를 긴 입력에도 강제하지 않는다. token capacity가 같아도 요청당 KV 요구가 커지면 admission 가능한 요청 수가 달라진다.

```text
필요 KV pages ≈ 현재 실제로 저장한 고유 prefix/sequence pages
                + 실행 중 요청들의 생성 증가분
                + allocator·workspace 안전 여유
```

공유 prefix를 요청마다 중복 계산하거나, future output의 전체 KV가 반드시 미리 할당된다고 가정하지 않는다. 실제 allocator 정책을 확인한다. 각 workload에서 가능한 C와 안전한 memory margin을 탐색하고, 최고 안전점과 인접점을 확인한다.

### 16.3 모델 행렬

| 모델 | 비교 | 제한 |
|---|---|---|
| 480B 주력 | REF-COMMON / EPOCH / TUNED-EXISTING | GPU-only FP8가 현재 4GPU 구성에서 불가능했던 사실과 분리 |
| 이력의 235B | GPU-only / 기존 hybrid / EPOCH 동일 원칙 적용 | 정확한 checkpoint·정밀도·TP 설정은 원시 기록에서 복구 |
| 이력의 30B | 안정적인 hybrid 경로를 확보한 경우만 보조 비교 | 기존 segfault 상태를 성능값으로 사용하지 않음 |

235B에서 GPU-only 우위였던 기존 반례는 삭제하지 않는다. 같은 가중치 정밀도·품질·GPU 수가 아니면 순수 실행 효율 비교가 아니라 자원·품질을 포함한 별도 비교로 표시한다. [S3 §2.6]

480B에서 GPU-only OOM을 `0 tok/s`로 두고 speedup을 계산하지 않는다. 8GPU GPU-only, 다른 양자화, 다른 모델을 추가할 수는 있지만 그 결과는 별도의 자원/품질 trade-off 표에 둔다. GPU-only가 들어가는 더 작은 모델을 억지로 메모리 제한해 이득을 만들 경우에는 인위적 capacity-limited 실험이라고 명시한다.

### 16.4 자원 민감도

E04의 CPU 48/72/96 workers 결과를 재사용한다. 추가 실험은 같은 node의 여유 GPU가 다른 작업을 수행하는 상황이 아니라, 먼저 격리된 조건에서 진행한다. 두 번째 실제 하드웨어가 있으면 별도 일반화로 추가하되, CPU core 제한만으로 다른 CPU 아키텍처를 실험했다고 주장하지 않는다.

hot-80/88의 과거 실패 실험을 전부 다시 돌리지 않는다. 기존 자료를 경계 사례로 보존한다. 새로운 graph 메모리 확보로 hot-100 등이 실제로 들어갈 가능성을 확인하려면 먼저 allocator 실측으로 검증하고, 기존 KV를 희생하는 거래와 구분한다. 이것은 선택적 확장이지 필수 성공 조건이 아니다. [S1 §6, §9]

### 판정

주력 W0 외 최소 두 조건에서 기전과 성능 효과가 재현되는지를 우선 확인한다. GPU-only가 가능한 모델에서 손해라면 그 경계를 명시한다. 부호가 바뀌는 이유를 CPU cold byte, KV 압박, batch 크기, GPU 병목으로 설명한다.

산출물: `generalization_matrix.csv`, `applicability_boundaries.md`, Figure 11–12 workload·자원별 성능 경계.

---

## 17. EXP-E11 — EPOCH 구성 선택 규칙의 알고리즘화와 검증

**선택적 확장이다. 기존 EPOCH 성과의 재현을 이 작업의 성공에 종속시키지 않는다.**

### 17.1 필요한 이유

현재 EPOCH의 실현형은 유효한 시스템 구성이다. 논문에서 "특정 옵션 조합을 찾았다"를 넘어 "어떤 조건에서 어떤 메모리·실행 구성을 선택해야 하는지 제시했다"를 주장하려면, 선택 규칙과 검증이 필요하다.

FP8 KV·graph 버킷·mixed는 기존 기능이므로 그 자체를 새 알고리즘이라고 부르지 않는다. EPOCH의 추가 기여 후보는 **CPU expert의 가중치 읽기 비용과 GPU 메모리 기회비용을 함께 반영한 구성 선택**이다. 이 절의 정책은 새 제안이며, 아직 구현되거나 신규성이 입증된 것으로 취급하지 않는다. [W2; W4]

### 17.2 초기 범위를 작게 제한

처음에는 hot-96, 통과한 KV dtype, 고정한 D0 또는 Dτ, mixed off를 유지한다. 다음 12개 이내의 후보만 사용한다.

```text
graph 최대 batch: 160 / 192 / 224
버킷 간격 정책: 약 16 / 약 32  (해당 max 포함, 실제 목록을 명시)
prefill chunk: 4096 / 8192
```

필수 작은 batch bucket과 padding 처리는 사용 중인 구현과 호환되게 설정한다. 버킷 목록을 바꾸면 capture가 실제로 어떤 그래프를 만드는지 확인한다. 물리적으로 불가능한 후보는 비용모델의 OOM 필터로 제외한 뒤 실제 부팅으로 확인한다.

동적 graph 재캡처, live expert migration, token rendezvous는 도입하지 않는다. 먼저 **배포 시 정적 구성 선택**만으로 검증한다. h=hot expert 수, q=KV dtype까지 확대하는 것은 초기 검증 후의 선택 사항이다.

### 17.3 비용모델

구성 x를 `(h, q, graph_set, chunk, admission_cap, kernel_policy)`라 한다. memory budget은 rank별로 판정한다.

```text
M_total_rank(x, workload)
  = M_weight_rank(h)
  + M_KV_rank(q, actual_pages)
  + M_graph/workspace_rank(graph_set, chunk, batch)
  + M_runtime_other_rank

모든 rank에서 M_peak <= 실제 사용 가능 메모리 - 검증된 안전 여유
```

공유 allocator 때문에 graph와 workspace가 중복 계산되지 않도록 E02 실측을 사용한다. 하드코딩된 "expert당 70GB/96" 같은 평균으로 전체 모델 비용을 대신하지 않는다.

CPU expert 실행 비용은 다음 형태를 출발점으로 한다.

```text
T_cpu_layer = dispatch/queue 비용
              + schedule_cost({t_e(m_e, kernel, socket)}, 실제 worker 배치)

t_e = 고정비 + weight streaming 항 + 행 연산 항 + contention 항
```

기록의 `8 + 58 + 9.5*m µs`는 해당 오라클의 측정 상수이며, 모든 m·phase·thread 수에 선형 외삽하지 않는다. AMX/AVX 전환과 cache 효과를 E03의 piecewise curve로 반영한다. 두 소켓의 expert 시간을 모두 더한 값을 CPU wall time으로 사용하지 않는다. [S1 §9]

step 시간은 GPU·CPU의 실제 의존성 DAG와 join을 반영한다. 단순 `CPU 총시간 + GPU 총시간` 또는 모든 layer에 일괄 `max(CPU,GPU)`를 적용하지 않는다. mixed 모드의 경우 별도로 검증된 모델이 없으면 선택 범위에서 제외한다.

### 17.4 선택 목적

```text
선택 x* = argmax predicted sustainable output throughput(x)
          subject to memory feasibility,
                     validated quality mode,
                     TTFT/TPOT SLO safety margin
```

예측 throughput은 작은 보정 실험과 route/행 분포의 통계로 만든다. 실제 serving에서 아직 도착하지 않은 요청의 routing, 미래 정답 token, held-out 평가의 최적 C를 입력으로 사용하지 않는다.

개별 미래 expert를 정확히 예측한다고 가정하지 않는다. 사용 가능한 workload 정보는 입력·출력 길이 분포, 도착률 추정, 과거 별도 calibration routing과 실제 memory footprint다. 새로운 workload에서 예측 범위를 벗어나면 안전한 검증 구성으로 돌아가고 이유를 기록한다.

### 17.5 비교 알고리즘

| 정책 | 정의 |
|---|---|
| 고정 EPOCH-B | 현재 최종 B를 모든 workload에 사용 |
| memory-only | 안전한 KV/admission capacity를 가장 크게 만드는 구성 |
| 균일/무작위 탐색 | 동일한 실제 측정 예산으로 후보를 탐색 |
| 단일요인 탐색 | KV/graph/chunk를 순차적으로 조정 |
| EPOCH 비용모델 선택 | CPU streaming + GPU memory + 지연을 함께 반영 |
| 전수탐색 oracle | feasible 후보 전부를 측정한 사후 최선, 온라인 정책 아님 |

비용모델의 calibration·profiling 비용도 탐색 예산에 포함한다. 후보 수만 같고 한 정책은 더 많은 GPU 시간을 쓰는 비교를 하지 않는다. 총 GPU/CPU wall time, server restart 수, profile 수를 모두 보고한다.

### 17.6 학습·평가 분리와 지표

calibration용 prompt·route trace와 held-out workload를 분리한다. 최소한 입력 길이 또는 도메인 하나는 calibration에 없는 조건으로 남긴다. hot map을 평가 workload에서 다시 학습하지 않는다. 성능모델의 같은-trace 적합과 외부 예측을 별도 표로 작성한다.

```text
selection_regret = (Y_oracle - Y_selected) / Y_oracle
```

Y는 같은 SLO의 sustainable output throughput이다. 선택한 구성이 SLO를 위반하면 해당 셀을 단순 평균 속도비에 섞지 말고 `infeasible selection`으로 집계한다.

제안 목표는 held-out 선택 regret 중앙값 ≤5%, 최악 ≤10%, 검증된 안전 제약 위반 0이다. 이는 신규성이나 논문 채택을 보장하지 않는 실험 목표다. 예측 오차와 순위뿐 아니라 **실제로 선택한 구성의 성능**을 제시한다.

### 판정

모든 workload에서 고정 B가 이미 최적이고 선택 정책의 이득도 없다면, 불필요한 adaptive algorithm 주장을 추가하지 않는다. 비용모델이 동일 예산 기존 탐색보다 유리하고 다른 조건에서도 좋은 구성을 선택한다면 독립 기여 후보가 된다. 그때 정확한 입력·결정 규칙·복잡도·실행 비용·가장 가까운 선행연구와의 차이를 명시한다.

산출물: `selector_spec.md`, `calibration_split.json`, `selector_predictions.csv`, `selector_regret.csv`, Figure 13 구성 선택의 일반화·탐색 비용.

---

## 18. EXP-E12 — 장시간 안정성과 부하 변화

### 목적

짧은 burst에서의 최고치가 아니라 지속 가능한 서빙 시스템이라는 증거를 확보한다.

### 실행

품질 검증을 통과한 EPOCH-CORE, EPOCH-OPS, 강한 기준선을 각각 60분 이상 실행한다. 먼저 고정 도착률을 해당 구성의 검증된 SLO 용량보다 낮게 설정하고, 별도 동일 입력 실험에서는 같은 절대 도착률로 비교한다. 구성마다 다른 상대 도착률 결과를 같은 부하 비교라고 부르지 않는다.

추가 시나리오는 한 실행 안에서 `낮은 부하 → 검증 용량 근처 → 낮은 부하`로 바뀌는 고정 도착열이다. W6의 길이 혼합을 사용한다. 부하 수준과 전환 시각은 실행 전에 파일로 고정한다.

### 측정

메모리 reserved/allocated, KV occupancy/retraction, CPU queue depth, worker progress, request timeout, 처리량·TTFT·TPOT 추이, graph fallback/capture, GPU clock·throttle, CPU frequency를 저장한다. callback-free completion ID의 중복·누락·오래된 buffer 소비도 검사한다.

에너지/출력 token은 계측할 수 있으면 보조 표로 추가하지만, CPU 사용률이나 전력 절감으로 원래 처리량 목표를 바꾸지 않는다.

### 판정

목표는 오류·데드락·잘못된 결과 소비 0, 지속적인 메모리 증가 부재, 부하가 낮아진 뒤 backlog 회복이다. workload를 고정한 후반 처리량이 초반 정상상태 대비 5% 이상 지속 하락하면 thermal, NUMA, cache, queue 누적을 조사한다. 일시적인 변동 하나로 실패를 선언하지 말고 원시 시계열을 남긴다.

산출물: `soak_timeseries.csv`, `stability_incidents.md`, Figure 14 장시간 처리량·지연·메모리.

---

## 19. 통계·성공 기준·중단 기준

### 19.1 반복 단위와 신뢰구간

처리량의 기본 독립 반복 단위는 서로 다른 workload seed/도착열을 가진 run이다. 같은 token 수천 개를 독립 처리량 표본으로 세지 않는다. 동일 seed의 기준선·EPOCH를 짝지어 속도비를 계산한다.

p99는 요청 간·시간 간 상관이 있을 수 있으므로 run 단위 재표집 또는 run 안에서 시간 block을 보존한 bootstrap을 사용한다. 제안 분석은 10,000회 재표집과 95% 신뢰구간이다. 단, 5 run 자체가 적은 표본이라는 한계는 유지되므로 경계적인 결론은 반복을 늘린다.

paired ratio의 산술평균과 기하평균을 혼용하지 않는다. 주 결과는 run별 원시값과 기하평균 속도비·CI를 제시하고 절대 token/s도 병기한다. 지연은 per-run p99와 합친 요청 분포의 p99를 모두 저장하되, **per-run p99의 평균을 전체 p99라고 부르지 않는다.**

주 확인 비교는 `480B·W0에서 EPOCH와 TUNED-EXISTING의 최대 처리량`으로 사전 등록하고, 인과 대조군 REF-COMMON 비교는 별도로 둔다. 여러 SLO·workload의 개별 유의성을 동시에 주장할 때는 다중비교 보정을 적용하거나 탐색적 결과로 명시한다.

여러 C 중 가장 빠른 점을 고른 뒤 같은 run으로 우위를 확정하지 않는다. 탐색용 seed에서 후보를 고르고, 선택된 점과 인접점을 독립 확인 run으로 평가한다. 신뢰구간이 겹친다고 곧바로 동등하다고 판단하지 말고, 차이 또는 비율의 짝 CI를 사용한다.

### 19.2 가설별 판단

| 가설 | 주 증거 | 제안 실용 목표 | 미달 시 조치 |
|---|---|---|---|
| H1 최대 처리량 개선 | E01/E06 | 통제 기준선 대비 +15% 이상; CI 하한 >0% | 실제 이득과 CI 보고. EPOCH-X 기각과 혼동하지 않음 |
| H2 capacity의 인과 효과 | E02 | cap 제거에 따른 이득이 반복 재현 | 개선 설명을 직접 실행·GPU 효율 등으로 재분해 |
| H3 CPU streaming 효율 | E03 | bytes/useful-row 10% 이상 감소 및 useful rows/s 증가 | 해당 기전 주장 축소; 처리량 성과는 유지 |
| H4 실제 CPU/GPU 동시 기여 | E04 | 결과 소비 정상, 유효 중첩·CPU 민감도 확인 | 단순 저장 효과와 연산 기여를 구분해 서술 |
| H5 동일 SLO의 용량 개선 | E07 | 사전 정의한 SLO 일부에서 +10% 이상, CI 하한 >0% | 적용 범위를 batch/완화 SLO로 명시 |
| H6 품질 조건 | E09 | 구조 오류 0, 선택한 품질 한계 통과 | 영향받는 정밀도/deferral 모드만 분리 |
| H7 일반화 | E10 | W0 외 최소 두 workload에서 효과 재현 | 특정 workload·메모리 체제의 결과로 한정 |
| H8 선택 알고리즘 | E11 | held-out regret·예산 조건 통과 | 선택적 알고리즘 주장을 추가하지 않음 |

이 표의 +10%, 10% byte 감소 등은 새로운 목표이며 기존 EPOCH 목표를 덮어쓰지 않는다. "목표 미달"과 "관측된 효과 없음"도 구분한다.

### 19.3 실험을 중단하거나 무효화해야 하는 경우

정확성 결함, 서로 다른 workload/precision, 실제 적용되지 않은 env, NUMA 배치 불일치, profiler 과도한 간섭, 숨은 client queue, thermal/다른 job 간섭으로 비교 조건이 깨진 경우는 해당 **run 또는 비교**를 무효화한다. 원인과 제외 사유를 사전에 정한 규칙대로 남긴다.

OOM, SLO 미달, 일관된 성능 손해는 실패 결과로 보존한다. 유리하지 않다는 이유로 제외하지 않는다. 특정 셀에서 이득이 없다고 전체 EPOCH를 기각하지 않는다.

---

## 20. 구현해야 할 계측·분석 작업

새로운 대형 runtime을 만들기보다 현재 패치에 관측 지점을 추가한다.

| 작업 | 위치·역할 | 주의점 |
|---|---|---|
| config dumper | launch 직후 실제 args/env·graph 목록·dtype 저장 | secret를 포함한 전체 env dump 대신 allowlist |
| memory recorder | rank별 weight/KV/capture/workspace 계측 | 공유 allocator 중복 합산 금지 |
| expert counters | worker dispatch 전후 useful/padding/phase별 행·invocation 집계 | TP·socket shard 중복 여부 명시 |
| handoff trace | enqueue/start/done/consume ID와 timestamp | profiling용 전역 synchronize 금지 |
| arrival replay | 계획 도착열 재생 및 planned/sent/server-admit 기록 | client cap이 도착률을 조용히 낮추지 않도록 함 |
| token timing | first/last·가능한 token event 저장 | multi-token chunk의 ITL 불확실성 표시 |
| paired analyzer | run별 ratio/CI·SLO·backlog·통제 셀 분석 | 실패 run의 원시 결과 보존 |
| correctness replay | 고정 activation·routing 및 incremental decode | 실제 serving 품질 검사와 구분 |

### 20.1 결과 디렉토리 규약

아래는 제안 경로이며 현재 존재하는 파일이라고 가정하지 않는다.

```text
eval/results/epoch_paper/
  source_records/
  preregistration/
    claims.json
    workload_manifest.json
    baseline_patch_ledger.csv
    frozen_matrix.yaml
  EXP-E00/
  EXP-E01/
  ...
  EXP-E12/
  runs/<run_id>/
    manifest.json
    server_args_resolved.json
    server_stdout.log
    benchmark_summary.json
    request_events.jsonl
    token_events.jsonl
    expert_counters.jsonl
    hardware_counters.csv
    memory_snapshots.jsonl
    trace_metadata.json
    outcome.json
  analysis/
    all_runs.csv
    paired_comparisons.csv
    infeasible_cells.csv
    figures/
```

### 20.2 manifest 최소 예시

다음 JSON은 템플릿이다. null은 복구·계측 전 값이며 임의로 채우지 않는다.

```json
{
  "experiment_id": "EXP-E02",
  "run_id": null,
  "config_id": "A11",
  "mode": "PERF",
  "hardware": {
    "node": "violet-h100-016",
    "gpu_count": 4,
    "tp": 4,
    "cpu_workers": 96,
    "cpu_pools": 2,
    "gpu_uuids": null,
    "worker_affinity": null
  },
  "software": {
    "sglang_reported_version": "0.5.18",
    "kt_kernel_reported_version": "0.7.0.post1",
    "sglang_commit": null,
    "kt_commit": null,
    "benchmark_client_version": null,
    "container_digest": null,
    "patch_hash": null
  },
  "inference": {
    "model_reported": "Qwen3-Coder-480B-A35B-Instruct",
    "gpu_weight_format": "FP8",
    "cpu_weight_format": "AMXINT4",
    "kv_dtype": "fp8_e5m2",
    "kv_scale_hash": null,
    "hot_experts": 96,
    "hotmap_hash": null,
    "graph_bs": [32, 64, 96, 128, 160, 192, 224],
    "chunked_prefill_size": 4096,
    "mixed": false,
    "delayer": false,
    "quality_mode": "D_tau_0.25",
    "actual_kv_pool_tokens_per_rank": null,
    "admission_cap": null
  },
  "workload": {
    "id": "W0",
    "input_tokens_target": 512,
    "output_tokens_target": 128,
    "prefix_semantics": null,
    "input_token_ids_hash": null,
    "arrival_trace_hash": null,
    "client_concurrency": 224,
    "seed": 42,
    "eos_policy": null,
    "cache_start_state": null
  },
  "result": {
    "status": "planned",
    "infeasible_reason": null,
    "measurement_start": null,
    "measurement_end": null
  }
}
```

### 20.3 집계 CSV 필수 열

```text
experiment_id,run_id,config_id,workload_id,quality_mode,seed,restart_id,
status,infeasible_reason,
client_concurrency,admission_cap,actual_decode_bs_p50,actual_decode_bs_p90,
kv_pool_tokens_min_rank,kv_occupancy_peak,
num_planned,num_sent,num_completed,num_failed,
input_tokens,output_tokens,measurement_seconds,
output_tokens_per_second,job_output_tokens_per_second,
ttft_user_p50_ms,ttft_user_p95_ms,ttft_user_p99_ms,
tpot_p50_ms,tpot_p95_ms,tpot_p99_ms,
goodput_req_s,goodput_tok_s,
cpu_useful_rows,expert_invocations,rows_per_invocation,
logical_weight_bytes,system_dram_read_bytes,system_dram_write_bytes,
dram_bytes_per_output_token,dram_bytes_per_useful_row,
cpu_useful_overlap_ratio,cpu_exposed_wait_ms,
padding_rows,discarded_rows,kv_retractions,graph_fallback_count,
backlog_slope,trace_overhead_ratio,manifest_hash
```

계산할 수 없는 필드는 0이 아니라 null로 저장한다. 예를 들어 trace가 없는 PERF run의 overlap을 0으로 기록하면 "중첩 없음"이라는 잘못된 의미가 된다.

---

## 21. 실행 명령의 사용 원칙

### 21.1 기존 B 재현 명령

아래는 [S1 §8]의 명령을 파일 경로 변수로 바꾼 재현용 템플릿이다. **이 문서 작성 환경에서 해당 서버를 실행한 것은 아니다.** 모델·가중치·hot map 경로를 채우고 E00의 환경·commit 확인 후 사용한다.

```bash
#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?GPU FP8 model path is required}"
: "${KT_WEIGHT_PATH:?CPU INT4 weight path is required}"
: "${HOTMAP_PATH:?Frozen hotmap path is required}"

test -d "$MODEL_PATH"
test -d "$KT_WEIGHT_PATH"
test -f "$HOTMAP_PATH"

export KT_AMX_MIN_QLEN=1000000
export KT_AMX_MIN_ROWS=3
export KT_CALLBACK_FREE=1
export KT_COLD_DEFER=1
export KT_COLD_TAU=0.25
export KT_AVX_RB=1
export KT_AVX_PF=2
export KT_FUSE_QIN=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --tp 4 \
  --attention-backend triton \
  --cuda-graph-backend-prefill disabled \
  --context-length 32768 \
  --kt-weight-path "$KT_WEIGHT_PATH" \
  --kt-method AMXINT4 \
  --kt-cpuinfer 96 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts 96 \
  --init-expert-location "$HOTMAP_PATH" \
  --ep-dispatch-algorithm dynamic \
  --kt-max-deferred-experts-per-token 8 \
  --kv-cache-dtype fp8_e5m2 \
  --mem-fraction-static 0.94 \
  --chunked-prefill-size 4096 \
  --max-total-tokens 143360 \
  --cuda-graph-max-bs 224 \
  --cuda-graph-bs 32 64 96 128 160 192 224
```

부팅 후 기록의 `pin_nonkt.sh`를 별도 terminal에서 수행하되, 실제 worker·비-worker affinity를 확인한다. 그 스크립트의 내용은 첨부되지 않았으므로 이 문서에서 재구성하지 않는다. CPU ID 범위를 다른 node에 그대로 적용하지 않는다.

`--enable-mixed-chunk` 추가는 EPOCH-OPS의 변화다. D0, K0, G0, admission/KV cap 대조군은 **복구한 resolved args와 현재 코드의 실제 parsing**에 맞춰 별도 설정 파일로 만든다. 존재가 확인되지 않은 `--epoch-*` 같은 CLI를 새로 가정하지 않는다.

### 21.2 실험 전 도움말·버전 고정

```bash
python3 -m sglang.launch_server --help > sglang_help.txt
vllm bench serve --help > vllm_bench_serve_help.txt
python3 -m pip freeze > python_packages.txt
nvidia-smi -q > nvidia_smi_q.txt
nvidia-smi topo -m > nvidia_topology.txt
lscpu -e > lscpu_topology.txt
```

실행 중인 container와 host가 다르면 각각의 정보와 수집 위치를 구분한다. 소스 경로가 여러 개일 때 `pip` 버전만으로 실제 import된 라이브러리를 확인하지 말고 module path·commit·빌드 hash까지 기록한다.

SGLang의 현재 공식 문서는 실제 인자 확인에 `launch_server --help`를 안내하며, KV dtype·mixed·graph 관련 옵션을 제공한다. 그러나 온라인 문서의 새 기능이 0.5.18의 패치 빌드에 존재한다는 의미는 아니다. 최종 권위는 고정한 소스와 그 실행의 resolved configuration이다. [W2]

### 21.3 벤치 harness 요구사항

현재 자료에는 sonnet dataset 파일 경로, OpenAI endpoint 종류, benchmark client 정확한 버전이 없다. 따라서 임의의 endpoint·dataset 옵션을 채운 완성 명령을 제시하지 않는다. 기존 성공한 benchmark 명령을 기반으로 다음을 명시적으로 바꾼다.

| 목적 | 요구 동작 |
|---|---|
| 과거 burst 재현 | 기존 `3×C`, seed42, 입력512/출력128/prefix100 |
| 지속 포화 | 정해진 C를 완료 보충 방식으로 유지; 고정 측정창 |
| open-loop | 파일의 계획 도착 시각에 요청 전송; client backlog 계측 |
| tail 통계 | TTFT/TPOT/가능한 ITL의 p50/p95/p99 및 요청별 원시값 저장 |
| SLO | 전체 계획 도착 분모로 timeout·거절 포함 |
| token 처리량 | 측정창 내 token event 또는 동등한 계측 필요 |

현재 vLLM 문서의 percentile·goodput 옵션은 참고할 수 있지만, 기존 클라이언트가 요구한 raw event를 저장하지 않으면 harness를 보완한다. 기본 summary JSON만으로 위의 모든 지표가 자동 계산된다고 가정하지 않는다. [W1]

---

## 22. 최소 실행 묶음과 확장 순서

전체 행렬을 한 번에 돌리지 않는다. 아래 묶음 순으로 가며, 확인된 핵심 셀만 반복 수를 늘린다.

### 묶음 A — 성과 보존과 비교 가능성 확보

EXP-E00, EXP-E01, EXP-E09의 구조적 검증을 먼저 수행한다. 최소 출력은 `HIST-769 / B-C224 / B-C64` 재현표와 configuration diff다. HIST 복구 실패는 기록하고 controlled baseline 작업을 계속한다.

### 묶음 B — EPOCH의 기전을 결정하는 실험

EXP-E02의 2×2와 capacity cap, EXP-E03의 DDR·행 계측을 수행한다. 최소 셀은 공통 C64, 각 구성 최적 C, A11의 기준 용량 cap이다. 여기서 **메모리 확대가 실제 CPU 연산 효율 개선으로 연결되는지**가 결정된다.

### 묶음 C — 논문의 주 성능표

EXP-E06에서 최대 처리량 곡선을 만들고, EXP-E07에서 후보 SLO 경계점만 길게 확인한다. 같은 C, 각 구성 최적 C, 동일 SLO의 세 표를 완성한다. EXP-E09의 품질 결과와 같은 mode를 사용한다.

### 묶음 D — 리뷰어가 물을 기전·일반화

EXP-E04/E05/E08/E10/E12를 대표점 위주로 수행한다. 새로운 data point마다 전체 factorial을 반복하지 않는다. 혼합 실행·CPU scaling의 기존 관측은 조건이 맞으면 재사용하되, 원시 기록과 정의가 확인돼야 한다.

### 묶음 E — 독립적인 구성 선택 알고리즘 주장

EXP-E11은 EPOCH의 실현형을 정적 튜닝 조합보다 일반적인 선택 방법으로 확장할 필요가 있을 때 수행한다. 앞선 실험이 실패한 것처럼 서술하면서 다른 이름의 기법으로 갈아타지 않는다.

---

## 23. 논문 구성과 실험의 연결

### 23.1 제목·주장의 범위

현재 실현형에 맞는 가제:

> **EPOCH: Memory-Budgeted Concurrency for CPU–GPU MoE Serving**

이 제목은 연구 프레이밍 제안이며, 기존 파일에서 EPOCH 약어의 공식 풀네임이 확인된 것은 아니다. 새로운 풀네임을 임의로 확정하지 않는다.

가능한 기여는 다음처럼 증거 수준에 맞춰 작성한다.

| 기여 | 필요한 실험 | 작성 가능한 내용 |
|---|---|---|
| 시스템 관찰 | E02/E03/E04 | CPU cold expert의 낮은 행 밀도와 streaming 비용, GPU memory 배분의 영향 |
| 시스템 설계 | E02/E05 | KV·graph·실행 패치가 동시성·유효 CPU 계산에 미치는 인과적 구성 원리 |
| 성능 평가 | E01/E06/E07/E09 | 최대 처리량과 동일 SLO 용량, 품질 조건을 분리한 개선 |
| 적용 경계 | E08/E10/E12 | phase·길이·모델·자원에 따른 이득/손해와 안정성 |
| 선택 알고리즘, 선택적 | E11 | CPU 비용·GPU memory를 반영하는 재현 가능한 구성 선택 규칙 |

EPOCH의 성과를 인정하는 것과 학술적 신규성을 보장하는 것은 다르다. 최종 제출 전에는 실제 authored code/decision rule을 가장 가까운 기존 연구와 비교한다. 이 문서의 외부 확인은 **기존 옵션과 평가 방법을 구분하기 위한 제한적 확인**이며 포괄적인 신규성 조사가 아니다.

### 23.2 본문 그림의 최소 세트

| 그림 | 내용 | 대응 실험 |
|---|---|---|
| Fig.1 | rank별 weight/KV/graph/workspace와 실제 capacity | E02 |
| Fig.2 | cap on/off에 따른 동시성·CPU 효율·처리량 | E02/E03 |
| Fig.3 | 유효 expert 행 분포와 실제 DRAM bytes/useful-row | E03 |
| Fig.4 | CPU/GPU 유효 계산과 노출 대기 timeline | E04 |
| Fig.5 | E×R 기여와 대표 ablation | E05 |
| Fig.6–7 | 구성별 C sweep 및 최대 처리량·지연 경계 | E06 |
| Fig.8 | 동일 SLO 서비스 용량과 tail 지연 | E07 |
| Table.10 | 품질 모드별 정확성·비열등성 | E09 |
| Fig.11–12 | 일반화·반례 | E10 |

mixed의 동일 작업량, 선택 정책, 장시간 안정성은 지면과 실제 결과에 따라 본문 또는 부록에 배치한다. 기각된 EHES/SER/GRC는 "추가 최적화의 한계"로 보존하되, EPOCH 주 성능 결과의 기각 근거로 사용하지 않는다.

### 23.3 리뷰어 질문과 대응 증거

| 예상 질문 | 대응 |
|---|---|
| 동시성만 키운 것 아닌가? | capacity-cap 대조, 기존 구성이 같은 C를 지원하지 못하는 이유, 각자 최적 C 비교 |
| FP8가 빨라진 것 아닌가? | K0/F × G0/GS, 동일 C·동일 token capacity 대조, 같은 FP8끼리 graph 효과 |
| 기존 기능을 켠 것 아닌가? | patch provenance, TUNED-EXISTING, 필요시 E11의 held-out 선택 검증 |
| CPU는 저장만 한 것 아닌가? | 유효 CPU rows/s, 결과 소비, worker scaling, useful overlap |
| mixed 15.5배는 분자만 늘어난 것 아닌가? | 같은 P,D의 별도/합산 실행, 실제 DRAM·CPU 시간 |
| 44%가 정확도와 교환된 것 아닌가? | D0/Dτ·KV 정밀도 분리, incremental decode·과제 비열등성 |
| 짧은 burst의 착시 아닌가? | 긴 포화 run, open-loop arrival, backlog·timeout 처리 |
| sonnet 전용 튜닝 아닌가? | 같은 길이 다른 도메인, 다른 길이, 고정 hot map의 held-out 검증 |
| GPU-only보다 항상 좋은가? | 235B 반례와 480B memory-limited 주장을 분리 |

---

## 24. 제출 전 완료 체크리스트

- [ ] 과거 769/1110.3과 새 측정값을 구분했고 A/B 구성을 섞지 않았다.
- [ ] 모든 주 비교에 정확성 수정이 공통 적용됐다.
- [ ] KV dtype·scale, hot map, graph 목록, 실제 pool capacity가 고정됐다.
- [ ] 같은 C와 각 구성의 최적 C를 모두 비교했다.
- [ ] 최대 처리량 기준선도 동일한 탐색 기회를 받았다.
- [ ] capacity-cap 대조로 용량 확대 경로를 확인했다.
- [ ] CPU 유효 연산량과 실제 DRAM counter를 수집했다.
- [ ] rows/invocation과 물리 weight read를 혼동하지 않았다.
- [ ] mixed는 같은 P/D 작업량으로 검증했다.
- [ ] CPU/GPU 중첩에서 spin·대기 kernel을 유효 연산으로 세지 않았다.
- [ ] Poisson 계획 도착과 실제 전송의 차이를 계측했다.
- [ ] TPOT p99와 ITL p99, 평균·중앙값을 혼용하지 않았다.
- [ ] 실패·timeout·OOM·SLO 미달 셀을 숨기지 않았다.
- [ ] 품질 동등성·비열등성·bit 동일성을 구분했다.
- [ ] FP8 KV 검증에 실제 cached incremental decode를 포함했다.
- [ ] 적어도 5개의 독립 확인 run과 CI를 확보했다.
- [ ] sonnet 외 workload와 GPU-only가 유리한 반례를 포함했다.
- [ ] 기각된 EPOCH-X 확장을 EPOCH 전체 기각으로 표현하지 않았다.
- [ ] 19.7배 개발 성과와 독립 메커니즘의 이득을 분리했다.
- [ ] 결과를 보지 않은 상태에서 고정한 기준과 사후 탐색 결과를 구분했다.

**가장 먼저 실행할 일은 새로운 cohort scheduler 개발이 아니라, E00의 구성 고정 → E01의 성과 재현 → E02의 capacity-cap 대조 → E03의 실제 CPU 연산·DDR 계측이다.** 이 네 단계가 EPOCH의 44% 개선을 설명 가능한 연구 결과로 만드는 중심이다.

---

## 25. 근거 자료와 참조 범위

### 사용자 제공 자료

**[S1] `PERFORMANCE_RECORD_20260909_02.md`**  
최신 성능 기록. §1 EPOCH 목표/실현형, §2 운영점 A/B, §3 정확도, §4 도착률, §5 기여, §6 기각, §7 상한, §8 실행 명령, §9 EPOCH-X 평가. 이 문서의 주 근거다.

```text
SHA-256: c1a0fb96fad6fde405de236c14191006347d3e1a861251f93e1078e0d9781aff
크기: 13,361 bytes
```

**[S2] `PERFORMANCE_RECORD_20260909.md`**  
EPOCH-X 평가가 추가되기 전 성능 기록. 결과의 시간 순서와 기존 기여를 확인하는 보조 자료다.

```text
SHA-256: 1065bd062297d44edacd548de968e984c91bc541a82617d01ca33b79ea519091
크기: 10,464 bytes
```

**[S3] `CPU_UTILIZATION_HISTORY_20260909.md`**  
이전 CPU/GPU 활용 연구 이력. CPU attention, NUMA/pinning 결함, GPU-only 반례, speculative decoding 등의 실패 및 기존 기반을 확인했다. 최신 EPOCH 결과로 과거의 판단을 소급 수정하지 않았다.

```text
SHA-256: f1bf4ade45ee189a3b619d562201e0f93ef59a283974356b77fb3aa7980b5990
크기: 17,766 bytes
```

S1에 나열된 원시 실험 디렉토리·harness·patch 파일 자체는 이번에 제공되지 않았다. 해당 내용이 필요한 곳은 실험의 확인 항목으로 명시했으며, 직접 검증했다고 주장하지 않는다.

### 외부 공식 자료 — 2026-09-09 확인

**[W1] vLLM 공식 `bench serve` 문서.**  
request-rate와 max-concurrency의 상호작용, percentile·goodput 옵션을 확인했다. 사용 중인 client 버전은 별도 고정해야 한다.

```text
https://docs.vllm.ai/en/latest/cli/bench/serve/
```

**[W2] SGLang 공식 Server Arguments.**  
실제 help를 통한 인자 확인, KV·메모리·graph·mixed 관련 옵션이 기존 시스템에 존재함을 확인했다. 온라인 최신 기능을 패치 버전의 지원으로 대체하지 않는다.

```text
https://docs.sglang.ai/advanced_features/server_arguments.html
https://docs.sglang.io/docs/advanced_features/server_arguments
```

**[W3] Intel 공식 Performance Counter Monitor 저장소.**  
memory bandwidth·NUMA 등 하드웨어 계측 도구를 확인했다. 실제 CPU에서의 counter support·권한·단위는 실험 전 검증 대상이다.

```text
https://github.com/intel/pcm
```

**[W4] Agrawal et al., Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve, OSDI 2024.**  
chunked prefill과 decode를 결합해 처리량·지연을 다루는 기존 연구다. EPOCH를 단순한 prefill/decode 혼합의 최초 제안으로 포지셔닝하지 않기 위한 비교 근거다. 논문의 수치나 하드웨어 결과를 본 실험의 예상 성능으로 가져오지 않았다.

```text
https://www.usenix.org/conference/osdi24/presentation/agrawal
```

---

## 최종 실행 원칙

**EPOCH의 기존 개선은 출발점이다. 추가 실험의 목적은 그 성과를 없애는 것이 아니라, 어느 구성 요소가 어떤 자원 제약을 바꾸어 얼마나 유효한 CPU·GPU 연산을 늘렸는지 입증하는 것이다.**

최대 처리량, 동일 지연의 서비스 용량, CPU 연산 효율, 품질을 각자의 실험으로 분리하면 성공한 부분을 보존하면서도 과장 없는 논문을 만들 수 있다. 후속 확장의 실패는 그 확장의 결과로 남기고, EPOCH 전체의 판정으로 확대하지 않는다.
