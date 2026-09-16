# CPU MoE Offloading 후속 검증 지시서 — IDE_071 후속·실행량 제한형

> 문서 ID: `cpu_offload_no_03_compact_verification`  
> 작성 기준: 2026-09-16, Asia/Seoul  
> 입력: 사용자 제공 `FULL_REPORT-1.md`(IDE_071), `RESULT.md`(IDE_070), 이전 실행량 제한형 지시서.  
> 목적: **S2의 RB=0 결합, skip-empty 해제 대조, 오류 재현, 동일 문항 출력 비교만 수행한다.**  
> **이 문서가 이번 캠페인의 전체 범위다. 이전 12단계·compact 미실행 셀·IDE_071의 OUT_OF_SCOPE 88개를 이어서 실행하지 마라.**

## 1. 실행 원칙과 이번에 확인할 항목

### 1.1 반드시 지킬 규칙

성능 비교는 **V0·V1·V2 세 구성**으로 제한한다. 정확성 대조용 Q0 한 구성과 오류 발생 시에만 실행하는 D0 한 구성을 더해, 서로 다른 실행 구성은 **최대 5개**다. 새 체크포인트·커널·배치 알고리즘·서빙 버전은 추가하지 않는다.

**일반 성능 벤치 최대 11회, 실패 입력 재현·연속 부하·진단·복구를 합한 부하 세션 최대 18회, 서버 부팅 시도 최대 10회**다. 이전 compact의 20회/24회/14회 상한보다 낮은 이번 상한을 적용한다. 준비·정확성 검사는 제4절의 별도 예산에 포함한다. 남은 예산을 소진하려고 실험을 추가하지 않는다.

**개별 실패나 낮은 처리량을 이유로 전체 실행을 포기하지 마라.** 실패한 프로세스는 제한된 절차로 정리하고 다음 독립 항목을 처리하라. 범위 내 실행, 실패·미실행 기록, 최종 MD 저장, 승인된 GitHub 게시 확인, MD 다운로드 제공까지 완료하라. 이 지시는 무제한 재시도나 안전·접근 권한 우회를 허용하지 않는다.

**실제 실험 시작 후 30분마다 중간 보고하라.** 벤치 실행과 독립된 경량 보고 루프를 사용한다. 문서 작성만으로 실험이나 알림이 시작되는 것은 아니다.

**등록 범위 완료 또는 예산 도달 이후에는 평가·원인 추측·권고·채택 판단을 작성하지 마라.** 모든 설정, 측정값, 문항별 출력과 기계적 채점 결과, 오류, 실행·게시 상태만 최대한 상세하게 MD로 저장하라. 실행 중 의존성·건강 상태 확인과 조건부 분기 판단은 허용하되 입력과 규칙을 기록한다.

### 1.2 자료에서 확인된 사실과 이번 검증의 구분

| 자료의 관측 사실 | 이번에 확인할 것 | 이번 결과로 자동 주장하지 말 것 |
|---|---|---|
| IDE_071의 S2 확인 반복: SHORT_COLD C64, 805.80 ± 5.31 tok/s. [S1 §5.1] | 같은 설정·요청 조건의 현재 재현값 | 모든 워크로드의 대표 성능 |
| B3에서 RB=0은 787.14 ± 11.34 tok/s. S2와의 결합 실측은 보고되지 않음. [S1 §5.2] | V0 대 V1에서 RB 설정 하나의 차이 | 두 개선율의 단순 합산·곱셈 |
| S2 load1024 a1은 워밍업 중 inf/nan assert, a2는 1024요청 성공. [S1 §10·11] | 초기 실행·요청 재사용에서의 오류 유무와 재현 입력 | 재시도 성공을 결함 해결로 표현 |
| B3 계열에서 deferral8일 때 skip-empty ON/OFF 처리량 차이가 관측됨. [S1 §5.2] | 동일 S2 배치에서 skip-empty만 해제한 V2 | skip-empty를 오류 원인으로 단정 |
| S2만 GSM40 38/40, 대응 구성의 같은 문항 결과는 없음. [S1 §9] | V0·V1·V2·Q0의 동일 소규모 문항 출력 | 품질 동등성·정확한 추론의 입증 |
| 레이어·expert·step 자료는 NOT_COLLECTED. [S1 §8] | 기존 로그로 확인 가능한 범위만 수집 | CPU 대역폭 포화·연산 병목 해소의 확정 |

여기서 S2는 `S2_b3_v2_nu5952__confirm`을 뜻한다. 이번 배치 예산·문항 수·watchdog·분기 규칙은 **새로 제안하는 제한형 검증 설계**이며 기존 실험의 관측값이 아니다.

### 1.3 제외 범위

NUMA·코어 수·pinning 재탐색, AMX threshold·AVX prefetch 전수 탐색, B4 재튜닝, GPU-only 재벤치, AWQ/W4, FP8 KV 전환, attention/MoE backend 교체, TP/EP·router·speculative decoding, 동시성·길이 전체 sweep은 `OUT_OF_SCOPE`다. 새 계측 프레임워크, 원인 규명을 위한 무제한 코드 수정, 여러 구성의 장시간 시험도 이번 범위가 아니다.

## 2. 구성 동결과 등록 셀

### 2.1 사전 점검 — 추가 대규모 실험 없이 한 번만 수행

기존 `eval/ide071/` 하네스와 승인된 저장소를 사용한다. 새 캠페인 ID를 생성하고 기존 결과를 덮어쓰지 않는다. IDE_072라는 식별자를 사용할 때에도 이미 존재하는지 먼저 확인한다.

S2의 원본 `launch_cmd.sh`, 부팅 당시 환경, 적용 패치, `status.json`, 벤치 요청 manifest를 먼저 읽는다. 보고서 요약의 잘린 JSON을 복사하지 않는다. 원본 파일·실제 프로세스 적용값·보고서가 충돌하면 각각 보존하고 `SOURCE_CONFIG_CONFLICT`를 기록한다. 충돌하는 필드를 조용히 수정하거나 추정값으로 메우지 않는다.

소스 revision, 로컬 diff, kt 실행 바이너리, 모델 snapshot, hotmap, 층별 예산, 하네스, 요청 manifest를 동결한다. 매 부팅마다 수백 GB 가중치를 다시 해시하지 말고 기존 shard manifest를 재사용한다. 작은 설정·패치·바이너리·요청 파일의 SHA256은 실제 계산한다.

`KT_AVX_RB`와 `KT_CF_SKIP_EMPTY_IMM`은 **로컬 소스에서 파싱 방식·기본값·실제 적용 경로**를 확인한다. S2 명령에 RB 변수가 없다는 사실을 RB=0 또는 RB=1이라고 바꾸어 적지 않는다. 문자열 `"0"`도 존재 여부만 확인하는 구현에서는 OFF가 아닐 수 있으므로, 확인된 해제 방법을 사용한다. 부팅 시 읽는 환경변수를 실행 중 변경하고 적용됐다고 기록하지 않는다.

환경은 명시적 whitelist로 재구성한다. 이전 B4의 `KT_COLD_DEFER`, `KT_COLD_TAU`, AMX threshold, FP8 KV 설정 등이 우연히 상속되지 않도록 한다. 의도한 변수와 실제 적용값을 모두 저장하고, 지원하지 않는 설정을 만들거나 버전을 올려 해결하지 않는다.

### 2.2 V0 — S2 공통 구성

| 항목 | 고정값 또는 확인 방법 |
|---|---|
| GPU | H100 80GB ×4, 물리 index 0/1/2/3, TP4. 실제 UUID도 저장 |
| CPU | 양 소켓, cpuinfer96, threadpool2, turbo OFF·기존 2.0 GHz 정책 유지 |
| 모델 | Qwen3-Coder-480B-A35B-Instruct-FP8, snapshot `003f183a92fbe5b9a8325aaa8b2ae797c91dd90f` |
| CPU expert | `/models/kt/qwen3-480b-int4`, AMXINT4 |
| hotmap | `/models/kt/ide070/hotmap_v2.json` |
| 층별 예산 | `/models/kt/ide070/layer_budget_5952.json`, 기존 per-layer 패치 적용 |
| 슬롯 검증 | 62개 층의 실제 배치값 및 합계5952 확인. uniform96으로 대체하지 않음 |
| callback-free / skip-empty | ON / ON, 원본 S2와 같은 경로 |
| deferral | `--kt-max-deferred-experts-per-token 8` |
| pinning | 원본 S2의 비-kt pinning 그대로. kt 코어·비-kt TID affinity 실측 저장 |
| KV | dtype은 S2 원본의 auto 및 실제 해석값, max-total-tokens40960, mem-fraction0.95 |
| graph | decode max64, prefill disabled. 실제 capture 목록·padding 동작 저장 |
| 나머지 | attention triton, context32768, dispatch dynamic, speculation OFF. prefill 크기 등 미명시 기본값도 확인하여 동결 |

다음은 [S1 §4]의 S2 명령을 바탕으로 한 구성 명세다. **pinning과 per-layer 패치 적용은 이 명령 외부의 기존 절차까지 함께 재현해야 한다.** 실행 전 원본과 대조하고, `PLAN_RESOLVED.md`에는 생략 없는 실제 명령을 저장한다.

```bash
KT_CALLBACK_FREE=1 \
KT_CF_SKIP_EMPTY_IMM=1 \
KT_GPU_EXPERTS_PER_LAYER=/models/kt/ide070/layer_budget_5952.json \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m sglang.launch_server \
  --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f \
  --served-model-name q480 --host 127.0.0.1 --port 30000 \
  --tp 4 --attention-backend triton --trust-remote-code \
  --context-length 32768 \
  --kt-weight-path /models/kt/qwen3-480b-int4 \
  --kt-method AMXINT4 --kt-cpuinfer 96 --kt-threadpool-count 2 \
  --kt-num-gpu-experts 96 \
  --init-expert-location /models/kt/ide070/hotmap_v2.json \
  --kt-max-deferred-experts-per-token 8 \
  --ep-dispatch-algorithm dynamic \
  --cuda-graph-max-bs 64 --cuda-graph-backend-prefill disabled \
  --mem-fraction-static 0.95 --max-total-tokens 40960
```

### 2.3 허용 구성 — 이 목록 밖의 셀은 생성하지 않음

| ID | 부모 | 유일한 변경 | 수행 역할 |
|---|---|---|---|
| `V0_S2` | IDE_071 S2 | 없음 | 현재 대조군·성능 확인·재현·동일 문항 출력 |
| `V1_S2_RB0` | V0 | `KT_AVX_RB=0`만 적용 | S2에서 RB0 추가 이득과 출력 변화 확인 |
| `V2_S2_SKIP_OFF` | V0 | skip-empty만 OFF | deferral8을 유지한 오류·출력·성능 대조 |
| `Q0_S2_SKIP_OFF_DEF0` | V2 | deferral8→0만 변경 | deferral 해제 대조. 품질 세션만 실행하고 일반 성능 sweep은 하지 않음 |
| `D0_GRAPH_OFF` | 실제 오류가 난 V0 또는 V1 또는 V2 | `--disable-cuda-graph`만 적용 | 조건부 진단1회. 일반 성능 결과에 합치지 않음 |

V0/V1은 RB, V0/V2는 skip-empty, V2/Q0는 deferral만 다르게 한다. Q0도 CPU INT4·GPU FP8의 같은 배치를 쓰므로 **고정밀 정답 모델이나 완전한 정확성 기준선으로 부르지 않는다.** V0/Q0는 두 설정이 다르므로 단일 요인 효과로 해석하지 않는다.

V1과 V0의 실제 RB 경로가 같으면 V1은 `DUPLICATE_EFFECTIVE_CONFIG`로 종료한다. 임의로 다른 RB 값을 찾아 대체하지 않는다. Q0에서 deferral0의 의미를 소스와 적용값으로 확인할 수 없으면 `SEMANTICS_UNVERIFIED`를 남긴다. 확인을 위해 새 구현을 시작하지 않는다.

D0는 이번 캠페인에서 해당 오류가 발생한 경우에만 생성한다. 과거 보고서에 오류가 있었다는 이유만으로 자동 실행하지 않는다. 부모는 **V0→V1→V2 순으로 오류가 관측된 첫 구성**으로 고정한다. callback-free나 다른 변수를 함께 끄지 않는다. graph OFF가 미지원이거나 실행 의미를 확인할 수 없으면 `UNSUPPORTED` 또는 `SEMANTICS_UNVERIFIED`로 기록한다.

## 3. 워크로드·출력 비교·오류 재현

### 3.1 주 성능 조건 — SHORT_COLD를 기준으로 고정

| 필드 | 값 |
|---|---|
| workload ID | `SHORT_COLD` |
| dataset / 목표 입출력 | sonnet / 512·128 tokens |
| concurrency / 요청 수 | 64 / 256 |
| prefix / seed | **0 / 20260916** |
| EOS / 도착률 | ignore_eos=True / request-rate=inf |
| cache | 매 반복 전 engine cache flush, 워밍업 후에도 flush |
| 지연 지표 | TTFT·TPOT·ITL·E2EL, p50/p90/p95/p99 |

**이전 compact 지시서의 prefix100 대신 IDE_071의 실제 SHORT_COLD 조건을 사용한다.** LEGACY_070·COMPACT_ALT·COMPACT_LOAD 수치를 이 표와 합치지 않는다.

원본 요청 manifest·token ID를 확보해 한 번 고정하고 모든 V 구성에 동일하게 사용한다. seed는 client 요청 생성과 server sampling을 구분해 기록한다. 같은 seed만으로 동일 입력이나 같은 GPU 실행 순서를 보장했다고 쓰지 않는다.

IDE_071의 동일 manifest에서 보고된 합계는 input129237·output32768 tokens다. 새 측정값을 이 수치로 덮어쓰지 말고 실제값을 비교 기록한다. 입력 해시가 다르면 `INPUT_MANIFEST_CHANGED`로 표시하고 과거 수치와 동일 입력 비교로 분류하지 않는다. 과거와 다른 입력을 쓰더라도 이번 V0·V1·V2 내부의 요청 파일은 동일하게 유지한다.

클라이언트의 streaming·sampling·EOS·stop·timeout 설정을 원본 파일로 복원한다. 보고서가 제공하지 않는 기본값은 확인하여 기록한다. cache flush 응답과 요청 drain 완료를 저장하고, flush 실패 시 `CACHE_PREP_FAILED`로 처리한다. 실행 중인 요청이 있을 때 cache를 초기화하지 않는다.

### 3.2 별도 입력과 요청 수 제한 연속 부하

`COMPACT_ALT`는 V0·V1의 확인 반복이 완료된 경우 각각1회만 실행한다. [S1 §7]의 C64·256요청·512/128·prefix100·seed20261002·ignore_eos=True·매회 flush 조건을 고정한다. 다른 seed만으로 calibration 데이터와 독립이라고 단정하지 않는다.

`COMPACT_LOAD`는 최대1회다. C64·1024요청·512/128·prefix100·seed20261003·ignore_eos=True·시작 전 flush 조건을 사용한다. **1024는 총 요청 수이며 동시성이나 출력 길이가 아니다.** 시간을 늘려 장시간 시험으로 바꾸지 않는다.

부하 대상은 기계적으로 정한다. V1이 실제로 다른 구성이며 정규 성능·재현·출력 세션에서 프로세스 종료·비유한값 오류·미완료 요청 없이 실행됐으면 V1, 그렇지 않고 V0가 같은 실행 요건을 충족했으면 V0, 둘 다 아니면 `NOT_TRIGGERED_NO_ELIGIBLE_TARGET`다. 정답률을 근거로 운영 적합성을 판단하지 않으며 이 선택은 배포 결정이 아니다.

### 3.3 실패 입력 재현 — V0·V1·V2 각각 최대1세션

각 구성의 **첫 부팅**에서 smoke와 워밍업 전체 요청·응답·순서를 기록한다. 가능하면 IDE_071 `S2_b3_v2_nu5952__load1024/a1`의 실제 워밍업 입력과 설정을 사용한다. 원본이 없으면 새로 고정한 C16·32요청으로 대체하고 `ORIGINAL_FAILURE_INPUT_UNAVAILABLE`를 명시한다. 입력을 복원했다고 추정하지 않는다.

첫 주 벤치 종료 후, 같은 서버에서 위의 최대32개 입력을 C16으로 한 번 다시 보낸다. 이것이 `REPLAY` 한 세션이다. 과거 실패가 초기 실행에서 발생했다는 사실을 보존하고 `FRESH_STARTUP`과 `WARM_SERVER_REPLAY`를 구분한다. 요청 순서·cache 상태·서버 재시작·실제 batch가 다르면 완전히 동일한 실행을 재현한 것으로 표기하지 않는다.

초기 워밍업에서 서버가 종료되면 이어지는 주 벤치·REPLAY는 전송하지 않는다. `NOT_RUN_PRECONDITION_FAILED`로 기록한다. **서버가 죽은 상태에서 1024개 요청을 보내 오류 숫자만 만드는 실행을 금지한다.** 실패한 워밍업 자체가 보존해야 할 관측 자료다.

오류가 다시 발생하지 않으면 `NOT_REPRODUCED_WITHIN_BUDGET`로 기록한다. 이 기록을 결함 해결이나 안정성 보장으로 바꾸지 않는다. 32개를 넘어 입력 최소화 탐색이나 여러 부팅 반복을 자동으로 추가하지 않는다.

### 3.4 동일 문항 출력 비교 — 총80문항 상한 유지

기존 GSM40에서 **사전 고정한 동일20문항을 V0·V1·V2·Q0에 각각1회**, 총80문항 실행한다. 이전 compact의 총80문항 상한은 유지하면서 비교 구성을 늘리는 설계다. 각 구성을 GSM40 전체로 실행하지 않는다.

원본 GSM40의 문항 ID를 정렬한 뒤 앞20개를 결과를 보지 않고 선택한다. 순서와 입력·정답·chat template·하네스·채점 함수 해시를 저장한다. ID가 없으면 원본40개 배열의 첫20개를 사용하고 선택 방법을 기록한다. 원본40문항을 복원할 수 없으면 새 데이터셋을 받지 말고 `BLOCKED_DEPENDENCY`로 종료한다.

workload는 `GSM20_PAIRED`로 별도 명명한다. 원본의 greedy 설정과 chat template를 유지하되 출력 한도는1024 tokens, 동시성은 최대4로 고정한다. 실제 지원 옵션으로 동일 조건을 구현하고 API·하네스 설정을 저장한다. 이는 **과거 GSM40 38/40과 분모·출력 한도가 다른 검증**이다.

문항별 전체 생성 텍스트, 가능한 token ID, 추출 답, 정답, 정오, finish reason, 생성 길이, truncation, 요청 오류를 저장한다. V0–V1, V0–V2, V2–Q0의 같은 문항을 나란히 놓고 출력·추출 답의 일치/불일치만 기록한다. 생성 문구가 다르다는 이유만으로 계산 오류로 판정하지 않는다.

문항별 반복은 없다. 실패 문항 재생성으로 총80문항을 넘기지 않는다. 오류·미완료·채점 불가·토큰 제한 도달을 별도 집계하고, 성공 문항만 남겨 분모를 숨기지 않는다. **GSM20의 정오 데이터는 기록하되 품질 동등·저하 없음·정확 추론 같은 결론은 쓰지 않는다.**

## 4. 실행 순서와 예산

### 4.1 실행량 표

| 구분 | 최대 세션 수 | 구체적 배분 |
|---|---:|---|
| 주 조건 1차 측정 | 3 | V0·V1·V2 각1회, 256요청 |
| 별도 부팅 확인 반복 | 6 | V0·V1 각각3회, 256요청. 1차값과 별도 집계 |
| 별도 입력 | 2 | V0·V1 COMPACT_ALT 각각1회 |
| **일반 성능 소계** | **11** | V2는 추가3회 확인 없음 |
| 실패 입력 REPLAY | 3 | V0·V1·V2 각각1회, 최대32요청 |
| 연속 부하 | 1 | 정해진 대상1개, 최대1024요청 |
| 조건부 D0 진단 | 1 | 최대32요청, 추가 성능 sweep 없음 |
| 복구 재시도 예비 | 2 | 캠페인 전체 합계. 실패한 시도도 소비 |
| **부하 세션 전체 상한** | **18** | 11+3+1+1+2. 미사용 예산은 이월·대체하지 않음 |

| 준비·출력 항목 | 별도 상한 |
|---|---|
| 서버 부팅 시도 | **10회**: 필수 첫 부팅4 + V0/V1 확인 부팅2 + 조건부 D0 1 + 필요 시 부하 대상 복귀1 + 복구2 |
| 워밍업 | 성공한 부팅당 최대1세트, C16·32요청, 전체 최대10세트 |
| 부팅 smoke | 기존 greedy4문항, 성공한 부팅당1세트, 전체 최대10세트 |
| GSM20_PAIRED | 네 구성×20문항, **총80문항**, 반복·재시도 없음 |
| 클라이언트 내장 probe | 호출당 최대1회, 실제 전송 수 별도 기록. 불필요한 자동 워밍업 반복 금지 |
| 새 계측 패치·커널 빌드 | 0회 |
| 기존 profiler 진단 | D0 한 세션에서 기존 도구1종만. 미설치면 설치하지 않음 |

모든 상한은 완료 시간 예측이 아니라 **실행량 제한**이다. 부팅 실패도 boot를 소비하며, 부하를 실제 시작한 뒤 실패하면 해당 세션 예산을 소비한다. 시작하지 못한 후속 측정은 미실행 상태로 남긴다.

재시도 세션은 `kind=RETRY`, `retry_of=<원run_id>`로 집계한다. 원본 PERF/REPLAY/LOAD/DIAG 세션과 별도의 시도이며, 총합에 두 번 더하지 않는다. 워밍업·부팅 복구도 같은 캠페인 재시도2회에 포함한다. 품질 문항은 이 예비 예산으로 추가 실행하지 않는다.

### 4.2 실행 순서

```text
manifest·실제 설정·입력·보고 루프 확인
 → V0 첫 부팅: smoke·warmup → SHORT_COLD 1회 → REPLAY 1회
 → V1 첫 부팅: smoke·warmup → SHORT_COLD 1회 → REPLAY 1회
 → V2 첫 부팅: smoke·warmup → SHORT_COLD 1회 → REPLAY 1회 → GSM20
 → Q0 첫 부팅: smoke·warmup → GSM20
 → V0 새 부팅: SHORT_COLD 3회 → COMPACT_ALT 1회 → GSM20
 → V1 새 부팅: SHORT_COLD 3회 → COMPACT_ALT 1회 → GSM20
 → 정해진 대상에서 COMPACT_LOAD 1회
 → 오류 조건을 충족한 경우에만 D0 1회
 → 예산·파일 정합성 확인 → 상세 MD → GitHub → MD 다운로드 제공
```

확인 대상은 사전에 V0·V1로 고정한다. 후보 선별을 위해 B3·B4·GPU8을 다시 부팅하지 않는다. V1이 중복·미지원이거나 1차 실행 실패 시 확인 부팅을 억지로 수행하지 않는다. 오류가 난 V 구성의 새 실행은 확인 반복이라는 이름으로 재시도 한도를 우회하지 않는다.

같은 서버에서 가능한 출력 검사·별도 입력·연속 부하는 묶어 부팅을 줄인다. 같은 CPU·GPU·DRAM을 공유하는 벤치를 병렬 실행하지 않는다. 부팅을 줄이려고 **부팅 시 읽는 설정을 live 변경**하는 방식은 금지한다. 기록 압축·대용량 해시·게시를 성능 측정과 동시에 돌리지 않는다.

V0·V1의 확인3회가 끝나면 차이가 작아도 추가5회·9회로 늘리지 않는다. 1차와 확인 반복을 합쳐 표본 수가 충분하다고 꾸미지 말고 별도 표로 남긴다. 이 설계는 짧은 재현·짝비교이며 전역 수렴·장시간 서비스 안정성을 확인하는 실험이 아니다.

## 5. 실패 처리·조건부 진단·30분 중간 보고

### 5.1 단계별 건강 상태와 유한한 복구

각 부하 앞뒤에 프로세스·health·미완료 요청을 확인한다. `BOOT → SMOKE → WARMUP → CACHE_PREP → MEASURE/REPLAY/QUALITY → DRAIN` 중 선행 단계가 실패하면 다음 단계는 전송하지 않는다. 정규 벤치의 반환코드0만으로 성공이라고 하지 말고 요청 완료 수·서버 상태·출력 수를 함께 보존한다.

동일 장애의 재시도는 최대1회, 전체 복구는 최대2회다. 실제 실패 입력·manifest·명령·로그를 보존한 뒤 같은 설정으로 새 프로세스를 띄운다. 실패 시도의 결과를 성공 재시도 값으로 덮어쓰지 않는다. 재현 목적으로 실패한 설정을 다시 부팅하는 경우도 예산에 포함한다.

초기 watchdog는 부팅600초, 일반·REPLAY·진단 부하300초, LOAD·GSM20 세션600초다. health가 명백히 실패하면 timeout까지 기다리며 요청을 계속 보내지 않는다. 자동 연장은 금지한다. 실제 watchdog, 발동 시각, 미완료 요청 수를 기록한다. 이 값은 장애 감지를 위한 제안 상한이다.

Xid·ECC·열·전원 위험이나 다른 사용자 작업 간섭이 있으면 해당 실행을 차단한다. 본 캠페인의 프로세스만 정리하고, host 재부팅·전역 GPU reset·BIOS 변경·권한 우회는 하지 않는다. 전체 환경이 불가하면 `BLOCKED_ENVIRONMENT`로 남기고 확보한 데이터·MD·게시 가능 부분을 끝까지 처리한다.

### 5.2 NaN/Inf·메모리 접근 오류의 증거 보존

로그가 해당 오류를 보고하면 요청ID, 입력 원문·token ID, sampling, seed, 최초 발견 시각, 부모 구성, 코드/바이너리 해시, rank, 서버 stdout/stderr, stacktrace, 마지막 성공 요청, stage를 저장한다. `NaN`, `Inf`, negative probability, CUDA assert, illegal memory access를 가능한 범위에서 원문대로 구분한다.

**마지막 sampler의 probability 오류를 최초 NaN 발생 지점이라고 단정하지 않는다.** 원인 파일·레이어·expert는 실제 로그나 기존 진단이 뒷받침하는 경우에만 사실로 기록한다. 증거가 없으면 `FIRST_NONFINITE_LOCATION_NOT_COLLECTED`다.

D0는 실제 실패 부모의 graph만 끈 별도 진단이다. 입력·순서·cache·나머지 설정을 유지하고 최대32개 요청으로 한 번 재현한다. 기존 진단 기능이 있으면 유한값 관련 로그·GPU stream 활동·작업 완료 이벤트 중 실제 얻을 수 있는 자료를 저장한다. 계측 기능이 없으면 그 항목은 `NOT_COLLECTED`이며 새 패치를 만들지 않는다.

`nan_to_num`, clipping, sampler 변경, 실패 요청 무시, CPU 결과 생략으로 오류를 감추지 않는다. 디버깅을 위해 동작이 바뀐 D0의 처리량을 주 성능 평균에 넣지 않는다. graph OFF에서 오류가 없었다는 관측만으로 graph를 원인이라고 판정하지 않는다.

### 5.3 30분 보고와 종료 상태

시작 시각부터 monotonic 기준1800초마다 경량 루프가 `PROGRESS.md`를 갱신하고 실제 사용 가능한 승인된 전달 수단으로 보고한다. 보고 시점이 부하 실행 중이어도 그 부하를 끊지 않는다.

```text
Asia/Seoul 시각 / 경과시간 / 현재 cell·stage·run_id
PERF 사용/11 / REPLAY 사용/3 / LOAD 사용/1 / DIAG 사용/1 / RETRY 사용/2
전체 부하 세션 사용/18 / 부팅 사용/10 / 워밍업·smoke 수 / GSM 처리문항/80
완료·실패·blocked·미실행 수 / 직전 원측정값
오류 stage·원문·파일 위치 / 다음 등록 항목
저장된 최신 MD / 중간보고 실제 전달 성공·실패
```

전달 기능이 없으면 `REPORT_DELIVERY_UNAVAILABLE`를 기록한다. PROGRESS.md에만 저장하고 사용자에게 보고했다고 말하지 않는다. 종료가30분보다 빠르면 추가 대기하지 않고 종료 보고한다. 보고 실패 때문에 나머지 데이터 수집·저장을 포기하지 않는다.

모든 등록 항목을 증거가 있는 완료·실패·차단·조건미충족 상태로 정리한 뒤, 캠페인 상태를 `BOUNDED_VALIDATION_COMPLETE`, `VALIDATION_BUDGET_REACHED`, `BLOCKED_ENVIRONMENT` 중 실제에 맞게 기록한다. 오류 재현 여부·정확성 미측정 등은 별도 상태로 병기한다. 유한 범위 종료를 ‘수렴 완료’, ‘최적 구성 확정’, ‘운영 검증 통과’로 바꾸지 않는다.

## 6. 원데이터 저장과 MD 작성

### 6.1 한 실행을 여러 파일에서 다르게 집계하지 않음

`execution_events.jsonl`을 단일 실행량 원장으로 사용한다. 시도 시작 전에 run_id·kind·cell·attempt·parent·config_hash·input_hash·세션 예약을 기록하고 종료 시 상태·실측 전송 수·성공·실패·미전송 수를 추가한다. 하네스 재시작 시 같은 캠페인 원장을 읽고 사용량을0으로 초기화하지 않는다.

최종 회수는 이 원장에서 다시 계산한다. `GSM=0`인데 실제 출력 파일이 존재하거나, RETRY 종료보다 앞선 시각이 최종 END가 되는 등 불일치를 검사한다. 요청·부팅·부하 세션은 서로 다른 단위다. 서버 부재 때문에 보내지 않은 요청을 전송 실패와 구분한다.

요약 생성 과정에서 환경변수·JSON·명령·파일경로를 자르지 않는다. 긴 내용은 Markdown 코드블록이나 별도 파일로 옮기고 링크한다. 각 상대 경로의 존재를 검사하여 `a1server.full.log.gz` 같은 잘못 결합된 경로가 만들어지지 않게 한다. 과거 자료의 오기는 원본을 보존하고 별도의 정합성 항목으로 기록한다.

### 6.2 필수 원기록

| 분류 | 저장할 항목 |
|---|---|
| 설정·환경 | 전체 launch/bench 명령, 비밀정보 제거 환경, 기본값/실효값, GPU UUID, 소스·diff·바이너리·hotmap·예산·입력 해시 |
| 실제 적용 | rank·TID별 affinity, 62층 expert 예산 및 합계, KV 실제 dtype·용량, graph 목록, RB/skip/deferral 적용 근거 |
| 시간 | 캠페인·셀·부팅·smoke·warmup·flush·측정·drain·품질·재현·게시 시작/종료, 실제 벽시계·monotonic 경과 |
| 성능 | 반복별 요청 수, 성공/실패/미전송, 실제 input/output tokens, output/total tok/s, TTFT/TPOT/ITL/E2EL 원값·percentile |
| 요청별 | 입력ID·내용/안전한 저장위치, 가능한 token ID, 출력 전문, finish reason, 시각, 길이, truncation, 오류 원문 |
| 경량 시계열 | 기존 CPU 코어군별 busy, GPU util·HBM·전력, RAM, affinity 시작/종료. 기존 로그의 queue·batch·retraction도 수집 가능한 것만 |
| 정확성 | GSM20 네 구성의 동일 문항 출력·추출답·정답·정오·불일치·미완료. 원본40개 중 선택된 ID 명시 |
| 오류·진단 | 모든 실패·재시도 로그, 최초 보고 stage, 재현 입력, D0의 수집 창과 변경점, 미수집 사유 |

수집되지 않은 내부 계측은 `null + 사유`로 남긴다. 오류 로그가 없었다는 사실을 모든 레이어의 nonfinite_count=0으로 대신 기록하지 않는다. 상세 데이터 수집을 이유로 실험 축을 늘리거나 새로운 무거운 계측을 정규 벤치에 넣지 않는다.

### 6.3 수치 집계와 표현

모든 반복 원값을 먼저 보존한다. 동일 workload·config·code·precision·sampling·cache 조건의 valid 반복만 평균·중앙값·최소·최대·표본표준편차(ddof=1)로 집계한다. 표본1개는 sd=null이다. 실패한 실행도 원값과 실패 내역을 별도 표에 그대로 남긴다.

`mean_of_rep_p95`와 요청을 합쳐 재계산한 `pooled_p95`를 구분하고 percentile 알고리즘·대상 요청 수를 명시한다. client가 실제 측정한 정의를 보존하며 streaming chunk 간격을 관측했는데 token 간격을 직접 측정했다고 바꾸지 않는다.

일반 성능, REPLAY, LOAD, D0, GSM20은 별도 표로 둔다. 1차와 새 부팅 확인3회도 별도 표다. 과거 IDE_071 수치는 출처가 있는 참고 표에만 두고 새 반복처럼 통계에 합치지 않는다.

정규 출력 처리량의 분모는 실제 benchmark duration이다. 캠페인 전체 시간은 부팅·준비·진단·품질·게시를 포함해 별도 기록한다. 기계적 차이·비율을 기록할 수는 있지만 ‘유의하게 개선’, ‘안정적’, ‘병목 해결’ 같은 평가 문장을 붙이지 않는다.

### 6.4 최소 산출물

```text
PLAN_RESOLVED.md                   # 확정된 범위·설정·의존성·예산·workload
PROGRESS.md                        # 시작·30분 주기·종료의 상태와 전달 기록
FULL_REPORT.md                     # 설정·전 반복·요청·출력·실패·실행량·실행시간
RESULT.md                          # 수치표·실행 상태·출처만, 해석 없음
execution_events.jsonl             # 단일 실행량 원장
branch_trace.jsonl                 # 조건부 실행 규칙·입력·분기 결과
cells/<cell>/<attempt>/...         # 전체 명령·raw 결과·로그·요청별 파일
quality/paired_question_results.md # 네 구성×동일20문항의 비교 원기록
failures/                         # 실패 입력·원문·재시도·진단 증거
artifact_manifest.jsonl           # path·bytes·SHA256·실제 저장 및 게시 위치
SHA256SUMS
PUBLISH_RECEIPT.md                 # commit·원격·파일·다운로드 확인
```

FULL_REPORT에는 모든 설정과 반복표를 포함하고, 요청별 출력이 길면 별도 MD shard와 JSONL을 연결한다. 본문에 ‘별도 파일 참고’만 쓰고 실제 위치·해시를 누락하지 않는다. 수행하지 않은 항목은 원인과 종료 상태를 기록한다.

## 7. 종료 후 GitHub 게시와 다운로드 제공

**실험의 제한 범위가 끝나면 평가·판단을 덧붙이지 말고, 데이터를 보존·검사·게시하는 작업으로 마무리하라.** 결과 표는 cell/attempt/run 순서로 배치한다. 최고 구성 선정, 배포 권고, 원인 추정, 추가 실험 제안은 최종 MD와 최종 답변에 포함하지 않는다.

게시 대상은 기존 승인된 저장소·remote다. 실제 연결·쓰기 권한과 branch를 확인하고, 새 공개 저장소 생성·공개 범위 변경·force push를 하지 않는다. 인증정보, 사용자 식별정보, 모델 가중치, 배포 권한 없는 자료는 게시하지 않는다. 필요한 비식별화는 원본 보존 위치와 처리 내역을 기록한다.

MD·작은 재현 자료를 커밋하고 push한다. 큰 raw파일은 승인된 저장 위치와 SHA256을 MD에 남긴다. 서버에만 존재하는 파일을 GitHub에서 받을 수 있다고 말하지 않는다. 원격 commit SHA, 대상 MD 존재, 가능하면 원격 파일 내용·해시를 읽어 반영을 확인한다.

`PUBLISH_RECEIPT.md`에는 데이터 commit SHA와 확인 시각을 적는다. 영수증 자체를 별도 커밋할 경우 data_commit과 receipt_commit을 구분하고, 자기 자신의 해시를 자기 파일에 무한히 갱신하지 않는다. 게시 권한·연결 문제가 있으면 원문을 남기고 `PUBLISH_BLOCKED`로 보고하되, 로컬 MD 생성·전달을 생략하지 않는다.

마지막에는 **실재하는 FULL_REPORT.md와 RESULT.md의 다운로드 링크**를 제공한다. 전체 MD shard·raw 묶음이 있으면 실제 파일을 만든 뒤 접근 가능한 위치도 제시한다. sandbox 링크는 그 경로에 실제 파일을 생성·확인한 경우에만 사용한다. 사용 가능한 GitHub 원문/파일 링크 역시 실제 존재·접근 범위를 확인한다. 가짜 링크·경로만 있는 안내로 대체하지 않는다.

최종 응답은 다음 사실만 담는다.

```text
캠페인 상태 / 완료·실패·차단·조건미충족 항목 수
실제 성능·재현·부하·진단·재시도·부팅·정확성 실행량
GitHub 게시 상태 / data commit SHA / 원격 MD 위치
FULL_REPORT.md·RESULT.md 다운로드 링크 / raw·실패 입력 저장 위치
```

## 8. 입력 자료와 출처

- **[S1]** `FULL_REPORT-1.md`, IDE_071 FULL_REPORT. §3·4 구성과 실제 명령, §5 반복값, §7 workload, §8 미수집 자료, §9 정확성, §10·11 오류·재시도, §12 실행시간, §13 파일 인벤토리.
  - SHA256: `ed93206e10ddb2c6a39c5ac64207f672c120a35479b7c97dbf87ade22f2149f4`
- **[S2]** `RESULT.md`, IDE_070. §0 기존 모델·정밀도, §4 층별 예산, §5 비동기 실행 구성. 이전 기록은 원본을 보존하며 현재 측정에 합산하지 않음.
  - SHA256: `07144c375090e647fa1baee6b730db8ff61c116cbf25383eb252362993f3eba3`
- **[S3]** `cpu_offload_no_02_compact_실험지시서.md`. 실행량 제한, 유한한 복구, 30분 보고, 종료 후 사실만 저장, GitHub·다운로드 규칙을 계승. 이번 문서가 범위·대조군·workload·세부 예산을 대체함.
  - SHA256: `3e5077e04e5a639494ad492f4936d45bbf18372cda9fd39724293130cdc166ef`

위 SHA256은 이 지시서 작성 시 제공된 로컬 파일에서 계산한 값이다. 실행 환경에서 원본 증거 파일의 경로와 해시를 확인하고, 다른 사본이면 차이를 기록한다. **이 지시서의 생성은 실험 수행·GitHub 게시·30분 알림 설정의 완료를 의미하지 않는다.**
