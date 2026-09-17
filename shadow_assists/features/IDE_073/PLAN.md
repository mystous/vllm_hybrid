# CPU MoE Offloading: 공식 기본 구성·최고 처리량 구성 비교 및 구간별 계측 지시서

> 문서 ID: `cpu_offload_no_04_two_models_baseline_opt_probe`  
> 작성일: 2026-09-17 · 실행 기록 시간대: `Asia/Seoul`와 UTC offset 병기  
> 대상: H100 80GB ×8, Xeon Platinum 8480+ ×2, Host DRAM 2TB의 단일 서버  
> 작업: **대표 MoE 모델 2종에서 GPU-only 8장, 공식 가이드 기반 오프로딩 4장, 기존 최고 처리량 설정을 적용한 오프로딩 4장을 비교하고, 최적화 구성의 단계별 시간을 계측한다.**  
> 산출: **평가 문장 없는 상세 결과 MD와 원시 데이터, 30분 보고 이력, Git 게시 영수증, 실제 다운로드 가능한 MD.**  
> 이 문서는 실행 지시서다. 문서 생성만으로 대상 서버의 실험이나 정기 보고가 시작되는 것은 아니다.

---

## 1. 범위와 절대 준수 사항

### 1.1 이번 실험이 답할 질문

1. 선택한 **동일 모델·동일 GPU 체크포인트**가 CPU 전문가 오프로딩 없이 H100 8장에서 실행되고, Hot/Cold 부분 상주로 H100 4장에서 실행되는가?
2. GPU 4장 조건에서 **KTransformers 공식 가이드 기반 기본 구성**과 **기존 최고 처리량 구성**의 처리량·지연·메모리·출력은 각각 어떻게 측정되는가?
3. 최적화 구성에서 **요청 대기, GPU 연산, CPU 전달, CPU 작업 대기·연산, 결과 반환, 결합·통신**에 각각 얼마의 시간이 쓰이는가?
4. CPU와 GPU의 작업이 얼마나 겹치고, 다음 층의 진행을 실제로 늦추는 **노출 대기시간**은 어느 경로에서 발생하는가?

이는 새 설정의 전수 탐색이 아니다. **동결한 구성의 비교와 계측**이다. 이전 지시서의 미실행 목록을 이어서 실행하거나, 여유 예산으로 새 모델·새 옵션 조합을 추가하지 않는다.

### 1.2 구조의 전제

**Hot expert의 실행 가중치는 GPU HBM에 상주하고 GPU가 계산한다. Cold expert의 실행 가중치는 같은 서버의 Host Memory에 상주하고 CPU가 계산한다.** GPU는 Attention·Router·KV cache를 담당한다. Cold 경로에는 MoE 입력 활성값, 선택 ID, 라우팅 가중치 등 해당 구현이 요구하는 데이터가 전달되고, 결과가 GPU의 결합 경로로 돌아온다.

- 기본·최적화 두 오프로딩 구성 모두 GPU 상주 전문가 수를 0으로 만들지 않는다.
- `Hot/Cold`는 고정된 별도 모델 종류가 아니라 배치 집합이다. 기본 `uniform` 배치가 실제 고빈도 전문가를 모두 포함한다고 전제하지 않는다.
- 모델의 원래 top-k, 전체 routed expert 수, shared expert 의미를 임의로 바꾸지 않는다.
- **Host Memory에 가중치를 보관하다가 매번 GPU로 옮겨 계산하는 일반 weight offload**를 CPU expert 실행과 동일하게 취급하지 않는다.
- 공식 native backend가 긴 prefill에서 layerwise GPU prefill을 사용하면, 이 구간은 Cold 가중치 전송이 생길 수 있는 **별도 실행 모드**로 기록한다. 이를 모든 단계가 CPU expert 실행인 것처럼 숨기지 않는다. [W3]
- CPU에 GPU 상주 전문가의 원본·변환본 사본이 남는 구현도 있다. 실행 배치와 backing copy를 구분하고 실제 메모리 점유를 기록한다.

### 1.3 끝까지 진행한다는 의미

**낮은 처리량, 개별 셀의 오류, 일부 profiler의 실패를 이유로 전체 캠페인을 중단하지 않는다.** 등록된 독립 항목을 끝까지 처리하고, 실패·미실행의 원인과 증거를 남긴 뒤 데이터 정리·Git 게시·MD 전달까지 수행한다.

다만 이 지시는 무한 재시도, GPU reset, BIOS 잠금 해제, 접근 권한 우회, 타인의 프로세스 종료, 원시 데이터 삭제를 허용하지 않는다. 장비·권한·디스크 등 필수 조건이 막히면 영향을 받는 셀을 `BLOCKED_*`로 닫고 가능한 나머지 항목 및 산출물 정리를 계속한다. **모든 셀에 최종 상태가 생긴 것과 모든 실험이 성공한 것은 다르다.**

---

## 2. 모델 선정과 8장/4장 실행 자격 확인

### 2.1 우선 대상 모델 두 종

| 모델 | 정확한 GPU 체크포인트 | 선정 근거와 현재 확인 수준 |
|---|---|---|
| Qwen3-Coder-480B-A35B | `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8` | 기존 자료에 GPU-only TP8+EP8와 CPU expert TP4의 실제 실행 기록이 있다. 최고 처리량 구성 재현의 기준 모델이다. [S1][S2][W5] |
| GLM-4.7 | `zai-org/GLM-4.7-FP8` | 공개 대형 MoE 계열이며 KTransformers에 `FP8_PERCHANNEL` 공식 실행 예가 있고, 모델 측 SGLang TP8 안내가 있다. **본 서버·본 워크로드에서 GPU-only 8장 및 오프로딩 4장 성공은 이번에 확인한다.** [W3][W6][W7] |

두 모델은 공개 배포·서빙 지원과 메모리 규모를 고려해 고른 대표 대상이다. **세계 사용량 1·2위라고 단정하지 않는다.** 참고용으로 준비 시점의 공식 모델 카드·배포 지원·가능하면 다운로드 지표를 저장하되, 다운로드 수를 실제 운영 사용량으로 바꾸어 표현하지 않는다.

GLM의 공개 config에는 92개 hidden layer, 초기 dense layer 3개, routed expert 160개, shared expert 1개, top-k 8, per-channel FP8이 기록되어 있다. 실제 MoE layer 목록과 MTP layer의 적재·실행 여부는 선택한 revision에서 확인한다. **Qwen의 62층 예산 파일을 GLM에 복사하지 않는다.** [W7]

### 2.2 모델별 사전 자격 게이트

실행 모델 수는 두 종으로 고정한다. 모델·revision·precision을 도중에 조용히 바꾸지 않는다.

1. 체크포인트 revision, tokenizer, config, tensor dtype·shape·scale, shard index를 고정한다. 디스크 파일 용량과 실행 시 HBM 요구량을 구분한다.
2. 실제 `nvidia-smi`의 GPU별 총량·가용량을 기록한다. 공칭 80GB×4/8만으로 적재 성공을 선언하지 않는다.
3. GPU-only 8장 참조군을 기동해 smoke와 주 워크로드를 수행한다. CPU expert, `cpu-offload-gb`, Unified Memory paging에 의한 가중치 오프로딩 등은 사용하지 않는다. CPU의 토큰화·스케줄링은 GPU-only에서도 허용되는 정상 동작이다.
4. 동일 체크포인트의 GPU-only 4장 적재 가능성을 tensor 메모리 산정으로 먼저 확인한다. 불명확할 때만 **기동 확인 1회/모델**를 허용한다. 실패 로그를 남기며 OOM을 처리량 0으로 표시하지 않는다. 기동되면 정해진 소규모 요청으로 실행 여부만 확인한다.
5. 기본·최적화 오프로딩 구성이 각각 GPU 4장만 사용하며 Hot/Cold 배치와 CPU expert 실행이 실제 활성화됐는지 확인한다. 의도한 GPU 외의 UUID에서 해당 모델 프로세스가 실행되면 자원 조건 불일치다.
6. 8장 참조군 실패 또는 4장 하이브리드 필수 경로 미지원이면 `MODEL_ELIGIBILITY_UNCONFIRMED`로 기록한다. 데이터를 없애거나 제3 모델을 자동 추가하지 않는다.

GPU-only 4장도 성공하면 그 사실을 그대로 기록한다. 의도적으로 KV 공간을 과도하게 키워 실패시키거나, “오프로딩만이 4장 실행을 가능하게 한다”고 쓰지 않는다. GPU-only 4장 확인은 **용량 자격 검사**이지 네 번째 성능 비교군이 아니다.

---

## 3. 공식 기본값과 실험용 정규화 값을 분리한다

### 3.1 ‘공식 기본 구성’의 정의

KTransformers 공식 안내는 등록 모델에 `kt run`의 모델별 기본값을 사용하고, 수동 실행에서는 모델·CPU ISA·가중치 형식에 맞는 backend를 선택하도록 한다. GPU 상주 수는 HBM 여유에 따라 정하고, activation 통계가 없으면 `uniform`에서 시작하도록 설명한다. 따라서 **모든 모델·H100 4장에 공통인 단 하나의 공식 숫자 조합은 확인되지 않았다.** [W1][W2]

이번 기본 비교군의 정식 표기는 다음과 같다.

> **KT-BASIC4 — KTransformers 공식 가이드 기반, H100 4장 및 공통 워크로드에 맞춘 기본 구성**

기존 실험의 612.1 tok/s 구성은 hotmap·deferral4가 들어간 내부 기준선이다. 이를 KTransformers 무수정 기본값으로 재명명하지 않는다. [S1]

`BASELINE_PROVENANCE.md`에 모든 설정을 아래 네 범주로 구분한다.

| 범주 | 의미 |
|---|---|
| `OFFICIAL_MODEL_RECIPE` | 해당 모델의 공식 실행 예에서 확인한 값 |
| `OFFICIAL_GENERAL_GUIDE` | CPU 물리 코어, NUMA pool, backend·placement 등 공식 일반 원칙 |
| `HARDWARE_WORKLOAD_ADAPTATION` | GPU4, 동일 KV·동시성·context 등 이번 실험을 위해 고정한 값 |
| `LOCAL_OPTIMIZATION` | hotmap·층별 배치 패치·callback-free·skip-empty·RB 등 우리 측 변경 |

**기본 구성에 `LOCAL_OPTIMIZATION`을 섞지 않는다.** 공식 upstream에 이미 들어간 기능을 무조건 우리 독자 기능이라고 부르지도 않는다. 소스 diff와 실행 경로로 구분한다.

### 3.2 기본 구성 — 모델별 수치

아래는 실행 전 동결할 **기본 비교군 명세**다. 표의 ‘자원 정규화’는 공식이 이 하드웨어에 권고한 숫자라는 뜻이 아니다.

| 설정 | Qwen 기본 | GLM 기본 | 근거·분류 |
|---|---|---|---|
| GPU / TP | 4장 / 4 | 4장 / 4 | 자원 정규화 |
| GPU 체크포인트 | 지정 FP8 revision | 지정 FP8 revision | 모델 카드 |
| CPU expert backend | `AMXINT4` | **`FP8_PERCHANNEL`** | Qwen은 공식 AMX 변환 경로를 선택; GLM은 모델별 공식 native 예 사용 [W3][W4] |
| CPU weight path | 같은 FP8 revision에서 만든 KT INT4 | GPU와 같은 native FP8 revision | backend별 적재 형식 |
| CPU inference workers | **112 물리 코어 기준** | **100** | Qwen: 일반 AMX 가이드의 물리 코어 기준; GLM: 해당 공식 예 [W3][W4] |
| threadpool count | 2 | 2 | 실제 NUMA node 수가 2임을 확인 |
| 층별 GPU routed expert 수 | **96** | **80** | Qwen96: 기존 H1004 적재 기록에 따른 자원 정규화, 공식 보편 기본값 아님; GLM80: 공식 예 |
| 초기 GPU 배치 | `uniform`, activation hotmap 미사용 | `uniform` | 공식 수동 배치 출발점 |
| runtime dynamic expert update | OFF | **ON** | Qwen은 선택한 AMX 기본 경로; GLM은 공식 native 예. 실행 중 변경 이력 저장 |
| deferred experts / token | **2** | **2** | 공식 일반 예·보수적 권고 범위에서 고정. 무손실이라고 전제하지 않음 |
| callback-free / skip-empty / RB 로컬 override | 미설정 | 미설정 | 최적화 env 상속 금지 |
| non-KT 스레드 별도 pinning | 없음 | 없음 | upstream 기본 affinity 그대로 측정 |
| Attention backend | `triton` | `flashinfer` | Qwen은 기존 환경 호환·자원 정규화; GLM은 모델별 공식 예 |
| native layerwise prefill threshold | 해당 경로 비적용 | **2048** | GLM 모델별 예. 실제 분기 기록 |
| mixed chunk | 공식 AMX 경로의 해석값을 동결 | ON | GLM 공식 예 |
| shared expert fusion | 구현 기본값 확인·고정 | disabled | GLM 공식 예; shared 결과 누락·중복 금지 |
| 기타 공개 기능 | 코드 기본값을 덤프하여 고정 | `enable-p2p-check`, 필요 시 `fp8-gemm-backend=triton` 등 공식 예를 확인·고정 | 없는 CLI를 만들어 전달하지 않음 |

기존 자료의 Qwen INT4 backend는 AMX라는 설정명을 쓰지만 decode에서 AVX-512 VNNI 경로가 관측됐다. 설정 이름으로 실행 ISA를 판정하지 않는다. [S3]

**GLM의 per-channel FP8과 Qwen의 block FP8은 다른 형식이다.** 같은 `fp8` 파일명만 보고 converter·scale layout을 교환하지 않는다. `kt run`에 모델 alias가 실제 등록돼 있지 않으면 임의의 alias를 만들지 말고 명시적 SGLang-KT 실행 명령을 생성한다.

### 3.3 공통 자원·워크로드 정규화

| 항목 | 기본4 / 최적화4 | GPU-only8 |
|---|---|---|
| context length | 32,768 | 동일 |
| 주 성능 세션 KV 총 토큰 | **40,960** | 동일 40,960 |
| KV dtype | BF16로 통일. `auto`이면 실제 해석값 확인 | 동일 |
| static memory fraction | 0.95 | 0.90 |
| decode CUDA graph | ON, max batch 64 | 동일 |
| prefill CUDA graph | disabled | 동일 |
| max running requests | 64 | 동일 |
| chunked prefill | Qwen: 기존 최고 구성에서 실제 해석값을 복원해 세 군 공통. 미복원 시 기본4·참조8만 명시값4096으로 등록하고 최고 재현군은 미확인 표시 | 같은 모델의 확정값 |
| GLM chunked prefill | 4096 | 동일 |
| speculative decoding / MTP generation | OFF | OFF |
| 외부 모델 fallback / 별도 request router | OFF | OFF |
| CPU turbo·주파수, GPU power cap | 현재 기록값 유지, 임의 튜닝 금지 | 같은 정책 |

GLM 공식 native 예의 TP8·static fraction0.75·chunk16384·max-running4·KV100000을 위 표로 바꾸는 것은 **명시적인 실험 자원 정규화**다. 원문 명령과 변경 diff를 둘 다 보존한다. 공식 예를 그대로 실행했다고 표기하지 않는다.

이 실험은 공통 서비스 용량에서의 비교다. GPU-only8에 KV131072를 준 과거 결과와도 구분한다. 모델 내부 kernel·fusion의 필수 차이는 manifest에 남긴다. 공식 기준선과 최적화군의 CPU precision이나 software가 다르면 **전체 구성 비교**이며 단일 옵션의 인과 효과가 아니다.

### 3.4 기본군의 실행 명령 골격

다음 명령은 **공식 예를 그대로 복사한 명령이 아니라, 제3.2~3.3절의 기본군 명세를 명시적으로 적용한 실행 템플릿**이다. 준비 단계에서 실제 checkout의 `--help`와 argument parser로 검증하고, 지원되지 않는 명칭은 의미가 같은 정식 옵션으로만 매핑한다. 원형 명령·매핑 diff·최종 argv를 보존한다. 지원되지 않는 기능을 그냥 삭제한 뒤 동일 구성이라고 기록하지 않는다.

기본군은 독립된 공식 checkout/컨테이너에서 실행한다. 최적화군에서 상속될 수 있는 `KT_CALLBACK_FREE`, `KT_CF_SKIP_EMPTY_IMM`, `KT_GPU_EXPERTS_PER_LAYER`, `KT_AVX_RB`, `KT_AVX_PF`, `KT_FUSE_QIN`, `KT_COLD_DEFER`, `KT_COLD_TAU`, `KT_AMX_MIN_QLEN`, `KT_AMX_MIN_ROWS`의 **로컬 override를 제거**하고, 해당 upstream의 실제 기본 동작을 기록한다. `=0`과 unset을 같게 취급하지 않는다.

Qwen 기본군:

```bash
# 변수는 준비 단계에서 검증한 로컬 snapshot·변환본·공통 prefill 값이다.
: "${QWEN_FP8_DIR:?validated model snapshot required}"
: "${QWEN_KT_INT4_DIR:?validated KT INT4 weights required}"
: "${QWEN_PREFILL_CHUNK:?resolve and freeze before starting}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server \
  --model-path "$QWEN_FP8_DIR" \
  --served-model-name qwen480 --host 127.0.0.1 --port 30000 \
  --tp 4 --context-length 32768 --trust-remote-code \
  --attention-backend triton \
  --kt-weight-path "$QWEN_KT_INT4_DIR" \
  --kt-method AMXINT4 --kt-cpuinfer 112 --kt-threadpool-count 2 \
  --kt-num-gpu-experts 96 --kt-expert-placement-strategy uniform \
  --kt-max-deferred-experts-per-token 2 \
  --kv-cache-dtype auto --max-total-tokens 40960 \
  --mem-fraction-static 0.95 --max-running-requests 64 \
  --chunked-prefill-size "$QWEN_PREFILL_CHUNK" \
  --cuda-graph-max-bs 64 --cuda-graph-backend-prefill disabled
```

GLM 기본군:

```bash
: "${GLM_FP8_DIR:?validated per-channel FP8 model snapshot required}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server \
  --model-path "$GLM_FP8_DIR" \
  --served-model-name glm47 --host 127.0.0.1 --port 30000 \
  --tp 4 --context-length 32768 --trust-remote-code \
  --kt-weight-path "$GLM_FP8_DIR" \
  --kt-method FP8_PERCHANNEL --kt-cpuinfer 100 --kt-threadpool-count 2 \
  --kt-num-gpu-experts 80 --kt-expert-placement-strategy uniform \
  --kt-enable-dynamic-expert-update \
  --kt-max-deferred-experts-per-token 2 \
  --attention-backend flashinfer --fp8-gemm-backend triton \
  --enable-p2p-check --disable-shared-experts-fusion \
  --enable-mixed-chunk --kt-gpu-prefill-token-threshold 2048 \
  --kv-cache-dtype auto --max-total-tokens 40960 \
  --mem-fraction-static 0.95 --max-running-requests 64 \
  --chunked-prefill-size 4096 \
  --cuda-graph-max-bs 64 --cuda-graph-backend-prefill disabled
```

`--kv-cache-dtype auto`는 이 명세에서 BF16으로 해석되는지 확인한다. 그렇지 않으면 해당 버전이 지원하는 BF16 명칭으로 명시하며 기록한다. graph 옵션, dynamic/deferral 조합의 허용 여부도 버전에 따라 확인한다. 명령이 파싱됐다는 사실만으로 기능 활성화를 인정하지 않고 실제 dtype·layer 배치·분기까지 검증한다. GPU-only8 명령에서는 KT 옵션을 제거하고 같은 모델의 attention·KV·prefill 조건과 TP8/지원되는 EP 설정을 적용한다. 최종 명령은 모두 argv 배열로 저장하며 셸 출력 절단을 금지한다.

---

## 4. 최고 처리량 구성: 무엇을 재현하고 무엇을 이식하는가

### 4.1 Qwen — 기존 최고 처리량 구성의 복원과 공통 조건 재측정

`OPT4-QWEN`은 사용자 자료의 **V1 = S2 + RB 활성**을 복원한다. 제3.3절에서 공통 서비스 조건을 명시한 부분은 원래 실행과의 diff를 남기고, 전체 effective config가 같지 않으면 완전 동일 조건 재현이 아니라 **최고 처리량 구성의 공통 조건 재측정**으로 표기한다. 기존 SHORT_COLD C64 기록은 844.0 ± 14.9 tok/s(3회)다. 이 값은 목표 강제값이나 예상값이 아니라 **과거 관측값**으로만 사용한다. 별도 조건의 887.1·897.8을 주 기준값으로 대체하지 않는다. [S1]

| 설정 | OPT4-QWEN |
|---|---|
| 모델 snapshot | `003f183a92fbe5b9a8325aaa8b2ae797c91dd90f` |
| CPU expert | `/models/kt/qwen3-480b-int4`, `AMXINT4`; 원본 revision·변환기·scale 확인 |
| SGLang | `0.5.18`, `71de97b2…` 기반 + 기존 로컬 패치 |
| KT | 기존 `0.7.0.post1` 소스 빌드와 기존 로컬 패치·실행 바이너리 |
| GPU / TP | GPU0–3 / TP4 |
| CPU workers / pools | 96 / 2 |
| KT worker physical cores | 0–47, 56–103; 실제 토폴로지로 확인 |
| non-KT pinning | 48–55, 104–111과 해당 물리 코어의 HT 형제. 기존 `pin_nonkt.sh`를 복원·검증 |
| nominal GPU experts | 96, 단 실제값은 아래 per-layer override |
| hotmap | `/models/kt/ide070/hotmap_v2.json` |
| per-layer budget | `/models/kt/ide070/layer_budget_5952.json` |
| 총 GPU expert 슬롯 | 62층 합계 **5,952**; 최솟값75 / 최댓값142 |
| `KT_CALLBACK_FREE` | `1` |
| `KT_CF_SKIP_EMPTY_IMM` | `1` |
| `KT_GPU_EXPERTS_PER_LAYER` | 위 예산 JSON 경로 |
| `KT_AVX_RB` | **문자열 `0`을 설정. 당시 소스는 변수 존재로 ON을 판단했음** |
| `kt-max-deferred-experts-per-token` | **8** |
| KV / memory | BF16(실제 dtype 확인), 40,960 / 0.95 |
| graph | decode ≤64, prefill disabled |
| Attention / dispatch | `triton` / `dynamic` |
| 나머지 | 과거 실행 당시 해석값을 복원하여 동결. 다른 캠페인의 옵션을 섞지 않음 |

**`KT_AVX_RB=0`은 이 기록에서 OFF가 아니다.** 새로운 소스에서 환경변수 파싱 의미가 달라졌으면 문자열과 실제 분기를 모두 저장하고, 역사적 바이너리를 재현하거나 변경을 명시한다. 변수 존재 여부·숫자 파싱을 구분한다.

층별 예산 원자료는 다음과 같다. 원본 JSON을 우선 사용하고, 아래 배열은 검산용이다. 모델 layer ID 순서를 반드시 확인한다. [S3 §4]

```json
[142,117,107,116,106,82,113,96,89,98,88,98,95,98,97,89,88,90,95,107,89,89,86,81,85,89,88,85,85,90,79,95,93,75,80,87,87,90,89,105,97,99,91,100,90,97,104,87,105,102,97,102,104,100,102,93,98,101,106,104,113,102]
```

동일 슬롯 수가 실제 HBM 바이트 수의 완전한 동일성을 뜻하지 않는다. alignment, scale, padding, runtime buffer를 따로 기록한다. hotmap이 없거나 per-layer 패치가 적용되지 않으면 uniform96으로 조용히 대체하지 말고 `OPT_REPRODUCTION_BLOCKED`로 남긴다.

실행 명령 구성 요소는 아래와 같다. 이는 **당시 소스·패치·pinning까지 복원해야 성립하는 명세**다. 실제 인자를 생략 없는 argv 배열과 셸 명령 둘 다 저장한다.

```bash
KT_CALLBACK_FREE=1 \
KT_CF_SKIP_EMPTY_IMM=1 \
KT_GPU_EXPERTS_PER_LAYER=/models/kt/ide070/layer_budget_5952.json \
KT_AVX_RB=0 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m sglang.launch_server \
  --model-path /models/hub/models--Qwen--Qwen3-Coder-480B-A35B-Instruct-FP8/snapshots/003f183a92fbe5b9a8325aaa8b2ae797c91dd90f \
  --served-model-name qwen480 --host 127.0.0.1 --port 30000 \
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

`max-running-requests` 등 공통 조건은 실제 해석값을 확인한 뒤 명시한다. 과거 명령에 없던 값을 추가할 때에는 자원 정규화 diff에 남긴다.

### 4.2 GLM — 최고 구성의 모델 간 이식 검증

**기존 최고 처리량은 Qwen에서 얻은 결과다. GLM에서도 이미 최적이라고 주장하지 않는다.** 두 번째 모델은 `OPT4-GLM-TRANSFER`로 명명하고, 같은 최적화 메커니즘이 다른 MoE 구현에서도 동작하는지 비교한다.

| 항목 | GLM 최적화 이식 구성의 명세 |
|---|---|
| GPU 모델 | 기본 GLM과 동일한 FP8 checkpoint·revision |
| CPU kernel | 기존 V1의 AMXINT4/RB 경로를 적용하려면 **해당 GLM per-channel FP8의 올바른 INT4 변환·로더 지원을 먼저 확인** |
| CPU workers / pools | 96 / 2, V1과 같은 물리 코어·non-KT 배치 정책 |
| Hot/Cold 배치 | GLM 전용 calibration 통계·layer 목록으로 생성. Qwen hotmap 사용 금지 |
| 상주 예산 | 기본 GLM의 총 GPU routed-expert weight 바이트 예산을 기준으로 고정. 같은 shape이면 기본80×실제 MoE층 수와 같은 슬롯 총량 |
| per-layer 할당 | 기존 빈도 기반 예산 배분 알고리즘을 1회 적용. 추가 알고리즘·threshold sweep 금지 |
| callback-free / skip-empty / deferral / RB | V1의 ON / ON / 8 / ON을 실제 실행 경로에서 검증 |
| KV / graph | 제3.3절의 공통 정규화 유지 |
| Attention·shared 처리 | GLM 구현의 의미와 지원 backend를 유지. Qwen 모듈을 억지로 대입하지 않음 |

**중요한 형식 차이:** 공식 GLM 기본군은 native `FP8_PERCHANNEL`, 기존 최고 구성은 CPU `AMXINT4`다. 따라서 GLM 비교는 precision·로더·배치·핸드오프까지 포함한 구성 비교가 된다. 결과 표에 GPU dtype, CPU 저장 dtype, CPU 연산 dtype, scale 방식과 실제 kernel을 모두 기록한다.

GLM 전용 준비 절차:

1. 고정한 소스에서 `Glm4MoeForCausalLM`의 KT 연결, routed/shared expert 처리, per-channel scale 변환, activation 함수 지원을 확인한다.
2. 기존 공식 converter/loader가 지원하는 경로만 사용한다. `kt quant --help`와 실제 converter 코드를 확인하기 전에는 Qwen용 `-i fp8` 명령을 그대로 실행하지 않는다.
3. 대표 routed expert의 변환 전·후 dequantized weight 차이와 동일 활성값의 expert 출력 차이를 기계적으로 저장한다. shared expert는 별도 항목으로 검사한다. scale 축·shape·offset·전문가 ID 매핑을 검증한다.
4. **INT4 이식이 지원되지 않으면 `OPT_TRANSFER_BLOCKED_FORMAT`로 기록한다.** native FP8 경로에서 RB 환경변수만 설정한 뒤 RB가 적용됐다고 기록하지 않는다. 임의의 새 quantizer 개발이나 native-only fallback을 ‘기존 최고 설정’이라고 대체하지 않는다.
5. calibration 입력은 128요청 1세션, 입력512·출력64·concurrency16으로 제한한다. 코드와 일반 텍스트를 포함하고 성능·정확성 입력과 원문 기준 중복을 배제한다. 공유된 prefix도 기록한다.
6. calibration 후 배치 파일·원자료·생성 알고리즘을 해시하고 고정한다. 평가 입력을 보고 다시 hotmap을 바꾸지 않는다.
7. 실제 layer 목록은 config·서버 초기화·모듈 traversal로 대조한다. 초기 dense 3층과 MTP 제외 여부를 고려하지 않고 92개 층에 모두 expert를 배치하지 않는다.

Qwen의 기존 hotmap도 사용한 calibration의 출처를 기록한다. 과거 Sonnet 기반 calibration과 이번 Sonnet 벤치의 중복 가능성을 숨기지 않는다. 새로운 입력에서의 일반화 성능이라고 부르지 않으며, 별도의 prefilling·정확성 입력을 함께 보존한다.

### 4.3 코드·설정 복원 게이트

기본군은 우리 성능 패치가 없는 공식 checkout 또는 그에 해당하는 검증된 경로, 최적화군은 기존 성능 패치가 있는 checkout을 사용한다. **가능하면 같은 upstream 기반 commit과 dependency 환경을 공유하되 디렉터리·컨테이너를 분리**한다. Git worktree나 독립 이미지로 원래 환경을 보존한다.

- 공개 base commit으로 복원할 수 있는지 확인하고, `git diff`, submodule revision, wheel·실행 `.so` SHA256, compiler flags를 저장한다.
- 공식 패키지/소스가 기존 V1 기반과 달라지면 `SOFTWARE_BASE_DIFFERS`를 기록한다. 이를 순수한 설정 변경 비교로 해석하지 않는다.
- 최신 버전을 기존 정상 컨테이너에 덮어 설치하지 않는다. 실험 도중 버전을 올리거나 원본 환경을 파괴하지 않는다.
- `KT_AVX_RB`, `KT_CALLBACK_FREE`, `KT_CF_SKIP_EMPTY_IMM`, `KT_GPU_EXPERTS_PER_LAYER`, deferred 옵션의 파싱 위치·실제 실행 분기·관련 함수 경로를 `effective_feature_paths.json`에 기록한다.
- deferred 전문가가 현재 토큰에 모두 반영되는지, 이전 결과를 쓰는지, 누락·근사를 허용하는지 소스로 구분한다. **보수적 deferral2도 무손실이라고 가정하지 않는다.** 판별하지 못하면 `SEMANTICS_UNVERIFIED`를 기록한다.
- 존재하지 않는 옵션을 무시하고 서버가 뜬 경우를 성공으로 처리하지 않는다. intended와 effective config를 대조한다.
- 같은 expert를 CPU와 GPU가 중복 처리하거나, 어느 쪽도 처리하지 않는 선택이 있는지 논리 expert ID 기준으로 검증한다.

---

## 5. 등록 구성·실행량·재시도 상한

### 5.1 성능 비교군은 총 6개

| 모델 | GPU-only 참조 | 공식 기본 오프로딩 | 최고 구성 재현/이식 |
|---|---|---|---|
| Qwen3-Coder-480B | `Q-GPU8` | `Q-KT-BASIC4` | `Q-OPT4` |
| GLM-4.7 | `G-GPU8` | `G-KT-BASIC4` | `G-OPT4-TRANSFER` |

GPU-only는 같은 모델 FP8의 GPU8·TP8 실행이다. Qwen은 기존 참조의 EP8을 사용한다. GLM도 지원되는 EP8 경로를 우선하되, 별도 EP 지원이 없으면 TP8의 실제 MoE parallel 설정을 그대로 기록한다. TP와 EP를 둘 다 지정했다고 GPU를 64장 쓴다는 뜻으로 해석하지 않는다.

### 5.2 고정된 성능 워크로드

| 워크로드 | 입력 / 출력 목표 | 동시성 | 요청 수 | 반복 / 구성 | 목적 |
|---|---:|---:|---:|---:|---|
| `MAIN_SHORT` | 512 / 128 | 64 | 256 | **3회** | 처리량·지연의 주 비교 |
| `LOW_CONCURRENCY` | 512 / 128 | 1 | 16 | **1회** | 단일 요청 생성 지연 |
| `LONGER_PREFILL` | 4096 / 128 | 8 | 32 | **1회** | prefill 길이 증가 시의 실행 경로 |

**6구성 × 5회 = 일반 성능 30회**다. `MAIN_SHORT`는 기존 SHORT_COLD와 동일한 cache·prefix·seed 정책을 사용하되, 원본 요청 manifest가 달라지면 과거와 동일 입력 재현으로 부르지 않는다.

별도 진단은 **최적화 구성당 최대4세션, 두 모델 합계8세션**이다. 기본 예정 부하 세션은 38개이며, 전체 재시도·조건부 진단은 **추가6세션 이내**로 제한한다. **성능·진단·재시도 합계 상한44세션**이다.

준비 및 품질은 다음과 같이 별도 집계한다.

| 항목 | 상한 |
|---|---|
| 서버 부팅 | 자격 검사·재부팅·profiler·재시도 포함 **20회** |
| GPU-only4 자격 부하 | 기동됐을 때만 모델당1세션, 4요청×출력32 이하 |
| GLM calibration | 128요청×출력64, 1세션 |
| smoke | 부팅당4문항, 출력32 이하 |
| 워밍업 | 부팅당1세션, concurrency16·32요청·출력32 이하 |
| 동일 문항 출력 비교 | 구성당20문항, 총 **120문항**, 출력1024 이하, concurrency4 |
| 모델 변환 | 새 GLM 변환1회 + 실패 시 원인·수정 diff가 있는 재시도1회 |
| 새로운 최적화 옵션 탐색 | **0회** |

계측 코드 준비·GLM adapter 호환성 수정은 합계90분을 상한으로 둔다. 이 한도에서 포착하지 못한 세부 구간을 거짓 시간값으로 채우지 않는다. 우선 실제 Cold 입출력·CPU 작업·Hot GPU·결합 대기 경계를 계측하고 세부 GEMM 분해는 지원 범위에서 추가한다. 실제 지원 경로가 없으면 미지원 상태를 남긴다. 모델 다운로드·정상 진행 중인 변환의 시간은 따로 기록하며, 정지로 오인해 반복 재기동하지 않는다. 정상 준비 작업도 제10절의 진행 감시와 명시적 최대 시간을 적용한다.

**남은 예산을 채우려고 불필요한 시험을 추가하지 않는다. 예산 도달은 최적화 수렴의 증명이 아니다.** 예상 완료 시간은 시작 후 실제 준비·부팅·측정 시간을 기반으로 보고한다. 새 수백 GB 가중치 다운로드·변환을 제외하고 무조건 1시간 안에 끝난다고 약속하지 않는다.

### 5.3 실행 순서와 재현성

모델별 GPU8 자격 확인 → 기본4 → 최적화4를 수행하되, 모델마다 기본4·최적화4의 측정 순서를 반대로 배치하거나 사전에 고정한 순서를 기록한다. GPU 수가 겹치는 서버는 동시에 실행하지 않는다.

최적화 구성은 첫 기동에서 주 비교1회와 보조 부하를 수행하고, 새 프로세스에서 주 비교2회를 추가해 재기동 재현을 확인한다. 기본·참조의 반복은 동일 기동 안에서 수행할 수 있으며 boot ID를 명시한다. 프로파일러를 붙이지 않은 결과만 일반 성능 표에 넣는다.

GLM 준비가 막혀도 Qwen 비교·계측·산출물 처리는 계속한다. 실패한 모델을 성공한 모델로 바꾸어 보고하지 않는다.

---

## 6. 입력·생성·캐시·자원 조건

### 6.1 성능 입력 동결

주 워크로드는 Sonnet 기반 입력512·출력128, prefix0, seed20260916, request-rate 무제한, `ignore_eos=True`를 사용한다. `MAIN_SHORT`의 각 구성은 **같은 모델 안에서 요청 텍스트·token ID·순서가 동일한 manifest**를 사용한다. 모델 간 tokenizer 차이를 기록한다. 토큰 수를 맞추기 위해 모델별 입력이 달라지면 이를 숨기지 않는다. [S2 §7]

이번 성능 측정의 sampling은 `temperature=0`, `top_p=1`, 추가 top-k 제한 없음으로 동결하고, 성능 강제 길이에는 `ignore_eos=True`를 사용한다. 이는 비교 재현성을 위한 이번 프로토콜이며, 과거 원시 manifest와 다르면 같은 이름만으로 과거 결과와 합치지 않는다. 서버와 클라이언트의 sampling seed를 분리하고 temperature·top-p·top-k·stop·EOS·chat template·thinking 모드를 모두 명시한다. 성능용 forced-length 생성과 품질용 정상 EOS 생성을 구분한다. 성능 본 측정은 두 모델 모두 raw completion 경로(`/v1/completions`)와 저장된 프롬프트를 사용하며 chat template을 암묵적으로 추가하지 않는다. GLM의 explicit thinking 제어가 필요한 별도 품질 요청에서는 공식 template의 `enable_thinking=False`를 사용하고 세 비교군에 동일하게 적용한다. 엔진이 해당 제어를 무시하면 실제 동작을 기록하고 조용히 다른 prompt로 대체하지 않는다.

매 반복 전 모든 이전 요청을 drain한 뒤 engine KV/prefix cache를 비우고 응답을 저장한다. kernel/JIT 워밍업 후에도 cache를 비운다. **engine cache flush와 OS page cache 제거는 다르다.** OS `drop_caches`나 가중치 재다운로드는 하지 않는다. 동적 expert 배치가 켜진 기본군에서는 cache flush가 expert 배치까지 초기화하는지 확인하고, 각 반복의 시작·종료 배치를 저장한다. 배치 초기화가 지원되지 않으면 adaptive state의 연속성을 명시한다.

### 6.2 저간섭 공통 계측

모든 성능군에 동일한 저빈도 계측을 사용한다: GPU2초, CPU/메모리2초 이상. startup·warmup·measure·drain의 경계를 별도 event로 남긴다. 시스템 전체 표본과 벤치 구간 표본을 둘 다 보존한다.

GPU별 UUID·SM utilization·메모리·전력·SM/메모리 clock·throttle reason, CPU per-core busy·주파수·package 전력, RSS/PSS·NUMA 배치·swap·page fault를 가능한 범위에서 수집한다. 실제 측정되지 않은 값은 0이 아니라 `null`과 사유 코드로 남긴다.

CPU2소켓·GPU4장은 같은 물리 서버다. 비교 중 소켓 하나를 다른 실험에 할당하거나 GPU4장을 추가 작업에 사용하는 것은 금지한다. 클라이언트·수집기·reporter의 실행 CPU와 자원 사용도 기록한다.

---

## 7. 최적화 구성의 구간별 프로브

### 7.1 먼저 실제 함수·스트림을 매핑한다

프로브 위치는 추정한 함수명이 아니라 **실제로 실행되는 Python/C++/CUDA 경로**에서 찾는다. 다음 위치를 조사하고 실제 파일·line·함수·PID·TID·rank·stream을 `probe_map.md`에 대응한다.

- SGLang scheduler, model runner, decode CUDA graph runner.
- 모델의 Attention, MoE router/top-k, KT wrapper의 submit·forward·sync/merge 경로.
- CPU expert dispatch, NUMA master·worker, quantize/dequantize·pack·matmul·reduce kernel.
- callback-free go/done signal, pinned slot ring buffer, CPU–GPU memcpy·stream wait 경로.
- GPU hot expert kernel, result combine, TP communication, residual·sampling.

`AMXINT4`라는 옵션만 보고 모든 CPU kernel을 AMX로 태깅하지 않는다. 실제 심볼·ISA 경로를 태깅한다. `dynamic dispatch`와 runtime weight migration도 별개 항목이다.

### 7.2 필수 계측 구간

| ID | 구간 | 필수 시점·수치 |
|---|---|---|
| S00 | 요청 수신·토큰화·스케줄러 대기 | receive, tokenize begin/end, enqueue, admitted, batch ID |
| S01 | Attention·MoE 입력 준비 | 실제 GPU begin/end, token rows, context 길이, KV read 관련 kernel |
| S02 | Router·top-k·Hot/Cold 분리 | GPU begin/end, logical expert ID·routing weights, pack·dispatch 시간 |
| S03 | CPU 입력 전달 | D2H enqueue와 실제 begin/end, bytes, source/dest, stream, pinned 여부 |
| S04 | CPU 작업 전달·큐 대기 | submit, 입력 준비 완료, worker pickup, go 신호, polling·blocking 구분 |
| S05 | CPU expert 계산 | expert별 rows, kernel 이름, quant/dequant/pack·GEMM·reduce 구간, worker/NUMA |
| S06 | CPU 완료·결과 반환 | 마지막 worker 완료, CPU reduce 완료, done 신호, H2D enqueue·begin/end·bytes |
| S07 | GPU Hot expert 계산 | GPU begin/end, expert별 실제·padding rows, grouped GEMM·reduce |
| S08 | 결과 결합 전 의존 대기 | Hot-ready, Cold-result-on-GPU-ready, 기타 prerequisite-ready, join-ready |
| S09 | 결과 결합·TP 통신·잔차 | combine kernel, collective 종류·bytes·rank 도착/완료, residual begin/end |
| S10 | logits·sampling·출력 전달 | sampling GPU 시간, detokenization, stream chunk 송신·client 수신 |
| S11 | graph·메모리 부수 경로 | capture/replay ID·batch size, graph miss/eager fallback, KV eviction/retraction·recompute |
| S12 | 동적 배치·layerwise prefill | weight movement bytes/time, map before/after, 임시 GPU expert 실행 및 정밀도 |

Shared expert가 존재하는 모델은 routed top-k 통계와 별도로 해당 실행 장치·kernel·시간·결합 지점을 남긴다. deferred 결과는 `source_step/token`, `consume_step/token`, produced/consumed/skipped/stale 상태와 연관 ID를 기록한다. 품질 계약을 확인하지 않은 채 늦은 결과를 삭제하도록 바꾸지 않는다.

### 7.3 단계 시간의 정확한 의미

CPU 호출 전후 wall clock만으로 GPU 연산 시간이라고 기록하지 않는다. GPU 실행은 비동기이므로 CPU enqueue 시간과 실제 device 시간을 분리한다. GPU는 CUDA event 또는 CUPTI/Nsight device activity, CPU는 monotonic clock을 사용한다. [W8][W9]

- 성능 경로에 단계마다 `cudaDeviceSynchronize`, `torch.cuda.synchronize`, 추가 barrier를 넣지 않는다.
- CUDA event는 미리 할당하고 결과는 완료 후 회수한다. Event 삽입도 overhead가 있음을 기록한다.
- Python NVTX 범위가 graph capture 때 한 번 실행된 것을 모든 replay의 layer 시간으로 복제하지 않는다.
- Graph ON의 최적화 경로를 유지한 상태에서 node-level trace 또는 replay 식별이 가능한 프로브를 사용한다. eager로 바꾼 결과만으로 graph 실행의 병목을 설명하지 않는다.
- graph replay·pinned ring buffer에는 batch/step/slot generation ID를 넣어 이전 token의 기록과 섞이지 않게 한다.
- 서로 다른 CUDA device event나 CPU clock의 raw 값을 직접 빼지 않는다. CUPTI 공통 타임라인 또는 명시적 clock alignment와 오차 범위를 기록한다.
- 프로브 수집용 메모리를 사전 할당한다. kernel·worker hot loop에서 매 token마다 문자열 포맷, 파일쓰기, 전역 lock, 메모리 재할당을 하지 않는다.
- 샘플링 여부·대상 step 규칙·drop count를 기록한다. 누락된 이벤트를 시간0으로 메우지 않는다.

### 7.4 노출 대기·중첩·임계 경로

같은 layer·batch의 결합에 대해, **동일하게 정렬된 GPU 타임라인**에서 다음 값을 정의한다.

```text
cold_ready = CPU 결과가 GPU에서 소비 가능한 시각
other_ready = Hot 경로 및 Cold 외 모든 결합 선행조건이 준비된 시각
cold_exposed_wait = max(0, cold_ready - other_ready)

hot_ready = Hot 경로 결과가 소비 가능한 시각
other_than_hot_ready = Cold 및 Hot 외 모든 결합 선행조건 준비 시각
hot_exposed_wait = max(0, hot_ready - other_than_hot_ready)

join_scheduling_gap = combine_actual_start - max(all_prerequisite_ready)
```

CPU의 done 시각을 Cold-result-on-GPU-ready로 대체하지 않는다. 실제 결과복사·stream dependency 완료까지 확인한다. 의존조건을 모두 포착하지 못하면 위 값은 `PARTIAL_DEPENDENCY_MEASUREMENT`로 기록하고 완전한 임계 경로라고 부르지 않는다. 음수 join gap은 그대로 보존하고 `CLOCK_OR_MAPPING_INCONSISTENT`를 기록한다.

단계별 GPU/CPU 시간이 겹치면 단순 합이 layer wall time을 넘을 수 있다. 아래를 분리한다.

- `busy_time`: 실제 작업 구간의 시간.
- `wall_span`: 해당 작업 집합의 시작~끝.
- `overlap_union/intersection`: 동일 clock·범위의 interval 집합으로 계산.
- `exposed_wait`: 다음 연산의 실제 진행을 막은 의존 대기.
- `exclusive_critical_path_time`: 관측 dependency graph에서 중복 없이 할당한 시간. 알고리즘과 동률 규칙을 저장.

96개 worker 시간이나 4개 GPU rank 시간을 단순 합해 노드 벽시계 비율로 사용하지 않는다. 전체 평균뿐 아니라 **모든 layer의 p50/p95/p99, 최대 대기 layer, rank별 차이, Cold 미선택 step/선택 step**을 분리한 표를 만든다. 표는 측정값의 기계적 정리이며 ‘병목 해결’, ‘포화’, ‘채택’ 같은 평가 문장은 작성하지 않는다.

---

## 8. 프로파일링 4세션/모델과 계측 간섭 확인

| 세션 | 조건 | 수집 내용 | 일반 처리량에 포함 |
|---|---|---|---|
| D0 `PROBE_OFF` | 최적화 설정, 512/128·C64·128요청 | 저빈도 시스템 계측만, probe 비활성 | 아니오, 간섭 비교 대조 |
| D1 `LIGHT_PROBE` | D0와 같은 요청·설정 | 단계 metadata·CPU job 구간·희소 GPU events | 아니오 |
| D2 `GPU_TIMELINE` | 같은 최적화 graph 실행, 준비된 요청 부하 | NVTX+CUDA activity, prefill·decode 구간, 전체 TP rank·copy·collective | 아니오 |
| D3 `CPU_TIMELINE` | 같은 최적화 설정·요청 | worker/master native profile, IPC·cycles, DRAM·NUMA/UPI, OS 대기 | 아니오 |

D1은 모든 MoE층을 포함하되 decode step을 우선 1/8 샘플링한다. 동일 step에서 모든 필요한 선행·후행 이벤트를 함께 수집한다. 오버헤드가 큰 경우 준비된 1/16 정책으로 한 번만 낮출 수 있고, 다시 실행하면 추가6세션 예산에서 차감한다.

D2는 **모델 적재 전체가 아니라 실제 요청 구간의 prefill 최대8초, decode 최대12초**를 수집한다. 대상은 benchmark client가 아니라 serving scheduler·GPU worker process tree다. 각 rank에 실제 CUDA activity가 있는지 확인한다. 문서 작성 시 확인한 Nsight 기능은 설치 버전의 `--help`와 대조한다. `--cuda-graph-trace=node`는 graph 내부를 보여 주지만 간섭이 클 수 있다. profiler 종료로 serving process가 자동 종료되지 않도록 `--kill none` 등 설치 버전의 해당 동작을 확인한다. [W8][W10]

D3는 native stack sample을 99Hz 수준에서 시작하고 길이를20초로 제한한다. `py-spy`·고주파 `perf`·GPU trace를 동시에 겹쳐 과도한 간섭을 만들지 않는다. `perf stat`의 process/TID 대상은 HTTP 프로세스가 아니라 KT worker와 scheduler임을 확인한다. PMU time_enabled/time_running·multiplex 비율·실제 단위를 저장한다. PCM과 perf가 같은 PMU를 충돌 사용하면 별도 구간으로 나누고 같은 순간의 수치라고 합치지 않는다. [W11]

메모리 대역폭은 GB/interval인지 GB/s인지 도구 출력의 정의를 확인한다. 실측 ceiling이 없는 상태에서 ‘DDR 대역폭50%’를 계산하지 않는다. worker wall time·thread CPU time·compute·spin·blocked를 구분하고, 전체 CPU busy40%를 ‘CPU60% 여유’로 환산하지 않는다.

D1/D2/D3의 소요시간·출력량을 D0와 별도 표에 기록한다. 다음 식의 원값과 비율을 저장한다.

```text
throughput_ratio = diagnostic_output_tps / probe_off_output_tps
elapsed_ratio = diagnostic_elapsed_s / probe_off_elapsed_s
```

수집기가 일부를 못 측정하면 해당 단계에 `NOT_COLLECTED`, 원인, 영향 범위를 남긴다. **CPU·GPU 사용률만 수집하고 단계별 계측 완료라고 표시하지 않는다.**

---

## 9. 동일 문항 출력과 오류 기록

성능과 별도로 고정한20문항을 6구성에 동일하게 적용한다. GSM8K 등 자동 정답 추출이 가능한 소규모 공개 문항을 사용하고 dataset revision·문항 ID·입력·정답·추출 함수를 저장한다. 성능용 입력과 혼합하지 않는다. 모델별 chat template·thinking 모드·greedy 설정은 비교군 사이에서 같게 유지한다.

문항별 원문, 생성 전문, token IDs(가능한 경우), 추출답, exact-match, EOS/length 종료, 미완료·오류를 저장한다. 정답수·전문 일치수·추출답 일치수는 기계적으로 집계할 수 있다. **‘품질 통과’, ‘동등’, ‘서비스 가능’이라는 평가는 작성하지 않는다.** GPU8도 절대 정답 모델이 아니라 대조 출력이다.

기존 최고 구성의20문항 비교에서 S2 재현군은20/20, RB 추가군은19/20이며 전문은11/20만 일치했다. 해당 사실 때문에 이번 출력 원문 보존을 필수로 한다. 새로운 결과를 과거 정답수로 채우지 않는다. [S1]

NaN/Inf·illegal access·비정상 종료 발생 시:

1. 실패를 감지한 최초 시각, 요청 원문·생성 인자·순서, boot ID·batch/step ID, seed, cache·배치 상태를 보존한다.
2. 처음 비유한값이 검출된 계층·tensor·kernel을 가능한 범위에서 기록한다. 실패 뒤의 연쇄 CUDA 오류를 최초 원인으로 단정하지 않는다.
3. 그 측정을 `FAILED_RUNTIME`으로 닫고 새 attempt에 재시도한다. 부분 성공의 출력량·경과시간·오류 수는 원시 값으로 보존한다.
4. 서버가 죽은 워밍업 뒤에는 정규 벤치를 시작하지 않는다.
5. 실패하지 않았으면 `NOT_REPRODUCED_WITHIN_BUDGET`이며 원인 해결이라는 뜻이 아니다.

---

## 10. 중단 없는 실행·복구·30분 보고

### 10.1 상태와 watchdog

모든 항목을 manifest에 선등록하고 상태를 전이시킨다.

```text
PLANNED → PREPARING → STARTING → SMOKE → WARMUP → MEASURING
        → DRAINING → SAVED → COMPLETE

실패 분기:
FAILED_BOOT / FAILED_RUNTIME / TIMED_OUT / CONFIG_MISMATCH /
UNSUPPORTED / BLOCKED_AUTH / BLOCKED_HARDWARE / BLOCKED_FORMAT
```

완료되지 않은 후속 항목은 `NOT_RUN_DEPENDENCY` 또는 `NOT_RUN_BUDGET`과 구체적 부모를 기록한다. 실패 로그를 덮어쓰거나 성공한 attempt만 남기지 않는다.

| 작업 | 기본 한도·복구 규칙 |
|---|---|
| boot | 15분. 가중치 적재 진행이 계속 확인되면 1회15분 연장하고 기록 |
| 일반 성능 세션 | 최대20분; 전체 진전·완료 요청·GPU 작업 상태 기록 |
| 진단 부하 | 최대10분, 실제 profiler capture는 제8절 상한 적용 |
| 데이터 다운로드 | resume 사용, 15분간 수신 byte 증가가 없으면 네트워크 확인1회; 총8시간 상한 |
| 변환 | checkpoint/progress 사용, 15분간 진전 없음+계산·I/O 없음 시 점검; 총4시간 상한 |
| 실패 재시도 | 동일 셀1회, 캠페인 합계6회 부하/6회 재기동 이내. 총부팅20회 한도도 동시 적용 |
| profiler 장애 | 그 profiler만 종료. 다른 수집기·일반 성능·산출물 작업 계속 |
| Git transient error | 최초+재시도2회. 인증 실패는 무한 재시도하지 않음 |

타임아웃 뒤 프로세스를 정리할 때에는 **본 캠페인 소유 PID·컨테이너만** 종료한다. `pkill python`, 전체 GPU reset, 타 작업의 namespace 종료는 금지한다. 새 프로세스 기동 전에 이전 작업의 GPU allocation 해제와 살아 있는 PID를 확인한다.

### 10.2 30분 중간 보고 — 문서가 아니라 실제 실행 루프에 구현

준비 시작을 `T0`로 잡고 **T0+30분, +60분, +90분…**에 보고한다. 다운로드·변환·boot가 오래 걸려도 해당 단계의 진행을 보고한다. 각 셀 종료 시에도 상태를 저장하며, 마지막에는 완료/차단 보고를 즉시 추가한다.

수행기는 main 실험과 독립된 경량 reporter를 실행한다. monotonic clock으로 다음 보고 deadline을 계산해 실험 세션 경계에 밀리지 않게 한다. 장시간 작업 때문에 보고 루프가 멈추는 구조로 만들지 않는다. 원시 로그를 매번 언어 모델에 전부 읽히지 말고 상태 JSON에서 작은 요약을 만든다.

보고 내용을 다음 두 곳에 동일하게 남긴다.

- `PROGRESS.md`와 `progress/progress.jsonl`에 append.
- 실행 환경에 승인된 사용자 메시지 채널이 있으면 같은 내용을 전송. 채널이 없으면 `REPORT_DELIVERY_UNAVAILABLE`와 로컬 저장 경로를 기록하고, 전송했다고 주장하지 않는다.

보고 레코드:

```text
보고 예정/실제 시각, 누적 경과시간
현재 모델/구성/boot ID/세션/단계
등록/완료/실패/차단/대기 항목 수
이번30분 완료한 실행 ID와 측정된 출력량·경과시간
현재 다운로드 byte/변환 진행률/완료 요청수(해당할 때)
GPU별 HBM, CPU RSS, 사용가능 디스크, 마지막 진행 시각
발생 오류·재시도 횟수와 저장한 증거 경로
다음 manifest 항목
잔여 시간 추정치와 계산 근거, 추정 불가면 null
```

‘순조롭다’, ‘개선됐다’, ‘성능이 나쁘다’ 같은 평가를 쓰지 않는다. 지연된 보고를 과거 시각으로 소급 생성하지 않는다. 누락되면 예정시각·누락시간을 남긴다. **결과 MD에도 이30분 보고 이력을 전부 포함한다.**

---

## 11. 필수 원시 데이터와 스키마

### 11.1 산출물 디렉터리

```text
<campaign>/
  PLAN.md
  PLAN_RESOLVED.md
  BASELINE_PROVENANCE.md
  CONFIG_DIFF.md
  RUN_MANIFEST.json
  RUN_STATE.json
  FULL_REPORT.md
  FULL_RAW_DATA.md
  PROGRESS.md
  COMPLETION_STATUS.md
  PUBLISH_RECEIPT.md
  README.md
  evidence/
    sources/                 # 공식 문서 snapshot·조회일·hash
    hardware/                # CPU/GPU/NUMA/clock/메모리/권한
    software/                # commit·diff·package·compiler·binary hash
    models/                  # revision·config·tokenizer·shard index·변환 로그
  config/<model>/<profile>/
    launch_cmd.sh
    argv.json
    env_allowlist.json
    resolved_server_args.json
    effective_feature_paths.json
    expert_map_start.json
    layer_budget.json
    affinity.json
  inputs/<model>/
    perf_requests.jsonl
    calibration_requests.jsonl
    quality_requests.jsonl
    hashes.json
  runs/<model>/<profile>/<boot>/<session>/<attempt>/
    server.log
    bench.log
    status.json
    requests.jsonl
    chunks.jsonl
    metrics_original.json
    timeseries/*.csv
  profiles/<model>/<session>/
    probe_map.md
    events.jsonl
    cpu_jobs.csv
    gpu_activities.csv
    dependencies.csv
    expert_dispatch.csv
    layer_step_raw.csv
    layer_stage_aggregate.csv
    critical_path_intervals.csv
    observer_overhead.csv
    original_trace_files/
  quality/<model>/<profile>/
    outputs.jsonl
    paired_outputs.md
  failures/<attempt>/
  progress/progress.jsonl
  raw_md/part-0001.md ...
  ARTIFACT_INDEX.csv
  SHA256SUMS.txt
```

### 11.2 최소 event 및 요청 필드

```text
events:
  run_id, model_revision, config_hash, boot_id, pid, tid, rank,
  gpu_uuid, stream_id, numa_node, clock_domain, clock_alignment_id,
  batch_id, step_id, layer_id, expert_logical_id, expert_physical_id,
  stage, begin_ns, end_ns, parent_event_id, correlation_id,
  input_rows, padded_rows, bytes, dtype, kernel_name,
  graph_id, replay_id, slot_id, slot_generation,
  sampled, sampling_rule, dropped_events, status

requests:
  request_id, input_text, input_token_ids, input_hash,
  submit_ns, first_chunk_ns, last_chunk_ns, finish_reason,
  output_text, output_token_ids, input_tokens, output_tokens,
  ttft_ms, tpot_ms, e2el_ms, http_status, error,
  generation_params, cache_status, thinking_mode

expert_dispatch:
  layer_id, step_id, logical_expert_id, execution_device,
  selected_tokens, routed_weight_sum, actual_compute_rows,
  deferred_count, produced_result_count, consumed_result_count,
  stale_or_omitted_count, map_generation, weight_precision
```

전체 input·output에 민감정보가 없는 사전 공개/합성 데이터를 사용한다. API token·SSH key·인증 헤더·비공개 사용자 입력은 저장소에 올리지 않는다. 환경변수 전체 `env`가 아니라 성능·실행 관련 allowlist만 기록한다.

Streaming chunk 하나가 항상 token 하나인 것은 아니다. chunk timestamp밖에 없으면 `inter_chunk_latency`로 기록하고 token-level ITL로 바꾸지 않는다. tokenizer 재계산값과 서버 usage 값을 둘 다 남긴다. `TPOT=(E2EL-TTFT)/(output_tokens-1)`은 output_tokens>1일 때만 계산한다.

### 11.3 FULL_REPORT.md와 FULL_RAW_DATA.md

**FULL_REPORT.md**에는 읽을 수 있는 상세 표를 넣는다. 단독으로 읽어도 실험 모델·정밀도·GPU수·기본/최적화 차이·측정 조건을 알 수 있어야 한다.

필수 장 순서:

1. 모델 자격 검사와 실행 상태.
2. 공식 원문 설정, H1004 자원 정규화, 최적화 설정의 전체 diff.
3. 실제 실행 환경·모델 revision·모든 명령·effective feature.
4. 모든 반복의 성공·실패·입출력 token·duration·처리량·지연.
5. 모든 layer·rank·stage의 구간 시간, 노출 대기·overlap·critical-path 표.
6. 모든 layer의 expert hit/Cold 선택·rows 분포·deferred 통계.
7. CPU·GPU·DRAM·NUMA 시계열의 원값과 경계.
8. 계측 ON/OFF 간섭과 누락 범위.
9. 동일 문항별 출력·기계적 채점·짝비교.
10. 모든 오류·재시도·미실행 항목.
11. 30분 보고 이력 전체.
12. 파일 색인·Git 게시 결과·다운로드 파일 목록.

**FULL_RAW_DATA.md**에는 설정·요청별 결과·step/layer/event·로그 등 텍스트 원자료를 누락 없이 묶는다. 너무 크면 `raw_md/part-*.md`로 분할하되 FULL_RAW_DATA.md를 전체 색인으로 제공한다. 머리·꼬리만 남기거나 `...`로 잘라 놓은 파일을 ‘전체 원데이터’라고 부르지 않는다. profiler 바이너리는 원본 파일·분할 archive·SHA256·재조립 방법으로 보존하고, MD에는 바이너리의 전체 텍스트 export를 연결한다.

집계 규칙:

- 토큰/초는 출력 token 합계 ÷ 첫 요청 제출~마지막 응답까지의 실제 벽시계 시간.
- 입력+출력 처리량과 출력 처리량을 분리한다.
- 모든 반복 원값, 평균·중앙값·최소·최대·표본표준편차(ddof=1)를 기록한다. n=1의 sd는 null.
- 요청 pooled percentile과 반복별 percentile 평균을 다른 열로 구분한다. percentile 알고리즘도 명시한다.
- 실패·timeout·관측 누락은 원시 값 그대로 보존하고 정상 반복 통계와 분리한다.
- GPU8 vs4의 장당 환산값을 전체 시스템 효율·비용 절감이라고 해석하지 않는다.
- **최종 결과물에 평가·원인 추측·성능 우열 선언·권고·운영 채택·후속 연구 제안을 쓰지 않는다.** `정상 종료`, `오류 발생`, `미수집` 같은 실행 상태는 허용한다.

---

## 12. Git 푸시와 다운로드 — 완료 조건의 일부

기존 승인된 실험 저장소를 사용한다. 정확한 저장소 주소가 지시서에 제공되지 않았으므로 **현재 실험 작업 경로의 Git remote·branch·권한을 먼저 확인**한다. 임의 저장소를 만들거나 공개 범위·가시성을 바꾸지 않는다.

1. 기존 작업물을 보존하고 충돌하지 않는 `feat/cpu-offload-two-models-<date>` 브랜치를 만든다. main에 직접 push하거나 force push하지 않는다.
2. 지시서, frozen config, 입력 manifest, converter/probe patch, 하네스, 데이터, 보고서를 보존한다. 새 계측 코드의 기능 변경과 관측 변경 diff를 분리한다.
3. 저장 전 비밀정보 검사를 수행하고, 모델 가중치 전체·라이선스상 재배포 불가 자료·font binary는 commit하지 않는다. 가중치는 출처·revision·허용된 hash manifest로 재현한다.
4. trace 등 대용량 결과는 승인된 Git LFS 또는 저장소 정책에 맞는 분할압축 파일로 게시한다. LFS pointer만 올라가고 실제 객체가 누락된 상태를 완료로 취급하지 않는다. 대용량이라고 로컬 경로만 적어 두고 전부 게시했다고 쓰지 않는다.
5. 결과 commit을 만들고 명시한 remote·branch로 push한다. `git ls-remote` 등으로 원격 head와 local commit 일치를 확인한다. 사용자 작업을 자동 rebase하거나 강제로 덮어쓰지 않는다.
6. `PUBLISH_RECEIPT.md`에 remote(credential 제거), branch, data commit, push exit code·시각, 원격 검증값, artifact hash를 기록한다. 영수증 자체는 후속 commit으로 게시할 수 있으며 그때 최종 원격 head도 검증한다. 자기 자신을 포함하는 hash의 순환 요구를 만들지 않는다.
7. 인증·네트워크·LFS 제한으로 게시하지 못하면 `PUBLISH_BLOCKED`와 오류 전문을 남기고 **로컬 다운로드 자료는 반드시 완성**한다. push 성공으로 꾸미지 않는다.
8. 결과 MD와 raw MD/분할 묶음을 사용자에게 전달 가능한 실제 경로로 복사하고 파일 존재·크기·UTF-8 한글·압축 무결성·링크를 확인한다.

최종 사용자 전달물은 최소 다음 네 가지다.

- 상세 결과 **`FULL_REPORT.md` 다운로드**.
- **`FULL_RAW_DATA.md` 또는 전체 분할 raw MD ZIP 다운로드**.
- config·입력·원시 로그·trace·하네스·30분 기록을 포함한 전체 자료 archive 다운로드.
- Git remote/branch/commit·게시 확인 상태. 실패했다면 실패 상태와 로컬 산출물 링크.

일반 웹 주소나 로컬 경로 문자열만 적지 말고, 실행 환경이 실제 제공하는 첨부·다운로드 링크를 사용한다. sandbox 링크는 해당 파일이 그 환경에 실제 존재할 때만 제공한다.

---

## 13. 최종 완료 체크

아래 항목을 `true/false/status`와 증거 경로로 기록한다. 체크리스트는 결과의 품질 평가가 아니라 실행·기록 완료 여부다.

- [ ] 두 모델의 정확한 revision, GPU8·GPU4 자격 상태를 기록했다.
- [ ] 공식 가이드와 실험 자원 정규화·로컬 최적화를 분리했다.
- [ ] 두 offload 구성에서 GPU resident expert 수가 0이 아니다.
- [ ] 최고 설정의 actual feature, hotmap·layer budget·RB 파싱 의미를 검증했다.
- [ ] GLM 이식 성공/미지원 여부와 per-channel FP8·INT4 형식 차이를 기록했다.
- [ ] 주 성능30회와 진단8회 계획의 각 항목에 완료·실패·미실행 상태가 있다.
- [ ] 일반 성능 수치와 profiler 부착 수치를 섞지 않았다.
- [ ] GPU 실행 시간과 CPU enqueue 시간을 구분했다.
- [ ] Hot/Cold 전달·연산·결합·대기 및 graph replay·rank 구간을 기록했다.
- [ ] 미계측 구간과 probe drop·overhead를 공개했다.
- [ ] 동일 문항별 출력·오류·정밀도 차이를 보존했다.
- [ ] 30분 보고를 실제 실행 중 작성했고 모든 기록을 결과 MD에 넣었다.
- [ ] 최종 MD에는 평가·추측·권고를 넣지 않았다.
- [ ] 모든 raw data 또는 원본 archive·hash·재조립 방법을 보존했다.
- [ ] Git push와 remote 확인을 수행했거나 차단 원인을 명시했다.
- [ ] 실제 다운로드 가능한 MD와 전체 archive를 제공했다.

`COMPLETED`는 필수 작업과 게시·전달이 모두 끝났을 때 사용한다. 일부 실행이 실패했지만 모든 항목을 처리했다면 `COMPLETED_WITH_FAILURES`; 실행 또는 게시 필수 조건이 막혔다면 해당 `BLOCKED_*`와 별도의 artifact completion 상태를 기록한다.

---

## 14. 근거 자료와 출처 구분

### 사용자 제공 실험 자료

- **[S1]** `EXPERIMENT_SUMMARY_20260916.md` — §0~4: V1 처리량·설정·정확성·미재현 상태. 요약의 ‘최적화 후 CPU 병목’ 추정은 이번 측정의 확정 사실로 사용하지 않는다.
- **[S2]** `FULL_REPORT-1.md` — IDE_071, §2~7: 실제 실행 명령, S2·R0·GPU8 구성, SHORT_COLD(prefix0)의 조건, 반복별 수치. 잘린 표보다 완전 명령·원본 argv를 우선하며 충돌은 별도 보존한다.
- **[S3]** `RESULT.md` — IDE_070, §1·4~5: 이전 CPU 계측 범위, 층별5952 예산, callback-free 구현 설명. 이전 기준선의 CPU 계측을 현재 최적화군의 측정값으로 재사용하지 않는다.
- **[S4]** `FULL_REPORT.md` — IDE_069, §2~4: 대상 서버와 Qwen FP8·INT4 및 GPU4/8 실행 기록.

### 공식 공개 자료 — 조회일 2026-09-17

- **[W1] KTransformers Inference**  
  https://ktransformers.net/en/docs/inference  
  `kt-kernel + sglang-kt` 실행 경로, 모델별 registry와 수동 실행 구분.
- **[W2] KTransformers Expert Placement**  
  https://ktransformers.net/en/docs/optimization-techniques/expert-placement  
  uniform 출발점, GPU 상주 수·dynamic update·보수적 deferred 설정.
- **[W3] KTransformers Native Precision Tutorial**  
  https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/Native-Precision-Tutorial.md  
  GLM-4.7의 `FP8_PERCHANNEL`, 명시적 native 예, layerwise prefill 구분. 실제 실행 시 commit-permalink로 동결한다.
- **[W4] KT-Kernel 공식 가이드**  
  https://kvcache-ai.github.io/ktransformers/en/kt-kernel/kt-kernel_intro.html  
  AMXINT4/8 경로, 물리 코어·NUMA pool, deferred 예시와 권고 범위. 문서의 예시32를 H1004에 그대로 복사한 것이 아니다.
- **[W5] Qwen3-Coder-480B-A35B-Instruct-FP8 모델 카드**  
  https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8
- **[W6] GLM-4.7-FP8 모델 카드**  
  https://huggingface.co/zai-org/GLM-4.7-FP8
- **[W7] GLM-4.7-FP8 config**  
  https://huggingface.co/zai-org/GLM-4.7-FP8/blob/main/config.json  
  MoE·dense·shared·MTP 구조 및 per-channel scale. 모델 카드와 tensor index의 parameter/size 표현이 다르면 원문을 별도 기록한다.
- **[W8] NVIDIA Nsight Systems User Guide**  
  https://docs.nvidia.com/nsight-systems/UserGuide/index.html  
  구간 제한, CUDA/NVTX trace, node-level graph 및 수집 overhead, process 종료 동작.
- **[W9] NVIDIA CUDA C++ Programming Guide, asynchronous execution**  
  https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html
- **[W10] NVIDIA CUDA Graph profiling 주의사항**  
  https://docs.nvidia.com/dl-cuda-graph/troubleshooting/performance-issues.html
- **[W11] Intel Performance Counter Monitor**  
  https://github.com/intel/pcm
- **[W12] KTransformers Support Matrix**  
  https://ktransformers.net/en/docs/support-matrix  
  모델·checkpoint·method·hardware·entry 조합에 따라 검증 범위가 다름.

문서의 실행 횟수·timeout·공통 KV 예산·프로브 설계·보고·게시 규칙은 **이번 캠페인을 위한 설계**다. 공식 권고 또는 이미 측정된 결과로 표기하지 않는다.
