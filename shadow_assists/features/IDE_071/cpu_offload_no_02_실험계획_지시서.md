# CPU MoE Offloading 후속 실험 계획 및 실행 지시서

> 문서 ID: `cpu_offload_no_02`  
> 실험 ID 제안: `IDE_071` — 기존 저장소에서 사용 중이면 다음 빈 번호를 사용한다.  
> 근거 자료: `FULL_REPORT.md`의 IDE_069, `RESULT.md`의 IDE_070.  
> 대상: 동일 서버 안의 CPU·DRAM과 H100 GPU를 함께 사용하는 Qwen3-Coder-480B-A35B 하이브리드 추론.  
> 목표: 기존 개선안과 추가 설정 최적화를 함께 탐색하고, 수렴 또는 사전 정의한 탐색 범위 소진 후 전체 측정 데이터를 상세한 Markdown으로 보존한다.

---

## 0. 실행자에게 내리는 필수 지시

**이 문서를 단순 검토하거나 계획만 다시 작성하지 말고, 실험 환경에 접근 가능한 실행자는 아래 절차를 실제로 수행하라. 기존 실험에서 제시한 변경뿐 아니라 이 문서의 추가 설정 탐색까지 포함하라.**

1. 사전 점검, 기준선 재현, 개별 요인 실험, 조합 탐색, 수렴 확인, 최종 반복 측정, 정확성 데이터 수집, 장시간 실행, 파일 검증, GitHub 게시, 다운로드 제공까지 수행하라. 일부 수치가 좋아졌다는 이유로 중간 종료하지 마라.
2. **실험 시작부터 30분 단위로 중간 보고하라.** 실행 중인 셀이 끝나기를 기다렸다가 보고하지 마라. 실험 실행기와 독립된 보고 루프를 사용하며, 실험은 계속 진행하라. 상세 요구는 제17절을 따른다.
3. 개별 셀의 OOM, 크래시, 시간 초과, 컴파일 오류, 지원되지 않는 조합을 전체 캠페인 종료 사유로 사용하지 마라. 해당 셀을 기록하고 제한된 재현·복구 절차를 수행한 뒤, 독립적으로 실행 가능한 다음 항목을 계속하라.
4. 권한 부족, BIOS 잠금, 지원되지 않는 하드웨어, 위험한 장치 상태를 우회하지 마라. 실행 불가능한 항목은 실제 시도·오류·의존성을 남겨 `BLOCKED_*` 또는 `UNSUPPORTED`로 종료하고 나머지를 완료하라. **실행 불가능 항목이 있는 상태를 ‘모든 실험 성공’으로 표현하지 마라.**
5. **수렴 확인 및 필수 후속 측정이 끝나면 평가·해석·의견·권고·채택 판단을 작성하지 마라.** “병목은 무엇이다”, “이 구성이 최적이다”, “운영에 적합하다”, “효과가 없다” 등의 결론을 붙이지 마라. 측정값, 계산식, 설정, 코드 변경 사실, 오류 원문, 실행 상태만 기록하라.
6. **실험 데이터는 요약표만 남기지 말고 최대한 상세하게 Markdown으로 저장하라.** 셀별·반복별·요청별·레이어별 데이터를 보존하고, 대용량 원본은 별도 파일과 체크섬으로 연결하라. 제18절의 기록 규격을 적용하라.
7. **완성한 MD, 하네스, 설정, 패치, 데이터 색인을 기존 프로젝트의 승인된 GitHub 저장소에 커밋하고 푸시하라.** 원격 반영을 확인하고 커밋 SHA와 파일 위치를 남겨라. 무단 공개 저장소 생성, 비밀정보 게시, 강제 푸시를 하지 마라.
8. **사용자가 바로 받을 수 있는 실제 MD 파일의 다운로드 링크를 제공하라.** 로컬 경로를 적거나 “GitHub에 있다”고 말하는 것으로 대신하지 마라. 파일 존재와 다운로드 가능성을 확인하라.
9. 최종 응답에는 완료·실패·차단 항목의 수, 실제 GitHub 반영 상태, MD 다운로드 링크, 원시자료 위치만 간단히 적어라. 결과 해석은 덧붙이지 마라.

이 지시서의 30분 보고는 **실험을 실행하는 프로세스/에이전트의 동작 요구**다. 이 문서 작성만으로 실험 서버가 시작되거나 별도의 알림 서비스가 등록되는 것은 아니다.

### 0.1 ‘끝까지 완료’의 정의

완료란 등록된 모든 필수 셀과 조건부 항목이 증거를 갖춘 종료 상태에 도달하고, 결과 파일 생성·검증·게시·전달 절차까지 수행한 상태다. 다음은 완료로 보지 않는다.

- 일부 시리즈만 실행한 뒤 나머지를 ‘추후 진행’으로 남긴 경우.
- 실패한 서버에 벤치를 계속 보내 생긴 0 tok/s를 정상 성능값으로 평균한 경우.
- 아직 `PENDING`, `RUNNING`, `RETRY_PENDING`인 셀이 남은 경우.
- 탐색 횟수 제한에 도달했을 뿐인데 ‘수렴했다’고 기록한 경우.
- GitHub 푸시가 실패했는데 업로드 완료로 기록한 경우.

단, 안전·권한·환경 제약으로 실행할 수 없는 셀은 그 근거를 남긴 종료 상태로 정리할 수 있다. 이는 측정 성공과 구분한다. 물리적 서버 장애로 모든 실행이 불가능해진 경우에도 이미 수집한 데이터를 보존하고 보고하며, 실행하지 않은 측정을 생성하지 않는다.

---

## 1. 기존 자료에서 계승할 사실과 재현 대상

아래는 **기존 보고서에 기재된 값**이다. 이번 실험의 측정값으로 재사용하지 않는다. 서로 다른 캐시 상태·설정·동시성의 값은 하나의 비교군으로 합치지 않는다. [S1][S2]

| 기존 구성/항목 | 보고서 기재값 또는 실행 사실 | 후속 실험에서 보존할 구분 |
|---|---|---|
| IDE_069 TP4 hot96, def4, KV40,960 | C64 약 642 ± 7 tok/s | IDE_069 기록 |
| IDE_070 `a0_base_turboOFF` | C64 629.9 ± 9.2 tok/s | 기존 설정의 재현 기준 |
| IDE_070 `u96v2` | C64 663.4 ± 13.2 tok/s | hotmap v2, uniform96 |
| IDE_070 패치 적용 `nu5952` | C64 696.7 ± 18.9 tok/s | 층별 비균일 배치, 총 5,952슬롯 |
| IDE_070 `d1_cf` | C64 669.6 ± 7.4 tok/s | callback-free 단독 |
| IDE_070 `d2_cf_skip8` | 1회차 128/256 요청 실패, 이후 서버 종료 | 성능 평균이 아니라 실패 재현 대상 |
| IDE_070 `d3_cf_skip8_pin` | C64 781.3 ± 8.0 tok/s | callback-free·skip-empty·def8·pinning 조합 |
| IDE_070 `d4_epoch` | C64 816.0 ± 29.2 tok/s | 여러 옵션을 동시에 변경한 별도 구성 |
| IDE_070 `d4_epoch`, C224 | 1,089.19 tok/s, 단일 측정, flush 수행 | C64 및 기존 warm-cache 반복과 분리 |
| IDE_069 GPU-only TP8+EP8 | C64 2,031.11 tok/s | GPU 수가 다른 참조 구성 |
| IDE_070 AWQ 하이브리드 | graph on/off 모두 초기 실행 실패 기록 | 처리량 값 없음 |
| turbo ON | root에서도 쓰기 거부, BIOS 잠금 | 같은 잠금이 유지되면 반복 우회 금지 |
| 프로파일 수집 | py-spy·perf 동시 실행 구간 517.21 tok/s | 일반 성능 측정과 분리 |
| 초기 `nu5952` 실행 | 실제 per-layer 패치 미적용 | 비균일 배치 결과로 재분류 금지 |

### 1.1 D4 재현 시 임의로 추가하면 안 되는 옵션

D4의 기록된 환경변수는 다음과 같다. 명령에 없는 D3의 `KT_CF_SKIP_EMPTY_IMM=1` 또는 `pin_nonkt.sh`를 D4에 임의로 추가하지 않는다. 실제 과거 하네스에 숨은 환경 상속이 있었는지는 별도로 확인하고, 확인된 경우 그 사실을 기록한다. [S2, 제5절]

```text
KT_AVX_RB=1
KT_AVX_PF=2
KT_FUSE_QIN=1
KT_AMX_MIN_QLEN=1000000
KT_AMX_MIN_ROWS=3
KT_CALLBACK_FREE=1
KT_COLD_DEFER=1
KT_COLD_TAU=0.25

hotmap = hotmap_mixed_0.25.json
max_deferred_experts_per_token = 8
KV dtype = fp8_e5m2
max_total_tokens = 143360
mem_fraction_static = 0.94
chunked_prefill_size = 4096
graph batches = [32, 64, 96, 128, 160, 192, 224]
```

`KT_AMX_MIN_QLEN`, `KT_AMX_MIN_ROWS`, `KT_COLD_DEFER`, `KT_COLD_TAU`의 의미는 이름으로 추측하지 않는다. 해당 빌드의 소스에서 조건문·기본값·적용 단계·연산 생략 여부를 확인한다. `hotmap_mixed_0.25`의 0.25가 무엇을 뜻하는지도 생성 코드를 확인한다.

### 1.2 기존 계측을 이어받을 때의 규칙

- cold expert 선택 수는 `한 레이어의 한 토큰당`, `전체 62층을 통과한 토큰당`, `전체 route assignment당`을 구분한다. 같은 단위를 사용하지 않은 수치를 비교하지 않는다.
- 프로파일의 샘플 비중을 곧바로 전체 벽시계 시간의 비중으로 변환하지 않는다.
- `/proc/stat` 전체 논리 CPU 평균, 물리 코어 사용률, kt 워커 CPU 사용량을 각각 기록한다.
- `perf`가 표시한 `/sec`를 그대로 시스템 벽시계 초당 횟수로 사용하지 않는다. 원시 count와 실제 측정 창 길이를 함께 저장한다.
- PCM 대역폭 단위는 해당 실행 버전의 헤더·출력 정의로 검증한다. 이미 GB/s인 값을 다시 샘플 간격으로 나누지 않는다.
- 요청 동시성 증가에 따른 성능 변화와 설정 변경에 따른 성능 변화를 분리한다.

---

## 2. 탐색 범위 및 구성 계열

### 2.1 세 종류의 변경을 분리한다

| 분류 | 범위 | 수행 방식 |
|---|---|---|
| `CONFIG` | 설치된 CLI·환경변수, thread affinity, KV·graph·스케줄링·커널 선택 | 먼저 존재와 효과를 검증한 뒤 비교 |
| `PATCH` | 핸드오프, 빈 작업 처리, 행 수별 커널 선택, 비균일 배치, 작업 묶음·캐시 등 코드 변경 | 독립 패치·단위 테스트·빌드 해시를 붙여 비교 |
| `FORMAT` | FP8 KV, GPU W4, CPU/GPU 가중치 출처 변경, 근사 경로 | 동일 수치 표현을 유지한 변경과 분리하고 출력·정확성 데이터를 수집 |

### 2.2 연산 의미별 계열

`deferral=0`을 포함한 기준을 마련하고, 각 기능의 소스를 읽어 아래 계열을 부여한다.

- `SEMANTICS_PRESERVING`: 선택된 expert의 필요한 계산을 모두 수행하고 모델의 데이터 의존성을 유지하는 경로. 부동소수점 합산 순서 차이 가능성은 별도 기록한다.
- `QUANTIZATION_CHANGED`: KV 또는 가중치 표현이 바뀌는 경로.
- `APPROXIMATE`: expert 결과 생략·대체·지연 사용 등 수치적 의미를 바꾸는 경로.
- `SEMANTICS_UNVERIFIED`: 구현 의미를 아직 확인하지 못한 경로. 확인 전에는 보존형 계열로 승격하지 않는다.

**deferral·tau·skip-empty라는 이름만으로 무손실 또는 근사라고 확정하지 않는다.** 정확성 수치 수집은 실험 중 측정 활동으로 수행하되, 최종 문서에서 품질 저하 여부나 서비스 채택 여부를 논평하지 않는다.

### 2.3 실행 순서

```text
P0 환경·옵션·코드 확인
 → P1 기준선·대조군 재현
 → P2 callback-free/skip-empty/pinning/deferral 분해
 → P3 CPU 커널·스레드 설정
 → P4 KV·HBM·CUDA graph
 → P5 스케줄링·attention·GPU MoE·collective 설정
 → P6 층별 expert 배치·적응형 배치
 → P7 NUMA·메모리·런타임 확인
 → P8 GPU W4·병렬 구성
 → P9 조건부 비동기 확장·speculative decoding 재시험
 → P10 조합 탐색·수렴 확인
 → P11 최종 반복·장시간·정확성 데이터 수집
 → 상세 MD·원본 색인 생성·검증
 → GitHub 게시·MD 다운로드 제공
```

의존성이 없는 항목은 앞 단계의 특정 셀 실패와 무관하게 진행한다. 성능 측정 자체는 같은 CPU·GPU·DRAM을 공유하는 다른 측정과 동시에 수행하지 않는다.

---

## 3. P0 — 환경 동결과 옵션 검증

### 3.1 실험 시작 전에 저장할 정보

서버·컨테이너·저장소마다 다음을 기록한다.

```text
host / UTC와 Asia/Seoul 시각 / kernel / OS
CPU socket·core·SMT·NUMA topology / 각 TID의 affinity
CPU governor / no_turbo / Bzy_MHz / power 및 thermal 제한
DIMM 구성·실제 configured speed / NUMA node별 메모리
GPU UUID·물리 index·컨테이너 index / NVLink·PCIe topology
GPU driver·CUDA·NCCL·PyTorch·SGLang·kt-kernel·FlashInfer·Triton 버전
컨테이너 image digest / cpuset.cpus·cpuset.mems / CPU quota / shm / memlock
cgroup의 실제 제한 / 부모 cgroup 제한 / 실험 외 프로세스 유무
SGLang·kt-kernel git SHA / git diff / 빌드 옵션 / 공유 라이브러리 SHA256
모델·토크나이저·quant config snapshot / shard manifest와 파일 크기
hotmap·layer budget·quantization scale 파일과 SHA256
기존 원격 저장소·브랜치·작업 트리 변경사항
```

보고서의 컨테이너 이름이 현재도 존재한다고 가정하지 않는다. `sgl-kt`, `vllm-h100`의 존재·실행 상태를 검증하고, 기존 프로젝트 하네스로 복구한다. Docker Engine이 없는 환경에서는 실제 `nerdctl/containerd` 경로를 사용한다. [S1][S2]

### 3.2 옵션 등록부를 만든다

`OPTION_REGISTRY.md`와 `option_registry.json`을 작성한다. 각 항목에 다음을 넣는다.

```text
name / category / source_kind
installed_cli_supported / parser_allowed_values / resolved_default
source_file / line_range / source_commit
runtime_application_probe / effective_value
requires_patch / compatible_model / compatible_GPU / compatible_KV_dtype
conflicts / source_reference / check_timestamp / evidence_file
```

`source_kind`는 `LOCAL_MEASURED`, `LOCAL_CODE`, `PUBLIC_DOC_CANDIDATE`, `NEW_PATCH_PROPOSAL` 중 하나다. 공개 문서에 존재하는 옵션이 로컬 SGLang 0.5.18과 수정본에서 작동한다고 가정하지 않는다. 공식 문서도 설치 환경의 `--help` 확인을 안내한다. [W1]

**검증 순서:** 로컬 `--help` → parser/source → 최소 부팅 → 실제 선택된 backend·값 로그 → 경로 실행 카운터/작은 진단 실행. CLI가 조용히 무시되거나 자동 fallback한 셀은 의도한 설정의 측정으로 인정하지 않는다.

특히 다음을 확인한다.

- 기존 `--cuda-graph-max-bs`, `--cuda-graph-bs`와 단계별 graph 인자의 별칭·우선순위.
- `--max-total-tokens`를 명시했을 때 실제 pool 크기와 `--mem-fraction-static`의 관계.
- `--max-prefill-tokens`가 context 설정·chunk 설정 때문에 실질적으로 변경되지 않는지.
- `--moe-runner-backend`가 kt wrapper 내부 GPU 경로에도 적용되는지.
- `KT_*` 변수의 unset과 0이 같은 동작인지.
- `KT_AMX_MIN_QLEN`과 `KT_AMX_MIN_ROWS`의 조건 결합 방식.
- graph padding 토큰이 CPU cold 작업 목록에 잘못 포함되는지.
- 모든 TP rank가 cold expert 연산을 중복 수행하는지, 아니면 설계된 shard/단일 실행인지.
- `deep_gemm.disabled`가 유지되는 이유와 현재 사용 가능한 GPU MoE backend.

### 3.3 코드와 의존성 변경 원칙

기존 작업 트리를 초기화하거나 덮어쓰지 않는다. 별도 worktree/브랜치와 실험용 컨테이너 또는 복제 환경을 사용한다. `pip install -U`로 기존 환경을 일괄 갱신하지 않는다.

업스트림 버전 비교가 필요하면 별도 `VERSION` 시리즈로 실행한다. 모델, 입력, 설정, 패치의 차이를 명시하고 기존 버전의 설정 실험과 합쳐 해석하지 않는다. 변경한 코드가 실제 import·로드된 위치와 `.so` 해시를 부팅마다 기록한다.

---

## 4. 공통 하네스·측정 조건

### 4.1 기준 구성 R0

아래 설정을 기존 환경에서 실제 적용되는 명령으로 해석해 저장한다. 전체 명령은 `launch_cmd.sh`, 값은 `effective_config.json`에 각각 보존한다. [S2]

```yaml
model: Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8
model_revision: 003f183a92fbe5b9a8325aaa8b2ae797c91dd90f
cpu_weights: /models/kt/qwen3-480b-int4
cpu_method: AMXINT4
gpus: [0, 1, 2, 3]
tp: 4
context_length: 32768
attention_backend: triton
cpuinfer: 96
threadpool_count: 2
num_gpu_experts_per_layer: 96
hotmap: /models/kt/ide069/hotmap.json
max_deferred_experts_per_token: 4
ep_dispatch_algorithm: dynamic
cuda_graph_max_bs: 64
prefill_cuda_graph: disabled
mem_fraction_static: 0.95
max_total_tokens: 40960
kv_cache_dtype: auto
speculative_decoding: disabled
```

이 설정 외의 의도하지 않은 환경변수를 상속하지 않는다.

`environment.before.txt`, `environment.effective.txt`을 저장하고, 인증정보 등은 공개 산출물에서 제거한다.

### 4.2 워크로드 계열

| 계열 | 입력/출력 토큰 | 용도 | 기본 동시성 |
|---|---:|---|---|
| `LEGACY_070` | sonnet 512/128, prefix100, seed42, 요청4×C | 과거 측정 프로토콜 재현, 반복 사이 flush 없음 | C64 |
| `SHORT_COLD` | 512/128, 새 요청 집합 | 주 설정 탐색, 매 반복 전 KV/prefix cache 초기화 | C64 |
| `PREFIX_WARM` | 공통 prefix100 + 서로 다른 suffix, 총512/128 | prefix 재사용 조건 측정 | C64 |
| `DECODE_HEAVY` | 512/1024 | decode 비중이 큰 조건 | C16·64 |
| `PREFILL_HEAVY` | 4096/128 | prefill·혼합 스케줄링 | C16·64 |
| `LONG_CONTEXT` | 16384/256 | KV·attention·정확성 데이터 | C1·4·16 |
| `CODE_MIX` | 고정된 코드 작업·길이 manifest | hotmap의 별도 입력 집합 측정 | C16·64 |
| `MIXED_SERVICE` | 위 요청의 사전 고정 비율·길이 분포 | finite-rate 및 장시간 실행 | 사전 등록값 |

표의 입력 크기는 목표 크기다. 토크나이저 적용 후 실제 입력·출력 토큰 수를 저장한다. 긴 입력을 조용히 잘라내거나 출력 길이를 줄여 셀을 성공시키지 않는다. 메모리 부족은 실패·제약 데이터로 기록한다.

`CODE_MIX`와 `MIXED_SERVICE`의 원문 출처, 라이선스, 고정 요청 ID, 길이, checksum을 먼저 등록한다. 임의 생성 입력이면 생성 절차와 원문을 함께 보관한다. 공개 벤치 결과와 자체 생성 입력을 같은 지표명으로 혼용하지 않는다.

### 4.3 캐시·워밍업

모델 로딩, 커널 JIT, graph capture를 끝내고 동일한 준비 부하를 1회 실행한다. `SHORT_COLD`에서는 그 뒤 엔진의 cache flush를 수행하고 반환값·캐시 점유를 확인한다. 지원되지 않으면 서버 재시작 등 검증된 대체 절차를 사용하고 비용·차이를 기록한다.

`PREFIX_WARM`은 측정할 전체 프롬프트를 미리 요청하지 않는다. 공통 prefix만 준비하고, 측정 suffix는 서로 다르게 한다. `LEGACY_070`의 동일 seed·무flush 결과는 그 이름 그대로 별도 표에 남긴다.

**엔진 KV cache flush와 OS page cache 제거를 혼동하지 않는다.** 공유 서버에서 `/proc/sys/vm/drop_caches`, swapoff, 재부팅을 성능 준비 절차로 사용하지 않는다.

hotmap 작성용 `CALIBRATION`, 설정 선택용 `TUNING`, 마지막 측정용 `HOLDOUT` 입력 ID를 분리한다. HOLDOUT은 hotmap 작성·튜닝·KV scale 보정에 사용하지 않는다.

### 4.4 반복·순서·벤치 범위

- 탐색 셀: 최소 3회. 동일한 요청 manifest를 비교 대상 셀끼리 짝지어 사용한다.
- 최종 후보: 최소 5회. 변동성 규칙에 따라 최대 9회까지 추가한다.
- 설정 효과 비교는 동일 C·동일 요청·동일 cache policy·동일 precision 계열에서 수행한다.
- 셀 순서는 고정 seed로 무작위화하고, 6개 셀 또는 90분마다 대조군을 다시 측정한다. 30분 중간 보고와는 독립 규칙이다.
- 기본 legacy 요청 수는 4×C이며, 확인 측정은 16×C 또는 독립적인 steady-state 300초 부하로 구분해 실행한다. 짧은 요청 wave 결과와 steady-state 결과를 합산하지 않는다.
- 서버 재부팅·graph/JIT 초기화 시간은 서비스 준비 시간으로 별도 저장한다. 요청 성능 표에는 명시적으로 제외하되, 전체 실험 시간에서도 삭제하지 않는다.
- `ignore_eos`, temperature, seed, stop 조건, stream 설정을 고정하고 저장한다. 출력128을 채우지 못한 요청은 실제 출력 수와 종료 사유를 기록한다.

### 4.5 측정 경로 분리

`PERF_CLEAN`에는 무거운 profiler, expert 전체 recorder, 매 토큰 파일 기록, debug 동기화를 넣지 않는다. 동일한 저부하 계측 집합만 사용한다.

`DIAGNOSTIC`에는 PCM, perf, NVTX/Nsight, recorder, sanitizer를 필요에 따라 **분리 실행**한다. 같은 구성을 profiler 없이 재실행한 값도 남긴다. 프로파일 영향을 포함한 처리량을 일반 처리량 평균에 넣지 않는다.

각 실행 창에는 `boot`, `warmup`, `cache_prepare`, `measure_start`, `measure_end`, `drain`, `shutdown`의 벽시계·monotonic 타임스탬프를 둔다. PCM·perf의 창을 요청 부하 창과 정확히 연결한다.

---

## 5. P1 — 재현 기준선과 대조군

다음 셀은 후속 탐색 결과와 무관하게 실행한다. 모든 값은 이번 실행에서 다시 측정한다.

| 셀 ID | 구성 | 필수 실행 |
|---|---|---|
| `R00` | R0, 기존 hotmap·def4 | LEGACY_070 C64 3회 + SHORT_COLD C64 3회 |
| `R01` | R0에서 def0, cold-defer/tau 관련 추가 경로 해제 | SHORT_COLD C64 3회, 정확성 기준 출력 |
| `R02` | 과거 D1을 그대로 재현 | LEGACY_070 C64 3회 |
| `R03` | 과거 D3를 그대로 재현 | LEGACY_070 C64 3회 + SHORT_COLD C64 3회 |
| `R04` | 과거 D4를 그대로 재현 | LEGACY_070 C64 3회 + SHORT_COLD C64 3회 |
| `R05` | hotmap v2, uniform96 | SHORT_COLD C64 3회 |
| `R06` | hotmap v2, 비균일5,952, 실제 패치 적용 확인 | SHORT_COLD C64 3회 |
| `R07` | GPU-only TP8+EP8, 같은 모델·토크나이저 | 기존 GPU8 사용 권한·가용성 확인 후 C32·64 3회 |

`R07`은 GPU4 실험과 GPU 자원이 다른 참조군이다. GPU 수·CPU 자원·precision·KV budget을 함께 표기한다. 미사용 GPU4~7이 현재 다른 작업에 배정되어 있으면 선점하지 않고 해당 셀을 차단 상태로 기록한다.

`R06`은 레이어62개 각각의 적용 expert 수, 합계5,952, 실제 생성된 텐서 shape, 물리↔논리 expert mapping을 로그로 확인한다. 환경변수 설정이나 패치 스크립트의 성공 종료만으로 적용을 확인하지 않는다.

---

## 6. P2 — callback-free·skip-empty·pinning·deferral 분해

### 6.1 핵심 요인 실험

R0의 모델·hotmap·KV·graph·CPU 코어 수는 고정한다. 다음 유효 조합을 모두 수행한다.

| 요인 | 수준 |
|---|---|
| callback-free | OFF / ON |
| skip-empty | OFF / ON, 단 callback-free ON일 때만 ON 허용 |
| 비-kt 스레드 pinning | OFF / ON |
| max deferred experts/token | 0 / 4 / 8 |

총 18개 구성이다. `OFF`의 표현은 소스 확인 후 unset 또는 0 중 정확한 방법을 사용한다. 셀 ID에 네 요인의 값을 모두 넣는다. 각 셀 SHORT_COLD C64 3회를 실행하고, `def0`과 `def8`의 추가 출력 비교 데이터를 수집한다.

pinning ON에서는 기본 kt 코어0–47,56–103을 먼저 확인한다. 실제 값이 같을 때 비-kt 스레드는 남은 물리 코어48–55,104–111 및 **그 물리 코어의 HT 형제**에 배치한다. kt 물리 코어의 HT 형제를 비-kt 전용 코어로 잘못 취급하지 않는다. 코어 번호를 일반화해서 다른 머신에 그대로 적용하지 않는다. [S2]

기록 대상은 TP0만이 아니라 전체 TP rank, kt master, worker, poller, scheduler, tokenizer, detokenizer, 통신 helper, 벤치 클라이언트다. TID가 다시 생성되면 재적용하고 적용 직후 및 측정 종료 시 affinity를 저장한다.

### 6.2 D2 실패 재현·진단

D2 형태의 셀은 먼저 profiler 없이 재현한다. 실패 시 첫 오류부터 전체 로그를 보존하며 `py-spy failed` 같은 사후 오류만 원인처럼 남기지 않는다.

진단 순서는 다음과 같다.

1. 같은 설정의 새 서버로 재현하고 exit code, signal, OOM/cgroup 이벤트, GPU 오류, stderr를 수집한다.
2. 동일 precision·hotmap에서 callback-free/skip-empty/deferral/pinning을 한 개씩 분리한다.
3. 진단 전용 빌드에서 empty 작업 완료 통지, go/done 상태 전이, ring slot generation, 작업 ID, 완료 횟수, buffer 재사용 시점을 기록한다.
4. CPU 측 동기화·GPU stream 완료·호스트 메모리 가시성 규칙을 해당 API와 구현 기준으로 확인한다. 단순 CPU atomic 교체만으로 CPU–GPU 가시성이 보장된다고 가정하지 않는다.
5. 필요한 경우 최소 재현 입력에서 Compute Sanitizer와 CPU sanitizer를 별도로 사용한다. 해당 실행의 성능값은 일반 성능 표와 분리한다. [W9]

무작정 대기 시간을 늘리거나 `sleep`을 넣어 크래시를 숨기지 않는다. 그런 변경은 명시적인 별도 셀로만 기록한다.

### 6.3 overlap scheduler와의 교차 실험

앞선 행렬에서 완전한 요청 처리가 확인된 callback-free OFF/ON 구성 각각에 대해 엔진 overlap scheduler ON/OFF를 비교한다. 설치본이 지원하면 `--disable-overlap-schedule`의 유무로 조절한다. [W1]

CPU expert 핸드오프의 비동기화와 SGLang scheduler overlap은 서로 다른 기능으로 취급한다. 두 기능을 묶어 ‘비동기 ON/OFF’ 한 항목으로 기록하지 않는다.

### 6.4 P2 필수 추가 계측

```text
레이어·step별 empty CPU 작업 수
CPU 작업 제출 수 / 실제 expert 계산 수 / 완료 통지 수
poll 횟수 / 시간 조회 횟수 / sleep·yield·spin 시간
ring slot 사용 수·재사용 generation·high-water mark
동일 작업의 중복 실행·미완료·오래된 완료 통지 수
각 rank의 CPU 작업 관여 여부
입력 복사·CPU 실행·출력 복사·combine 대기 시간
```

모든 카운터를 매 토큰 파일에 동기 출력하지 않는다. 진단 모드의 메모리 버퍼에 모아 측정 종료 후 기록한다.

---

## 7. P3 — CPU expert 커널·스레드·런타임 설정

### 7.1 D4 커널 옵션을 따로 시험한다

먼저 선택된 P2 구성에 대해 다음 옵션을 **각각 하나씩 변경**한다. 이어서 유효한 이진 옵션은 제한된 조합 실험으로 확인한다. 아래 숫자는 탐색 후보값이며, 설치된 로컬 코드가 허용하는 범위로 매핑한다. [S2]

| 항목 | 탐색 후보 | 필수 확인 |
|---|---|---|
| `KT_AVX_RB` | unset / 0 / 1 | 실제 의미·기본값·실행된 커널 |
| `KT_AVX_PF` | unset / 0 / 1 / 2 / 4 | 값의 단위·지원 범위·메모리 prefetch 동작 |
| `KT_FUSE_QIN` | unset / 0 / 1 | 입력 quantization이 실제 결합되는지 |
| `KT_AMX_MIN_ROWS` | unset / 1 / 2 / 3 / 4 / 8 / 16 | expert별 실제 입력 행 수와 dispatch 경계 |
| `KT_AMX_MIN_QLEN` | unset / 32 / 64 / 128 / 256 / 512 / 1024 / 1000000 | qlen의 의미, MIN_ROWS와의 조건 관계 |
| kt CPU worker 수 | 80 / 88 / 96 / 104 / 108 / 112 | 실제 thread 수·소켓 분할·비-kt 전용 코어 유무 |

기본값과 같은 유효 설정은 해시로 중복 제거한다. 지원 범위를 벗어난 값은 임의로 parser를 완화하지 않고 `UNSUPPORTED_VALUE`로 기록한다.

`KT_AMX_MIN_QLEN=1000000`을 ‘AMX 가속 ON’으로 표기하지 않는다. 실제 경로를 소스로 확인하고 AVX/AMX dispatch 횟수로 기록한다.

### 7.2 expert 입력 크기 기반 microbenchmark

서버 동시성 C를 expert 행 수 M으로 대신하지 않는다. prefill·decode·speculative verification의 실제 `(layer, expert, M, K, N)` 분포를 먼저 수집한다.

| 축 | 값/방법 |
|---|---|
| M | 1,2,3,4,8,16,32,64,128 및 실측 빈도가 높은 추가 값 |
| weight shape | 모델의 gate/up/down 실제 shape 및 TP shard shape |
| CPU 경로 | 현재 dispatch, 지원되는 AVX 경로, 지원되는 AMX 경로 |
| thread/team 크기 | 1,2,4,8,16,24,48의 실현 가능한 수준 |
| 메모리 상태 | 동일 weight 재사용 / 여러 expert weight 순환 |
| 실행 | warmup 후 충분한 반복, raw sample 저장, E2E와 분리 |

양자화 unpack, activation quantization, matmul, SiLU·곱셈, 출력 rescale·combine 시간을 나눠 저장한다. 모든 경우 동일 입력·동일 가중치·동일 정답 기준을 사용한다. 타일 padding·유효 행 마스크·scale 처리의 정확성도 기록한다.

새로운 M별 dispatch 정책을 구현하면 기존 정책과 별도 patch ID로 구분한다. 작은 M에서 여러 expert를 한꺼번에 처리하거나 expert별 team 크기를 줄이는 정책도 시험하되, 선택된 모든 expert 기여도를 유지한다.

### 7.3 비-kt 스레드와 병렬 라이브러리

`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, PyTorch intra/inter-op thread 수, tokenizer 병렬성을 확인한다. kt 자체 threadpool과 다른 병렬 계층을 구분한다.

탐색 후보는 라이브러리별 기존값/1/2/4다. 한꺼번에 모든 변수를 바꾸지 않는다. 관련 라이브러리가 실제 호출되지 않는 경우 `NO_EFFECTIVE_PATH`로 남기고 해당 축을 중복 탐색하지 않는다. 설정 때문에 kt thread 수까지 변경되었으면 별도 구성으로 기록한다.

worker112에서는 남는 물리 코어가 없을 수 있다. 이 구성을 worker96+전용 비-kt 코어 구성과 비교할 때 코어 자원 차이를 명시한다. HT를 끄거나 커널 boot 옵션을 바꾸는 작업은 이 설정 탐색에 포함하지 않는다.

### 7.4 polling 및 작업 묶음 — 로컬 옵션이 없으면 PATCH

다음은 이미 구현된 플래그라고 가정하지 않는다. 지원 여부를 확인하고 없으면 명시적인 패치로 구현한다.

- 매 iteration의 시간 조회를 줄인 polling: 시간 확인 간격1/8/32/128 poll.
- bounded spin 후 yield 또는 event 대기로 전환: spin budget 0/10/50/100µs.
- expert별 작업을 작은 task로 과분할하지 않는 chunk/team 정책.
- worker별 로컬 완료 카운터와 최종 합산, false sharing을 피한 상태 배치.
- 한 step 안에서 같은 expert에 배정된 행을 모으는 exact grouped 실행.

CPU 사용률 감소만 측정하지 말고 제출→시작 지연, 완료→GPU 재개 지연, context switch, 처리량, p99를 함께 저장한다. future token 또는 다음 레이어의 미완료 입력을 미리 사용하지 않는다.

---

## 8. P4 — KV·HBM·CUDA graph 공동 탐색

### 8.1 먼저 메모리 항목을 따로 계측한다

GPU별로 모델 로딩 직후, hot expert 배치 직후, KV 할당 직후, graph capture 직후, warmup 직후, prefill peak, decode peak, 종료 직전의 메모리 사용량을 저장한다.

```text
dense/attention weights
GPU expert weights + quantization metadata
KV data + scale + page/index metadata
CUDA graph private pool / captured input-output buffer
GEMM·attention workspace
activation / temporary tensor
PyTorch allocated·reserved / NVML used / non-PyTorch 사용량
```

설정값이 아니라 실제 tensor bytes와 allocator peak를 기준으로 비교한다. 동일한 `max-total-tokens`에서 mem-fraction만 변경했는데 실제 할당이 같다면 그 사실을 기록한다.

### 8.2 메모리 예산의 계산 참고값

다음은 기존 모델 config로부터 계산한 **배열 본체의 크기**이며 실제 HBM 측정값이 아니다. scale·padding·workspace·복제 비용은 제외한다. [S1, 모델 config]

```text
FP8 expert 1개, 한 레이어, TP4 rank당
  = 3 × 6144 × 2560 × 1 byte / 4
  = 11,796,480 byte = 11.25 MiB

BF16 KV 1 token, TP4 rank당, KV head가 균등 shard된 경우
  = 2(K,V) × 62 layers × 8 KV heads × 128 head_dim × 2 byte / 4
  = 63,488 byte = 62 KiB

동일 조건의 FP8 KV 본체 = 31 KiB/token/rank
```

실제 모델 경로가 KV head를 복제하거나 다른 layout을 쓰면 식을 수정하고 수정 근거를 남긴다. 이 계산으로 OOM 부팅을 대신하거나 ‘반드시 들어간다’고 선언하지 않는다.

### 8.3 탐색 축

| 축 | 1차 후보값 | 실행 규칙 |
|---|---|---|
| KV dtype | auto/BF16, fp8_e5m2, fp8_e4m3 | backend별 지원 및 scale 실제 로딩 확인 |
| KV token pool | 24,576 / 40,960 / 57,344 / 81,920 / 114,688 / 143,360 | 같은 HBM 조건에서 가능한 값만 수행 |
| mem-fraction | 0.90 / 0.92 / 0.94 / 0.95 / 0.96 | 실제 할당·런타임 peak 기록, 상한 강제 확대 금지 |
| decode graph max batch | 64 / 96 / 128 / 224 / 256 | 실제 활성 batch 및 capture 목록과 구분 |
| graph capture 목록 | 아래 G0~G3 | 같은 max batch에서 목록 효과를 별도 비교 |
| graph padding | 기본 / padding 비활성 | 설치본 지원 및 eager fallback 여부 기록 |
| prefill graph | disabled / 지원되는 piecewise·breakable 중 최소 구성 | kt host 경로 호환성 검증 후 별도 패치/설정군 |

FP8 KV는 attention backend의 지원 조합, 실제 scale, 변환·복원 경로를 확인한다. 공개 문서는 backend와 결합되지 않은 복원 경로 및 scale 누락을 주의하도록 안내한다. 해당 경고를 성능·정확성 측정 항목으로 반영한다. [W3][W4]

KV 실험은 두 종류로 나눈다.

- **token budget 고정:** dtype만 바꾸고 최대 토큰 수를 고정한다.
- **HBM budget 고정:** dtype별 실제 KV bytes를 맞추고 토큰 수 증가를 허용한다.

두 결과를 하나의 ‘FP8 효과’로 합치지 않는다. dtype 변경과 동시에 hot expert 수를 늘리는 실험은 별도 조합 셀이다.

### 8.4 CUDA graph 목록

```text
G0: 해당 설치본의 기본 capture 목록
G1: [1, 2, 4, 8, 16, 24, 32, 48, 64]
G2: G1 + [80, 96, 128]
G3: G2 + [160, 192, 224, 256]
```

지원하지 않는 batch 값은 local parser 및 kernel 제약을 기록하고 제외한다. C64에서 G1과 G3를 비교해 사용하지 않는 capture buffer 비용을 기록한다. G3가 많은 graph를 담는다는 이유로 우선 적용하지 않는다.

**반드시 수행할 경계 실험:** 충분한 KV를 확보한 동일 구성에서 client C63/64/65/80을 실행하고 graph max64 대128을 비교한다. 다음을 각각 기록한다.

```text
실제 decode batch 크기 histogram
선택된 capture batch 크기 / padding token 수
graph replay / eager fallback 횟수
KV 부족에 따른 retraction·재prefill 횟수
CPU empty/cold route 작업 수
각 step의 wall time
```

기존 C80/C96 변화는 KV와 graph 경계가 함께 달라졌을 수 있으므로, 이번 실험에서는 한 원인으로 미리 확정하지 않는다. [S1, 제5.6절]

### 8.5 예산 경계의 세밀한 탐색

첫 단계에서 실행된 경계값 주변에만 token pool을 4,096 또는 8,192 단위로 추가한다. hot expert 예산·KV·graph 세 축의 전체 Cartesian product를 만들지 않는다. 대신 한 축씩 고정하고 인접값을 탐색한 뒤 P10에서 제한된 교차 조합을 수행한다.

OOM 재시도에서는 실패 설정을 덮어쓰지 않는다. 조정한 KV·mem-fraction·graph는 새 cell ID를 부여한다.

---

## 9. P5 — 엔진 설정 추가 최적화

이 절은 기존 제안 외에 수행할 **설정 중심 탐색**이다. 공개 문서의 옵션 이름은 후보를 식별하기 위한 것이며, 로컬 지원 여부와 실제 사용 경로를 P0에서 확인한다. [W1][W2][W3]

### 9.1 prefill·decode·요청 스케줄링

| 설정 | 후보값 | 비교 조건·추가 저장 데이터 |
|---|---|---|
| chunked prefill | 기존값, 512,1024,2048,4096,8192 | SHORT_COLD/PREFILL_HEAVY, actual chunk histogram |
| prefill batch 요청 상한 | 기존값,1,4,8,16 | 실제 prefill 요청 수·token 수 |
| max prefill tokens | 기존값 및 로컬에서 의미 있는 4096/8192/16384 | 적용값이 context/chunk로 무시되지 않는지 |
| mixed chunk | OFF/ON | prefill+decode 혼합 batch, CPU expert 행 수 |
| max running requests | 32,48,64,80,96,128,160,192,224 | client concurrency와 별도 축 |
| schedule conservativeness | 0.6,0.8,1.0,1.2,1.4 | retraction, queue, 실제 KV usage |
| schedule policy | fcfs/lpm 및 현재값 | 길이·prefix 분포 고정 |
| continuous decode steps | 1,2,4,8 | TTFT·ITL·scheduler 호출 수 |
| scheduler receive interval | 1,2,4 | 신규 요청 대기시간·poll 횟수 |
| overlap scheduler | ON/OFF | P2 교차 결과를 재사용하되 최종 후보에서 재측정 |

최대 running requests를 낮추면서 client 동시성도 같이 낮추지 않는다. 기본 C64에서는 서버 상한32/48/64를 비교하고, 확장 실험에서는 client C128/224에 대해 상한64/96/128/224를 비교한다. 실제 admit 시점과 queue 대기를 저장한다.

continuous decode steps와 receive interval은 따로 시험한 뒤 각각 상위 후보를 교차한다. 새 요청을 오래 기다리게 만들어 처리량만 높아진 경우도 원시 TTFT/queue 데이터로 그대로 보존한다.

`--enable-mixed-chunk`가 현재 kt wrapper에서 지원되지 않으면 자동으로 다른 모델 경로로 바꾸지 않는다. 최소 재현·지원 제약을 기록한다.

### 9.2 attention backend·page size

| 항목 | 후보 |
|---|---|
| attention | triton / fa3 / flashinfer 중 로컬 H100·Qwen3Moe 지원 경로 |
| 단계별 조합 | 공통 backend / prefill fa3 + decode flashinfer / 지원되는 역조합 |
| Triton KV splits | 1,2,4,8,16 |
| KV page size | backend가 허용하는 1,16,32,64 |
| FP8 KV 연계 | 해당 backend가 지원하는 dtype만 연결 |

Qwen3-Coder의 실제 attention 구조에 맞는 경로만 시험한다. DeepSeek 전용 MLA/DSA 설정이나 Blackwell 전용 kernel을 H100에 이름만 보고 적용하지 않는다. page size 변경 시 prefix hit token 수·내부 단편화·KV metadata를 함께 저장한다. [S1][W3]

backend 변경은 먼저 def0·BF16 KV·graph OFF의 최소 실행으로 shape·정확성·실제 dispatch를 확인한 다음 graph·FP8 KV와 결합한다. fallback했다면 요청한 backend와 실행된 backend를 모두 기록한다.

### 9.3 GPU hot expert 커널과 dense FP8 GEMM

다음 두 경로를 별도로 기록한다.

```text
GPU hot expert MoE runner
attention projection 등 dense/block-FP8 GEMM runner
```

기존 자동값, 명시적 Triton, 로컬에서 H100과 해당 quant shape를 지원하는 DeepGEMM 또는 추가 runner를 비교한다. kt wrapper가 선택값을 우회하면 해당 셀은 `NO_EFFECTIVE_PATH`로 기록하고, wrapper 연결 패치는 별도 실험으로 만든다.

`deep_gemm.disabled`를 원래 컨테이너에서 즉석으로 이름 변경하지 않는다. 복제 환경에서 버전·빌드·JIT 조건을 확인해 활성화하고, 커널 준비 시간을 별도로 저장한다. GPU에 있는 expert 일부만 계산하는 mask, invalid expert ID, padded row 처리까지 검사한다. [S1][W1]

### 9.4 TP collective·통신 설정

먼저 실제 collective 이름·메시지 크기·호출 수·소요시간을 수집한다. 이후 다음 순서로 실행한다.

1. 현재 custom all-reduce 대 `--disable-custom-all-reduce`를 사용한 NCCL 경로.
2. 설치본과 H100에서 지원되는 all-reduce+residual/RMSNorm fusion, 별도 셀.
3. 실제 NCCL 경로가 사용되는 경우에만 기본 algorithm 대 지원되는 Ring/Tree 등의 제한된 강제값.
4. NVLS 등 추가 경로는 설치본·토폴로지·소스의 지원을 확인한 경우에만 후보로 등록한다.

NCCL 변수는 현재 설치 버전 기준으로 검증하고 실험 종료 후 제거한다. 단일 노드 NVLink 통신을 IB/socket 통신이라고 가정해 네트워크 변수를 무작정 늘리지 않는다. 사용자 설정이 통신 backend에 도달하지 않으면 그것도 기록한다. [W5]

### 9.5 API·tokenizer·클라이언트

토크나이저/디토크나이저 worker1/2/4, 지원되는 batch encode, 클라이언트 connection reuse를 개별 시험한다. 벤치 프로세스는 서버 kt core와 겹치지 않는 예비 코어에 고정하고, GPU8 비교 등 코어 배치가 달라지는 셀에서는 동일 자원 제한을 명시한다.

주 성능 측정의 stream interval은1로 유지한다. interval2/4가 필요하면 `TRANSPORT_BUFFERING` 실험으로 분리한다. 응답을 묶어 전송해서 달라진 client ITL과 내부 token 생성시간을 구분한다.

디버그 로깅·요청 본문 로그·실시간 expert recorder의 ON/OFF에 따른 계측 부하도 한 번 비교한다. 정식 측정에는 같은 logging 수준을 사용한다. 커널 launch를 동기화하는 디버그 환경변수는 성능 경로에 남기지 않는다.

---

## 10. P6 — expert 배치·HBM 활용 확장

### 10.1 같은 예산에서 배치 방식 비교

먼저 총5,952슬롯을 고정하고 다음을 실행한다. 각 슬롯은 `(layer, logical_expert)`이며 TP shard 수를 슬롯 수에 중복 합산하지 않는다.

| 셀 계열 | 배치 |
|---|---|
| `PL00` | 기존 hotmap v1, uniform96 |
| `PL01` | 기존 hotmap v2, uniform96 |
| `PL02` | 기존 hotmap v2, 기존 층별 예산5,952 |
| `PL03` | 새 CALIBRATION 트레이스, 호출 빈도 기반 동일 예산 |
| `PL04` | 새 CALIBRATION 트레이스, 층별 cold 비율을 고려한 동일 예산 |
| `PL05` | 새 CALIBRATION 트레이스, 노출된 CPU 대기시간 감소량 기반 동일 예산 |

`PL05`는 새 알고리즘 제안이며 이미 효과가 측정된 방법이 아니다. 다음 식은 후보 정렬의 정의로 사용한다.

```text
priority(layer, expert)
  = 추정된 step critical-path 지연 감소량 / 추가 GPU 상주 bytes
```

호출 빈도가 높다는 이유만으로 지연 감소량이 크다고 대입하지 않는다. CPU 처리시간, GPU 완료 시점, CPU 결과를 기다린 시간, expert 공동 선택 패턴을 트레이스에서 계산한다. 전체 trace histogram만으로 전문가 한 개를 옮겼을 때의 효과를 정확히 알 수 없으면 근사 모델·가정을 명시하고 실제 셀로 확인한다.

### 10.2 단계별·입력 유형별 map

prefill과 decode의 route 빈도·CPU 시간을 따로 수집한다. 동일한 고정 map에서 시작하고 다음만 추가한다.

- prefill/decode 비용 가중치를 바꿔 만든 **정적 map**: 가중치0.25/0.5/0.75. 기존 `hotmap_mixed_0.25`와 동일 의미라고 가정하지 않는다.
- CODE_MIX 전용 calibration map과 여러 작업이 혼합된 calibration map.
- map을 바꾸지 않는 상태에서 HOLDOUT 입력 유형별 측정.

prefill마다 weight를 바꿔 올리는 동적 구현은 이 정적 map 실험과 분리한다. weight 이동량·변환·동기화 비용을 포함하지 않은 수치를 동적 배치 성능으로 기록하지 않는다.

### 10.3 전체 상주 예산과 KV의 교환

후보 슬롯 총량은5,456/5,704/5,952/6,076/6,200이다. FP8 GPU weight 기준이며, 실제 HBM 한계 내에서만 수행한다. uniform88/92/96/98/100 또는 같은 합계의 비균일 배치를 사용하고, 레이어별 expert 수는0~160 및 kernel 정렬 제약을 만족시킨다.

첫 비교는 KV pool을 고정한다. 두 번째 비교는 전체 HBM budget을 고정한다. 배치만 바뀐 효과와 KV를 줄여서 더 많은 weight를 넣은 효과를 구분한다. 과거 hot112의 OOM 기록을 단순히 재확인하기 위한 반복 실행은 하지 않는다. 다른 dtype·다른 실제 footprint가 성립했을 때만 새 셀로 등록한다. [S1]

### 10.4 GPU expert cache·prefetch·이동 — PATCH

정적 배치 실험이 끝나면 다음 최소 구현을 시험한다.

| 항목 | 후보값/규칙 |
|---|---|
| 동적 cache budget | GPU당0/512MiB/1GiB/2GiB, 같은 전체 HBM 예산에서 확보 |
| 교체 주기 | 64/256/1024 decode steps |
| 한 epoch의 이동량 | GPU당128MiB 이하로 시작, 실제 비용 기록 |
| 갱신 근거 | 최근 실제 route의 빈도·CPU 비용, 미래 정답 route 사용 금지 |
| prefetch 비교 | 없음 / 과거 정보 기반 / 진단용 oracle을 별도 표시 |

고정 expert 메모리를 줄여 cache 공간을 확보한 경우 제거된 슬롯 수까지 기록한다. expert tensor 포인터·shape·scale·mapping을 변경하는 작업은 graph 실행과 충돌하지 않는 명시적 안전 시점에만 수행한다. graph 재capture가 필요하면 그 시간과 요청 지연을 포함한다.

cache miss에서는 원래 선택된 expert를 CPU에서 계산한다. GPU에 없는 expert를 임의의 hot expert로 대체하지 않는다. prefetched weight가 사용되지 않은 경우의 전송량도 저장한다. pinned weight buffer와 CPU INT4→GPU FP8/W4 변환 경로의 출처를 명확히 한다.

### 10.5 hot expert 복제·routing-aware scheduling

GPU별 hot expert 복제는 기존 TP shard와 별개의 EP/dispatch 구현이 필요한지 확인한다. 지원되는 경우에만 복제0/소규모1안으로 먼저 비교한다. 같은 logical expert의 기여도를 중복 합산하거나 총 상주 bytes를 늘리고 동일 예산이라고 표기하지 않는다.

routing-aware scheduling은 **모델 router의 top-k 선택 변경이 아니라 요청·작업 배치 순서 변경**으로 구현한다. 과거의 expert 사용 패턴 또는 이미 계산된 route를 이용하며, 요청별 최대 추가 대기0/1/2/5ms를 두고 지연·공정성 데이터를 저장한다. 지연 예산은 실험 후보이지 사용자 서비스 SLA가 아니다.

---

## 11. P7 — NUMA·메모리 대역폭·CPU/GPU 런타임

### 11.1 현재 설정을 다시 확인할 항목

| 항목 | 실행 |
|---|---|
| turbo·frequency | no_turbo와 실제 Bzy_MHz 확인. 잠금 유지 시 ON 시도는 한 번만 기록하고 종료 |
| CPU power·thermal | 실제 package power, throttle 사유, 주파수 시계열 저장 |
| GPU power·clock | 기존 power limit·실제 SM/memory clock·throttle 이유 확인 |
| cgroup·memlock | quota, cpuset 상속, memory.limit, locked/pinned memory 제한 확인 |
| 컨테이너 shared memory | 기존 설정·사용량 기록, 필요한 증설은 해당 실험 컨테이너만 변경 |
| GPU topology | 기존0,1,2,3을 기본값으로 유지, 필요 시0,1,4,5 대조군1회 |

GPU clock 고정·power limit 변경은 원래 값과 복구 절차가 있고 해당 장치를 독점할 권한이 있을 때만 실행한다. 후보는 현재값과 장치가 보고하는 허용 clock의 최대 안정 실행값이다. 제조사 제한을 넘기거나 다른 작업을 종료하지 않는다. 높은 clock을 지정한 것과 실제 달성된 clock을 구분한다.

### 11.2 NUMA 실험

P2/P3 이후 선택된 동일 구성에서 기존 policy, interleave, 소켓별 explicit local allocation을 비교한다. `membind0`, pool1, single-socket은 기존 측정과 연결할 대조군으로 제한한다. [S2]

threadpool 수를 NUMA 수와 무관하게 늘리지 않는다. threadpool2의 master·worker별 실제 weight allocation과 접근 위치를 확인한다. first-touch, weight packing/repacking, mmap을 구분하고 CPU task를 다른 소켓에 이동시키는 경우 remote 접근을 함께 수집한다.

### 11.3 지속 가능한 DRAM 기준 측정

실험 서버의 추론 프로세스를 정상 종료한 상태에서 메모리 microbenchmark를 실행한다. Intel MLC 또는 검증된 streaming 도구를 사용하고 버전·명령·부작용을 기록한다. CPU prefetcher 등 전역 상태를 변경하는 도구 옵션은 승인 없이 사용하지 않는다. [W6][W7]

다음 조건을 분리한다.

```text
소켓0 단독 / 소켓1 단독 / 양 소켓
실제 kt와 동일한 코어 수·affinity
local / remote / interleaved
읽기 중심 / 읽기+쓰기 / 여러 expert weight 순환
LLC보다 충분히 큰 buffer / buffer bytes·page size
```

도구가 측정한 대역폭과 CPU expert 실제 실행 구간의 대역폭을 원시값으로 나란히 저장한다. 서로 다른 read/write 패턴의 값을 하나의 포화율로 합치지 않는다. 전체 inference 구간 평균과 CPU expert active 구간 평균을 따로 기록한다.

### 11.4 memory allocation·page 실험

기존 allocator, 실험 프로세스 범위의 first-touch/NUMA local allocation, 지원되는 `madvise` 기반 hugepage 요청을 비교한다. 실제 hugepage 할당량과 page fault를 확인한다. 전역 THP 설정 변경, swapoff, DIMM 교체·속도 변경, BIOS 변경, 시스템 재부팅은 별도 승인 없이 수행하지 않는다.

PyTorch GPU allocator 옵션은 fragmentation이나 allocation 실패가 관측된 경우에만 별도 셀로 비교한다. allocator 변경으로 총 메모리가 달라졌으면 그 값을 기록한다. OOM 메시지의 일반 제안을 자동으로 모든 셀에 적용하지 않는다.

---

## 12. P8 — GPU W4 경로와 TP/EP 구성

### 12.1 AWQ 실패 경로의 단계별 분리

IDE_070 AWQ 실행에는 처리량 측정에 도달하지 못한 기록이 있다. 부팅 실패를 W4 처리량 값으로 변환하지 않는다. [S2, 제6절]

아래 순서를 지킨다.

| 단계 | 조건 | 필수 산출 |
|---|---|---|
| `W00` | 기존 AWQ checkpoint·revision·quant config 확인 | tokenizer·weight shape·group size·scale·zero point manifest |
| `W01` | AWQ GPU-only TP4, 작은 KV, graph OFF, kt 경로 없음 | 최소 요청1/2/4/8/16의 부팅·정확성·커널 로그 |
| `W02` | W01에서 로컬 지원 dtype 및 AWQ runner 변경 | 요청값·실행값·fallback·오류 |
| `W03` | GPU-only 경로의 graph ON | capture batch·padding·안정성 |
| `W04` | 최소 kt 하이브리드 연결, hot96, graph OFF | logical/physical ID·masked expert·CPU/GPU combine |
| `W05` | 하이브리드 graph ON | 작은 batch부터 graph 호환 확인 |
| `W06` | hot96/112/128/144 및 비균일 예산 | 실제 HBM·CPU route 수·요청 성능 |
| `W07` | 가능하면 GPU-only all160과 하이브리드 비교 | 같은 W4 표현의 GPU4 비교군 |

AWQ dtype 후보는 checkpoint·backend가 지원하는 `half`/`bfloat16` 등이며, 실제 허용값을 먼저 확인한다. `awq`, `awq_marlin`, `moe_wna16` 등의 이름이 parser에 있다고 해서 현재 Qwen3Moe·kt mask 경로가 지원된다고 간주하지 않는다. shape·scale·invalid-ID 경로를 최소 실행으로 검증한다. [W1][W8]

GPU-only TP4에도 실제 메모리 용량이 부족하면 작은 KV·graph OFF로 한 번 재확인하고 그 사실을 기록한다. 다른 장수를 사용한 진단은 별도의 `GPU_COUNT_CHANGED` 셀이다.

illegal memory access는 최소 재현 입력·graph OFF 상태에서 실제 첫 CUDA 오류를 수집하고 필요한 경우 Compute Sanitizer를 사용한다. 오류가 activation/combine에서 보고되었다는 이유만으로 그 kernel이 최초 원인이라고 확정하지 않는다. [W9]

### 12.2 양자화 출처

기존 CPU INT4는 FP8 스냅샷에서 변환한 것이고, AWQ GPU checkpoint는 다른 변환 경로다. 이 조합은 `MIXED_WEIGHT_PROVENANCE`로 표시한다. 가능한 경우 동일 원본에서 파생된 CPU/GPU 가중치 조합을 별도 제작·검증한다. 없는 변환 기능을 있다고 가정하지 않는다.

새 양자화 artifact를 만들면 원본 revision, 변환 도구·SHA, 옵션, calibration 데이터, 출력 shard checksum을 보존한다. 품질 측정 전후 checkpoint를 바꾸지 않는다. 모델 전체를 Git 일반 파일로 업로드하지 않는다.

### 12.3 TP/EP와 인스턴스 구성

다음은 조건부 구성 실험이다. 기존 TP4와 동일한 설정 변경 효과로 표기하지 않는다.

| 구성 | 실행 조건 |
|---|---|
| TP4, 기존 EP 경로 | 기본 비교군 |
| TP4+EP4 하이브리드 | kt wrapper의 CPU/GPU expert ownership·A2A·combine 지원 확인 |
| TP4에서 supported DP attention | Qwen3Moe와 kt 경로가 모두 지원되는 경우, KV 복제·통신 변화 기록 |
| TP2 하이브리드 단일 | weight·KV 실제 용량이 성립하는 경우만 |
| GPU4 = TP2 인스턴스2개 | 총 CPU cores·소켓·RAM budget을 TP4 비교군과 함께 기록 |
| GPU8 = TP4 인스턴스2개 | 기존 소켓별 구성, 가용성 확인 후 제한된 최종 참조 |
| GPU-only TP8+EP8 | R07과 동일 프로토콜 재측정 |

`--ep-dispatch-algorithm static`은 기존 a2a-none 구성에서 거부된 기록이 있으므로 같은 실패를 반복하지 않는다. A2A·EP 구성 자체를 새로 검증한 경우에만 별도 후보로 등록한다. [S1]

8GPU dual과 GPU-only는 **한 클라이언트·한 측정 창**에서 전체 요청을 보내 측정한다. 별도 벤치의 tok/s 두 값을 단순 합한 값은 `sum_of_independent_rates`로만 별도 보존하며 실제 단일 엔드포인트 처리량으로 대신하지 않는다.

---

## 13. P9 — 조건부 구조 개선과 speculative decoding

설정 탐색을 완료한 뒤 아래 항목도 수행한다. 구현이 없으면 작은 패치 단위로 진행하고, 최소 재현·지원 조건을 충족하지 못하면 정해진 종료 상태를 남긴다.

### 13.1 CPU–GPU overlap과 전송

진단 트레이스에서 다음 구간을 계측한다.

```text
GPU routing 종료
D2H 시작/종료
CPU queue enqueue/dequeue
CPU expert 계산 시작/종료
H2D 시작/종료
GPU hot expert 종료
combine 시작/종료
다음 레이어 실제 시작
```

서로 다른 CPU/GPU clock domain의 timestamp를 무보정으로 빼지 않는다. Nsight 등 동기화된 timeline 또는 명시적인 clock alignment를 사용한다.

그 다음 기본 경로와 아래 변경을 각각 비교한다.

- 선택된 CPU 작업만 모은 compact activation transfer.
- 같은 입력 행이 여러 cold expert로 향하는 경우 입력 중복 복사 제거.
- pinned buffer 사전 할당 및 재사용, NUMA-local staging buffer.
- input/output double buffer 또는 ring depth2/4/8.
- GPU hot expert 실행과 CPU expert 실행의 겹침, 불필요한 전역 synchronize 제거.
- 독립 요청 microbatch 사이의 파이프라인; microbatch16/32/64 중 실제 전체 batch에 맞는 값.

ring depth를 늘릴 때 generation 및 in-flight 참조 수를 검증한다. 아직 사용 중인 buffer 재사용, 늦은 CPU 결과의 다른 요청 결합, 다음 레이어 의존성 위반을 허용하지 않는다.

### 13.2 CPU expert batching·fusion

이미 같은 expert에 모이는 작업에 대해 재정렬·메모리 layout 변경·quantization fusion·grouped 실행을 비교한다. 가능한 경우 다음을 단계별 기록한다.

```text
pack/unpack 횟수·bytes
activation quantization 횟수
expert별 M histogram / 고유 expert 수
matmul 호출 수 / effective rows / padded rows
동일 weight 재사용 횟수 / DRAM bytes
scatter/combine 호출 수·시간
```

CPU 총 route 수를 줄여 보이게 하려고 모델의 top-k를 줄이지 않는다. 근사적인 route 변경을 별도 탐색할 때는 APPROXIMATE 계열과 완전한 정확성 측정을 요구한다.

### 13.3 speculative decoding 재시험

IDE_069의 EAGLE3·STANDALONE 결과를 그대로 반복하는 것이 아니라, 최종 CPU/GPU 실행 경로에서 verification 비용을 다시 측정한다. [S1]

EAGLE3 대상 draft의 모델 revision·tokenizer 호환성을 확인한다. 후보는 `num_steps=1/2/3`, `topk=1`, draft token 수2/3/4 중 **설치본이 허용하는 일관된 조합**이다. C1/4/16을 먼저 수행하고 C64 대조를 추가한다. 모든 조합의 무제한 교차는 하지 않는다.

필수 저장값은 draft latency, target verify latency, proposed/accepted tokens, accepted length histogram, CPU expert M·route 수, graph 경로, KV 점유, E2E latency다.

acceptance threshold를 임의로 완화하거나 검증을 생략해 성능을 높이지 않는다. 다른 acceptance 정책을 시험하려면 APPROXIMATE 계열의 별도 셀로 명시한다. STANDALONE4B는 최소 대조군1개로 제한하고, 새 코드·새 조건이 없으면 과거와 같은 광범위 sweep을 반복하지 않는다.

모델·kt·graph 조합이 지원되지 않으면 그 증거를 남기고 다음 단계로 진행한다. 지원되는 n-gram 방식은 로컬 호환성 확인 후 별도 소규모 대조군으로 등록할 수 있다.

---

## 14. P10 — 조합 탐색과 수렴 규칙

### 14.1 조합을 만드는 원칙

서로 다른 precision/의미 계열의 결과를 한 후보 집합으로 합치지 않는다. `SEMANTICS_PRESERVING`, `QUANTIZATION_CHANGED`, `APPROXIMATE`를 각각 탐색한다.

먼저 다음 묶음을 순서대로 결합한다.

```text
P2의 검증된 핸드오프·pinning 구성
 + P3의 CPU 커널·thread 구성
 + P4의 KV·graph 구성
 + P5의 scheduler·attention·GPU kernel 구성
 + P6의 expert 배치
 + P7의 유효 메모리/NUMA 설정
```

각 단계에서는 기존 anchor를 유지한 상태에서 하나의 묶음을 추가하고, 묶음 안의 주요 요인을 다시 한 번 제거한 ablation을 실행한다. P8의 W4·TP/EP 변경과 P9의 speculation은 다른 구조 계열에서 독립적으로 조합한다.

### 14.2 기계적인 후보 선택

실험을 계속 진행하기 위한 수치 기반 선택은 허용한다. 이는 최종 문서의 성능 평가·채택 판단과 다르다. 선택 규칙을 manifest에 먼저 저장하고 중간에 수치를 보고 임의 변경하지 않는다.

- 요청 수가 모두 완료되고 설정 적용 증거가 있는 셀만 동일 조건의 후보 집합에 넣는다.
- primary search는 SHORT_COLD C64의 처리량과 TTFT p95·TPOT p95를 함께 사용한다.
- 각 계열에서 처리량 순2개와 지연시간 측정상 비지배 후보 최대2개를 보존한다. 중복은 제거한다.
- 지연시간 제약 데이터는 TTFT p95 2/4/8초, TPOT p95 50/100/200ms의 조합별로 별도 기록한다. 이 값은 **실험용 관측 기준**이며 사용자 SLA로 선언하지 않는다.
- 정확성은 정해진 기계적 검증값을 함께 저장한다. shape 위반·NaN·잘못된 token ID·중복/누락 작업 등 구현 오류가 확인된 셀은 일반 성능 후보에서 제외하되 원시 데이터를 삭제하지 않는다.
- 근사·양자화 품질에 대한 허용 오차를 임의로 정해 ‘품질 동일’로 판정하지 않는다. 해당 계열을 분리해서 최종 측정을 계속한다.

선택과 제외의 이유는 `selection_trace.jsonl`에 입력 지표·규칙 ID·셀 ID로 남긴다. 사람의 평가 문장을 추가하지 않는다.

### 14.3 유한 탐색과 인접값

사전 정의된 지원 가능 후보를 1차 스크리닝하고, 최대4회의 refinement round를 수행한다. 한 round는 각 활성 축의 인접값과 현재 후보 조합을 모두 처리한 단위다.

예시 인접 탐색:

```text
cpuinfer: 기존 후보의 ±4 또는 ±8, 실제 코어 예산 이내
KV tokens: ±4096/8192, 실제 HBM 이내
graph: 관측 batch histogram에 나타난 경계 batch 추가/제거
chunked prefill: 선택값의 1/2 또는 2배, 등록 범위 이내
layer budget: 총 슬롯 ±62/124, 실제 HBM 이내
AMX row threshold: 관측 M 분포의 인접 경계
```

실험 셀은 정규화한 effective config hash로 dedup한다. 단, 다른 seed·cache policy·build·dataset·측정 창의 반복은 제거하지 않는다. 이미 실패한 조합을 다른 이름으로 계속 재시도하지 않는다.

### 14.4 수렴 판정의 계산 규칙

아래 기준은 **이번 계획에서 정하는 탐색 제어 규칙**이지, 이전 보고서에서 검증된 통계 결론이 아니다.

1. 비교 대상은 동일 workload·C·cache·precision·build 계열의 짝지어진 반복이다.
2. 최종 후보와 anchor는 최소5회 측정한다. run 단위 throughput 차이와 bootstrap 95% 구간을 고정 seed로 계산하고 원시 반복값을 남긴다.
3. 처리량 변화가2% 이상인 새 후보가 나타나면 확인 측정을 추가한다. 2%는 사전 설정한 refinement trigger다.
4. 새 후보가 기존 처리량 후보를 넘거나 사전 정의한 지연시간 비지배 집합을 바꾸면 다음 round를 수행한다. 오차가 커서 구분되지 않으면 최대9회까지 반복한다.
5. 지원되는 모든 활성 축의 인접값을 시험한 **연속2개 round에서**, 기존 후보 대비 가능한 처리량 증가의 95% 구간 상한이2% 이하이고 지연시간 관측 집합도 변하지 않으면 `CONVERGED_WITHIN_TESTED_DOMAIN`으로 기록한다.
6. 반복을9회까지 늘려도 불확실하면 `CONVERGENCE_UNRESOLVED_NOISE`로 기록한다. 전체 후보를 소진했으면 `SEARCH_DOMAIN_EXHAUSTED`, 4회 refinement만 소진했으면 `REFINEMENT_LIMIT_REACHED`로 구분한다.
7. 마지막 세 상태를 ‘수렴 완료’로 바꿔 쓰지 않는다. 어떤 상태이든 P11의 필수 측정과 산출물 생성·업로드·다운로드 제공은 끝까지 수행한다.

전역 최적을 찾았다고 주장하지 않는다. 새 후보 공간을 무한히 추가하지 않으며, 계획에 없던 확장은 별도 amendment ID·이유·유한한 범위를 기록한 후 진행한다.

---

## 15. P11 — 최종 측정·정확성 데이터·장시간 실행

### 15.1 최종 측정 대상

원래 R0와 각 계열의 최종 후보 최대4개, 실행 가능한 GPU-only 참조군을 포함한다. 후보가 하나도 실행되지 않은 계열도 해당 실패·차단 기록을 최종 문서에서 누락하지 않는다.

| 측정 | 실행 규격 |
|---|---|
| 비교 반복 | SHORT_COLD C64 최소5회, 필요 시9회 |
| 동시성 sweep | C1/4/8/16/32/48/64/80/96/128/160/192/224, 실제 자원 범위 |
| graph 경계 | C63/64/65/80, graph max와 KV를 분리 |
| 길이 sweep | SHORT_COLD, DECODE_HEAVY, PREFILL_HEAVY, LONG_CONTEXT |
| prefix 조건 | cold와 prefix-only warm을 분리 |
| steady-state | 후보별300초 부하, startup/drain 별도 |
| finite-rate | 512/128 기준2/4/6/8 requests/s, 각300초, 클라이언트 제출·대기 구분 |
| 장시간 | 각 계열 대표1개와 R0, MIXED_SERVICE 60분 |
| 최종 재현 | 새 서버 프로세스로 시작한 대표 셀의5회 반복 |

서버가 처리할 수 있는 양보다 높은 finite-rate에서도 계획된 종료까지 수집하되, 명시된 요청/queue 제한과 drain timeout을 사용한다. `request-rate inf`와 finite-rate 결과를 섞지 않는다. rate2/4/6/8은 실험 입력이지 추정된 서비스 용량이 아니다.

### 15.2 정확성·출력 데이터

‘평가하지 말라’는 지시를 ‘정확성 데이터를 수집하지 말라’로 해석하지 않는다. 수치·출력·테스트 실행 결과는 수집하되 최종 의견은 작성하지 않는다.

- 같은 입력·seed에 대한 token sequence, 종료 사유, 생성 길이, NaN/Inf·비정상 ID 여부.
- 구현 검증용 intermediate tensor의 absolute/relative error, 필요 시 logits 차이. 기록 가능한 범위·샘플링 기준을 고정한다.
- 기존 greedy4문항과 GSM40을 호환용으로 다시 실행한다.
- 설치·접근 가능한 고정 revision의 전체 GSM 데이터 및 코드 생성 테스트 집합을 추가한다. 전체 요청 수·subset 추출 규칙·채점기 버전을 기록한다.
- 코드 모델에 맞는 코드 작업은 프로젝트의 기존 하네스를 우선 사용하고, 없는 경우 데이터셋과 채점 방식을 먼저 고정한다. 이름만 적고 실행한 것으로 처리하지 않는다.
- 긴 입력의 고정된 정보 회수·정확한 출력 과제 데이터를 수집한다. 임의 사례를 일반적인 장문 성능 지표로 표현하지 않는다.

생성된 코드를 실행하는 테스트는 네트워크 차단·비특권·읽기 전용 기본 파일시스템·CPU/RAM/시간 제한을 둔 별도 sandbox에서 수행한다. 모델 출력을 호스트 셸에서 직접 실행하지 않는다.

전용 정답/로그는 라이선스와 개인정보 정책을 확인해 저장한다. 공개가 제한되는 데이터는 원본 로컬 보존, ID·hash·기계적 결과만 게시하는 방식으로 분리한다.

### 15.3 장시간 기록 항목

분 단위 요청수·완료수·실패수·token/s·latency histogram, GPU HBM allocated/reserved/used, CPU RSS, queue, ring occupancy, retraction, CPU/GPU 오류, frequency·power·thermal을 남긴다.

시작과 끝의 메모리 차이만으로 누수를 판정하지 않는다. 전체 시계열과 server restart 횟수, 정리되지 않은 request/task 수를 저장한다. 완료까지 실행했는지, 안전 문제로 해당 셀만 중단했는지 구분한다.

---

## 16. 중단 방지·오류 복구·완료 상태

### 16.1 실행기 상태 머신

```text
PENDING
  → PREFLIGHT
  → BOOTING
  → VERIFY_EFFECTIVE_CONFIG
  → WARMUP
  → CACHE_PREPARE
  → MEASURING
  → DRAINING
  → PERSISTING
  → COMPLETED

오류 발생
  → CAPTURE_FAILURE
  → CLEANUP_OWN_PROCESSES
  → RETRY_PENDING 또는 종료 상태
  → 다음 독립 셀
```

외부 loop에서 단일 명령 실패가 전체 프로그램 종료로 전파되지 않게 한다. 셀 runner 내부의 실패는 명시적으로 반환하고 최상위 runner가 처리한다. `set -e`를 사용한다면 셀 단위 예외 경계를 반드시 둔다.

### 16.2 종료 상태

| 상태 | 의미 |
|---|---|
| `COMPLETED` | 등록된 반복·요청·원본 저장을 완료 |
| `FAILED_BOOT` | 부팅 실패, 로그·exit 정보 보존 |
| `FAILED_RUNTIME` | 측정 중 종료·크래시 |
| `FAILED_REQUESTS` | 일부 또는 전체 요청 실패, 서버 상태 별도 |
| `TIMEOUT` | 사전 설정한 실행/무진행 한계 도달 |
| `INVALID_CONFIG` | 실제 설정이 계획과 다름, 패치 미적용 포함 |
| `INVALID_MEASUREMENT` | profiler 창·클라이언트·단위 등 계측 요건 미충족 |
| `UNSUPPORTED` | parser·모델·kernel·GPU 조합 미지원 |
| `NO_EFFECTIVE_PATH` | 설정은 수용됐지만 해당 실행 경로에서 사용되지 않음 |
| `BLOCKED_PERMISSION` | 권한·BIOS·게시 인증 등 제한 |
| `BLOCKED_RESOURCE` | 필요한 GPU·메모리·저장공간·자료를 사용할 수 없음 |
| `BLOCKED_DEPENDENCY` | 선행 구현·변환·호환성 조건 미충족 |
| `BLOCKED_SAFETY` | GPU/시스템 위험 상태로 해당 실행 불가 |

`UNSUPPORTED` 등은 완료한 측정 횟수에 합산하지 않는다. 분모에는 전체 등록 항목을, 상태별 집계에는 각 종료 상태를 표시한다.

### 16.3 재시도와 수정 범위

같은 effective config의 재시도는 최초 실행 외 최대2회다. 원인 확인용 최소 재현·패치는 해당 문제당 최대2개 수정 revision을 먼저 등록해 실행한다. 각 revision은 이전과 다른 cell/build ID를 갖는다.

단순히 수치가 기대보다 낮다는 이유로 다시 실행해 좋은 반복만 선택하지 않는다. 모든 반복을 보존한다. 같은 실패를 무한 재시도하지 않으며, 더 큰 구현 변경이 필요한 경우 `BLOCKED_DEPENDENCY`와 최소 재현 자료를 남기고 다른 시리즈를 완료한다.

실험 도중 새 컨테이너가 필요하면 기존 검증된 image digest를 사용한다. 벤치 클라이언트가 없는데 서버 profiler만 실행하는 상황은 `PREFLIGHT`에서 차단한다.

### 16.4 timeout과 장시간 명령

기본 server boot timeout은1,200초로 두고, 최초 JIT가 포함된 경우 manifest에1,800초를 명시할 수 있다. 모델 변환·다운로드·장시간 테스트는 별도 task timeout을 둔다. 로그의 실제 진척이 없는 상태를 무한히 기다리지 않는다.

일반 벤치는 같은 workload의 기준 실행시간을 바탕으로 `max(900초, 5×기준시간+300초)`를 초기 task timeout으로 정한다. 기준시간이 없으면1,800초를 사용한다. 60분 장시간 셀은 계획한 부하 시간과 별도 drain 여유를 적용한다. 이 값은 실행기의 장애 감지를 위한 한계이며 작업 소요시간 예측이 아니다.

새 token이 없다는 사실만으로 prefill 중인 서버를 종료하지 않는다. queue·CPU/GPU 진행·현재 phase·서버 heartbeat를 함께 확인한다. 무진행 timeout의 실제 값과 트리거 시각을 기록한다.

### 16.5 정리·재개

- 해당 cell이 생성한 PID/process group/container만 종료한다. `pkill python`, 전체 컨테이너 삭제, GPU 전체 reset을 사용하지 않는다.
- illegal memory access 이후 손상된 CUDA context로 다음 측정을 이어가지 않는다. 새 서버 프로세스에서 device health와 최소 요청을 확인한다.
- Xid·ECC·열·전원 등 위험 상태는 해당 GPU 실행을 차단하고 남은 안전한 작업과 데이터 보존을 진행한다. 안전상 필요한 중지는 감추지 않는다.
- 각 상태 전이를 atomic write한 `state.json`과 append-only journal에 남긴다. 재개 시 완료된 반복을 덮어쓰지 않는다.
- 원격 세션 단절에도 실행이 유지되도록 승인된 지속 실행 환경을 사용하고, watchdog·보고 프로세스·runner의 생존을 확인한다. 단순히 `&`를 붙인 것으로 지속 실행을 확인했다고 기록하지 않는다.
- 전역 설정을 변경한 셀은 종료·오류 경로 모두에서 원래 값으로 복구하고 read-back 값을 기록한다.

---

## 17. 30분 단위 중간 보고

### 17.1 보고 시각과 실행 방식

실험 시작 monotonic 시각을 기준으로1,800초 간격으로 보고한다. Asia/Seoul 및 UTC 타임스탬프를 함께 기록한다. 셀·컴파일·변환 작업이 오래 걸려도 보고 루프는 별도로 실행한다.

보고 루프는60초 이하 주기로 상태 파일을 읽고 다음1,800초 경계 도달 시 보고를 작성한다. 보고 때문에 현재 벤치나 profiler를 중단하지 않는다. 외부 알림 도구의 스케줄 제약에 의존하지 말고 실험 실행 환경 내부의 reporting loop를 사용한다.

한 번의 보고 실패로 전체 실험을 중지하지 않는다. 파일 보고를 먼저 저장하고, 승인된 전달 채널의 전송 성공/실패를 별도 기록한다. 사용자에게 실시간 전달 가능한 채널이 없으면 이를 `REPORT_DELIVERY_UNAVAILABLE`로 명시하며, 파일 저장만으로 사용자 전달을 했다고 주장하지 않는다.

### 17.2 저장 위치

```text
shadow_assists/features/IDE_071/PROGRESS.md
shadow_assists/features/IDE_071/progress/YYYYMMDDTHHMMSS+0900.md
shadow_assists/features/IDE_071/progress/delivery_log.jsonl
```

실제 실험 번호가 달라지면 위 경로도 일괄 변경한다. 과거 IDE_070의 로그에 이번 결과를 append하지 않는다.

### 17.3 보고 내용

보고에는 진행 사실과 raw 수치를 넣는다. ‘큰 성과’, ‘병목 확정’, ‘현재 최적’ 같은 판단을 넣지 않는다.

```markdown
# 중간 실행 보고 — <timestamp Asia/Seoul>

- campaign_id: <id>
- elapsed_seconds: <n>
- current_phase / cell_id / rep_id: <...>
- phase_state: <BOOTING|MEASURING|...>
- registered / completed / failed / blocked / pending: <counts>
- last_completed_cells: <cell IDs>
- latest_measurements: <cell ID, rep ID, workload, C, actual tok/s, TTFT/TPOT>
- request_completed / request_expected: <counts>
- latest_error: <시각, exit code, 오류 원문, log path 또는 없음>
- retries_and_recovery: <실행 사실>
- HBM / RAM / disk_available: <측정값·단위>
- latest_persisted_artifacts: <paths>
- next_registered_cells: <IDs>
- report_delivery_state: <saved|delivered|failed>
```

실험 종료가30분 경계와 무관하더라도 마지막 종료 보고를 한 번 더 남긴다. 최종 전달은 GitHub 게시 및 다운로드 검증이 끝난 후 수행한다.

---

## 18. 실험 데이터 저장 규격 — 최종 문서는 측정 사실만 기록

### 18.1 필수 디렉터리 구조

```text
cpu_offload_no_02_실험계획_지시서.md
shadow_assists/features/IDE_071/
  INPUT_SOURCES.md
  OPTION_REGISTRY.md
  PLAN_RESOLVED.md
  PROGRESS.md
  RESULT.md
  FULL_REPORT.md
  COMPLETION_STATUS.md
  PUBLISH_RECEIPT.md
  progress/
  reports/
    P01_baselines.md
    P02_handoff.md
    P03_cpu_kernel.md
    P04_memory_graph.md
    P05_engine_settings.md
    P06_placement.md
    P07_numa_runtime.md
    P08_w4_parallelism.md
    P09_structural_speculative.md
    P10_search_trace.md
    P11_validation.md
    request_tables/
    layer_tables/
  evidence/
    software_manifest.json
    source_changes.patch
    option_registry.json
    source_locations.md
  manifests/
    campaign.json
    cells.jsonl
    workload_manifest.json
    artifact_manifest.jsonl
    SHA256SUMS
  state/
    state.json
    journal.jsonl
    selection_trace.jsonl

eval/ide071/
  run_all.py
  run_cell.py
  report_progress.py
  verify_environment.py
  verify_effective_config.py
  build_manifest.py
  build_hotmap.py
  collect_metrics.py
  render_data_report.py
  validate_artifacts.py
  publish_results.py
  configs/
  tests/

eval/results/<campaign_id>/<cell_id>/<attempt_id>/<rep_id>/
  launch_cmd.sh
  bench_cmd.sh
  effective_config.json
  environment.redacted.txt
  status.json
  timestamps.json
  bench.stdout.log
  bench.stderr.log
  server.full.log.gz
  benchmark.raw.json
  requests.jsonl
  stream_events.jsonl
  metrics.json
  cpu_timeseries.csv
  gpu_timeseries.csv
  memory_timeseries.csv
  thread_affinity.json
  layer_step_metrics.jsonl
  expert_metrics.jsonl
  errors.jsonl
  diagnostic/
```

원래 프로젝트 구조를 유지해야 하면 대응 경로를 `PLAN_RESOLVED.md`에 기록한다. 파일명만 만들고 내용이 비어 있는 상태를 완료로 간주하지 않는다. 셀에 적용되지 않는 파일은 `not_applicable` 사유를 manifest에 남긴다.

### 18.2 메인 Markdown의 필수 내용

`FULL_REPORT.md`는 다음 순서로 구성한다.

1. 실험 식별자, 시작·종료 시각, 입력 자료·코드·환경의 해시.
2. 하드웨어·소프트웨어·컨테이너·모델·양자화·hotmap의 실제 구성.
3. 등록된 모든 셀의 상태 표. 미실행·실패·지원 불가도 포함.
4. **셀별 실제 전체 실행 명령과 환경변수**, 원래 기준 대비 설정 차이.
5. **모든 반복별** throughput, 요청수, 실제 input/output tokens, latency, cache, 메모리, CPU/GPU 값.
6. 반복 통계와 정확한 계산 규칙. 원시 반복을 대체하지 말 것.
7. 워크로드·precision·동시성·cache policy별 구분 표.
8. 레이어별·expert별·step별 카운터와 시계열 세부표/상세 MD 링크.
9. 정확성 측정 원문·문항 ID·기계적 일치 수·오류 크기·제약.
10. 크래시·OOM·timeout·지원되지 않는 설정·재시도·패치 revision의 시간순 기록.
11. 수렴 제어 로그, 종료 상태, 탐색하지 못한 등록 항목과 실제 제약.
12. 원본 파일 인벤토리, 크기, SHA256, 저장 위치, 게시 여부.

**포함 금지:** executive summary 형태의 평가, 실험 결과의 의미 해석, 원인 단정, 추천 구성, best practice 도출, 투자/도입 판단, 후속 제안, 임의의 예상 성능.

`RESULT.md`도 데이터 표·실행 상태·원본 색인만 담는다. “측정값만 기록”이라고 머리에 쓰고 본문에 해석을 추가하지 않는다. 표의 기본 정렬은 phase/cell/rep ID 순서로 하며, ‘최고’, ‘승자’, ‘실패한 아이디어’ 등의 순위를 붙이지 않는다.

### 18.3 셀·반복별 필수 필드

```text
campaign_id, phase_id, cell_id, attempt_id, rep_id
parent_cell_id, config_hash, code_sha, binary_sha, container_digest
workload_id, dataset_revision, prompt_manifest_hash, seed, cache_policy
precision_family, semantics_family, GPU_count, TP, EP
requested/effective cpuinfer, threadpool, affinity
requested/effective hotmap, per_layer_budget, KV dtype/tokens, graph settings
requested/effective backend, actual kernel names, fallback counters
start/end timestamps, warmup_seconds, boot_seconds, measurement_seconds
requests_submitted/completed/succeeded/failed/timed_out/cancelled
input_tokens_total, input_tokens_computed, prefix_hit_tokens
output_tokens_total, output_tokens_successful_requests, proposed/accepted_spec_tokens
output_tps_e2e, total_tps_e2e, phase-specific throughput
TTFT/TPOT/ITL/E2E latency mean/p50/p90/p95/p99/max
server_queue_delay, client_queue_delay, service_latency
request_rate_requested/actual, client_concurrency, server_running_requests
KV usage/retractions/re-prefill/OOM, graph_replay/eager/padding
CPU busy physical/SMT, task_clock, cycles, instructions, migrations, context_switches
DRAM local/remote counts, read/write bytes or rates, UPI counts/rates
GPU SM/memory utilization, HBM allocated/reserved/used/peak, power/clock
CPU/DRAM/GPU energy와 계측 가능 범위
status, exit_code, signal, first_error_timestamp, evidence_paths
```

필드가 측정되지 않았으면 null과 reason을 남긴다. 누락값을0으로 채우지 않는다. 과거 하네스가 출력한0은 `raw_reported_value`로 보존할 수 있지만 정규화된 값에는 별도 유효성 상태를 붙인다.

### 18.4 요청별 데이터

요청 ID, 입력 해시·길이, 제출·admit·첫 응답·마지막 응답 시각, actual output length, TTFT, TPOT, stream event 시각, 종료 사유, 오류를 남긴다. 요청 원문·생성문 저장은 데이터 공개 범위에 맞춘다.

요청별 상세 Markdown이 너무 크면2,000행 단위로 분할하고 `FULL_REPORT.md`에서 **전체 shard를 빠짐없이 연결**한다. JSONL/CSV 원본을 함께 보존하며, 대량 token event는 압축 원본과 schema·예시·checksum을 연결한다. 파일 크기를 이유로 raw data를 삭제하거나 평균만 남기지 않는다.

### 18.5 레이어·expert·step 데이터

각 레이어에 대해 총 입력 행 수, 실제 route assignment 수, GPU/CPU assignment 수, cold 비율, CPU unique expert 수, expert별 M histogram, AVX/AMX 횟수, queue·compute·copy·wait·combine 분포를 저장한다.

레이어 수62·top-k8을 사용하는 경우 단위는 다음처럼 명시한다.

```text
cold_routes_per_token_layer
cold_routes_per_token_all_layers
cold_fraction_of_route_assignments
fraction_of_token_layers_with_any_cold
fraction_of_steps_with_any_cold
```

TP rank가 같은 라우팅을 중복 기록한다면 논리 토큰 단위의 dedup 규칙과 rank별 원본을 모두 보존한다. padding token과 실제 token을 분리한다. calibration trace와 held-out trace는 별도 이름을 사용한다.

### 18.6 계산 규칙

```text
output_tps_e2e = 실제 수신 출력 token 수 / 해당 benchmark의 벽시계 측정시간

TPOT(request) = (마지막 token 시각 - 첫 token 시각) / (출력 token 수 - 1)
  단, 출력 token 수≤1 또는 token별 시각을 확정할 수 없으면 유효성·한계를 기록.

context_switches_per_wall_second = raw context switch count / 실제 계측 창 초

phase_cycles_per_output_token = 해당 phase cycles / 해당 phase의 실제 출력 token 수
  prefill이 섞인 cycles를 pure decode cycles로 표기하지 않음.

새 반복 통계의 표준편차 = sample standard deviation, ddof=1
```

client stream event가 여러 token을 묶어 전달한 경우 event 간격을 token별 ITL로 꾸며 넣지 않는다. 서버 내부 생성 timestamp가 없으면 해당 한계를 표기한다.

반복별 p95의 평균은 `mean_of_rep_p95`, 전체 요청을 합친 p95는 `pooled_request_p95`로 구분한다. failed run을 정상 run과 섞어 평균하지 않는다. 오류 상태에서 생성된 token과 duration은 별도 raw 필드에 보존한다.

성능 측정 창에는 첫 요청 제출부터 마지막 요청 종료 및 정의된 drain을 포함하는지 정확히 기록한다. 성공 요청만 골라 duration을 짧게 잘라내지 않는다.

### 18.7 원본 로그·큰 파일

server log 전체를 저장한다. 최종400행만 보관하지 않는다. perf.data, Nsight trace, recorder tensor, core dump는 승인된 저장 위치에 보존하고 checksum·파일 크기·수집 명령을 MD에 남긴다.

원본이 크면 압축·shard한다. Git의 용량 제한을 이유로 원본을 지우지 않는다. 저장공간이 부족하면 수집 중인 대용량 진단을 종료 상태로 보존하고 나머지 필수 성능 데이터를 우선 저장하며, 디스크를 가득 채우지 않는다.

### 18.8 결과 문서 검증

자동 검증기는 다음을 확인한다.

```text
등록된 셀 ID 집합 = 결과 상태 표의 셀 ID 집합
완료 반복 수와 raw benchmark 파일 수의 일치
요청 expected/actual/failed 합계와 raw JSONL 행 수 일치
모든 성능값의 raw data 경로와 계산식 존재
config hash·build hash·hotmap hash 누락 없음
단위·측정 창·precision·cache policy 누락 없음
Markdown 링크 대상 존재 / 분할 파일 누락 없음
모든 원본의 파일 크기·SHA256 검증
FAILED/UNSUPPORTED의 성능값을 정상 통계에 넣지 않았는지 확인
동일 이름으로 다른 설정을 덮어쓰지 않았는지 확인
예상값·가공된 결과·해석 문장이 최종 보고서에 추가되지 않았는지 확인
```

파일이 존재한다는 사실만으로 검증을 통과시키지 않는다. JSON 파싱, gzip 무결성, CSV/JSONL 행 수와 MD 표 행 수의 일관성도 확인한다.

---

## 19. GitHub 게시와 MD 다운로드 제공

### 19.1 게시 대상 확인

기존 프로젝트의 `origin`, 현재 브랜치, 접근 권한, 공개 범위를 확인한다. 저장소 URL·이름을 추측하거나 새 공개 저장소를 만들지 않는다. 실행자에게 GitHub 연동 도구가 제공되어 있으면 그 도구로 실제 읽기/권한 확인 후 쓰기 동작을 수행한다. 로컬 Git을 사용하면 기존 승인된 인증만 사용한다.

기존 브랜치의 사용자 변경을 덮어쓰지 않도록 별도 실험 브랜치를 사용한다. 예시 이름은 `feat/cpu-offload-ide071`이며 실제 사용명은 manifest에 남긴다. 브랜치 생성·푸시는 기존 승인된 저장소 범위에서 수행한다.

### 19.2 게시할 파일

지시서 사본, 환경/설정 등록부, 실험 하네스, 최소 재현·테스트, 패치, hotmap·layer budget·scale manifest, RESULT/FULL_REPORT/상세 MD, 진행 로그, 상태 표, 원본 인벤토리, 관리 가능한 크기의 원시 CSV/JSONL을 게시한다.

다음은 Git 일반 파일로 게시하지 않는다.

```text
모델 weight 전체
HF/GitHub/API token, SSH key, 인증 쿠키
credential이 포함된 remote URL·환경변수
허가되지 않은 사용자 입력·생성문·개인정보
대용량 perf/Nsight/core/tensor 원본을 무분별하게 추가한 파일
```

원시 로그의 비밀정보는 게시본에서만 마스킹하고, 마스킹 내역과 사유를 남긴다. 측정 숫자·설정값을 임의로 지우거나 바꾸지 않는다. 원본 보존이 허용되는 경우 접근 제한된 위치에 둔다.

대용량 원본은 기존 승인된 Git LFS 또는 artifact 저장소가 있으면 사용한다. 없으면 서버의 보존 경로·크기·checksum을 MD에 명시한다. 로컬 보존 경로를 외부 다운로드 링크로 가장하지 않는다.

### 19.3 커밋·푸시·검증 절차

```text
1. 결과 파일 생성 완료 및 데이터 검증
2. 게시 대상 경로를 명시하여 stage — 무차별 git add . 금지
3. staged diff·비밀정보·대형 파일 확인
4. data commit 생성
5. 승인된 origin의 실험 브랜치로 push — force push 금지
6. remote branch SHA가 게시한 SHA와 일치하는지 확인
7. 원격에서 FULL_REPORT.md·RESULT.md가 실제 조회되는지 확인
8. data commit SHA·게시 경로·검증 결과를 PUBLISH_RECEIPT.md에 기록
9. receipt를 별도 후속 commit으로 게시하고 그 SHA를 최종 전달
```

파일 내부에 자기 자신의 commit SHA를 넣으려고 무한 commit하지 않는다. data commit과 receipt commit을 구분한다. 원격 확인이 실패하면 로컬 commit만 존재하는 상태로 기록한다.

푸시 실패는 최대2회 복구·재시도하고 인증정보를 우회하지 않는다. 다른 필수 파일 생성과 다운로드 제공은 계속 완료한다. 최종 상태에는 `PUBLISH_FAILED` 또는 `BLOCKED_PERMISSION`을 명시한다. 실제 게시하지 못했는데 GitHub 링크를 만들어 제시하지 않는다.

### 19.4 다운로드 링크

최종 `FULL_REPORT.md`를 사용자가 직접 받을 수 있는 산출물 위치에 복사하고 존재·크기·SHA256을 확인한다. 사용자 인터페이스가 파일 첨부/다운로드를 제공하면 해당 기능을 사용한다. 실제 샌드박스 파일이 확인된 경우에만 sandbox 다운로드 링크를 제공한다.

GitHub를 통한 다운로드는 실제 커밋의 MD raw/download 위치를 확인해 제공한다. 접근 제한 저장소라면 사용자에게 해당 권한이 필요한 사실을 표시한다. 다운로드용 패키지를 함께 제공할 수 있으나 **개별 MD 다운로드 링크를 ZIP으로 대신하지 않는다.**

최종 링크 목록에는 최소 다음이 들어가야 한다.

```text
FULL_REPORT.md 직접 다운로드
RESULT.md 직접 다운로드
GitHub의 실제 data commit 또는 MD 파일 위치
원시자료 인벤토리 또는 승인된 artifact 위치
```

### 19.5 최종 전달 형식

```markdown
실험 종료 상태: <수렴/범위 소진/기타 정확한 상태 코드>
등록 <N>개 / 측정 완료 <N>개 / 실패 <N>개 / 지원 불가·차단 <N>개
GitHub 게시: <실제 성공·실패 상태>, data commit <SHA>, receipt commit <SHA>

[상세 실험 데이터 MD 다운로드](<실제로 확인된 링크>)
[결과 데이터 색인 MD 다운로드](<실제로 확인된 링크>)
[GitHub 기록](<실제로 확인된 링크>)
```

이 전달 뒤에 “성능이 개선되었다”, “이 설정을 권장한다” 등의 평가를 추가하지 않는다.

---

## 20. 완료 체크리스트

- [ ] IDE_069·070 원본과 코드·환경 해시를 저장했다.
- [ ] 공개 문서 후보와 로컬 구현 옵션을 구분하고 실제 적용을 확인했다.
- [ ] R0·def0·D1·D3·D4·비균일 배치를 재현하거나 실패 근거를 저장했다.
- [ ] D2 오류는 재현·분리·복구 기록을 남겼다.
- [ ] CPU 커널·스레드·KV·graph·scheduler·attention·GPU MoE 설정을 탐색했다.
- [ ] 층별 expert 배치와 총 HBM 예산 변경을 별도로 측정했다.
- [ ] NUMA·메모리 기준 측정 및 권한 제약을 기록했다.
- [ ] W4·TP/EP·구조 개선·speculation의 등록된 조건부 항목을 종료 상태로 정리했다.
- [ ] 수렴/잡음/범위 소진/횟수 제한을 구분해 기록했다.
- [ ] 수렴 이후에도 필수 최종 측정·정확성 데이터·장시간 실행을 완료했다.
- [ ] 30분 보고 파일과 실제 전달 로그를 남겼다.
- [ ] 모든 raw run·실패·재시도·실제 적용 설정을 보존했다.
- [ ] 최종 MD에 평가·해석·권고·채택 판단을 넣지 않았다.
- [ ] 상세 MD·분할 MD·원본 색인·checksum을 검증했다.
- [ ] 기존 승인된 GitHub 저장소에 실제 push하고 원격 SHA·파일을 확인했다. 실패했다면 그대로 표시했다.
- [ ] 실제 존재하고 받을 수 있는 FULL_REPORT.md와 RESULT.md 다운로드 링크를 제공했다.
- [ ] 임시 전역 설정을 복구했고 실험 외 사용자 작업을 변경하지 않았다.

**이 체크리스트와 실제 상태가 일치할 때만 최종 전달하라. 끝나지 않은 항목을 숨기거나 성공으로 대체하지 마라.**

---

## 21. 근거 자료와 적용 범위

### 21.1 사용자가 제공한 실측 자료

- **[S1] `FULL_REPORT.md` — IDE_069 상세 보고서.** 하드웨어·모델 config·기존 hotmap/graph/KV/dual/speculation/GPU-only 실험의 근거. 원문 해석 부분과 실측 표를 구분해서 사용한다.
- **[S2] `RESULT.md` — IDE_070 결과.** A~E 시리즈 설정·실측값·오류·프로파일·파일 경로의 근거. 특히 제4절의 패치 적용 여부, 제5절 D2~D4의 서로 다른 구성, 제6절 AWQ 초기 실행 실패를 그대로 보존한다.

원본 SHA256:

```text
FULL_REPORT.md
542723c46b41871871e02d456d344c23ea010ac64e03aacccebf9279fd0b9843

RESULT.md
07144c375090e647fa1baee6b730db8ff61c116cbf25383eb252362993f3eba3
```

### 21.2 추가 설정을 찾기 위해 확인한 공개 1차 자료

아래 자료는2026-09-15에 조회한 문서다. **현재 공개 문서의 설정이 사용자의 고정된 로컬 빌드에 존재하거나 작동한다는 보증이 아니다.** 실제 실험에서는 설치본의 source와 effective config를 최우선으로 확인하고 문서 snapshot도 함께 보관한다.

| ID | 자료 | 이 계획에서 사용하는 범위 |
|---|---|---|
| W1 | SGLang Server Arguments | CLI 후보명, 로컬 help/source 확인, scheduler·graph·MoE·kt 옵션 분류 |
| W2 | SGLang Hyperparameter Tuning | KV·graph·prefill·admission의 별도 관측 항목 |
| W3 | SGLang Attention Backend | 모델·GPU·KV dtype·page size·단계별 backend 조합 확인 |
| W4 | SGLang Quantized KV Cache | FP8 scale·backend 경로 확인 |
| W5 | NVIDIA NCCL Environment Variables | 실제 NCCL 경로에 한정한 통신 설정, 디버그 변수 복구 |
| W6 | Intel Performance Counter Monitor | DRAM·NUMA·CPU 계측 도구와 단위 검증 |
| W7 | Intel Memory Latency Checker | 지속 메모리 대역폭·지연 기준 측정 도구 |
| W8 | KTransformers kt-kernel README | CPU 커널·양자화·NUMA·SGLang 연결 지원 확인 |
| W9 | NVIDIA Compute Sanitizer | 오류 경로의 별도 진단 실행 |

```text
[W1] https://docs.sglang.io/docs/advanced_features/server_arguments
[W2] https://docs.sglang.io/docs/advanced_features/hyperparameter_tuning
[W3] https://docs.sglang.io/docs/advanced_features/attention_backend
[W4] https://docs.sglang.io/docs/advanced_features/quantized_kv_cache
[W5] https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
[W6] https://github.com/intel/pcm
[W7] https://www.intel.com/content/www/us/en/developer/articles/tool/intelr-memory-latency-checker.html
[W8] https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md
[W9] https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html
```

탐색 후보 숫자, 반복 수, 수렴 임계값, 실험 순서, 보고·게시 절차, 새로운 배치·캐시·polling 패치는 **이번 계획에서 제안한 실험 설계**다. 기존 실험의 검증된 사실이나 공개 문서의 권장값으로 인용하지 않는다.

---

## 부록 A. 셀 manifest 예시

아래는 하네스가 생성할 레코드의 예시이며, 실행 결과가 아니다.

```json
{
  "campaign_id": "IDE_071_<timestamp>",
  "phase_id": "P2",
  "cell_id": "P2_cf1_skip0_pin1_def0",
  "change_kind": "CONFIG",
  "parent_cell_id": "R01",
  "semantics_family": "SEMANTICS_UNVERIFIED",
  "precision_family": "GPU_FP8_CPU_INT4_KV_BF16",
  "workload_id": "SHORT_COLD",
  "cache_policy": "engine_cache_flush_before_each_rep",
  "client_concurrency": 64,
  "num_requests": 256,
  "repetitions_initial": 3,
  "env_requested": {
    "KT_CALLBACK_FREE": "1",
    "KT_CF_SKIP_EMPTY_IMM": null
  },
  "settings_requested": {
    "max_deferred_experts_per_token": 0,
    "nonkt_thread_pinning": true
  },
  "required_runtime_proofs": [
    "callback_free_path_used",
    "skip_empty_effective_state",
    "all_rank_tid_affinity",
    "per_layer_deferred_limit",
    "graph_replay_and_fallback_counts"
  ],
  "effective_config_sha256": null,
  "status": "PENDING",
  "evidence_files": []
}
```

null은 미확인 또는 미지정을 뜻한다. env의 null을 unset으로 처리하는 규칙은 하네스에서 명시하고, 실행 후 실제 값으로 별도 기록한다.

## 부록 B. 셀별 Markdown 기록 예시

```markdown
# <cell_id> — 실행 데이터

## 식별
campaign / attempt / rep / source SHA / binary SHA / config hash

## 요청한 설정과 실제 적용된 설정
<전체 값 표, 차이가 있으면 원시 로그 위치>

## 실제 실행 명령
<환경변수를 포함한 launch command와 benchmark command>

## 시간·상태
boot / warmup / cache prepare / measurement / drain / exit status

## 반복별 원시 측정
| rep | workload | cache | C | 성공/실패 | duration | 실제 in/out | output tok/s | TTFT p50/p95/p99 | TPOT p50/p95/p99 | HBM peak |
|---|---|---|---|---|---|---|---|---|---|---|

## 요청별 데이터
<상세 MD 전체 링크 및 JSONL 원본>

## 레이어·expert·시스템 데이터
<원시 수치 표, 단위, 계측 창, 파일 위치>

## 오류·재시도
<시각, exit code, 첫 오류 원문, 전체 log, 다음 attempt ID>

## 기계적 정확성 데이터
<입력 ID, 출력 해시, 일치 수, 오류 크기, 측정하지 못한 이유>

## 원본 색인
<파일, 크기, SHA256, 저장/게시 위치>
```

**위 양식 뒤에 결론·평가·추천 절을 추가하지 않는다.**
