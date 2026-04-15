# Hybrid 개선 방향 재작성 — 로그 확정 사실, 외부 근거, 가설 분리

**작성 시각**: 2026-04-15 08:58:58 KST  
**작성자**: Codex

## 범위

이 문서는 다음 세 부류를 분리해서 정리한다.

1. **로컬 실측과 코드로 확정된 사실**
2. **논문/공식 프로젝트 자료로 뒷받침되는 개선 방향**
3. **흥미롭지만 아직 우리 코드베이스에선 검증되지 않은 가설**

핵심 전제는 유지한다.

- **request-level hybrid 는 유지**
- 목표는 **CPU가 더 많은 request 를 처리하되 total wall time 을 줄이는 것**
- 따라서 핵심은 **CPU 동시 request 처리에서 진짜 batch scaling 을 만드는 것**

---

## 1. 로컬 실측으로 확정된 사실

### 1-1. 현재 request-level hybrid 의 실패 모드는 CPU tail 이다

H100x8:

- `cpu_max_num_seqs=1` 에서도 CPU request 소수가 마지막까지 남아 wall time 을 결정
- `cpu_max_num_seqs=16` 은 CPU 32 req tail 로 wall time 을 재앙적으로 키움

RTX3090:

- 1.5B hybrid: GPU-only 대비 악화
- 7B hybrid: GPU-only 대비 크게 악화
- 둘 다 CPU 4 req tail 이 마지막 wall time 을 결정

즉 현재 문제는 단순히 “CPU가 느리다”가 아니다.  
**여러 request 를 CPU에 동시에 줘도 per-request cost 가 충분히 내려가지 않아서, CPU inflight 증가가 throughput gain 이 아니라 tail 확대가 된다.**

### 1-2. `wave-batch + cpu-first` 는 현재 구조에선 가짜 동시성이다

로그상 공통 패턴:

- 메인 burst 초반에 CPU wave 를 먼저 채움
- GPU bulk 는 먼저 거의 끝남
- CPU wave drain 이 마지막 wall time 을 결정

이건 진짜 batching 이 아니다.

- 여러 req 를 함께 잡았지만
- CPU 커널/메모리 경로가 그 req 들을 실제로 효율적으로 합쳐 처리하지 못했다

따라서 현재 결론은 분명하다.

- `cpu_max_num_seqs` 증가 = 현재는 tail 증폭
- `wave-batch` = 현재는 failure amplifier
- batch-aware 계산 구조가 생기기 전까지 기본 전략이 되면 안 된다

### 1-3. CPU bring-up 자체는 이미 끝났다

코드/로그상 이미 확인된 것:

- CPU engine launch
- NUMA 선택
- `local_omp_cpuid`
- C++ thread pinning
- IPEX decode 경로 사용

따라서 다음 단계의 질문은 bring-up 이 아니라 이것이다.

> CPU가 여러 request 를 함께 처리할 때 왜 scaling 이 안 나오는가, 그리고 그 구조를 어떻게 바꿔서 CPU inflight 증가를 실제 completion gain 으로 바꿀 것인가

---

## 2. 외부 자료가 뒷받침하는 것

아래는 이번에 직접 확인한 외부 자료 기준으로 보강되는 방향이다.

### 2-1. CPU+GPU 협업 자체는 포기할 대상이 아니다

**NEO (MLSys 2025)** 는 GPU의 일부 attention compute 와 KV cache state 를 CPU로 오프로드해 GPU batch size 를 키우고 throughput 을 높인다. OpenReview 기준으로 H100에서 **최대 14% throughput 향상**을 같은 latency 조건에서 보고한다.  
출처: [OpenReview NEO](https://openreview.net/forum?id=umgy9tWBLA)

이게 곧바로 우리 구조에 이식된다는 뜻은 아니다. 하지만 한 가지는 확실하다.

- CPU+GPU 협업이 원리상 불가능한 게 아니라
- **잘 설계된 batching / pipelining / load-aware scheduling** 이 있으면 유효할 수 있다

즉 “현재 request-level hybrid 가 실패했으니 CPU는 포기”는 과도한 결론이다.

### 2-2. CPU 커널 구조를 바꾸면 batch 이득이 커질 수 있다는 실증이 있다

**T-MAC** 은 lookup-table 기반 mixed-precision matrix multiplication 으로 dequantization 기반 CPU low-bit 경로보다 큰 속도 향상을 보고한다. GitHub README 기준으로:

- 3B BitNet 에서 single-core 20 tok/s, 4-core 48 tok/s
- 기존 CPU low-bit baseline 대비 4~5x speedup
- Snapdragon X Elite 환경에서 특정 설정의 CPU 경로가 공개 NPU 수치보다 높게 나옴

출처: [microsoft/T-MAC](https://github.com/microsoft/T-MAC)

이 자료가 우리에게 주는 핵심 메시지는 이것이다.

- CPU가 원천적으로 batch / low-bit inference 에 불리한 게 아니다
- **데이터 재사용과 연산 치환을 잘 설계하면** CPU 쪽 성능 지형이 크게 달라질 수 있다

### 2-3. dataflow/pipeline 재설계가 decode throughput 을 크게 바꿀 수 있다

**T-MAN** 은 NPU 문맥이지만, 핵심 아이디어는 우리에게도 중요하다. Hugging Face papers 페이지 기준:

- unified table lookup
- fused dequantization
- **three-stage pipeline**
- decoding **3.1x** speedup, prefill **1.4x**, energy **84% 절감**

출처: [T-MAN paper page](https://huggingface.co/papers/2511.11248)

이 결과를 x86 CPU에 직접 대입하면 안 된다.  
하지만 방향성은 분명하다.

- 단순히 “더 빠른 GEMM 하나”가 아니라
- **prefetch / dequant / matrix work 의 파이프라인화**
- **중간 데이터 재배치 최소화**

가 decode 쪽 병목을 크게 바꿀 수 있다.

### 2-4. AMX + sparsity 는 실제 개선 여지가 있다

**SparAMX** 는 AMX와 unstructured sparsity 를 이용해:

- linear layer 에서 end-to-end latency **1.42x reduction**
- attention computation 에서 **1.14x speedup**

을 보고한다.

출처: [SparAMX paper page](https://huggingface.co/papers/2502.12444)

이건 “AMX/VNNI/희소성은 보조선일 뿐”이라고 보기 어렵다는 뜻이다.  
적어도 request-level hybrid 를 살리려면 이런 하드웨어 친화적 커널 개선은 본선에 가깝다.

---

## 3. 클로드 문서에서 가져올 것과 낮춰 쓸 것

### 3-1. 가져올 것

클로드 문서의 다음 방향은 유효하다.

- CPU batch 비효율의 원인을 dataflow 미설계로 본 점
- LUT 기반 GEMV/GEMM 계열 탐색
- kernel fusion
- pre-pack / layout-aware path
- 3-stage pipeline 류의 개념 차용
- AMX + sparsity 경로를 진지한 후보로 둔 점

즉 클로드 문서의 장점은 **HPC식 돌파구 목록을 구체적으로 제시했다는 것**이다.

### 3-2. 낮춰 써야 할 것

다만 다음은 아직 조심해서 다뤄야 한다.

- `compute 10ms + data 3069ms` 같은 분해는 현재 로컬 코드에서 직접 측정된 값이 아니라 모델링/추정에 가깝다
- `Huge Pages + WoQ INT8 만으로 H100 H1 wall 394s -> <200s` 같은 정량 기대치는 아직 근거가 약하다
- KTransformers의 일부 수치는 이번 패스에서 1차 소스 확인이 충분치 않다

따라서 새 로드맵에서는:

- **확정 수치**와
- **강한 가설**

을 분리해서 써야 한다.

---

## 4. 통합 로드맵

## 4-0. 경유지와 3축 성공 기준

클로드 v2 문서의 장점 중 하나는 “어디까지 왔는가”를 다축으로 판정하는 방식이다. 이 점은 받아들이는 것이 맞다.

이 문서에서는 다음 3축을 함께 본다.

- **속도 축**: CPU batch tok/s, per-request latency, per-step 시간
- **tail 축**: GPU bulk 완료 후 남는 CPU-only tail 길이, inflight 고착 여부
- **wall ratio 축**: `hybrid wall / gpu_only wall`

경유지는 다음처럼 둔다.

- **G1. CPU batch scaling 징후 확인**
  - `num_seqs` 증가에 따라 CPU batch tok/s 가 실제로 증가
  - tail 이 일방적으로 늘기만 하지 않고 감소 방향성이 보임
  - wall ratio 가 기존 hybrid 대비 개선
- **G2. routing 재평가 가능 구간**
  - CPU batch scaling 이 반복 측정으로 재현됨
  - CPU inflight 증가가 tail 폭증으로 이어지지 않음
  - routing 정책 차이가 total completion 차이로 나타남
- **G3. 최종 목표 구간**
  - CPU가 더 많은 request 를 처리
  - CPU-only tail 이 실질적으로 사라지거나 매우 짧아짐
  - hybrid wall 이 gpu_only 에 근접

중요한 규칙:

- 한 축만 좋아져도 경유 통과로 보지 않는다
- 최소한 속도 축과 tail 축이 같이 좋아져야 다음 단계로 간다

## 4-1. 1단계: CPU batch 가 왜 진짜 batch 가 아닌지 계측으로 깨기

이 단계는 최우선이다.

필수 실험:

1. `num_seqs=1/2/4/8/16` 에서 CPU-only tok/s 측정
2. per-request latency 감소율 측정
3. attention / FFN / sync / memory wait 비중 분해
4. NUMA-local vs non-local 경로 차이 측정

핵심 질문:

- req 수 증가가 실제 throughput scaling 으로 이어지는가
- 아니면 barrier 와 memory traffic 만 늘어나는가

이 단계 없이 `wave-batch`, `cpu_max_num_seqs`, router 정책을 논해도 의미가 약하다.

## 4-2. 2단계: 저위험 HPC식 구조 개선

여기서는 request-level hybrid 를 유지한 채, CPU batch 가 실제 batch 처럼 동작하게 만드는 쪽부터 봐야 한다.

우선순위 후보:

1. **Head Folding**
2. **VNNI pre-pack / load-once-pack-twice**
3. **batch-aware decode attention**
4. **QKV / Gate-Up fusion**
5. **NUMA-local multi-engine 강화**
6. **barrier / sync 감소**
7. **AMX / VNNI / AVX-512 dispatch 최적화**

이 단계의 목표:

- CPU request 1개를 조금 빠르게 만드는 것보다
- **여러 request 를 함께 넣었을 때 비용이 실제로 줄어드는지**를 확인하는 것

## 4-3. 3단계: 그 위에서 routing 재평가

전제:

- 2단계에서 CPU batch scaling 이 실제로 생겼을 것

그 다음에만 의미 있는 실험:

1. `cpu_max_num_seqs=1` baseline 재측정
2. `2`, `4`, `8` 순으로 확대
3. `wave-batch` vs `throughput-adaptive` 비교

이 단계의 성공 조건:

- CPU handled requests 증가
- CPU batch tok/s 증가
- CPU tail 감소
- total wall time 감소

즉 routing 은 1차 해결책이 아니라 **batch scaling 확보 이후의 증폭기**다.

## 4-4. 4단계: GPU batch ceiling 확대

이 단계는 CPU batch scaling 이 어느 정도 확보된 뒤 의미가 크다.

단기:

- `--kv-cache-dtype fp8`

중기:

- INT4 KV + CPU DRAM / LMCache

이 축의 의미는 GPU-only 최적화가 아니라:

- CPU가 더 많은 request 를 맡을 수 있는 total operating point 를 넓히는 것

이다.

## 4-5. Stop/Go 규칙

각 단계 종료 시 다음처럼 판단한다.

### 경우 1. 속도 축만 좋아지고 tail 축이 안 좋아짐

가능한 해석:

- CPU kernel 자체는 빨라졌지만 router 가 여전히 잘못 CPU wave 를 채움
- CPU prefill / decode 경계에서 직렬화가 남아 있음
- batch scaling 은 생겼지만 wave close 정책이 tail 을 다시 만듦

조치:

- routing / gate 파라미터 재검토
- `wave-batch` 유지 여부 재판단
- inflight / finished / drained 로그를 기준으로 상태 전이 다시 확인

### 경우 2. 속도 축 자체가 거의 안 좋아짐

가능한 해석:

- 해당 기법이 hot path 를 못 건드렸음
- memory traffic 이 줄지 않았음
- runtime 재배치 / sync / per-seq loop 가 여전히 지배적

조치:

- 다음 단계로 가지 않음
- 해당 기법은 보류하고 계측 단계로 되돌아감

### 경우 3. CPU handled requests 는 늘었는데 wall ratio 가 안 좋아짐

가능한 해석:

- 처리한 request 수는 늘었지만 CPU req 당 비용이 여전히 높음
- CPU 증가분이 pure throughput 이 아니라 tail 로 남음

조치:

- 성공으로 보지 않음
- `cpu_max_num_seqs` 확대 실험 중단
- batch scaling 재검토 후 다시 시도

즉 최종 판정 기준은 이것이다.

- **CPU가 더 많이 처리했다**
- **CPU batch tok/s 가 올랐다**
- **tail 이 줄었다**
- **wall ratio 가 좋아졌다**

이 네 가지 중 앞의 하나만 좋아졌다고 다음 단계로 넘어가면 안 된다.

## 4-6. 병행 트랙 명시

클로드 v2 문서는 “CPU 자체 최적화 먼저, 역할 재정의는 병행 트랙”이라고 정리한다. 이 점은 받아들일 만하다.

이 문서의 입장은 다음처럼 정리한다.

- **메인라인**
  - request-level hybrid 유지
  - CPU batch scaling 형성
  - routing 재평가
- **병행 트랙**
  - spec decode drafter
  - 장문/대형모델 전용 offload
  - 장거리 pipeline / 더 큰 구조 변경

즉 role-level 또는 speculative 방향이 틀렸다는 뜻은 아니다. 다만 **현재 ideation의 immediate mainline 은 request-level hybrid 를 살리는 것**이다.

---

## 5. 당장 낮춰야 할 것

다음은 immediate roadmap 에서 우선순위를 낮춰야 한다.

- `wave-batch` 기본 전략
- CPU batch scaling 확인 전 `cpu_max_num_seqs` 확대
- “CPU가 더 많은 request 를 맡는 것 자체”를 성공으로 보는 해석
- request-level 문제를 아직 풀지 못한 상태에서 role-level 전환을 메인 해법으로 미는 것

마지막 항목은 중요하다. role-level / speculative / 다른 역할 분업이 장기적으로 의미가 없다는 뜻은 아니다. 다만 **현재 ideation의 중심은 request-level hybrid 를 살리는 것**이므로 immediate mainline 에서 앞세우면 안 된다.

---

## 6. 최종 정리

이 시점의 가장 정확한 문장은 이것이다.

> 목표는 CPU가 더 많은 request 를 처리하는 것이다. 단, 그 request 들이 tail 로 남지 않도록 여러 request 를 함께 처리할 때 실제 계산 이득이 생기는 구조를 먼저 만들어야 한다.

즉:

- **CPU를 포기할 이유는 없다**
- **지금 방식의 wave-batch 는 실패했다**
- **본선은 single-request polishing 이 아니라 true batch scaling**
- **HPC식 dataflow / kernel / layout 개선이 이 문제의 정면 해법이다**

---

## 참고 자료

- 로컬:
  - [20260414_220744_codex_hybrid_improvement_directions_after_h100x8_rtx3090_log_review.md](/vllm_hybrid/ideation/20260414_220744_codex_hybrid_improvement_directions_after_h100x8_rtx3090_log_review.md)
  - [20260414_233407_claude_hybrid_improvement_from_log_analysis.md](/vllm_hybrid/ideation/20260414_233407_claude_hybrid_improvement_from_log_analysis.md)
  - [20260414_213415_codex_h100x8_log_analysis.md](/vllm_hybrid/eval/h100x8/20260414_213415_codex_h100x8_log_analysis.md)

- 외부:
  - NEO: https://openreview.net/forum?id=umgy9tWBLA
  - T-MAC: https://github.com/microsoft/T-MAC
  - T-MAN: https://huggingface.co/papers/2511.11248
  - SparAMX: https://huggingface.co/papers/2502.12444
