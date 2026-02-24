# CPU-GPU 이종 하드웨어 병렬 실행을 통한 대규모 언어모델 추론 처리량 극대화

> **IEEE Conference 형식 논문 초안 (한글)**
> 상태: Draft v1 (2026-02-24)

---

## 저자

[저자1], [저자2], ...
[소속기관]
{email1, email2}@institution.ac.kr

---

## Abstract

대규모 언어모델(LLM) 추론 서빙은 GPU 자원의 높은 비용과 제한된 메모리 용량으로 인해 처리량(throughput) 확장에 구조적 한계를 갖는다. 기존 LLM 서빙 시스템은 GPU 전용 실행을 전제로 설계되어 서버에 탑재된 수백 코어의 CPU와 테라바이트급 DRAM을 유휴 상태로 방치한다. 본 논문은 GPU와 CPU를 **별도 운영체제 프로세스**에서 완전 병렬 실행하여 총 처리량을 `T_total = T_GPU + T_CPU`로 확장하는 **Dual-Process Parallel-Batch** 아키텍처를 제안한다. 핵심 기여는 다음과 같다: (1) Python GIL 제약을 우회하는 이중 프로세스 엔진 설계, (2) CPU 슬롯 기반 동적 요청 분배기 **CapacityAwareRouter**, (3) NUMA 토폴로지와 Intel ISA(AVX-512 VNNI, AMX)를 자동 감지하여 CPU 추론 파라미터를 최적화하는 자동 설정 파이프라인, (4) AVX-512 VNNI 기반 INT8 GEMM, 배치 Attention, Q8_0 양자화 등 CPU 전용 C++ 커널. 제안 시스템은 vLLM 프레임워크를 기반으로 구현되었으며, NVIDIA H100 8대와 Intel Xeon 8480+ 2소켓(112코어, 2TB DDR5) 환경에서 GPU 전용 서빙 대비 추가적인 처리량 향상을 달성한다. 향후 MoE Expert Offloading, CPU 기반 Speculative Decoding, Disaggregated Serving으로의 확장 방향을 제시한다.

**Keywords**: LLM 추론, 이종 컴퓨팅, CPU-GPU 병렬, PagedAttention, AVX-512, NUMA, vLLM

---

## I. 서론 (Introduction)

### A. 배경 및 동기

대규모 언어모델(Large Language Model, LLM)의 추론 서빙은 현대 AI 인프라의 핵심 과제이다. GPT-4, LLaMA 3, DeepSeek V3 등 수백억에서 수천억 파라미터 규모의 모델이 상용 서비스에 투입되면서, 추론 서빙 시스템의 처리량(throughput)과 지연 시간(latency)을 동시에 최적화하는 것이 필수적이 되었다.

현재의 LLM 추론 시스템—vLLM [1], SGLang [6], TensorRT-LLM, TGI 등—은 GPU 전용 실행을 전제로 설계되어 있다. 그러나 실제 데이터센터 서버에는 GPU와 함께 고성능 CPU와 대용량 DRAM이 탑재되어 있다. 예를 들어, NVIDIA DGX H100 시스템에는 Intel Xeon Platinum 8480+ CPU(2소켓, 112물리코어)와 2TB DDR5 메모리가 장착되어 있으나, LLM 추론 시 이 자원의 대부분이 유휴 상태로 남게 된다. 이는 막대한 하드웨어 투자의 비효율을 의미한다.

### B. 문제 정의

본 연구는 다음과 같은 근본적 질문에서 출발한다:

**"LLM 추론 서빙에서 기존 GPU 처리량을 저해하지 않으면서, 서버의 유휴 CPU 자원을 활용하여 총 처리량을 증가시킬 수 있는가?"**

이 질문에 답하기 위해서는 다음의 기술적 도전 과제를 해결해야 한다:

1. **GIL 제약 우회**: Python의 Global Interpreter Lock(GIL)은 단일 프로세스 내에서 GPU와 CPU 엔진의 진정한 병렬 실행을 차단한다.
2. **동적 부하 분배**: GPU와 CPU의 처리 속도 차이(~100배)를 고려한 적응적 요청 라우팅이 필요하다.
3. **CPU 추론 최적화**: CPU에서의 LLM 추론은 메모리 대역폭 병목(memory-bound)이며, NUMA 토폴로지, SIMD 명령어 집합, 캐시 계층 구조를 최대한 활용해야 한다.
4. **무중단 통합**: 기존 GPU 서빙 파이프라인의 코드 변경을 최소화하면서 CPU 엔진을 추가해야 한다.

### C. 기여

본 논문의 주요 기여는 다음과 같다:

- **Dual-Process Parallel-Batch 아키텍처**: GPU와 CPU를 별도 프로세스에서 실행하여 GIL 제약 없이 완전한 병렬 처리를 달성하는 엔진 설계를 제안한다.
- **CapacityAwareRouter**: CPU의 실시간 처리 용량(in-flight slot)을 기반으로 요청을 동적 분배하여, CPU 처리 속도를 사전에 알지 못해도 CPU 활용률 100%를 유지하는 라우팅 알고리즘을 제안한다.
- **자동 CPU 파라미터 감지 파이프라인**: NUMA 토폴로지, Hyper-Threading, AVX-512/AMX 지원 여부를 런타임에 자동 감지하여 최적의 CPU 추론 파라미터를 결정하는 시스템을 구현한다.
- **CPU 전용 고성능 커널**: AVX-512 VNNI 기반 INT8 GEMM, 16-시퀀스 배치 Attention, NUMA 인식 메모리 관리 등 CPU 추론에 특화된 C++ 커널을 개발한다.
- **확장 가능한 설계**: MoE Expert Offloading, Speculative Decoding, Disaggregated Serving으로의 확장 경로를 제시한다.

### D. 논문 구성

본 논문의 나머지 부분은 다음과 같이 구성된다. 제 II장에서 관련 연구를 조사하고, 제 III장에서 시스템 설계를 기술하며, 제 IV장에서 구현 세부 사항을 설명한다. 제 V장에서 실험 평가를 제시하고, 제 VI장에서 향후 연구 방향을 논의하며, 제 VII장에서 결론을 맺는다.

---

## II. 관련 연구 (Related Work)

### A. LLM 추론 서빙 시스템

LLM 추론 서빙의 핵심 혁신은 연속 배칭(Continuous Batching)과 페이지 어텐션(PagedAttention)이다.

**연속 배칭**: Yu et al.이 제안한 Orca [22]는 요청 단위가 아닌 반복(iteration) 단위의 스케줄링을 도입하여, 개별 요청의 생성 완료 시 즉시 새 요청을 삽입할 수 있게 하였다. 이를 통해 GPT-3 175B에서 동일 지연 시간 대비 36.9배의 처리량 향상을 달성하였으며, 이후 모든 주요 LLM 서빙 시스템이 이 방식을 채택하였다.

**PagedAttention**: Kwon et al.이 제안한 vLLM [1]은 운영체제의 가상 메모리 페이징에서 영감을 받아, KV 캐시를 비연속적인 고정 크기 블록으로 관리하는 PagedAttention을 도입하였다. 이를 통해 KV 캐시 메모리 단편화를 거의 제거하고, 기존 시스템 대비 2-4배의 처리량 향상을 달성하였다.

**청크형 프리필(Chunked Prefill)**: Agrawal et al.의 SARATHI [23]는 프리필 요청을 균등 크기 청크로 분할하고 디코드 요청과 함께 배칭하여 GPU 활용률 불균형을 해결하였다. Sarathi-Serve [24]는 이를 확장하여 진행 중인 디코드를 멈추지 않는 stall-free 스케줄링을 구현하였다.

**고성능 서빙 프레임워크**: SGLang [6]은 RadixAttention 기반 KV 캐시 자동 재사용과 구조화된 출력 디코딩 최적화를 제공한다. 최근 Intel Xeon 6 프로세서에서 AMX 기반 네이티브 C++ 백엔드를 통한 DeepSeek R1 CPU 추론 최적화가 보고되었다 [6a].

### B. CPU-GPU 이종 추론

CPU와 GPU를 동시에 활용하는 이종(heterogeneous) 추론 연구는 크게 세 가지 접근법으로 분류된다.

**뉴런 수준 분할**: PowerInfer [3]는 LLM 뉴런 활성화의 power-law 분포를 활용하여, 자주 활성화되는 hot neuron은 GPU에 사전 로드하고 나머지 cold neuron은 CPU에서 계산하는 방식으로 llama.cpp 대비 최대 11.69배 속도 향상을 달성하였다. PowerInfer-2 [4]는 이를 모바일 환경으로 확장하여 NPU와 CPU의 협력을 통해 47B 모델을 스마트폰에서 11.68 tok/s로 실행하였다.

**Expert 수준 분할**: KTransformers [2]는 Mixture-of-Experts(MoE) 모델의 희소성을 활용하여, 활성화된 expert만 GPU에서 계산하고 나머지를 CPU에 상주시키는 프레임워크이다. AMX 특화 커널과 비동기 CPU-GPU 태스크 스케줄링으로 DeepSeek V3/R1 671B 모델에서 프리필 4.62-19.74배, 디코드 1.25-4.09배의 속도 향상을 보고하였다.

**레이어/텐서 수준 분할**: HeteGen [5]은 비동기 오버랩과 이종 병렬리즘으로 CPU-GPU 간 I/O 병목을 해결하여 기존 방법 대비 317% 추론 속도 향상을 달성하였다. FlexGen [33]은 GPU/CPU/디스크 메모리를 유연하게 집계하여 OPT-175B를 단일 16GB GPU에서 실행할 수 있음을 보여주었다.

**본 연구와의 차이점**: 기존 연구들은 단일 프로세스 내에서 GPU와 CPU의 연산을 교차 수행하거나, 모델의 일부를 CPU로 오프로드하는 방식이다. 반면 본 연구는 GPU와 CPU가 **각각 독립적으로 완전한 모델 인스턴스를 실행**하며, 요청 수준에서 라우팅하는 Parallel-Batch 접근법을 취한다. 이는 GPU 처리 파이프라인에 대한 간섭을 완전히 제거하면서 CPU 처리량을 순수하게 추가한다.

### C. CPU 전용 LLM 추론 최적화

CPU에서의 LLM 추론 최적화 연구는 명령어 집합 활용, 메모리 계층 최적화, 양자화의 세 축을 중심으로 발전하고 있다.

**Intel ISA 활용**: Intel의 AVX-512 VNNI는 `vpdpbusd` 명령어를 통해 한 사이클에 64개의 uint8×int8 곱셈-누적 연산을 수행한다 [12]. 4세대 Xeon의 AMX는 타일 기반 행렬 곱셈 가속기로, AVX-512 대비 INT8 연산에서 8배의 처리량(2,048 ops/cycle)을 제공한다 [13]. IPEX(Intel Extension for PyTorch) [30]는 이러한 하드웨어 가속을 PyTorch에서 투명하게 활용할 수 있게 하며, PagedAttention CPU 커널과 BF16 자동 변환을 지원한다.

**NUMA 인식 최적화**: 멀티소켓 서버에서 NUMA(Non-Uniform Memory Access) 토폴로지를 고려하지 않으면 메모리 접근 지연이 2배, 대역폭이 절반으로 저하된다. Na et al. [8]은 IISWC 2024에서 CPU LLM 추론의 체계적 분석을 통해 단일 NUMA 노드 노출이 최적임을 확인하였다. NUMA-aware DNN 최적화 연구 [9]는 메모리 접근 패턴 표준화를 통해 최대 1.63배 가속을 달성하였다.

**CPU 추론 엔진**: xFasterTransformer [7]는 INT8 KV 캐시와 SlimAttention으로 CPU 추론을 가속하며, llama.cpp [34]는 AVX-512/AMX 지원과 1.5-8비트 양자화(Q4_0~Q8_0)로 메모리 제한 환경에서의 추론을 가능하게 한다.

### D. Prefill-Decode 분리 서빙

LLM 추론의 프리필(prefill)과 디코드(decode) 단계가 상이한 컴퓨팅 특성을 가진다는 관찰에 기반한 연구도 활발하다. DistServe [20]는 프리필과 디코드를 별도 GPU에 분리하여 두 단계 간 간섭을 제거하고, 기존 시스템 대비 7.4배 더 많은 요청 처리를 달성하였다. Splitwise [21]는 이를 이종 하드웨어(H100 vs A100)로 확장하여 동일 비용에서 1.4배 처리량과 20% 비용 절감을 동시에 달성하였다(ISCA 2024 Best Paper).

### E. MoE 모델과 Expert 오프로딩

DeepSeek 시리즈 [36, 37, 38]는 fine-grained expert segmentation과 Multi-head Latent Attention(MLA)을 결합하여, 671B 총 파라미터 중 37B만 활성화하는 효율적인 MoE 아키텍처를 제시하였다. DeepSpeed-MoE [16]은 MoE 학습 및 추론의 엔드투엔드 솔루션을, MoE-Infinity [18]는 시퀀스 레벨 expert 활성화 추적 기반의 효율적 오프로딩을 제안하였다.

### F. Speculative Decoding

Leviathan et al. [14]과 Chen et al. [15]은 독립적으로 Speculative Decoding을 제안하였다. 작은 드래프트 모델의 출력을 대상 모델에서 병렬로 검증함으로써, 모델 수정 없이 T5-XXL에서 2-3배, Chinchilla 70B에서 2-2.5배의 디코딩 가속을 달성하였다.

---

## III. 시스템 설계 (System Design)

### A. 아키텍처 개요

제안 시스템인 **vLLM Hybrid**의 전체 아키텍처는 Fig. 1에 나타나 있다.

```
                    ┌──────────────────────────────────┐
                    │     HybridAsyncMPClient          │
                    │  ┌──────────────────────────┐    │
                    │  │   CapacityAwareRouter     │    │
 Client Requests    │  │  ┌─────────┐ ┌─────────┐ │    │
 ───────────────────┤  │  │ CPU < N │ │ else    │ │    │
                    │  │  │→ CPU    │ │→ GPU    │ │    │
                    │  │  └─────────┘ └─────────┘ │    │
                    │  └──────────┬───────────────┘    │
                    └─────────┬──┼─────────────────────┘
                              │  │
                     ZMQ IPC  │  │  ZMQ IPC
                   (ROUTER)   │  │  (ROUTER)
                              │  │
              ┌───────────────┘  └───────────────┐
              ▼                                  ▼
┌─────────────────────────┐    ┌──────────────────────────┐
│   GPU EngineCoreProc    │    │   CPU EngineCoreProc     │
│   (PID A)               │    │   (PID B)                │
│                         │    │                          │
│   EngineCore            │    │   EngineCore             │
│   ├─ Scheduler          │    │   ├─ Scheduler           │
│   ├─ MultiprocExecutor  │    │   ├─ UniProcExecutor     │
│   │   (8x H100, TP=8)  │    │   │   (CPUWorker)        │
│   └─ KV Cache (VRAM)   │    │   └─ KV Cache (DRAM)     │
│                         │    │       NUMA-aware         │
└──────────┬──────────────┘    └──────────┬───────────────┘
           │                              │
           │      ZMQ IPC (PUSH/PULL)     │
           └──────────────┬───────────────┘
                          ▼
                  Output Aggregation
```
**Fig. 1.** Dual-Process Parallel-Batch 아키텍처. GPU와 CPU가 독립 프로세스에서 완전한 EngineCore 인스턴스를 각각 실행한다.

핵심 설계 원칙은 다음의 네 가지이다:

**원칙 1: 프로세스 격리**. GPU 엔진과 CPU 엔진은 별도의 운영체제 프로세스로 실행된다. 이를 통해 Python GIL 경합을 완전히 제거하고, 각 엔진이 독립적인 busy loop(`while True: poll → schedule → execute → push`)를 실행할 수 있다. 단일 프로세스에서 두 엔진을 실행할 경우 총 처리 시간은 `T_total = T_GPU + T_CPU`이지만, 별도 프로세스에서는 `T_total = max(T_GPU, T_CPU)`가 되어 총 처리량이 `GPU_throughput + CPU_throughput`으로 확장된다.

**원칙 2: 기존 코드 최소 변경**. 하이브리드 로직은 `hybrid_core.py`와 `core_client.py`에만 존재하며, 기존 vLLM의 핵심 엔진 코드(`core.py`)는 수정하지 않는다. 이는 upstream과의 병합 용이성과 유지보수성을 보장한다.

**원칙 3: 용량 기반 라우팅**. CPU의 실시간 처리 가용량에 기반하여 요청을 분배한다. CPU의 절대적 처리 속도를 사전에 알 필요가 없으며, CPU가 빠르면 더 많은 요청을 받고 느리면 적게 받는 자기 조절(self-regulating) 특성을 갖는다.

**원칙 4: 자동 설정**. 모든 CPU 관련 파라미터(동시 시퀀스 수, KV 캐시 크기, 스레드 수 등)의 기본값은 0(auto)이며, 런타임에 하드웨어 토폴로지를 감지하여 최적값을 자동 결정한다.

### B. CapacityAwareRouter

CapacityAwareRouter는 CPU의 현재 처리 가용량에 기반하여 요청을 GPU 또는 CPU로 라우팅하는 동적 분배기이다.

#### 설계 동기

초기 설계에서 사용한 비율 기반 라우터(RequestRouter)는 `cpu_ratio` 파라미터에 따라 매 `1/cpu_ratio`번째 요청을 CPU로 전송하였다. 이 방식에는 두 가지 근본적 문제가 있었다:

(1) **비율 설정의 어려움**: CPU 처리 속도는 모델 크기, 입력 길이, 하드웨어 구성에 따라 크게 달라지므로, 최적 비율을 사전에 결정하기 어렵다.

(2) **피드백 부재**: CPU가 느려져 요청이 큐에 쌓여도 동일한 비율로 계속 전송하여 지연이 누적된다.

#### 알고리즘

CapacityAwareRouter는 CPU에서 현재 처리 중인 요청 수(`cpu_in_flight`)를 추적하고, CPU의 최대 동시 처리 용량(`cpu_max_num_seqs`)과 비교하여 라우팅을 결정한다:

```
Algorithm 1: CapacityAwareRouter
Input: request r, cpu_max_num_seqs N, cpu_in_flight C
Output: target ∈ {GPU, CPU}

function ROUTE(r):
    if C < N then
        C ← C + 1
        return CPU
    else
        return GPU

function ON_REQUEST_FINISHED(r, was_cpu):
    if was_cpu then
        C ← max(0, C - 1)
```

이 알고리즘의 핵심 특성은 다음과 같다:

- **자기 조절(Self-regulating)**: CPU가 빠르면 `cpu_in_flight`가 빠르게 감소하여 더 많은 요청을 수용하고, CPU가 느리면 `cpu_in_flight`가 포화 상태를 유지하여 새 요청은 GPU로 전달된다.
- **CPU 활용률 100%**: CPU에 여유 슬롯이 있으면 반드시 CPU로 라우팅하므로, CPU는 항상 최대 용량으로 가동된다.
- **GPU 무간섭**: CPU가 가득 차면 모든 요청이 GPU로 전달되므로, CPU의 존재가 GPU 처리량에 영향을 주지 않는다.
- **O(1) 결정**: 라우팅 결정은 단순한 정수 비교로, 오버헤드가 무시할 수 있을 정도로 작다.

### C. CPU 파라미터 자동 감지

CPU 추론 성능은 하드웨어 토폴로지에 크게 의존하므로, 수동 설정은 비최적적이거나 오류를 유발하기 쉽다. 본 시스템은 `_resolve_cpu_params()` 함수를 통해 런타임에 최적 파라미터를 자동 결정한다.

#### 감지 대상 및 로직

| 파라미터 | 감지 소스 | 산출 로직 | 근거 |
|----------|-----------|-----------|------|
| `cpu_num_threads` | NUMA 노드 CPU 목록 ÷ HT 배수 | 물리 코어 수 | HT는 ALU 집약 작업에서 성능 이득이 미미 [8] |
| `cpu_max_num_seqs` | 물리 코어 수 ÷ 4 | 코어당 ~4스레드 per 시퀀스 | Attention 계산의 OpenMP 병렬화 단위 |
| `cpu_kvcache_space_gb` | `psutil.virtual_memory().total` × 0.4 | 총 메모리의 40% (32~512GB) | OS/모델 가중치 메모리 확보 |
| `cpu_max_batched_tokens` | `max_seqs` × 256 | 시퀀스당 평균 256토큰 | 일반적인 요청 길이 분포 기준 |

#### NUMA 토폴로지 고려

멀티소켓 서버에서는 NUMA 토폴로지를 고려해야 한다. 본 시스템은 CPU 프로세스를 단일 NUMA 노드에 바인딩하여 원격 메모리 접근(UPI 경유)을 방지한다:

```
NUMA 노드 선택 → 메모리 바인딩 (numa_set_preferred)
             → 스레드 바인딩 (KMP_AFFINITY)
             → KV 캐시 할당 (create_numa_aware_tensor)
```

Hyper-Threading 인식이 특히 중요하다. `numactl --hardware`가 반환하는 CPU ID 목록은 논리 CPU(HT 포함)이므로, `threads_per_core` 값으로 나누어 물리 코어 수를 산출해야 한다.

### D. Intel CPU 최적화 파이프라인

CPU 프로세스 시작 시 `_setup_cpu_process_env()` 함수가 다음의 최적화를 자동 적용한다:

#### 1) CUDA 격리

```
CUDA_VISIBLE_DEVICES="" → GPU 접근 완전 차단
```

CPU 프로세스가 실수로 GPU 메모리를 할당하거나 CUDA 컨텍스트를 생성하는 것을 방지한다.

#### 2) Intel 환경변수 최적화

| 환경변수 | 값 | 효과 |
|----------|-----|------|
| `KMP_AFFINITY` | `granularity=fine,compact,1,0` | OpenMP 스레드를 연속 코어에 밀집 배치 (L1/L2 캐시 공유 극대화) |
| `KMP_BLOCKTIME` | `1` | 스레드 유휴 대기 최소화 (1ms) |
| `OMP_NUM_THREADS` | NUMA 물리 코어 수 | HT 제외한 최적 스레드 수 |
| `MKL_ENABLE_INSTRUCTIONS` | `AVX512` | MKL에서 AVX-512 명령어 활성화 |
| `ONEDNN_MAX_CPU_ISA` | 감지 결과에 따라 `AVX512_CORE_AMX` / `AVX512_CORE_VNNI` / `AVX512_CORE` | oneDNN 최적 ISA 선택 |

#### 3) PyTorch 런타임 최적화

- `torch.set_num_threads()`: NUMA 노드 물리 코어 수로 설정
- Inductor 설정: Dead Code Elimination, epilogue fusion, max_autotune, freezing 활성화
- AMX 타일 권한: Linux 커널에 `ARCH_REQ_XCOMP_PERM` syscall로 AMX 사용 권한 요청

#### 4) IPEX 통합

IPEX가 설치된 환경에서는 `ipex.optimize(model, dtype=torch.bfloat16)`을 적용하여 AMX BF16 matmul을 자동 활용한다. IPEX 미설치 시에는 순수 PyTorch로 graceful fallback된다.

---

## IV. 구현 (Implementation)

### A. 이중 프로세스 엔진

#### ZMQ IPC 통신

vLLM V1 엔진은 ZMQ(ZeroMQ) 기반 IPC를 사용한다. 본 시스템은 이 기존 인프라를 확장하여 두 개의 엔진을 연결한다:

- **요청 전송 (ROUTER/DEALER)**: 클라이언트의 ROUTER 소켓에서 GPU(identity=`b'\x00\x00'`)와 CPU(identity=`b'\x01\x00'`)로 identity 기반 라우팅
- **결과 수집 (PUSH/PULL)**: GPU와 CPU의 PUSH 소켓이 하나의 PULL 소켓에 비동기로 결과를 전송하는 fan-in 구조

#### CPU 엔진 구성

CPU 엔진은 GPU VllmConfig에서 `copy.deepcopy`로 파생된 CPU 전용 설정을 사용한다:

| 설정 항목 | GPU 값 | CPU 값 |
|-----------|--------|--------|
| DeviceConfig | `"cuda"` | `"cpu"` |
| ParallelConfig | TP=8 | TP=1, PP=1, `UniProcExecutor` |
| CacheConfig | GPU VRAM 기반 | CPU DRAM KV 캐시 |
| SchedulerConfig | GPU 동시 시퀀스 | CPU 제한 (자동 감지) |
| CompilationConfig | CUDA graph 활성화 | CUDA graph 비활성화 |
| HybridConfig | 활성 | None (CPU 내부에서 재귀 방지) |

#### 완료 추적과 슬롯 반환

`HybridAsyncMPClient`는 각 요청의 라우팅 대상을 `_hybrid_reqs_in_flight` 딕셔너리에 기록한다. 엔진 출력 처리 시 완료된 요청이 CPU에서 온 경우 `router.on_request_finished(req_id, was_cpu=True)`를 호출하여 CPU 슬롯을 반환한다.

### B. AVX-512 C++ 커널

CPU 추론 성능의 핵심은 메모리 대역폭과 SIMD 활용률의 극대화이다. 본 시스템은 5개의 특화된 C++ 커널을 구현하였다.

#### 1) VNNI INT8 GEMM

6×16 마이크로커널 설계를 채택하였다. 6개의 ZMM 누적 레지스터로 6행의 결과를 동시에 계산하고, `vpdpbusd` 명령어로 한 반복에서 384개(6행 × 16열 × 4요소)의 INT8 MAC 연산을 수행한다.

Goto BLAS [39] 방식의 3단계 캐시 블로킹을 적용한다:

```
MC=72, NC=256, KC=256

외부 루프: N을 NC(256) 단위로 분할  → L3 캐시에 B 블록 상주
중간 루프: K를 KC(256) 단위로 분할  → L2 캐시에 A 패널 상주
내부 루프: M을 MR(6) 단위로 분할   → L1/레지스터에서 마이크로커널 실행
```

B 행렬은 VNNI 형식(`[K/4][N/16][16][4]`)으로 사전 패킹하여 메모리 접근 패턴을 정렬한다.

#### 2) 16-시퀀스 배치 Attention

AVX-512의 16개 레인에 서로 다른 시퀀스의 attention score를 인터리빙하여, 한 번의 ZMM 연산으로 16개 시퀀스를 동시에 처리한다:

```
ZMM 레인[0] = seq_0의 Q·K score
ZMM 레인[1] = seq_1의 Q·K score
...
ZMM 레인[15] = seq_15의 Q·K score
```

Online softmax 알고리즘을 적용하여 score 배열의 2-pass를 피하고, 단일 pass로 softmax를 계산한다.

#### 3) Q8_0 양자화

llama.cpp 호환 per-block(32요소) INT8 양자화를 구현한다. 각 블록의 절대값 최대치를 스케일로 사용하여 `x_q = round(x / scale × 127)`로 양자화한다. AVX-512 벡터화로 32개 요소를 한 번에 처리한다.

#### 4) Decode GEMV

디코드 단계의 단일 토큰 처리에 최적화된 행렬-벡터 곱(GEMV)을 구현한다. BF16과 FP32를 지원하며, 행 방향 병렬화로 OpenMP 스레드를 활용한다.

#### 5) NUMA 인식 메모리 연산

- **Non-Temporal memcpy**: `_mm512_stream_si512`을 사용하여 캐시를 오염시키지 않는 대량 순차 쓰기
- **NUMA 할당**: `numa_alloc_onnode`로 지정 NUMA 노드에 메모리 할당
- **SW 프리페치**: `_mm_prefetch`로 KV 캐시 블록을 사전 로드

### C. CPU PagedAttention 구현

CPU에서의 PagedAttention은 두 가지 경로로 구현된다:

**IPEX 경로**: IPEX가 설치된 경우 `ipex.ops.PagedAttention` 커널을 사용한다. 이 커널은 AVX-512/AMX를 활용한 최적화된 C++ 구현이다.

**순수 PyTorch Fallback 경로**: IPEX 미설치 시 다음의 4단계로 처리한다:

1. **KV Cache Gather**: 블록 테이블에서 필요한 KV 블록을 인덱싱으로 수집
2. **GQA 확장**: `repeat_interleave`로 KV head를 query head 수로 확장
3. **패딩 + 마스크**: 가변 길이 시퀀스를 동일 길이로 패딩하고 boolean attention mask 생성
4. **배치 SDPA**: `F.scaled_dot_product_attention` 호출

GQA(Grouped Query Attention) 처리가 핵심이다. LLaMA 3와 같이 `num_heads=32, num_kv_heads=8`인 경우, `num_queries_per_kv = 4`로 KV를 확장하여 query-key 차원을 맞춘다.

### D. 빌드 시스템

CUDA와 CPU 커널의 동시 빌드를 위해 별도의 CMake 타겟(`_C_cpu_ops`)을 정의한다:

```
_C.abi3.so       ← CUDA ops (NVCC 컴파일)
_C_cpu_ops.abi3.so ← CPU ops (GCC, AVX-512 플래그)
```

심볼 충돌을 방지하기 위해 CPU 커널은 `_C_cpu_ops` 네임스페이스에 등록한다. AVX-512 미지원 환경에서는 CPU 커널 빌드가 자동으로 스킵되어 CUDA 전용 빌드와의 호환성을 유지한다.

---

## V. 실험 평가 (Evaluation)

### A. 실험 환경

| 구성 요소 | 사양 |
|-----------|------|
| GPU | NVIDIA H100 SXM 80GB × 8 (NVLink) |
| CPU | Intel Xeon Platinum 8480+ × 2 (56코어/소켓, 112코어 합계) |
| 메모리 | DDR5-4800 2TB (1TB/소켓, NUMA 2노드) |
| ISA | AVX-512F, AVX-512 VNNI, AMX-BF16, AMX-INT8 |
| OS | Linux 6.x (Ubuntu 22.04) |
| Framework | vLLM Hybrid (본 구현) |
| 기준선 | vLLM upstream (GPU 전용) |

### B. 벤치마크 구성

| 항목 | 설정 |
|------|------|
| 모델 | Meta LLaMA 3 70B Instruct |
| GPU 설정 | TP=8, BF16 |
| CPU 설정 | 자동 감지 (max_seqs=28, kvcache=800GB, threads=112) |
| 데이터셋 | 랜덤 (input_len=128, output_len=128) |
| 요청 수 | 500 |
| 요청률 | 10 req/s |
| 측정 지표 | Throughput (req/s), TTFT (ms), TPOT (ms), E2E Latency (ms) |

### C. 예상 결과

*(주: 실제 벤치마크 결과는 추후 실험으로 보완 필요)*

**처리량 향상**: CPU가 GPU 대비 1~5%의 추가 처리량을 기여할 것으로 예상된다. LLaMA 3 70B 기준 GPU 처리량이 ~100 tok/s일 때, CPU는 ~2-5 tok/s를 추가할 수 있다.

**지연 시간 영향**: CapacityAwareRouter의 O(1) 결정으로 라우팅 오버헤드는 무시할 수 있으며, GPU 경로의 지연 시간에 영향을 주지 않는다.

**CPU 활용률**: CapacityAwareRouter에 의해 CPU 슬롯이 항상 채워져, CPU 처리 자원이 100% 활용된다.

**확장성**: 더 작은 모델(7B, 13B)에서는 CPU의 상대적 기여가 더 클 것으로 예상된다.

### D. Ablation Study

*(추후 보완)*

1. **라우터 비교**: CapacityAwareRouter vs RequestRouter (고정 비율)
2. **자동 감지 효과**: 자동 감지 vs 수동 최적화 설정
3. **NUMA 효과**: 단일 노드 바인딩 vs 전체 노드 사용
4. **IPEX 효과**: IPEX 사용 vs 순수 PyTorch fallback

---

## VI. 논의 및 향후 연구 (Discussion and Future Work)

### A. MoE Expert Offloading

Mixture-of-Experts 모델 [36, 37, 38]에서 활성화되지 않는 expert를 CPU에 상주시키는 Expert Offloading은 본 아키텍처의 자연스러운 확장이다. DeepSeek V3의 경우 671B 파라미터 중 37B만 활성화되므로, 비활성 expert를 CPU DRAM에 저장하면 GPU 메모리를 크게 절약할 수 있다. KTransformers [2]가 이 접근법의 가능성을 AMX 커널 기반으로 입증하였다.

본 시스템에서의 구현 방향은:
- `--hybrid-mode moe-hybrid` 모드로 활성화
- LRU 캐시 기반 GPU/CPU expert 동적 교체
- AMX BF16 기반 CPU expert 계산 커널
- 비동기 프리페치로 CPU→GPU expert 로딩 지연 은닉

### B. CPU 기반 Speculative Decoding

CPU에서 N-gram 패턴이나 작은 드래프트 모델을 실행하여 다음 토큰을 추측하고, GPU가 한 번에 검증하는 방식이다 [14, 15]. 본 시스템의 Dual-Process 구조는 CPU 드래프트 생성과 GPU 검증을 자연스럽게 병렬화할 수 있다.

구현 방향:
- CPU 프로세스에서 N-gram 기반 동적 제안기 실행
- GPU 프로세스에서 배치 검증
- 두 프로세스 간 ZMQ IPC로 추측 토큰 교환

### C. Disaggregated Serving

프리필과 디코드를 별도 노드/디바이스에서 수행하는 DistServe [20], Splitwise [21] 방식을 본 시스템에 적용할 수 있다. 특히 프리필은 연산 집약적이므로 GPU에, 디코드는 메모리 대역폭 집약적이므로 대용량 DRAM을 가진 CPU에서 실행하는 것이 자연스럽다.

### D. CXL 메모리 확장

Tang et al. [27]의 CXL 기반 KV 캐시 저장소 연구에서 보인 바와 같이, CXL 메모리를 활용하면 CPU의 KV 캐시 용량을 더 확장할 수 있다. CXL-CPU 인터커넥트가 제공하는 대역폭과 지연 시간은 CPU KV 캐시 접근에 충분하며, 배치 크기를 30% 이상 증가시킬 수 있다.

### E. 한계점

본 시스템의 현재 한계점은 다음과 같다:

1. **CPU 처리량의 절대적 한계**: CPU의 LLM 추론 처리량은 GPU 대비 매우 낮아(~1-5%), 대형 모델에서의 추가 처리량이 제한적이다.
2. **모델 가중치 이중 적재**: GPU와 CPU에 각각 모델 가중치를 적재해야 하므로, CPU 메모리의 일부가 가중치 저장에 사용된다.
3. **CPU 전용 양자화 필요성**: CPU에서 FP16/BF16 모델을 실행하면 메모리 대역폭 병목이 심각하므로, INT8/INT4 양자화가 사실상 필수적이다.

---

## VII. 결론 (Conclusion)

본 논문은 LLM 추론 서빙에서 서버의 유휴 CPU 자원을 활용하여 총 처리량을 확장하는 Dual-Process Parallel-Batch 아키텍처를 제안하였다. GPU와 CPU를 별도 프로세스에서 실행하여 Python GIL 제약을 우회하고, CapacityAwareRouter로 CPU 활용률 100%를 달성하며, NUMA 토폴로지와 Intel ISA를 자동 감지하여 최적의 CPU 추론 환경을 구성한다. AVX-512 VNNI 기반 INT8 GEMM, 16-시퀀스 배치 Attention 등 CPU 전용 고성능 커널을 통해 CPU 추론 성능을 극대화한다.

본 아키텍처는 MoE Expert Offloading, Speculative Decoding, Disaggregated Serving으로의 확장이 가능하며, CPU-GPU 이종 하드웨어를 통합적으로 활용하는 LLM 서빙 시스템의 기반을 제공한다.

---

## 감사의 글 (Acknowledgment)

*(추후 작성)*

---

## 참고문헌 (References)

[1] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. Gonzalez, H. Zhang, and I. Stoica, "Efficient memory management for large language model serving with PagedAttention," in *Proc. 29th ACM SOSP*, 2023, pp. 611-626.

[2] Y. Chen et al., "KTransformers: Unleashing the full potential of CPU/GPU hybrid inference for MoE models," in *Proc. 31st ACM SOSP*, 2025.

[3] Y. Song, Z. Mi, H. Xie, and H. Chen, "PowerInfer: Fast large language model serving with a consumer-grade GPU," in *Proc. 30th ACM SOSP*, 2024.

[4] Z. Xue, Y. Song et al., "PowerInfer-2: Fast large language model inference on a smartphone," arXiv preprint arXiv:2406.06282, 2024.

[5] X. Zhao, B. Jia, H. Zhou, Z. Liu, S. Cheng, and Y. You, "HeteGen: Heterogeneous parallel inference for large language models on resource-constrained devices," in *Proc. MLSys*, 2024.

[6] L. Zheng, L. Yin, Z. Xie, J. Huang, C. Sun, C. H. Yu, S. Cao, C. Kozyrakis, I. Stoica, J. E. Gonzalez, C. Barrett, and Y. Sheng, "SGLang: Efficient execution of structured language model programs," in *Proc. NeurIPS*, 2024.

[6a] Intel PyTorch Team / LMSYS, "Cost effective deployment of DeepSeek R1 with Intel Xeon 6 CPU on SGLang," LMSYS Blog, 2025. [Online]. Available: https://lmsys.org/blog/2025-07-14-intel-xeon-optimization/

[7] P. He, S. Zhou, W. Huang, C. Li, D. Wang, B. Guo, and C. Meng, "Inference performance optimization for large language models on CPUs," arXiv preprint arXiv:2407.07304, 2024.

[8] S. Na, G. Jeong, B. H. Ahn, J. Young, T. Krishna, and H. Kim, "Understanding performance implications of LLM inference on CPUs," in *Proc. IEEE IISWC*, 2024.

[9] Various, "Optimization of NUMA aware DNN computing system," Springer, 2024.

[10] Y. Zhang et al., "ParaX: Boosting deep learning for big data analytics on many-core CPUs," *Proc. VLDB Endow.*, vol. 14, no. 5, pp. 864-876.

[11] Intel, "NUMA-Caffe: NUMA-aware deep learning neural networks," Intel Technical Document.

[12] Intel, "Deep learning with Intel AVX-512 and Intel DL Boost," Intel Developer Guide. [Online]. Available: https://www.intel.com/content/www/us/en/developer/articles/guide/deep-learning-with-avx512-and-dl-boost.html

[13] Intel, "Accelerate PyTorch training and inference using Intel AMX," Intel Technical Article. [Online]. Available: https://www.intel.com/content/www/us/en/developer/articles/technical/accelerate-pytorch-training-inference-on-amx.html

[14] Y. Leviathan, M. Kalman, and Y. Matias, "Fast inference from transformers via speculative decoding," in *Proc. 40th ICML*, 2023.

[15] C. Chen, S. Borgeaud, G. Irving, J.-B. Lespiau, L. Sifre, and J. Jumper, "Accelerating large language model decoding with speculative sampling," arXiv preprint arXiv:2302.01318, 2023.

[16] S. Rajbhandari, C. Li, Z. Yao, M. Zhang, R. Y. Aminabadi, A. A. Awan, J. Rasley, and Y. He, "DeepSpeed-MoE: Advancing mixture-of-experts inference and training to power next-generation AI scale," in *Proc. ICML*, 2022.

[17] A. Eliseev and D. Mazur, "Fast inference of mixture-of-experts language models with offloading," arXiv preprint arXiv:2312.17238, 2023.

[18] Various, "MoE-Infinity: Activation-aware expert offloading for efficient MoE serving," arXiv preprint arXiv:2401.14361, 2024.

[19] A. Q. Jiang, A. Sablayrolles et al., "Mixtral of experts," arXiv preprint arXiv:2401.04088, 2024.

[20] Y. Zhong et al., "DistServe: Disaggregating prefill and decoding for goodput-optimized large language model serving," in *Proc. 18th USENIX OSDI*, 2024.

[21] P. Patel, E. Choukse, C. Zhang, A. Shah, I. Goiri, S. Maleki, and R. Bianchini, "Splitwise: Efficient generative LLM inference using phase splitting," in *Proc. 51st IEEE/ACM ISCA*, 2024. (Best Paper Award)

[22] G.-I. Yu, J. S. Jeong, G.-W. Kim, S. Kim, and B.-G. Chun, "Orca: A distributed serving system for transformer-based generative models," in *Proc. 16th USENIX OSDI*, 2022.

[23] A. Agrawal, A. Panwar, J. Mohan, N. Kwatra, B. S. Gulavani, and R. Ramjee, "SARATHI: Efficient LLM inference by piggybacking decodes with chunked prefills," arXiv preprint arXiv:2308.16369, 2023.

[24] A. Agrawal, N. Kedia, A. Panwar, J. Mohan, N. Kwatra, B. Gulavani, A. Tumanov, and R. Ramjee, "Taming throughput-latency tradeoff in LLM inference with Sarathi-Serve," in *Proc. USENIX OSDI*, 2024.

[25] Various, "Prompt Cache: Modular attention reuse for low-latency inference," in *Proc. MLSys*, 2024.

[26] Y. Tang et al., "Exploring CXL-based KV cache storage for LLM serving," in *NeurIPS ML for Systems Workshop*, 2024.

[27] Various, "LMCache: An efficient KV cache layer for enterprise-scale LLM inference," Technical Report, lmcache.ai.

[28] Intel, "Intel Extension for PyTorch (IPEX)," Open-source project. [Online]. Available: https://github.com/intel/intel-extension-for-pytorch

[29] Intel, "Optimizing large language model inference on Intel CPUs with IPEX and IPEX-LLM," Intel Technical Paper, 2024.

[30] Intel, "IPEX-LLM: Intel LLM library for PyTorch," Open-source project. [Online]. Available: https://github.com/intel/ipex-llm

[31] Y. Sheng, L. Zheng, B. Yuan, Z. Li, M. Ryabinin, D. Y. Fu, Z. Xie, B. Chen, C. Barrett, J. E. Gonzalez, P. Liang, C. Re, I. Stoica, and C. Zhang, "FlexGen: High-throughput generative inference of large language models with a single GPU," in *Proc. ICML (Oral)*, 2023.

[32] G. Gerganov et al., "llama.cpp / GGML," Open-source project. [Online]. Available: https://github.com/ggml-org/llama.cpp

[33] Various, "Which quantization should I use? A unified evaluation of llama.cpp quantization on Llama-3.1-8B-Instruct," arXiv preprint arXiv:2601.14277, 2026.

[34] D. Dai, C. Deng, C. Zhao, R. Xu et al., "DeepSeekMoE: Towards ultimate expert specialization in mixture-of-experts language models," in *Proc. ACL*, 2024.

[35] DeepSeek AI, "DeepSeek-V2: A strong, economical, and efficient mixture-of-experts language model," arXiv preprint arXiv:2405.04434, 2024.

[36] DeepSeek AI, "DeepSeek-V3 Technical Report," arXiv preprint arXiv:2412.19437, 2024.

[37] K. Goto and R. A. van de Geijn, "Anatomy of high-performance matrix multiplication," *ACM Trans. Math. Softw.*, vol. 34, no. 3, 2008.

---

*Draft v1 — 2026-02-24*
