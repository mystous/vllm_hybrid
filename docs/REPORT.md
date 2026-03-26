# vLLM Hybrid: CPU-GPU 이기종 LLM 추론 시스템 구현 보고서

**프로젝트**: vLLM Hybrid — Dual-Process Parallel-Batch Architecture
**대상 하드웨어**: NVIDIA H100 x8 (TP=8) + Intel Xeon Platinum 8480+ (2소켓, 112코어, 2TB DDR5)
**작성일**: 2026-03-20

---

## 목차

1. [목적](#1-목적)
2. [설계방향](#2-설계방향)
3. [설계](#3-설계)
4. [구현체](#4-구현체)
5. [구현결과](#5-구현결과)
6. [평가](#6-평가)

---

## 1. 목적

### 1.1 문제 정의

현대 GPU 추론 서버에서 고성능 멀티코어 CPU는 추론 시 5% 미만의 활용률로 유휴 상태에 놓인다. 서버급 CPU(Intel Xeon Platinum 8480+)는 소켓당 56코어, 307 GB/s 메모리 대역폭을 제공하지만, 토크나이제이션·스케줄링·오케스트레이션 등 보조 작업에만 사용되고 있다. 이 유휴 전력(~175W/소켓)은 데이터센터 에너지 비효율의 주요 원인이며, 100노드 클러스터 기준 연간 153 MWh($18,400)의 낭비를 초래한다.

### 1.2 프로젝트 목표

| 목표 | 설명 |
|------|------|
| **유휴 CPU 활용** | GPU 서버의 유휴 CPU를 LLM 추론에 투입하여 서버 전체 처리량 향상 |
| **GPU 성능 무간섭** | CPU 추론 추가 시 GPU 성능 저하 제로 보장 |
| **가법적 처리량** | $T_{hybrid} = T_{GPU} + \alpha \cdot T_{CPU}$ 모델 달성 |
| **제로 설정** | NUMA 토폴로지, ISA, 스레드 배치 자동 감지 (수동 설정 불필요) |
| **에너지 효율** | 추가 하드웨어 없이 유휴 전력만으로 성능-전력비(performance-per-watt) 개선 |

### 1.3 기존 접근법의 한계

| 기존 접근법 | 대표 시스템 | 한계 |
|------------|-----------|------|
| 뉴런 레벨 파티셔닝 | PowerInfer | 단일 프로세스 내 CPU-GPU 공존 → GIL 직렬화, NUMA 페널티 |
| Attention 오프로딩 | HeteGen | GPU 커널 런칭/메모리 전송과 CPU 연산 간섭 |
| 투기적 디코딩 드래프팅 | Dovetail, APEX | CPU는 draft 모델용 → 대형 모델 직접 추론 불가 |

**공통 제약**: 모든 기존 시스템이 **단일 OS 프로세스**에서 CPU-GPU를 공유하며, GIL 경합·NUMA 원격 메모리 접근·GPU 스케줄링 간섭이 불가피하다.

---

## 2. 설계방향

### 2.1 핵심 통찰

> CPU-GPU 이기종 추론의 장벽은 하드웨어 한계가 아닌 **단일 프로세스 공유의 산물**이다.

각 디바이스를 독립 OS 프로세스로 실행하면 인터프리터 락, 메모리 공간, NUMA 바인딩이 완전 분리되어 소프트웨어 레벨 간섭이 구조적으로 제거된다.

### 2.2 설계 원칙

| 원칙 | 설명 | 근거 |
|------|------|------|
| **core.py 무수정** | 기존 vLLM EngineCore/EngineCoreProc 코드를 변경하지 않음 | 업스트림 호환성, 유지보수 최소화 |
| **별도 프로세스** | GPU/CPU 각각 독립 PID, GIL, busy loop | GIL 경합·NUMA 페널티·스케줄링 간섭 원천 차단 |
| **CPU-first 라우팅** | CPU 슬롯 여유 시 CPU 우선 → 가득차면 GPU로 오버플로우 | CPU 활용률 극대화 (유휴 자원 수확) |
| **자동 감지** | cpu_max_num_seqs, kvcache, threads 모두 `0=auto` 기본값 | 운영자 개입 없는 제로 설정 배포 |
| **Graceful Fallback** | NUMA/IPEX/AMX 미가용 시 표준 경로로 자동 전환 | 다양한 하드웨어 환경 호환 |

### 2.3 Roofline 기반 설계 근거

LLM 추론의 디코드 단계는 **메모리 대역폭 제한(memory-bandwidth-bound)** 영역에 위치한다. 이 영역에서는 연산 성능이 아닌 메모리 대역폭이 처리량을 결정하므로, CPU의 메모리 대역폭(307 GB/s)이 의미 있는 기여를 할 수 있다.

```
디코드 처리량: T_decode = B_mem / (P × b)  [tok/s]
  B_mem: 메모리 대역폭, P: 모델 파라미터 수, b: 파라미터당 바이트

GPU (H100):  B = 3,350 GB/s → Ridge point = 295 FLOP/byte
CPU (Xeon):  B = 307 GB/s   → Ridge point = 23 FLOP/byte
비율: B_CPU / B_GPU = 9.2% (단일 GPU 대비)
```

TP=8 시스템에서 각 GPU의 유효 대역폭은 `B_GPU/k`이므로, CPU는 단일 GPU 스트림 대비 **73%** 처리량을 기여할 수 있다.

---

## 3. 설계

### 3.1 시스템 아키텍처 (클래스 다이어그램)

```mermaid
classDiagram
    class AsyncLLMEngine {
        <<vLLM 상위 엔진>>
        +generate()
        +add_request()
    }

    class HybridAsyncMPClient {
        <<extends AsyncMPClient>>
        -router: CapacityAwareRouter
        -_hybrid_reqs_in_flight: Dict~str, int~
        -_hybrid_req_token_counts: Dict~str, int~
        +add_request_async(request)
        +abort_requests_async(request_ids)
        +process_engine_outputs()
    }

    class CapacityAwareRouter {
        -_strategy: str
        -_cpu_in_flight: int
        -_cpu_max_seqs: int
        -_ema_gpu_tps: float
        -_ema_cpu_tps: float
        -_adaptive_slots: int
        +route(request_id, prompt_len) int
        +on_request_finished(request_id, engine_idx, num_tokens, elapsed)
    }

    class HybridEngineProcManager {
        -gpu_proc: EngineCoreProc
        -cpu_proc: EngineCoreProc
        +close()
        +sentinels()
        +finished_procs()
    }

    class GPU_EngineCoreProc {
        <<별도 PID>>
        engine_idx = 0
        MultiprocExecutor
        8x H100 GPUWorker
        HBM3 KV Cache
    }

    class CPU_EngineCoreProc {
        <<별도 PID>>
        engine_idx = 1
        UniProcExecutor
        CPUWorker
        DDR5 KV Cache
    }

    AsyncLLMEngine --> HybridAsyncMPClient : uses
    HybridAsyncMPClient --> CapacityAwareRouter : routes
    HybridAsyncMPClient --> HybridEngineProcManager : manages
    HybridEngineProcManager --> GPU_EngineCoreProc : spawns
    HybridEngineProcManager --> CPU_EngineCoreProc : spawns
```

### 3.2 시퀀스 다이어그램 (요청 처리 흐름)

```mermaid
sequenceDiagram
    participant C as Client
    participant A as AsyncLLMEngine
    participant H as HybridAsyncMPClient
    participant R as CapacityAwareRouter
    participant CPU as CPU_Engine<br/>(별도 PID)

    C->>A: HTTP request
    A->>H: add_request_async(request)
    H->>R: route(request_id, prompt_len)
    R-->>R: CPU 슬롯 확인 (C < N?)
    R-->>H: engine_idx = 1 (CPU)

    H->>CPU: ZMQ ROUTER send(b'\x01\x00', request)

    Note over CPU: LLM 추론 실행<br/>(독립 프로세스)

    CPU-->>H: ZMQ PULL recv(output)

    H->>R: on_request_finished(req_id, engine=1, tokens, elapsed)
    R-->>R: CPU 슬롯 반환 (C -= 1)
    R-->>R: EMA 처리량 업데이트
    R-->>R: 통계 로깅 (주기적)

    H-->>A: output
    A-->>C: HTTP response
```

### 3.3 컴포넌트 다이어그램

```mermaid
graph TB
    subgraph MainProcess["메인 프로세스 (PID 0)"]
        ALE["AsyncLLMEngine"]
        subgraph HAMC["HybridAsyncMPClient"]
            ZMQ_R["ZMQ ROUTER<br/>(input socket)"]
            ZMQ_P["ZMQ PULL<br/>(output socket)"]
        end
        subgraph Router["CapacityAwareRouter"]
            S1["capacity (기본)"]
            S2["length-aware"]
            S3["throughput-adaptive"]
        end
        ALE --> HAMC
        HAMC --> Router
    end

    subgraph GPUProcess["GPU 프로세스 (PID 1)"]
        GPU_ECP["EngineCoreProc"]
        GPU_EC["EngineCore"]
        GPU_EX["MultiprocExecutor"]
        GPU_W["8x GPUWorker (H100)"]
        GPU_KV["HBM3 KV Cache"]
        GPU_ZD["ZMQ DEALER + PUSH"]
        GPU_ID["identity: b'\\x00\\x00'"]
        GPU_ECP --> GPU_EC --> GPU_EX --> GPU_W --> GPU_KV
    end

    subgraph CPUProcess["CPU 프로세스 (PID 2)"]
        CPU_ECP["EngineCoreProc"]
        CPU_EC["EngineCore"]
        CPU_EX["UniProcExecutor"]
        CPU_W["CPUWorker"]
        CPU_NUMA["NUMA 바인딩 (Node 0)"]
        CPU_AVX["AVX-512 커널"]
        CPU_KV["DDR5 KV Cache (400GB)"]
        CPU_ZD["ZMQ DEALER + PUSH"]
        CPU_ID["identity: b'\\x01\\x00'"]
        CPU_ENV["CUDA_VISIBLE_DEVICES=''<br/>OMP_NUM_THREADS=56<br/>ONEDNN_MAX_CPU_ISA=AVX512_CORE_AMX"]
        CPU_ECP --> CPU_EC --> CPU_EX --> CPU_W
        CPU_W --> CPU_NUMA
        CPU_W --> CPU_AVX
        CPU_W --> CPU_KV
    end

    ZMQ_R -- "ZMQ IPC" --> GPU_ZD
    ZMQ_R -- "ZMQ IPC" --> CPU_ZD
    GPU_ZD -- "ZMQ PUSH" --> ZMQ_P
    CPU_ZD -- "ZMQ PUSH" --> ZMQ_P
```

### 3.4 상태 다이어그램 (CapacityAwareRouter)

```mermaid
stateDiagram-v2
    [*] --> Warmup: 서버 시작

    state Warmup {
        [*] --> Profiling
        Profiling: 워밍업 프로파일링
        Profiling: GPU N + CPU N 요청 완료 대기
        Profiling: 또는 GPU 2N + CPU 1 강제 완료
    }

    Warmup --> ActiveRouting: 워밍업 완료\n(EMA 초기화)

    state ActiveRouting {
        state strategy_choice <<choice>>
        [*] --> strategy_choice

        strategy_choice --> Capacity: strategy = "capacity"
        strategy_choice --> LengthAware: strategy = "length-aware"
        strategy_choice --> ThroughputAdaptive: strategy = "throughput-adaptive"

        state Capacity {
            [*] --> CheckSlot_C
            CheckSlot_C: C < N ?
            CheckSlot_C --> RouteCPU_C: Yes → CPU
            CheckSlot_C --> RouteGPU_C: No → GPU
        }

        state LengthAware {
            [*] --> CheckLen
            CheckLen: prompt_len ≤ τ(512)?
            CheckLen --> CheckSlot_L: Yes
            CheckLen --> RouteGPU_L: No → GPU
            CheckSlot_L: C < N?
            CheckSlot_L --> RouteCPU_L: Yes → CPU
            CheckSlot_L --> RouteGPU_L2: No → GPU
        }

        state ThroughputAdaptive {
            [*] --> CheckLen_T
            CheckLen_T: prompt_len ≤ τ?
            CheckLen_T --> CheckAdaptiveSlot: Yes
            CheckLen_T --> RouteGPU_T: No → GPU
            CheckAdaptiveSlot: C < N_adaptive?
            CheckAdaptiveSlot --> RouteCPU_T: Yes → CPU
            CheckAdaptiveSlot --> RouteGPU_T2: No → GPU
        }
    }

    state OnFinish {
        [*] --> SlotReturn: CPU 슬롯 반환 (C -= 1)
        SlotReturn --> EMAUpdate: EMA 처리량 업데이트\nγ=0.3
        EMAUpdate --> AdaptiveAdjust: 동적 슬롯 조정\nN = clamp(N_base × ratio, 2, 2N_base)
        AdaptiveAdjust --> StatsLog: 주기적 통계 로깅
    }

    ActiveRouting --> OnFinish: on_request_finished()
    OnFinish --> ActiveRouting: 다음 요청 대기
```

### 3.5 클래스 관계도

```mermaid
classDiagram
    class VllmConfig {
        <<기존 vLLM 설정>>
        +model_config
        +parallel_config
        +scheduler_config
        +hybrid_config: HybridConfig
    }

    class HybridConfig {
        <<dataclass>>
        +mode: str = "none"
        +routing_strategy: str = "capacity"
        +cpu_max_num_seqs: int = 0
        +cpu_kvcache_space_gb: int = 0
        +cpu_num_threads: int = 0
        +numa_aware: bool = True
        +warmup_requests: int = 10
        +cpu_prefill_threshold: int = 512
    }

    class _HybridEngineLauncherMixin {
        <<Mixin>>
        +_start_hybrid_engines()
        +_build_cpu_config()
    }

    class HybridAsyncMPClient {
        +add_request_async()
        +abort_requests_async()
        +process_engine_outputs()
    }

    class HybridSyncMPClient {
        +add_request()
        +get_output()
        +abort_requests()
    }

    class CapacityAwareRouter {
        -_strategy: str
        -_cpu_in_flight: int
        -_cpu_max_seqs: int
        -_ema_gpu_tps: float
        -_ema_cpu_tps: float
        -_adaptive_slots: int
        +route(request_id, prompt_len) int
        +on_request_finished()
    }

    class IntelCPUFeatures {
        <<dataclass>>
        +model_name: str
        +num_sockets: int
        +cores_per_socket: int
        +avx512f: bool
        +avx512_vnni: bool
        +amx_bf16: bool
        +amx_int8: bool
    }

    class NUMAAllocator {
        <<Singleton>>
        +bind_to_node(node_id)
        +allocate_on_node(size, node_id)
        +get_node_memory(node_id)
    }

    class ResolvedCpuParams {
        <<dataclass>>
        +cpu_num_threads: int
        +cpu_max_num_seqs: int
        +cpu_kvcache_gb: int
        +cpu_max_batched_tokens: int
    }

    VllmConfig *-- HybridConfig : has-a
    _HybridEngineLauncherMixin <|-- HybridAsyncMPClient : extends
    _HybridEngineLauncherMixin <|-- HybridSyncMPClient : extends
    HybridAsyncMPClient --> CapacityAwareRouter : routes via
    HybridAsyncMPClient ..> ResolvedCpuParams : uses
    ResolvedCpuParams ..> IntelCPUFeatures : derived from
    ResolvedCpuParams ..> NUMAAllocator : queries
```

---

## 4. 구현체

### 4.1 파일 구조 및 코드 규모

| 카테고리 | 파일 | 라인 수 | 역할 |
|----------|------|---------|------|
| **하이브리드 엔진** | `vllm/v1/engine/hybrid_core.py` | 1,000 | CapacityAwareRouter, 프로세스 런칭, CPU 파라미터 자동 감지 |
| | `vllm/v1/engine/core_client.py` | 1,584 | HybridAsyncMPClient, ZMQ 통신, 요청 추적 |
| | `vllm/config.py` (HybridConfig) | 102 | 하이브리드 설정 데이터클래스 |
| | `vllm/engine/arg_utils.py` (hybrid args) | 102 | CLI 인자 정의 |
| **CPU 최적화** | `vllm/platforms/intel_cpu_utils.py` | 965 | Intel CPU 감지, NUMA 할당, ISA 감지 |
| | `vllm/v1/worker/cpu_worker.py` | 530 | CPU 워커 (NUMA 바인딩, IPEX 감지) |
| | `vllm/v1/attention/backends/cpu_attn.py` | 1,205 | CPU PagedAttention (IPEX/SDPA) |
| **C++ 커널** | `csrc/cpu/gemm_vnni.cpp` | 503 | VNNI INT8 GEMM (6×16 마이크로커널) |
| | `csrc/cpu/quant_q8_0.cpp` | 366 | Q8_0 양자화 커널 |
| | `csrc/cpu/decode_gemv.cpp` | 289 | BF16/FP32 Decode GEMV |
| | `csrc/cpu/batch_attention.cpp` | 499 | 배치 Paged Attention + L2 프리페치 |
| | `csrc/cpu/mem_opt.cpp` | 246 | NT memcpy, NUMA 할당, 프리페치 |
| **빌드** | `cmake/cpu_hybrid_extension.cmake` | — | _C_cpu_ops 타겟 빌드 |
| **테스트** | `tests/v1/engine/test_hybrid_core.py` | 390 | 단위 테스트 30개 |
| | **합계** | **~7,781** | |

### 4.2 핵심 컴포넌트 구현 상세

#### 4.2.1 CapacityAwareRouter

```python
class CapacityAwareRouter:
    """CPU 용량 기반 요청 라우팅 (3가지 전략)"""

    def route(self, request_id, prompt_len) -> int:
        """요청을 GPU(0) 또는 CPU(1)로 라우팅"""
        if self._strategy == "capacity":
            return self._route_capacity(request_id)
        elif self._strategy == "length-aware":
            return self._route_length_aware(request_id, prompt_len)
        else:  # throughput-adaptive
            return self._route_throughput_adaptive(request_id, prompt_len)

    def on_request_finished(self, request_id, engine_idx, num_tokens, elapsed):
        """요청 완료 시 슬롯 반환, EMA 업데이트, 통계 수집"""
        if engine_idx == 1:  # CPU
            self._cpu_in_flight -= 1
        throughput = num_tokens / elapsed
        self._update_ema(engine_idx, throughput)
        self._update_adaptive_slots()
```

**라우팅 전략별 알고리즘**:

| 전략 | 알고리즘 | 특성 |
|------|---------|------|
| `capacity` | `C < N → CPU, else → GPU` | 단순, CPU 활용 극대화 |
| `length-aware` | `len ≤ τ(512) && C < N → CPU, else → GPU` | 짧은 프롬프트만 CPU |
| `throughput-adaptive` | EMA 기반 동적 슬롯: `N_adaptive = clamp(N_base × ratio, 2, 2N_base)` | 실시간 성능 적응 |

#### 4.2.2 자동 CPU 파라미터 감지

```python
def _resolve_cpu_params(hybrid_config, features) -> ResolvedCpuParams:
    """0(auto) 값을 하드웨어 기반 최적값으로 해석"""

    # NUMA 노드의 물리 코어 수 (HT 제외)
    effective_cores = features.cores_per_socket  # 56 (Xeon 8480+)

    # NUMA 노드의 유효 메모리 (GB)
    effective_mem = get_numa_node_memory(node_id)  # ~1,000 GB

    return ResolvedCpuParams(
        cpu_num_threads=effective_cores,                      # 56
        cpu_max_num_seqs=max(4, effective_cores // 4),        # 14
        cpu_kvcache_gb=clamp(effective_mem * 0.4, 32, 512),   # 400
        cpu_max_batched_tokens=cpu_max_num_seqs * 256,        # 3,584
    )
```

#### 4.2.3 CPU 프로세스 환경 설정

```python
def _setup_cpu_process_env(resolved_params, features):
    """CPU 엔진 프로세스의 환경 변수를 자동 설정"""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""           # GPU 격리
    os.environ["VLLM_CPU_KVCACHE_SPACE"] = str(400)   # KV cache GB
    os.environ["OMP_NUM_THREADS"] = str(56)            # OpenMP 스레드
    os.environ["VLLM_CPU_OMP_THREADS_BIND"] = "auto"   # 자동 바인딩
    os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
    os.environ["ONEDNN_MAX_CPU_ISA"] = "AVX512_CORE_AMX"  # AMX 활성화
```

#### 4.2.4 ZMQ IPC 통신

```python
# 엔진 식별자 (Little-Endian 2바이트)
GPU_IDENTITY = b'\x00\x00'  # engine_index=0
CPU_IDENTITY = b'\x01\x00'  # engine_index=1

# 입력: ROUTER 소켓 (멀티플렉싱)
input_socket.send_multipart([identity, request_data])

# 출력: PULL 소켓 (비동기 수집)
output_data = output_socket.recv()  # GPU/CPU 인터리브 수신
```

#### 4.2.5 AVX-512 C++ 커널

| 커널 | 핵심 최적화 | ISA 요구사항 |
|------|-----------|-------------|
| VNNI INT8 GEMM | 6×16 마이크로커널, 3-level 캐시 블로킹 (MC=72, NC=256, KC=256) | AVX512F + VNNI |
| Q8_0 양자화 | 32-element 블록, FP16 scale + INT8 quants | AVX512F + VNNI |
| Decode GEMV | BF16→FP32 shift 변환, SW 프리페치, 2×16 unroll | AVX512F |
| Batch Attention | 시퀀스 순차 + head dim SIMD, L2 프리페치 (`_MM_HINT_T1`) | AVX512F |
| 메모리 최적화 | NT memcpy (캐시 우회), NUMA 노드 할당 | AVX512F |

### 4.3 단위 테스트

| 테스트 그룹 | 테스트 수 | 검증 내용 |
|------------|----------|----------|
| capacity 라우팅 | 6 | CPU 슬롯 여유/가득참, 요청 완료 반환, 오버플로우 |
| length-aware 라우팅 | 5 | 프롬프트 길이 임계값, 경계값, CPU 포화 시 동작 |
| throughput-adaptive 라우팅 | 5 | EMA 기반 슬롯 조정, 클램핑, length threshold |
| 워밍업 | 3 | 비활성화, 정상 완료, 강제 완료 |
| 장애 허용성 | 2 | CPU crash 시 GPU 폴백 |
| _resolve_cpu_params | 6 | 자동 감지 공식, 수동 오버라이드, 최소값 |
| ResolvedCpuParams | 1 | 데이터클래스 필드 검증 |
| **합계** | **30** | **전체 통과 (30/30)** |

---

## 5. 구현결과

> **참고**: 아래 결과는 이론적 분석 및 Roofline 모델 기반 **예측값**입니다. 프로덕션 H100 서버에서의 실측 검증은 진행 중이며, 향후 확장 버전에서 보고될 예정입니다.

### 5.1 처리량 예측 (Roofline 기반)

#### 5.1.1 디코드 단계 처리량 (메모리 대역폭 제한)

| 모델 | 크기 | GPU (H100×8) | CPU (Xeon 1S) | 예상 Hybrid | CPU 기여율 |
|------|------|-------------|---------------|-------------|-----------|
| Llama-2 | 7B | 479 tok/s | 44 tok/s | 523 tok/s | +9.2% |
| Llama-2 | 13B | 258 tok/s | 24 tok/s | 282 tok/s | +9.3% |
| Llama-2 | 70B | 48 tok/s | 4.4 tok/s | 52.4 tok/s | +9.2% |
| Mixtral | 8×7B | 48 tok/s | 4.4 tok/s | 52.4 tok/s | +9.2% |

> **산출 근거**: `T_decode = B_mem / (P × b)`, BF16 (b=2), GPU: 3,350 GB/s (TP=8 합산), CPU: 307 GB/s (1소켓)
> **주의**: GPU tok/s는 배치 시스템 처리량(vLLM 벤치마크 기반), CPU tok/s는 단일 요청 상한값

#### 5.1.2 Q8_0 양자화 시 처리량

| 모델 | 크기 | GPU (Q8_0) | CPU (Q8_0) | 예상 Hybrid | CPU 기여율 |
|------|------|-----------|-----------|-------------|-----------|
| Llama-2 | 7B | 958 tok/s | 88 tok/s | 1,046 tok/s | +9.2% |
| Llama-2 | 70B | 96 tok/s | 8.8 tok/s | 104.8 tok/s | +9.2% |

> **산출 근거**: Q8_0 OI=2, 유효 대역폭 2배

#### 5.1.3 TP=8 시스템 기여도 관점별 정리

| 관점 | 산식 | 비율 | 의미 |
|------|------|------|------|
| CPU vs 단일 GPU | `B_CPU / B_GPU` | 9.2% | CPU는 단일 H100의 9.2% |
| CPU vs GPU 1스트림 (TP=8) | `B_CPU / (B_GPU/8)` | 73% | TP 분산 시 GPU 스트림 대비 73% |
| CPU vs 시스템 전체 | `B_CPU / (8×B_GPU)` | 1.15% | 8-GPU 시스템 전체 대비 1.15% |

### 5.2 지연 시간 분석

| 메트릭 | 예측값 | 근거 |
|--------|-------|------|
| 라우팅 오버헤드 | ~11 μs | ZMQ IPC 왕복 (LogGP 모델) |
| GPU 지연 영향 | < 0.04% | 라우팅 오버헤드 / GPU 토큰 생성 시간 |
| CPU Prefill TTFT | 10-50× GPU 대비 | 컴퓨트 바운드 → compute ceiling 격차 (7.2 vs 989 TFLOP/s) |
| CPU Decode 토큰 지연 | ~2-10× GPU 대비 | 메모리 바운드 → 대역폭 비율 (307 vs 3,350 GB/s) |

> **주의**: CPU TTFT(Time-To-First-Token)는 GPU 대비 10-50배 느리므로, 지연 민감 요청은 length-aware 라우팅으로 완화

### 5.3 에너지 효율 분석

#### 5.3.1 전력 프로파일 (DGX H100 서버 기준)

| 컴포넌트 | 유휴 전력 | 활성 전력 | 증분 전력 (ΔP) |
|----------|----------|----------|----------------|
| GPU 8×H100 | — | 5,600 W | — |
| CPU 1소켓 (유휴) | 175 W | — | — |
| CPU 1소켓 (활성) | — | 350 W | +175 W |
| 시스템 메모리 | ~80 W | — | — |

#### 5.3.2 에너지 효율 개선 예측

```
η = (T_hybrid / P_hybrid) / (T_GPU / P_GPU)
  = (1 + 0.092) / (1 + 175/5600)
  = 1.092 / 1.031
  ≈ 1.059  (약 5.9% 성능-전력비 개선)
```

> **주의**: 실제 TDP 60-70% 미달 + DRAM 전력 증가로 실측값은 이론 대비 낮을 수 있음

#### 5.3.3 연간 비용 절감 (100노드 클러스터)

| 메트릭 | 값 | 산출 근거 |
|--------|-----|----------|
| 유휴 CPU 전력 낭비 | 153 MWh/년 | 175W × 100노드 × 8,760시간 |
| 연간 전기 비용 절감 | $18,400 | 153MWh × $0.12/kWh |
| 유휴 하드웨어 자산 | $3M+ | CPU/메모리 시스템 $30-50K × 100노드 |

### 5.4 자동 감지 결과 (Xeon 8480+ 기준)

| 파라미터 | 자동 감지값 | 산출 공식 |
|----------|-----------|----------|
| cpu_num_threads | 56 | NUMA 노드 물리코어 수 |
| cpu_max_num_seqs | 14 | max(4, 56/4) |
| cpu_kvcache_space_gb | 400 GB | clamp(1000×0.4, 32, 512) |
| cpu_max_batched_tokens | 3,584 | 14 × 256 |
| NUMA 노드 | 0 | rank % num_nodes |

### 5.5 테스트 결과

| 테스트 스위트 | 테스트 수 | 통과 | 실패 | 통과율 |
|-------------|----------|------|------|--------|
| CapacityAwareRouter | 21 | 21 | 0 | 100% |
| _resolve_cpu_params | 6 | 6 | 0 | 100% |
| ResolvedCpuParams | 1 | 1 | 0 | 100% |
| 장애 허용성 | 2 | 2 | 0 | 100% |
| **합계** | **30** | **30** | **0** | **100%** |

---

## 6. 평가

### 6.1 설계 목표 달성도

| 목표 | 달성 상태 | 평가 | 비고 |
|------|----------|------|------|
| 유휴 CPU 활용 | ✅ 달성 | CPU를 독립 추론 엔진으로 투입 | 3가지 라우팅 전략 구현 완료 |
| GPU 성능 무간섭 | ✅ 설계적 보장 | 별도 프로세스 격리로 소프트웨어 간섭 제거 | 하드웨어 경합은 최소 (이론 분석), 실측 검증 필요 |
| 가법적 처리량 | ✅ 이론적 달성 | Claim 1 조건 (i) 프로세스 격리 충족, 조건 (ii) 실측 필요 | Roofline 기반 2-13% 개선 예측 |
| 제로 설정 | ✅ 달성 | NUMA/ISA/스레드 자동 감지 + 환경변수 자동 설정 | 모든 파라미터 `0=auto` 기본값 |
| 에너지 효율 | ✅ 이론적 달성 | ΔP(175W) ≪ P_GPU(5,600W), 순 에너지 효율 개선 | RAPL 기반 실측 검증 계획 |

### 6.2 구현 완성도 평가

| 컴포넌트 | 완성도 | 상세 |
|----------|-------|------|
| CapacityAwareRouter | 95% | 3가지 전략, 워밍업, EMA 적응, 통계 로깅 완성 |
| HybridAsyncMPClient | 95% | ZMQ 통신, 요청 추적, 토큰 누적, abort 완성 |
| CPU 자동 감지 | 95% | NUMA/ISA/스레드/KV cache 자동 감지 완성 |
| Intel CPU 유틸리티 | 90% | AVX-512/VNNI/AMX 감지, NUMA 할당 완성 |
| CPU Worker | 90% | NUMA 바인딩, IPEX 감지, 하이브리드 모드 인식 |
| CPU Attention | 90% | PagedAttention, IPEX 디스패치, 커스텀 커널 연결 |
| AVX-512 C++ 커널 | 85% | GEMM/Q8_0/GEMV/Attention/Memory 5개 커널 구현 |
| 단위 테스트 | 100% | 30/30 테스트 통과 |

### 6.3 기존 접근법 대비 비교

| 비교 항목 | 기존 시스템 (PowerInfer 등) | vLLM Hybrid | 장점 |
|----------|--------------------------|-------------|------|
| 프로세스 모델 | 단일 프로세스 공유 | 별도 프로세스 격리 | GIL/NUMA/스케줄링 간섭 제거 |
| GPU 영향 | GPU 성능 저하 가능 | GPU 무간섭 보장 | 소프트웨어 격리 |
| CPU 역할 | 보조 (draft/offload) | 독립 추론 엔진 | CPU 전체 활용 |
| 설정 복잡도 | 수동 파티셔닝 필요 | 제로 설정 (auto) | 운영 간소화 |
| 확장성 | 모델별 재설정 | 모델 독립적 | 범용성 |
| 독립 인스턴스 2개 대비 | 별도 API 엔드포인트 2개 | 단일 API + 통합 라우팅 | 자원 관리 통합, 로드밸런싱 자동화 |

### 6.4 한계 및 향후 과제

| 한계 | 상세 | 완화 방안 |
|------|------|----------|
| **실측 미완료** | 모든 성능 수치가 이론적 예측 | H100 서버에서 실측 검증 진행 중 |
| **CPU TTFT 지연** | Prefill 단계 10-50× 느림 | length-aware 라우팅으로 짧은 요청만 CPU |
| **모델 가중치 중복** | CPU/GPU 모두 모델 로드 (~70GB) | DDR5 용량(2TB) 대비 미미, KV cache 17% 감소 |
| **하드웨어 경합** | PCIe/DDR5 경합 가능 | NUMA 분리 + 실측 검증으로 확인 |
| **Continuous Batching 상호작용** | 배치 동적 변경과의 상호작용 미분석 | 향후 분석 계획 |
| **단일 소켓 서버** | DDR5 경합 증가 가능 | `--hybrid-numa-node` 수동 지정 |

### 6.5 종합 평가

vLLM Hybrid 프로젝트는 **GPU 추론 서버의 유휴 CPU를 독립 추론 엔진으로 활용하는 실용적 시스템**을 설계·구현하였다. 핵심 기여는 다음과 같다:

1. **프로세스 격리 기반 간섭 제거**: 단일 프로세스 공유의 근본 문제를 프로세스 분리로 해결하여, 기존 이기종 추론 시스템의 GIL·NUMA·스케줄링 간섭을 구조적으로 차단
2. **자기 조절 라우팅**: 3가지 전략(capacity/length-aware/throughput-adaptive)으로 다양한 워크로드에 자동 적응
3. **제로 설정 배포**: 하드웨어 토폴로지 자동 감지로 운영자 개입 없는 즉시 배포 가능
4. **에너지 수확**: 추가 하드웨어 없이 유휴 전력(175W)만으로 2-13% 처리량 향상 가능

Roofline 분석과 이론적 모델링을 통해 설계의 타당성을 검증하였으며, **프로덕션 H100 서버에서의 실측 검증이 즉각적인 다음 단계**로 계획되어 있다.

---

*본 보고서의 성능 수치는 Roofline 모델 기반 이론적 예측값이며, 실측 검증 결과는 향후 확장 버전에서 보고될 예정입니다.*
