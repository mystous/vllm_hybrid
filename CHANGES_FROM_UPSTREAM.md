# vLLM Hybrid: Upstream 대비 수정사항 및 기술 해설

> **Fork 기준점**: `3303f134e` — vLLM upstream (2025년 중반)
> **현재 HEAD**: `cd577b76d`
> **총 커밋**: 49개 (사용자 커스텀)
> **총 변경**: 117개 파일, +30,149 / -383 라인
> **타겟 하드웨어**: NVIDIA H100 x8 + Intel Xeon 8480+ (2S/112C/2T, 2TB DDR5)

---

## 목차

1. [변경 요약](#1-변경-요약)
2. [A. 하이브리드 엔진 (Dual-Process Parallel-Batch)](#a-하이브리드-엔진-dual-process-parallel-batch)
3. [B. Intel CPU 최적화 플랫폼](#b-intel-cpu-최적화-플랫폼)
4. [C. CPU 워커 및 모델 러너](#c-cpu-워커-및-모델-러너)
5. [D. CPU Attention 백엔드](#d-cpu-attention-백엔드)
6. [E. AVX-512 C++ 커널](#e-avx-512-c-커널)
7. [F. 분산 통신 (Heterogeneous)](#f-분산-통신-heterogeneous)
8. [G. Executor 및 Worker (V0 레거시)](#g-executor-및-worker-v0-레거시)
9. [H. 설정 및 CLI](#h-설정-및-cli)
10. [I. 미래용 컴포넌트](#i-미래용-컴포넌트)
11. [J. 빌드 시스템](#j-빌드-시스템)
12. [K. 기타 수정](#k-기타-수정)
13. [전체 파일 목록](#전체-파일-목록)

---

## 1. 변경 요약

### 아키텍처 변경

```
[Upstream vLLM]                        [vLLM Hybrid]
GPU-only EngineCore ─────→ GPU EngineCoreProc (별도 프로세스, PID A)
                           + CPU EngineCoreProc (별도 프로세스, PID B)
                           + CapacityAwareRouter (CPU 슬롯 기반)
                           + 자동 CPU 파라미터 감지 (NUMA/HT-aware)
                           + Intel AVX-512/AMX/IPEX 최적화 자동 적용
```

### 카테고리별 규모

| 카테고리 | 신규 파일 | 수정 파일 | 추가 라인 | 핵심도 |
|----------|----------|----------|----------|--------|
| A. 하이브리드 엔진 | 1 | 2 | ~1,245 | **핵심** |
| B. Intel CPU 플랫폼 | 2 | 3 | ~1,579 | **핵심** |
| C. CPU 워커/러너 | 0 | 2 | ~645 | **핵심** |
| D. CPU Attention | 1 | 1 | ~531 | **핵심** |
| E. AVX-512 C++ 커널 | 12 | 0 | ~2,452 | 중요 |
| F. 분산 통신 | 0 | 5 | ~327 | 레거시 |
| G. Executor/Worker (V0) | 1 | 6 | ~1,304 | 레거시 |
| H. 설정/CLI | 0 | 2 | ~221 | **핵심** |
| I. 미래용 컴포넌트 | 5 | 1 | ~2,378 | 보류 |
| J. 빌드 시스템 | 2 | 2 | ~185 | 중요 |
| K. 기타 수정 | 0 | 5 | ~39 | 보조 |

---

## A. 하이브리드 엔진 (Dual-Process Parallel-Batch)

### 기술 배경: 왜 별도 프로세스인가

#### Python GIL(Global Interpreter Lock) 문제

Python은 GIL로 인해 한 프로세스 내에서 동시에 하나의 스레드만 Python 바이트코드를 실행할 수 있다. vLLM의 `EngineCoreProc`는 busy loop 방식으로 `while True: poll → schedule → execute → push`를 반복하는데, GPU 엔진과 CPU 엔진이 같은 프로세스에 있으면:

```
같은 프로세스 (순차 실행):
T_total = T_GPU_step + T_CPU_step    ← GIL 경합으로 병렬화 불가

별도 프로세스 (진정한 병렬):
T_total = max(T_GPU_step, T_CPU_step)  ← 각 프로세스가 독립 GIL 보유
Throughput = GPU_throughput + CPU_throughput
```

#### ZMQ IPC 통신 패턴

vLLM V1 엔진은 이미 ZMQ(ZeroMQ) 기반 IPC(Inter-Process Communication)를 사용한다:

- **ROUTER/DEALER 패턴**: 클라이언트 → 엔진 요청 전송. ROUTER 소켓은 identity 기반 라우팅을 지원하므로, 동일 소켓에 여러 엔진을 연결 가능
- **PUSH/PULL 패턴**: 엔진 → 클라이언트 결과 전송. 여러 PUSH 소켓이 하나의 PULL 소켓에 비동기로 결과를 보내는 fan-in 구조

이 기존 인프라를 그대로 활용하여, GPU 엔진(identity=`b'\x00\x00'`)과 CPU 엔진(identity=`b'\x01\x00'`)을 하나의 ROUTER 소켓에 연결한다.

#### CapacityAwareRouter vs RequestRouter

**RequestRouter** (초기 구현): `cpu_ratio` 기반 라운드로빈. 매 `1/cpu_ratio`번째 요청을 CPU로 보냄.
- 문제: CPU 처리 속도를 모르면 비율 설정이 어려움. CPU가 느려서 요청이 쌓여도 계속 보냄.

**CapacityAwareRouter** (최종 구현): CPU의 현재 in-flight 요청 수 기반.
```python
if cpu_in_flight < cpu_max_num_seqs:
    cpu_in_flight += 1
    return "cpu"    # CPU에 여유 있으면 CPU로
return "gpu"        # 가득 차면 GPU로
```
- 장점: CPU 처리량을 모르는 상태에서도 최적 동작. CPU가 빠르면 많이 받고, 느리면 적게 받음 → **CPU 활용률 항상 100%**
- `on_request_finished()` 콜백으로 슬롯 반환

### 수정 파일

#### `vllm/v1/engine/hybrid_core.py` — **신규** (+765줄)

| 클래스/함수 | 역할 |
|------------|------|
| `is_hybrid_mode()` | VllmConfig가 hybrid parallel-batch 모드인지 판별 |
| `compute_auto_cpu_ratio()` | CPU/GPU 처리량 기반 최적 cpu_ratio 계산 (R = T_cpu / (T_gpu + T_cpu)) |
| `RequestRouter` | cpu_ratio 기반 라운드로빈 분배기 (레거시) |
| `CapacityAwareRouter` | CPU 슬롯 기반 분배기 — 여유 있으면 CPU, 없으면 GPU |
| `ResolvedCpuParams` | 자동 감지된 CPU 파라미터 (dataclass) |
| `_resolve_cpu_params()` | NUMA 토폴로지 기반 자동 파라미터 감지 (아래 상세) |
| `_setup_cpu_process_env()` | CPU 프로세스 환경 설정 — Intel 최적화, NUMA affinity |
| `_create_cpu_vllm_config()` | GPU VllmConfig에서 CPU용 config 안전 파생 |
| `run_cpu_engine_core()` | CPU 프로세스 진입점 (CUDA_VISIBLE_DEVICES="" 차단) |
| `HybridEngineProcManager` | GPU+CPU 프로세스 관리 (sentinels, shutdown) |
| `launch_hybrid_engines()` | GPU+CPU 프로세스 스폰 컨텍스트 매니저 |

**`_resolve_cpu_params()` 자동 감지 로직**:
```
psutil.cpu_count(logical=False)          → 물리 코어 수 (HT 제외)
psutil.virtual_memory().total             → 총 메모리
NUMAAllocator.get_node_info(node)         → NUMA 노드별 CPU 목록, 메모리
detect_intel_cpu_features().threads_per_core → HT 배수

effective_cores = numa_node_cores / threads_per_core   ← 핵심: HT 제거
cpu_num_threads     = effective_cores
cpu_max_num_seqs    = effective_cores / 4   (최소 4)
cpu_kvcache_space   = effective_mem * 0.4   (32~512GB)
cpu_max_batched_tokens = max_seqs * 256
```

**`_create_cpu_vllm_config()` 파생 로직**: GPU의 VllmConfig를 `copy.deepcopy`한 뒤 다음을 변경:
- `DeviceConfig`: `"cpu"` 명시
- `ParallelConfig`: TP=1, PP=1, 단일 프로세스 (`UniProcExecutor`)
- `CacheConfig`: CPU KV cache 크기
- `SchedulerConfig`: CPU 동시 시퀀스/토큰 제한
- `CompilationConfig`: CUDA graph 비활성화 (CpuPlatform이 DYNAMO_ONCE 설정)
- `HybridConfig`: None (CPU 엔진 내부에서는 hybrid 미사용)

#### `vllm/v1/engine/core_client.py` — 수정 (+274줄)

| 클래스/함수 | 변경 내용 |
|------------|----------|
| `MPClient._create_engine_launcher()` | **신규** 오버라이드 훅 — Hybrid에서 `launch_hybrid_engines()` 호출 |
| `MPClient._compute_core_engines()` | **신규** 오버라이드 훅 — 2개 CoreEngine (GPU+CPU) 반환 |
| `_HybridEngineLauncherMixin` | **신규** — GPU+CPU 프로세스 스폰 Mixin |
| `HybridAsyncMPClient` | **신규** — CapacityAwareRouter, `_hybrid_reqs_in_flight` 추적 |
| `HybridSyncMPClient` | **신규** — 동기 버전 |

**`HybridAsyncMPClient` 핵심 오버라이드**:
- `add_request_async()`: `router.route()` → identity 기반 ZMQ 전송
- `process_engine_outputs()`: `finished_requests`에서 CPU 슬롯 반환 (`on_request_finished`)
- `abort()`: CPU/GPU 양쪽 abort + in-flight 정리

#### `vllm/v1/engine/core.py` — 수정 (+5줄, -15줄)

hybrid 관련 코드 **완전 제거**:
- `run_engine_core()`의 hybrid 분기 제거
- `_handle_client_request()`의 None 체크 제거
- `process_input_sockets()`의 hybrid 라우팅 제거

**설계 원칙**: core.py는 upstream과 최소 차이를 유지. hybrid 로직은 hybrid_core.py와 core_client.py에만 존재.

---

## B. Intel CPU 최적화 플랫폼

### 기술 배경: Intel Xeon CPU의 LLM 추론 가속 기술

#### AVX-512 (Advanced Vector Extensions 512-bit)

AVX-512는 Intel Xeon Scalable Processor (Skylake-SP 이후)에서 지원하는 512비트 SIMD 명령어 집합이다.

| 확장 | 설명 | 활용 |
|------|------|------|
| **AVX-512F** | Foundation — 512비트 정수/실수 벡터 연산 | FP32 GEMM, 벡터 연산 |
| **AVX-512BW** | Byte/Word — 8/16비트 정수 연산 | INT8/BF16 변환 |
| **AVX-512DQ** | Doubleword/Quadword — 32/64비트 정수 확장 | 인덱싱, 마스크 |
| **AVX-512 VNNI** | Vector Neural Network Instructions | **INT8 dot product** (vpdpbusd) |
| **AVX-512 BF16** | BFloat16 변환/연산 | BF16 GEMM (vcvtne2ps2bf16) |

**AVX-512 VNNI**의 `vpdpbusd` 명령어는 한 사이클에 `uint8 × int8` 점곱을 4쌍 계산하여 INT32에 누적한다. 이것이 INT8 양자화 모델의 추론 속도를 크게 향상시키는 핵심이다.

```
vpdpbusd zmm0, zmm1, zmm2
// zmm0[i] += sum(zmm1[i*4+j] * zmm2[i*4+j]) for j=0..3
// 한 번에 16 × 4 = 64개의 uint8 × int8 곱셈
```

#### AMX (Advanced Matrix Extensions)

AMX는 Sapphire Rapids (4세대 Xeon, 2023~) 이후에서 지원하는 **타일 기반 행렬 곱셈 가속기**이다.

```
AMX 구성:
- 8개의 Tile 레지스터 (TMM0~TMM7): 각 1KB (16행 × 64바이트)
- TMUL (Tile Matrix Multiply Unit): 하드웨어 행렬 곱셈 유닛
- 지원 데이터: BF16 (AMX-BF16), INT8 (AMX-INT8)

성능 비교:
| 세대            | 명령어     | INT8 연산/사이클 |
|-----------------|-----------|-----------------|
| Ice Lake (3세대) | AVX-512   | 256 ops/cycle   |
| Sapphire (4세대) | AMX       | 2,048 ops/cycle | ← 8배 향상
```

AMX를 사용하려면:
1. **커널 지원**: Linux 5.16+ 필요 (XSTATE에 AMX 타일 포함)
2. **권한 요청**: `ARCH_REQ_XCOMP_PERM` syscall로 AMX 타일 사용 권한 획득
3. **oneDNN 설정**: `ONEDNN_MAX_CPU_ISA=AVX512_CORE_AMX`로 AMX 커널 활성화
4. **IPEX 활성화**: Intel Extension for PyTorch에서 AMX BF16 matmul 활용

#### IPEX (Intel Extension for PyTorch)

IPEX는 Intel이 제공하는 PyTorch 확장으로, CPU 추론에 최적화된 커널을 제공한다:

- **PagedAttention CPU 커널**: vLLM의 KV cache paged attention을 CPU에서 최적 실행
- **BF16 자동 변환**: `ipex.optimize(model, dtype=torch.bfloat16)`
- **Linear 융합**: Attention Q/K/V projection 융합
- **AMX 자동 활용**: oneDNN 백엔드를 통해 AMX BF16/INT8 자동 사용

IPEX 미설치 시 CPU decode는 Python 루프 기반 `F.scaled_dot_product_attention()`으로 fallback되어 **수십 배 느려진다**.

#### NUMA (Non-Uniform Memory Access)

멀티소켓 서버에서 각 CPU 소켓은 자신의 로컬 메모리에 접근할 때 가장 빠르다:

```
Xeon 8480+ (2소켓) NUMA 토폴로지:

       Socket 0                    Socket 1
  ┌──────────────┐           ┌──────────────┐
  │  56 cores    │           │  56 cores    │
  │  (112 HT)   │           │  (112 HT)    │
  └──────┬───────┘           └──────┬───────┘
         │ DDR5                     │ DDR5
  ┌──────┴───────┐           ┌──────┴───────┐
  │  ~1TB RAM    │◄══UPI══►│  ~1TB RAM    │
  │  (NUMA 0)    │  ~100GB/s │  (NUMA 1)    │
  └──────────────┘           └──────────────┘

로컬 접근: ~100ns, ~200GB/s (DDR5-4800)
원격 접근: ~200ns, ~100GB/s (UPI 경유)  ← 2배 느림
```

따라서 CPU 프로세스는:
1. **메모리 바인딩**: `numa_set_preferred(node)` → 할당되는 메모리가 해당 노드에 위치
2. **스레드 바인딩**: `KMP_AFFINITY` → OpenMP 스레드가 해당 노드의 코어에서만 실행
3. **두 바인딩이 같은 노드**: 메모리/스레드 노드 불일치 시 모든 메모리 접근이 UPI 경유 → 대역폭 절반

#### Hyper-Threading (HT) 주의사항

Xeon 8480+는 코어당 2개의 논리 스레드(HT)를 제공한다:
- 물리 코어 56개 × 2 = 논리 CPU 112개 (소켓당)
- `numactl --hardware`의 `cpu_ids`는 **논리 CPU** 목록 반환
- LLM 추론은 ALU/FPU 집약적이므로 HT 스레드가 성능을 크게 높이지 못함
- OpenMP 스레드 수를 **물리 코어 수**로 설정해야 최적 (`logical / threads_per_core`)

### 수정 파일

#### `vllm/platforms/intel_cpu_utils.py` — **신규** (+965줄)

| 클래스/함수 | 역할 |
|------------|------|
| `IntelCPUFeatures` (dataclass) | CPU 기능 플래그 및 토폴로지 (sockets, cores_per_socket, threads_per_core, avx512f, vnni, amx 등) |
| `detect_intel_cpu_features()` | `/proc/cpuinfo` + `lscpu` 파싱으로 CPU 기능 감지. `amx_bf16`/`amx-bf16` 양 형식 지원 |
| `NUMANodeInfo` (dataclass) | NUMA 노드별 정보 (cpu_ids, total_memory_bytes) |
| `NUMAAllocator` (singleton) | libnuma ctypes 바인딩 — `numa_set_preferred()`, `numa_alloc_onnode()`, interleave 모드 |
| `create_numa_aware_tensor()` | 지정 NUMA 노드에 PyTorch 텐서 할당 (custom allocator) |
| `is_ipex_available()` | IPEX import 가능 여부 확인 |
| `optimize_model_with_ipex()` | `ipex.optimize(model, dtype=bfloat16)` 래핑 |
| `configure_intel_optimizations()` | 환경변수 설정: KMP_AFFINITY, KMP_BLOCKTIME, MKL_ENABLE_INSTRUCTIONS, ONEDNN_MAX_CPU_ISA |
| `configure_pytorch_for_intel()` | PyTorch 런타임 설정: `torch.set_num_threads()`, Inductor config, AMX 타일 권한 |
| `setup_intel_cpu_environment()` | 통합 진입점 — NUMA + AVX-512 + AMX + IPEX 전체 설정 |

**`configure_intel_optimizations()` 환경변수 자동 설정**:
```bash
KMP_AFFINITY=granularity=fine,compact,1,0   # 스레드를 연속 코어에 밀집 배치
KMP_BLOCKTIME=1                              # 스레드 유휴 대기 1ms (busy wait 최소화)
KMP_TPAUSE=0                                 # tpause 비활성화 (latency 최소)
MKL_ENABLE_INSTRUCTIONS=AVX512               # MKL에서 AVX-512 명령어 사용
ONEDNN_MAX_CPU_ISA=AVX512_CORE_AMX           # AMX 지원 시 (최우선)
                   AVX512_CORE_VNNI          # VNNI만 지원 시
                   AVX512_CORE               # 기본 AVX-512
```

**`KMP_AFFINITY=granularity=fine,compact,1,0` 의미**:
- `granularity=fine`: 각 OpenMP 스레드가 단일 논리 프로세서에 바인딩
- `compact`: 같은 코어/소켓에 스레드를 밀집 배치 (L1/L2 캐시 공유 극대화)
- `1,0`: 오프셋 파라미터 (첫 번째 코어부터 시작)

#### `vllm/platforms/heterogeneous.py` — **신규** (+441줄)

GPU+CPU 이기종 플랫폼 추상화 (V0 heterogeneous pipeline 모드용).

| 클래스 | 역할 |
|--------|------|
| `HeterogeneousPlatform` | `CudaPlatform`과 `CpuPlatform`을 lazy 초기화, 디바이스 타입에 따라 동적 디스패치 |
| `HeterogeneousDeviceCommunicator` | Gloo 기반 분산 통신 + SHM 최적화 |

**동적 디스패치**: `get_attn_backend_cls()` 호출 시 현재 디바이스가 CUDA면 FlashAttention, CPU면 CpuSdpa 반환.

#### `vllm/platforms/cpu.py` — 수정 (+126줄, -23줄)

| 변경 | 내용 |
|------|------|
| AVX-512/AVX2 SIMD 감지 | `/proc/cpuinfo` 파싱 → Inductor `cpp.simdlen` 자동 설정 (16 for AVX-512, 8 for AVX2) |
| V1 fallback | `use_v1=False`일 때 ValueError 대신 CpuSdpaBackend 반환 |
| OpenMP 최적화 | `KMP_BLOCKTIME`, `KMP_TPAUSE`, `KMP_AFFINITY`, `MKL_ENABLE_INSTRUCTIONS` setdefault |

**`simdlen`의 의미**: PyTorch Inductor가 C++ 코드 생성 시 SIMD 벡터 길이를 결정. AVX-512는 512비트 = 16개의 float32, AVX2는 256비트 = 8개의 float32.

#### `vllm/platforms/__init__.py` — 수정 (+47줄, -13줄)

- `heterogeneous_platform_plugin()` 추가 — `VLLM_HETEROGENEOUS_PLATFORM=1` 환경변수로 활성화
- 플랫폼 감지 우선순위에서 heterogeneous를 첫 번째로 배치

---

## C. CPU 워커 및 모델 러너

### 기술 배경: OpenMP 스레드 바인딩

LLM 추론에서 CPU 워커의 성능은 **OpenMP 스레드 설정**에 크게 좌우된다:

```
OMP_NUM_THREADS: 총 OpenMP 스레드 수
  - 물리 코어 수로 설정 (HT 스레드 제외)
  - HT 스레드를 포함하면 오히려 성능 저하 (ALU/FPU 경합)

VLLM_CPU_OMP_THREADS_BIND: vLLM의 스레드-코어 바인딩 모드
  - "auto": NUMA 토폴로지 기반 자동 바인딩
  - 수동: "0-55" 등 CPU ID 범위 지정

VLLM_CPU_NUM_OF_RESERVED_CPU: 서빙 프레임워크용 예약 코어
  - 하이브리드 모드에서는 0 (CPU 프로세스가 전용이므로)
  - CPU-only 모드에서는 1~2 (HTTP 서버용)
```

vLLM의 `init_cpu_threads_env()` C++ 함수(`_C_utils`)가 이 환경변수를 읽어 pthread affinity를 설정한다.

### 기술 배경: 하이브리드 모드에서의 중복 초기화 방지

CPU 프로세스의 초기화 순서:

```
run_cpu_engine_core()
 ├─ _setup_cpu_process_env()         ← (1) 환경변수 설정 + Intel 최적화
 │   ├─ OMP_NUM_THREADS=56
 │   ├─ configure_intel_optimizations()  ← KMP/MKL/ONEDNN 설정
 │   └─ configure_pytorch_for_intel()    ← torch.set_num_threads(56)
 │
 └─ EngineCoreProc(cpu_config)
     └─ CPUWorker.__init__()
         └─ setup_intel_cpu_environment()  ← (2) 다시 호출?!
             └─ configure_pytorch_for_intel()
                 └─ torch.set_num_threads(112)  ← 전체 코어로 덮어씀!

문제: (2)가 (1)의 NUMA-aware 설정을 전체 코어 수로 덮어씀
해결: _is_hybrid_cpu_process 플래그로 (2) 스킵
```

### 수정 파일

#### `vllm/v1/worker/cpu_worker.py` — 수정 (+385줄, -15줄)

| 변경 | 내용 |
|------|------|
| `_is_hybrid_cpu_process` | `CUDA_VISIBLE_DEVICES==""` + `VLLM_CPU_KVCACHE_SPACE` 존재로 감지 |
| Intel 초기화 분기 | 하이브리드: feature detection + IPEX 경고만, CPU-only: full `setup_intel_cpu_environment()` |
| CpuPlatform 강제 | `vllm.platforms._current_platform = CpuPlatform()` + attention backend 캐시 클리어 |
| `_configure_inductor_for_intel()` | **신규** — Dead Code Elimination, epilogue fusion, max_autotune, freezing |
| `_configure_threads_for_numa()` | **신규** — 하이브리드: OMP_NUM_THREADS 존중, CPU-only: 물리코어 기반 (코어 낭비 없음) |
| `init_device()` 개선 | `init_cpu_threads_env` AttributeError 처리, gloo 강제, NUMA modulo 할당 |
| `compile_or_warm_up_model()` | 전면 재작성 — CPU용 `_dummy_run` 호출 |

**`_configure_inductor_for_intel()` 설정**:
```python
torch._inductor.config.dce = True           # Dead Code Elimination
torch._inductor.config.epilogue_fusion = True # Epilogue 연산 융합
torch._inductor.config.max_autotune = True   # 최적 커널 자동 탐색
torch._inductor.config.freezing = True       # 가중치 상수화 (추론 시)
torch.set_float32_matmul_precision('high')   # TF32/BF16 matmul 허용
```

#### `vllm/v1/worker/cpu_model_runner.py` — 수정 (+260줄, -13줄)

| 변경 | 내용 |
|------|------|
| NUMA 인식 생성자 | `numa_node` 파라미터, NUMAAllocator로 메모리 바인딩 |
| `_allocate_kv_cache_tensors()` | **신규** — `create_numa_aware_tensor()`로 NUMA 로컬 KV cache 할당 |
| `load_model()` 재작성 | CPU 디바이스 강제, attention selector CpuPlatform 패치, IPEX 최적화 |
| `profile_run()` | **신규** — CUDA graph 없는 CPU 전용 프로파일 |
| `_dummy_run()` | **신규** — CPU 전용 dummy forward pass |

**NUMA 인식 KV Cache 할당**: 2TB RAM이 NUMA 0과 NUMA 1에 분산되어 있을 때, CPU 워커가 NUMA 0에 바인딩되면 KV cache 텐서도 NUMA 0에 할당해야 한다. 그렇지 않으면 모든 KV cache 접근이 UPI를 경유하여 대역폭이 절반으로 줄어든다.

---

## D. CPU Attention 백엔드

### 기술 배경: PagedAttention on CPU

vLLM의 핵심 혁신인 **PagedAttention**은 KV cache를 고정 크기 블록(pages)으로 분할하여 메모리 단편화를 제거한다:

```
KV Cache 구조:
  key_cache:   [num_blocks, block_size, num_kv_heads, head_size]
  value_cache: [num_blocks, block_size, num_kv_heads, head_size]

블록 테이블: 각 시퀀스가 어떤 블록을 사용하는지 매핑
  block_table[seq_id] = [block_3, block_7, block_12, ...]
```

GPU에서는 `vllm::paged_attention_v1/v2` CUDA 커널이 이를 처리하지만, CPU에서는 해당 커널이 없다. 해결책:

1. **IPEX 있음**: `ipex.ops.PagedAttention` 커널 사용 (최적화된 C++ 구현)
2. **IPEX 없음**: 순수 PyTorch로 구현해야 함:
   - 블록 테이블에서 KV cache gather
   - 패딩 → boolean attention mask 생성
   - `F.scaled_dot_product_attention()` 호출 (PyTorch 네이티브 SDPA)

### 기술 배경: GQA (Grouped Query Attention)

Llama 2 이후 대부분의 LLM은 GQA를 사용한다:

```
GQA: num_heads=32, num_kv_heads=8 → 4개의 query head가 1개의 KV head를 공유
    num_queries_per_kv = num_heads / num_kv_heads = 4

처리: KV를 repeat_interleave로 확장
    K: [seq_len, 8, head_size] → [seq_len, 32, head_size]
    V: [seq_len, 8, head_size] → [seq_len, 32, head_size]
```

### 수정 파일

#### `vllm/v1/attention/backends/cpu_attn.py` — 수정 (+320줄, -20줄)

| 변경 | 내용 |
|------|------|
| IPEX import 강화 | `Exception` 추가 (IPEX 호환성 문제 무시) |
| `reshape_and_cache_flash()` | **재작성** — CUDA ops 대신 순수 PyTorch (block 인덱싱 + 텐서 할당) |
| `write_to_paged_cache()` | **신규** — CPU에서 KV를 paged cache에 쓰기 |
| `forward_decode()` 확장 | **CPU 배치 SDPA** — gather → pad → mask → `F.scaled_dot_product_attention()` |
| num_tokens != num_seqs | 시퀀스별 토큰 수 불일치 시 loop fallback |

**`forward_decode()` CPU 구현 핵심 로직**:
```python
# 1. 블록 테이블에서 KV cache gather
for seq_i in range(num_seqs):
    blocks = block_tables[seq_i, :num_blocks_needed]
    k_gathered = key_cache[blocks].reshape(-1, num_kv_heads, head_size)
    v_gathered = value_cache[blocks].reshape(-1, num_kv_heads, head_size)

# 2. GQA 확장 (num_kv_heads → num_heads)
k_gathered = k_gathered.repeat_interleave(num_queries_per_kv, dim=1)

# 3. Padding + boolean attention mask
padded_k = F.pad(k_gathered, (0, 0, 0, 0, 0, max_len - seq_len))
mask = torch.zeros(max_len, dtype=torch.bool)
mask[:seq_len] = True

# 4. 배치 SDPA (PyTorch 네이티브)
output = F.scaled_dot_product_attention(
    query, key, value, attn_mask=mask, scale=scale)
```

#### `vllm/attention/backends/cpu_stubs.py` — **신규** (+211줄)

V0 엔진용 CPU SDPA 백엔드 스텁 (CpuSdpaMetadata, CpuSdpaMetadataBuilder, CpuSdpaImpl, CpuSdpaBackend).

#### `vllm/attention/layer.py` — 수정 (+19줄)

CPU 텐서일 때 CUDA 전용 custom op(`vllm::unified_attention`)을 우회하여 direct call:
```python
use_direct = self.use_direct_call or query.device.type == "cpu"
```

---

## E. AVX-512 C++ 커널

### 기술 배경: SIMD 마이크로커널 설계

CPU에서 GEMM(General Matrix Multiply) 성능을 극대화하려면 **마이크로커널(micro-kernel)** 설계가 핵심이다:

```
6x16 마이크로커널 (VNNI INT8):
- 6개의 ZMM 누적기 (zmm0~zmm5): 6행의 결과를 동시 계산
- 1개의 ZMM 브로드캐스트 (zmm6): A 행렬의 4바이트를 16레인에 복제
- 1개의 ZMM 로드 (zmm7): B 행렬의 16×4 블록 로드

루프 바디 (핵심):
  zmm_a = _mm512_set1_epi32(*(int32_t*)&A[m][k])     // broadcast 4 bytes
  zmm_b = _mm512_load_si512(&B_packed[k/4][n/16])      // load 64 bytes
  zmm_c[m] = _mm512_dpbusd_epi32(zmm_c[m], zmm_a, zmm_b)  // u8×s8 dot product

  // 한 반복에서: 6행 × 16열 × 4요소 = 384개의 INT8 MAC 연산
```

**3단계 캐시 블로킹** (Goto BLAS 방식):
```
MC=72, NC=256, KC=256

외부 루프: N을 NC(256) 단위로 분할  → L3 캐시에 B 블록 상주
중간 루프: K를 KC(256) 단위로 분할  → L2 캐시에 A 패널 상주
내부 루프: M을 MR(6) 단위로 분할   → L1/레지스터에서 마이크로커널 실행
```

### 기술 배경: Non-Temporal Store

LLM의 KV cache 복사처럼 쓰기 전용 대량 메모리 전송에서는 **Non-Temporal Store**(`_mm512_stream_si512`)가 효과적이다:

```
일반 store: CPU → L1 → L2 → L3 → RAM (캐시 오염)
NT store:   CPU → RAM 직접 (Write Combining Buffer 경유, 캐시 우회)

장점: L3 캐시를 오염시키지 않아 다른 데이터의 캐시 히트율 유지
단점: 작은 데이터에는 오히려 느림 (캐시 활용 불가)
적합한 상황: KV cache 복사, 모델 가중치 로드 등 대량 순차 쓰기
```

### 수정 파일 (12개 신규, +2,452줄)

| 파일 | 줄수 | 내용 |
|------|------|------|
| `csrc/cpu/gemm_vnni.cpp` | 503 | **VNNI INT8 GEMM** — 6x16 마이크로커널, u8s8 형식, 3단계 캐시 블로킹, OpenMP M-타일 병렬화 |
| `csrc/cpu/gemm_vnni.hpp` | 101 | GEMM 헤더 — MR=6, NR=16, MC=72, NC=256, KC=256 상수, VNNI weight packing 함수 |
| `csrc/cpu/batch_attention.cpp` | 458 | **16-시퀀스 배치 Attention** — ZMM 레인 인터리빙으로 16개 시퀀스의 Q·K 점곱을 동시 계산, online softmax |
| `csrc/cpu/quant_q8_0.cpp` | 366 | **Q8_0 양자화** — per-block(32요소) INT8 양자화/역양자화, AVX-512 벡터화 |
| `csrc/cpu/decode_gemv.cpp` | 289 | **Decode GEMV** — BF16/FP32 행렬-벡터 곱 (decode 단계의 단일 토큰 처리 최적화) |
| `csrc/cpu/mem_opt.cpp` | 246 | **메모리 최적화** — NT memcpy, NUMA `numa_alloc_onnode`, SW 프리페치 (`_mm_prefetch`) |
| `csrc/cpu/torch_bindings_hybrid.cpp` | 129 | `_C_cpu_ops` 네임스페이스로 PyTorch 바인딩 (CUDA 빌드 시 충돌 방지) |
| `csrc/cpu/torch_bindings.cpp` | 79 | `_C` 네임스페이스로 PyTorch 바인딩 (CPU-only 빌드용) |
| `csrc/cpu/cpu_types_x86.hpp` | 29 | x86 SIMD 타입 정의 |
| `csrc/cpu/numa_utils.hpp` | 54 | libnuma C 래핑 |
| `csrc/cpu/cache.cpp` | 13 | KV cache reshape 유틸리티 |

**배치 Attention 최적화 (batch_attention.cpp)**: 일반적으로 시퀀스를 하나씩 처리하지만, AVX-512의 16개 레인에 서로 다른 시퀀스의 attention score를 배치하면:

```
ZMM 레인 인터리빙:
lane[0] = seq_0의 Q·K score
lane[1] = seq_1의 Q·K score
...
lane[15] = seq_15의 Q·K score

→ 한 번의 ZMM 연산으로 16개 시퀀스를 동시 처리
→ 메모리 대역폭 활용 극대화 (벡터화율 높음)
```

---

## F. 분산 통신 (Heterogeneous)

### 기술 배경: NCCL vs Gloo

| | NCCL | Gloo |
|---|------|------|
| **대상** | GPU-to-GPU | CPU 또는 GPU-CPU 혼합 |
| **성능** | NVLink/NVSwitch 최적화 | TCP/SHM 기반 |
| **제약** | 모든 참여 rank가 GPU 필요 | 모든 디바이스 지원 |

heterogeneous 모드(GPU+CPU가 같은 tensor parallel 그룹)에서는 NCCL을 사용할 수 없다(CPU rank가 존재). 따라서 Gloo 백엔드로 fallback하되, GPU 텐서를 CPU로 복사한 후 통신해야 한다.

### 수정 파일

#### `vllm/distributed/parallel_state.py` — 수정 (+252줄)

| 변경 | 내용 |
|------|------|
| `_IS_HETEROGENEOUS_MODE` | 전역 플래그 |
| `GroupCoordinator.__init__()` | GPU-only NCCL 그룹 + 전체 Gloo 그룹 분리 |
| `_all_reduce_out_place()` | GPU NCCL reduce → CPU bridge → Gloo all_reduce |
| `_all_gather_out_place()`, `broadcast()` | Gloo 기반 fallback |

**All-Reduce 브릿지 패턴**:
```
GPU rank 0: tensor.cuda() ──NCCL──→ reduced on GPU
                           │
                           └─ .cpu() ──Gloo all_reduce──→ 전체 rank
                                                          │
CPU rank 8: tensor.cpu() ─────────────────────────────────┘
```

#### `vllm/distributed/utils.py` — 수정 (+59줄)

- **비대칭 Pipeline Parallelism**: heterogeneous 모드에서 CPU rank에 1개 레이어만 할당 (GPU 28 레이어 vs CPU 1 레이어)

---

## G. Executor 및 Worker (V0 레거시)

V0 엔진의 heterogeneous pipeline parallelism 지원. **현재 parallel-batch 모드는 V1 엔진을 사용**하므로 이 코드는 레거시.

### `vllm/executor/parallel_batch_executor.py` — **신규** (+979줄)
- V0용 CPU 워커 래퍼 (NUMA 바인딩, IPEX, 스레드 설정)

### `vllm/executor/mp_distributed_executor.py` — 수정 (+44줄)
- heterogeneous 모드에서 GPU 수 체크 완화
- GPU/CPU rank별 환경변수 분리

### `vllm/worker/worker.py` — 수정 (+178줄)
- heterogeneous GPU 워커 지원

### `vllm/worker/worker_base.py` — 수정 (+40줄)
- `rank >= num_gpus` 판별로 CPU 워커 자동 할당

---

## H. 설정 및 CLI

### `vllm/config.py` — 수정 (+100줄)

`HybridConfig` dataclass 추가:

| 필드 | 기본값 | 설명 | 자동 감지 로직 |
|------|--------|------|--------------|
| `mode` | `"none"` | parallel-batch / moe-hybrid / none | - |
| `cpu_ratio` | `None` | CPU 배치 비율 | CapacityAwareRouter 사용 시 무시 |
| `cpu_num_threads` | `0` | OpenMP 스레드 수 | NUMA 노드 물리코어 수 |
| `cpu_kvcache_space_gb` | `0` | CPU KV cache GB | 총메모리 × 0.4 (32~512GB) |
| `cpu_max_num_seqs` | `0` | 최대 동시 시퀀스 | 물리코어 / 4 (최소 4) |
| `cpu_max_num_batched_tokens` | `0` | 최대 배치 토큰 | max_seqs × 256 |
| `numa_aware` | `True` | NUMA 최적화 | - |
| `numa_bind_node` | `None` | NUMA 노드 바인딩 | rank=0 기반 자동 선택 |

**기본값 0의 의미**: 모든 CPU 관련 파라미터의 기본값이 0이면 `_resolve_cpu_params()`가 시스템 하드웨어를 감지하여 최적값을 자동 결정한다. 사용자가 명시적으로 값을 설정하면 해당 값을 그대로 사용.

### `vllm/engine/arg_utils.py` — 수정 (+121줄)

CLI 인자 9개 추가:
```
--hybrid-mode {none,parallel-batch,moe-hybrid}
--hybrid-cpu-ratio FLOAT
--hybrid-cpu-max-seqs INT
--hybrid-cpu-kvcache-gb INT
--hybrid-cpu-threads INT
--hybrid-cpu-max-batched-tokens INT
--hybrid-cpu-dtype {bfloat16,float16,int8}
--hybrid-numa-aware / --no-hybrid-numa-aware
--hybrid-numa-node INT
```

---

## I. 미래용 컴포넌트

`--hybrid-mode moe-hybrid` 시 warning만 출력. 엔진 통합 미완성.

### MoE Expert Offload (`expert_offload.py`, +582줄)

**기술 배경**: Mixture of Experts (MoE) 모델(DeepSeek V2/V3 등)은 수백 개의 expert 중 소수만 활성화한다. 자주 사용되는 expert는 GPU에, 나머지는 CPU에 상주시켜 GPU 메모리를 절약하는 **Expert Offload** 전략.

- LRU 기반 GPU/CPU expert 분배
- 사용량 통계 기반 동적 교체

### N-gram Speculative Decoding (`ngram_proposer_dynamic.py`, +414줄)

**기술 배경**: CPU에서 N-gram 패턴을 학습하여 다음 토큰을 "추측"하고, GPU가 한 번에 검증하는 **Speculative Decoding**. GPU 처리량은 batch 크기에 덜 민감하므로, 여러 추측을 한 번에 검증하면 효과적.

### Disaggregated Serving (`coordinator.py` +641줄, `kv_transfer.py` +469줄)

**기술 배경**: Prefill(프롬프트 처리)과 Decode(토큰 생성)를 별도 노드에서 수행. Prefill은 연산 집약적(GPU), Decode는 메모리 대역폭 집약적(CPU 가능).

### Q8_0 양자화 (`q8_0.py`, +242줄)

**기술 배경**: llama.cpp 호환 per-block(32요소) INT8 양자화. AVX-512 VNNI의 `vpdpbusd` 명령어와 최적 조합.

---

## J. 빌드 시스템

### 기술 배경: 듀얼 타겟 빌드

vLLM은 원래 CUDA 전용 빌드(`_C.abi3.so`)만 생성한다. 하이브리드 모드에서는 CUDA 커널과 CPU 커널을 **동시에** 빌드해야 한다:

```
빌드 결과:
  vllm/_C.abi3.so         ← CUDA ops (기존 vLLM, NVCC로 컴파일)
  vllm/_moe_C.abi3.so     ← MoE ops (기존 vLLM)
  vllm/_C_cpu_ops.abi3.so ← CPU ops (AVX-512 커널, GCC로 컴파일)

문제: 같은 `_C` 네임스페이스에 CUDA와 CPU 함수를 등록하면 심볼 충돌
해결: CPU 커널을 별도 `_C_cpu_ops` 네임스페이스로 분리
```

### `cmake/cpu_hybrid_extension.cmake` — **신규** (+129줄)

- AVX-512 감지 (`-mavx512f -mavx512vnni -mavx512bw -mavx512dq`)
- 컴파일 플래그: `-O3 -funroll-loops -ffast-math`
- Phase 1-5 소스 파일 등록
- `optional=True`: AVX-512 미지원 환경에서 빌드 자동 스킵

### `CMakeLists.txt` — 수정 (+41줄)

- **NVTX 헤더 패치**: 최신 CUDA Toolkit(12.x)에서 PyTorch NVTX 호환성 문제 해결 (nvtx3 → nvToolsExt)
- `include(cmake/cpu_hybrid_extension.cmake)` 추가

---

## K. 기타 수정

### `vllm/_custom_ops.py` — 수정 (+17줄)

```python
HAS_CPU_OPS = False
with contextlib.suppress(ImportError):
    import vllm._C_cpu_ops
    HAS_CPU_OPS = True

def cpu_ops():
    """하이브리드 빌드: _C_cpu_ops, CPU-only: _C"""
    return torch.ops._C_cpu_ops if HAS_CPU_OPS else torch.ops._C
```

### `vllm/_ipex_ops.py` — 수정 (+14줄)

IPEX import 오류 처리 강화: `AttributeError`("module 'os' has no attribute 'exit'"), 버전 불일치 등을 graceful하게 처리.

### `vllm/platforms/interface.py` — 수정 (+3줄)

`from __future__ import annotations` 추가로 순환 import 방지.

---

## 전체 파일 목록

### 신규 파일 (소스 코드, 27개)

```
# 하이브리드 엔진 핵심
vllm/v1/engine/hybrid_core.py              (+765)

# Intel CPU 플랫폼
vllm/platforms/intel_cpu_utils.py           (+965)
vllm/platforms/heterogeneous.py             (+441)

# CPU Attention
vllm/attention/backends/cpu_stubs.py        (+211)

# AVX-512 C++ 커널
csrc/cpu/gemm_vnni.cpp                      (+503)
csrc/cpu/gemm_vnni.hpp                      (+101)
csrc/cpu/batch_attention.cpp                (+458)
csrc/cpu/quant_q8_0.cpp                     (+366)
csrc/cpu/decode_gemv.cpp                    (+289)
csrc/cpu/mem_opt.cpp                        (+246)
csrc/cpu/torch_bindings_hybrid.cpp          (+129)
csrc/cpu/torch_bindings.cpp                 (+79)
csrc/cpu/cpu_types_x86.hpp                  (+29)
csrc/cpu/numa_utils.hpp                     (+54)
csrc/cpu/cache.cpp                          (+13)

# 빌드
cmake/cpu_hybrid_extension.cmake            (+129)
cmake/cpu_extension.cmake                   (+10)

# 미래용 컴포넌트
vllm/executor/parallel_batch_executor.py    (+979)
vllm/engine/disaggregated/__init__.py       (+30)
vllm/engine/disaggregated/coordinator.py    (+641)
vllm/engine/disaggregated/kv_transfer.py    (+469)
vllm/model_executor/.../expert_offload.py   (+582)
vllm/v1/spec_decode/ngram_proposer_dynamic.py (+414)
vllm/model_executor/.../quantization/q8_0.py (+242)

# 스크립트
scripts/check_amx.sh                        (+131)
scripts/verify_hybrid_parallel.py           (+426)
```

### 수정된 파일 (소스 코드, 26개)

```
# 하이브리드 엔진
vllm/v1/engine/core_client.py               (+274)
vllm/v1/engine/core.py                      (+5/-15)

# CPU 워커
vllm/v1/worker/cpu_worker.py                (+385/-15)
vllm/v1/worker/cpu_model_runner.py          (+260/-13)
vllm/v1/worker/gpu_worker.py                (+33)
vllm/v1/worker/gpu_model_runner.py          (+60)

# CPU Attention
vllm/v1/attention/backends/cpu_attn.py      (+320/-20)
vllm/attention/layer.py                     (+19)

# 설정/CLI
vllm/config.py                              (+100)
vllm/engine/arg_utils.py                    (+121)

# 플랫폼
vllm/platforms/__init__.py                  (+47/-13)
vllm/platforms/cpu.py                       (+126/-23)
vllm/platforms/interface.py                 (+3)
vllm/platforms/cuda.py                      (+3)

# 분산 통신
vllm/distributed/parallel_state.py          (+252)
vllm/distributed/utils.py                   (+59)
vllm/distributed/shm_broadcast.py           (+8)
vllm/distributed/__init__.py                (+5)
vllm/distributed/communication_op.py        (+3)

# Executor/Worker (V0)
vllm/executor/mp_distributed_executor.py    (+44)
vllm/executor/ray_distributed_executor.py   (+10)
vllm/executor/ray_utils.py                  (+15)
vllm/v1/executor/multiproc_executor.py      (+39)
vllm/worker/worker.py                       (+178)
vllm/worker/worker_base.py                  (+40)

# 기타
vllm/_custom_ops.py                         (+17)
vllm/_ipex_ops.py                           (+14/-10)
vllm/__init__.py                            (+3)
vllm/core/interfaces.py                     (+1)
vllm/model_executor/.../quantization/__init__.py (+3)
CMakeLists.txt                              (+41)
setup.py                                    (+5)
```

---

*생성일: 2026-02-24*
