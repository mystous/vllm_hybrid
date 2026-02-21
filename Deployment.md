# vLLM Hybrid 배포 가이드

> **대상 환경**: NVIDIA H100 x8 + Intel Xeon Platinum 8480+ (2소켓, 112코어, 2TB RAM)
> **최종 업데이트**: 2026-02-21

---

## 목차
1. [사전 요구사항](#1-사전-요구사항)
2. [설치](#2-설치)
3. [빌드 (소스 빌드)](#3-빌드-소스-빌드)
4. [실행](#4-실행)
5. [자동 감지 동작](#5-자동-감지-동작)
6. [환경 변수](#6-환경-변수)
7. [아키텍처 개요](#7-아키텍처-개요)
8. [검증](#8-검증)
9. [벤치마킹](#9-벤치마킹)
10. [트러블슈팅](#10-트러블슈팅)

---

## 1. 사전 요구사항

### 하드웨어
- NVIDIA GPU (CUDA 12.1+, H100 권장)
- Intel Xeon CPU (AVX-512 지원 — Cascade Lake 이상)
- 64GB+ RAM (2TB 권장, KV cache용)

### 소프트웨어
- Linux (Ubuntu 22.04+ 권장)
- Python 3.10+
- CUDA Toolkit 12.1 이상

### 선택 의존성

| 패키지 | 용도 | 없으면? |
|--------|------|---------|
| `intel-extension-for-pytorch` (IPEX) | CPU PagedAttention 최적화 커널 | Python loop fallback (느림) |
| `numactl` + `libnuma-dev` | NUMA-aware 메모리/스레드 바인딩 | 단일노드 모드 |
| `psutil` | 자동 코어수/메모리 감지 | /proc 기반 fallback |

---

## 2. 설치

### 2.1 환경 구성

```bash
git clone git@github.com:mystous/vllm_hybrid.git
cd vllm_hybrid

# 가상환경 (uv 권장)
uv venv vllm_dev_prj --python 3.12 --seed
source vllm_dev_prj/bin/activate
```

### 2.2 의존성 설치

```bash
VLLM_USE_PRECOMPILED=1 uv pip install -U -e . --torch-backend=auto
uv pip install -r requirements/build.txt --torch-backend=auto
```

### 2.3 버전 호환성 (중요!)

PyTorch, torchvision, IPEX는 반드시 버전을 맞춰야 합니다.

| PyTorch | torchvision | IPEX | CUDA |
|---------|-------------|------|------|
| **2.8.0** | **0.23.0** | **2.8.0** | 12.1/12.4 |
| 2.7.x | 0.22.x | 2.7.x | 12.1/12.4 |

```bash
# 수동 설치 시
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu121
pip install torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu121
pip install intel-extension-for-pytorch==2.8.0
```

### 2.4 NUMA 지원 설치

```bash
# Ubuntu/Debian
sudo apt install -y numactl libnuma-dev

# 토폴로지 확인
numactl --hardware
```

---

## 3. 빌드 (소스 빌드)

### 3.1 CMake Preset 생성

```bash
python tools/generate_cmake_presets.py
```

### 3.2 NVTX 헤더 패치

`CMakeLists.txt` 73번째 줄 근처 (`Import torch cmake configuration` 주석 아래)에 추가:

```cmake
# Workaround for PyTorch NVTX headers issue with newer CUDA Toolkits
message(STATUS "Applying custom PyTorch NVTX headers workaround...")
if(NOT TARGET CUDA::nvToolsExt)
    message(STATUS "--> nvToolsExt Not found, looking for nvtx3.")
    if (NOT TARGET CUDA::nvtx3)
        message(STATUS "--> nvtx3 not found, adding library.")
        add_library(CUDA::nvtx3 INTERFACE IMPORTED)
        target_include_directories(CUDA::nvtx3 SYSTEM INTERFACE "${CUDAToolkit_INCLUDE_DIRS}")
        target_link_libraries(CUDA::nvtx3 INTERFACE ${CMAKE_DL_LIBS})
    endif()
    if (TARGET CUDA::nvtx3)
     add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
     target_compile_definitions(CUDA::nvToolsExt INTERFACE TORCH_CUDA_USE_NVTX3)
     target_link_libraries(CUDA::nvToolsExt INTERFACE CUDA::nvtx3)
     message(STATUS "--> Workaround applied.")
    else()
     message(STATUS "--> nvtx3 not found.")
    endif()
else()
    message(STATUS "--> Workaround not needed.")
endif()
```

### 3.3 빌드 및 설치

```bash
cmake --preset release
cmake --build --preset release --target install
```

### 3.4 빌드 결과

CUDA + CPU 커널이 동시에 빌드됩니다:

| 모듈 | 파일 | 내용 |
|------|------|------|
| `_C` | `vllm/_C.abi3.so` | CUDA ops (기존 vLLM) |
| `_moe_C` | `vllm/_moe_C.abi3.so` | MoE ops (기존 vLLM) |
| `_C_cpu_ops` | `vllm/_C_cpu_ops.abi3.so` | Phase 1-5 AVX-512 CPU 커널 (하이브리드용) |

`_C_cpu_ops`는 AVX-512 미지원 환경에서는 자동 스킵됩니다 (`optional=True`).

---

## 4. 실행

### 4.1 자동 감지 모드 (권장)

```bash
vllm serve facebook/opt-125m \
  --tensor-parallel-size 8 \
  --hybrid-mode parallel-batch
```

모든 CPU 파라미터가 자동 감지됩니다:
- `cpu_max_num_seqs`: 물리코어 / 4
- `cpu_kvcache_space_gb`: 총메모리 * 0.4
- `cpu_num_threads`: NUMA 노드 물리코어 수
- `cpu_max_num_batched_tokens`: max_seqs * 256

### 4.2 수동 오버라이드

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --tensor-parallel-size 8 \
  --hybrid-mode parallel-batch \
  --hybrid-cpu-max-seqs 28 \
  --hybrid-cpu-kvcache-gb 800 \
  --hybrid-cpu-threads 112 \
  --hybrid-cpu-max-batched-tokens 7168
```

### 4.3 CLI 옵션 전체 목록

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--hybrid-mode` | `none` | `parallel-batch` (활성화) / `none` (비활성화) |
| `--hybrid-cpu-ratio` | auto | RequestRouter 사용 시 CPU 비율 (CapacityAwareRouter와 동시 사용 불가) |
| `--hybrid-cpu-max-seqs` | 0 (auto) | CPU 최대 동시 시퀀스. CapacityAwareRouter의 용량 기준 |
| `--hybrid-cpu-kvcache-gb` | 0 (auto) | CPU KV cache 메모리 (GB) |
| `--hybrid-cpu-threads` | 0 (auto) | CPU OpenMP 스레드 수 |
| `--hybrid-cpu-max-batched-tokens` | 0 (auto) | CPU 최대 배치 토큰 수 |
| `--hybrid-cpu-dtype` | `bfloat16` | CPU 모델 데이터 타입 |
| `--hybrid-numa-aware` / `--no-hybrid-numa-aware` | True | NUMA 최적화 |
| `--hybrid-numa-node` | auto | 특정 NUMA 노드 바인딩 |

---

## 5. 자동 감지 동작

### 5.1 `_resolve_cpu_params()` 로직

```
입력: HybridConfig (기본값 모두 0)
  │
  ├─ psutil로 물리코어수, 총메모리 감지
  │
  ├─ NUMA 토폴로지 감지 (NUMAAllocator)
  │   ├─ numa_num_nodes: NUMA 노드 수
  │   ├─ target_numa_node: rank=0 기반 선택 (CPUWorker와 일치)
  │   ├─ logical_cpus_on_node: HT 포함 논리 CPU 수
  │   └─ numa_node_cores: HT 제거 물리 코어 수
  │       (logical_cpus / threads_per_core)
  │
  ├─ effective_cores = numa_node_cores (NUMA) 또는 physical_cores (비NUMA)
  ├─ effective_mem = node_memory (NUMA) 또는 total_memory (비NUMA)
  │
  └─ 결과:
      cpu_num_threads     = effective_cores
      cpu_max_num_seqs    = effective_cores / 4 (최소 4)
      cpu_kvcache_space_gb = effective_mem * 0.4 (32~512GB)
      cpu_max_batched_tokens = max_seqs * 256
```

### 5.2 Xeon 8480+ 예상 자동 감지값 (2소켓, 56코어/소켓, HT 2x)

| 파라미터 | NUMA 활성 | 비NUMA |
|----------|----------|--------|
| effective_cores | 56 (노드 0) | 112 (전체) |
| cpu_num_threads | 56 | 112 |
| cpu_max_num_seqs | 14 | 28 |
| cpu_kvcache_space_gb | ~400 (노드 0 메모리*0.4) | ~800 (전체*0.4) |
| cpu_max_batched_tokens | 3584 | 7168 |

### 5.3 `_setup_cpu_process_env()` 자동 설정

CPU 프로세스 시작 시 다음이 자동 설정됩니다:

```bash
# 기본 설정
CUDA_VISIBLE_DEVICES=""               # GPU 격리
VLLM_CPU_KVCACHE_SPACE=<auto>         # KV cache GB
OMP_NUM_THREADS=<auto>                # 물리 코어 수
VLLM_CPU_OMP_THREADS_BIND=auto        # 스레드 자동 바인딩
VLLM_CPU_NUM_OF_RESERVED_CPU=0        # 예약 코어 없음

# configure_intel_optimizations()에서 감지 기반 자동 설정:
KMP_AFFINITY=granularity=fine,compact,1,0
KMP_BLOCKTIME=1
KMP_TPAUSE=0
MKL_ENABLE_INSTRUCTIONS=AVX512
ONEDNN_MAX_CPU_ISA=AVX512_CORE_AMX    # AMX 지원 시
                   AVX512_CORE_VNNI   # VNNI만 지원 시
                   AVX512_CORE        # 기본 AVX-512

# configure_pytorch_for_intel()에서 설정:
# - torch.set_num_threads(OMP_NUM_THREADS)
# - Inductor: simdlen=16, epilogue_fusion, max_autotune
# - AMX 타일 권한 요청 (Linux 5.16+)
# - IPEX 활성화 (설치된 경우)
```

### 5.4 NUMA Affinity 일관성

CPU 프로세스에서 메모리와 스레드가 같은 NUMA 노드에 바인딩됩니다:

```
_setup_cpu_process_env()     → numa_set_preferred(node 0) → 메모리 바인딩
CPUWorker._get_autobind_cpu_ids() → local_rank=0 → NUMA 노드 0 → 스레드 바인딩
                                                          ↑
                                                   동일 NUMA 노드!
```

---

## 6. 환경 변수

### 6.1 자동 설정 (하이브리드 모드에서 코드가 설정)

| 변수 | 값 | 설정 위치 |
|------|-----|----------|
| `CUDA_VISIBLE_DEVICES` | `""` | `run_cpu_engine_core()` |
| `VLLM_CPU_KVCACHE_SPACE` | auto | `_setup_cpu_process_env()` |
| `OMP_NUM_THREADS` | auto | `_setup_cpu_process_env()` |
| `KMP_AFFINITY` | `granularity=fine,compact,1,0` | `configure_intel_optimizations()` |
| `KMP_BLOCKTIME` | `1` | `configure_intel_optimizations()` |
| `ONEDNN_MAX_CPU_ISA` | auto (감지 기반) | `configure_intel_optimizations()` |
| `VLLM_CPU_OMP_THREADS_BIND` | `auto` | `_setup_cpu_process_env()` |

### 6.2 수동 오버라이드 가능 (서버 시작 전 설정)

```bash
# Intel MKL/OpenMP (대부분 자동 설정됨)
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
export MKL_ENABLE_INSTRUCTIONS=AVX512

# vLLM CPU KVCache (수동 지정 시)
export VLLM_CPU_KVCACHE_SPACE=800

# NUMA 바인딩 (기본 auto)
export VLLM_CPU_OMP_THREADS_BIND=auto
```

---

## 7. 아키텍처 개요

### 7.1 프로세스 구조

```
vllm serve (main process)
│
├─ AsyncLLM
│   └─ HybridAsyncMPClient (또는 HybridSyncMPClient)
│       ├─ CapacityAwareRouter: CPU 슬롯 여유시 CPU, 가득차면 GPU
│       ├─ input_socket (ZMQ ROUTER, bind)
│       └─ output_socket (ZMQ PULL)
│
├─ GPU EngineCoreProc (별도 프로세스, PID A)
│   ├─ EngineCore(gpu_vllm_config)
│   ├─ Scheduler
│   ├─ MultiprocExecutor (TP=8, 8x H100)
│   └─ busy_loop: poll → schedule → execute → push
│
└─ CPU EngineCoreProc (별도 프로세스, PID B)
    ├─ EngineCore(cpu_vllm_config)  ← _create_cpu_vllm_config()로 파생
    ├─ Scheduler
    ├─ UniProcExecutor (CPUWorker)
    └─ busy_loop: poll → schedule → execute → push
```

### 7.2 CapacityAwareRouter 동작

```python
class CapacityAwareRouter:
    def route(request_id) -> "gpu" | "cpu":
        if cpu_in_flight < cpu_max_num_seqs:
            cpu_in_flight += 1
            return "cpu"
        return "gpu"

    def on_request_finished(request_id, was_cpu):
        if was_cpu:
            cpu_in_flight -= 1
```

CPU에 여유 슬롯이 있으면 항상 CPU로 라우팅 → **CPU 활용률 100% 유지**.

### 7.3 핵심 파일

| 파일 | 줄수 | 역할 |
|------|------|------|
| `vllm/v1/engine/hybrid_core.py` | ~750 | 라우터, CPU 파라미터 감지, 프로세스 관리 |
| `vllm/v1/engine/core_client.py` | ~1530 (확장) | Hybrid MPClient 클래스, 라우팅 훅 |
| `vllm/v1/engine/core.py` | (무수정) | EngineCore/EngineCoreProc |
| `vllm/v1/worker/cpu_worker.py` | ~530 | CPU 워커 (NUMA/Intel 최적화) |
| `vllm/platforms/intel_cpu_utils.py` | ~960 | Intel CPU 유틸리티 |
| `vllm/config.py` (HybridConfig) | ~80 | 하이브리드 설정 |

---

## 8. 검증

### 8.1 CPU 기능 감지

```bash
python -c "
from vllm.platforms.intel_cpu_utils import detect_intel_cpu_features, setup_intel_cpu_environment

print('=== CPU Feature Detection ===')
f = detect_intel_cpu_features()
print(f'CPU: {f.model_name}')
print(f'Topology: {f.num_sockets}S x {f.cores_per_socket}C x {f.threads_per_core}T')
print(f'AVX-512: {f.avx512f}, VNNI: {f.avx512_vnni}, BF16: {f.avx512_bf16}')
print(f'AMX-BF16: {f.amx_bf16}, AMX-INT8: {f.amx_int8}')

print()
print('=== Environment Setup ===')
config = setup_intel_cpu_environment(rank=0, world_size=1)
print(f'NUMA: {config[\"numa_enabled\"]}')
print(f'IPEX: {config[\"ipex_enabled\"]}')
print(f'AMX: {config[\"amx_enabled\"]}')
print('OK')
"
```

**Xeon 8480+ 예상 출력:**
```
CPU: Intel(R) Xeon(R) Platinum 8480+
Topology: 2S x 56C x 2T
AVX-512: True, VNNI: True, BF16: True
AMX-BF16: True, AMX-INT8: True
NUMA: True, IPEX: True, AMX: True
OK
```

### 8.2 하이브리드 모듈 확인

```bash
python -c "
from vllm.v1.engine.hybrid_core import (
    is_hybrid_mode, CapacityAwareRouter, _resolve_cpu_params
)
print('hybrid_core imports OK')

from vllm.v1.engine.core_client import (
    HybridAsyncMPClient, HybridSyncMPClient
)
print('core_client imports OK')

# CPU ops 빌드 확인
try:
    import vllm._C_cpu_ops
    print('_C_cpu_ops: available')
except ImportError:
    print('_C_cpu_ops: not built (CUDA-only build)')
"
```

### 8.3 프로세스 확인 (서버 실행 중)

```bash
# GPU + CPU 프로세스 확인
ps aux | grep -E "GPU_EngineCore|CPU_EngineCore"

# 로그에서 자동 감지값 확인
# "Resolved CPU params: max_seqs=14, kvcache=400GB, threads=56"
# "CapacityAwareRouter initialized: cpu_max_num_seqs=14"
# "Intel optimizations configured: AMX=True, AVX512=True, VNNI=True"
```

### 8.4 NUMA 확인

```bash
numactl --hardware
# Expected: 2 nodes, each with 56 CPUs

# NUMA 메모리 바인딩 확인
numastat -p <CPU_EngineCoreProc_PID>
```

---

## 9. 벤치마킹

```bash
# 서버 실행 후
python benchmarks/benchmark_serving.py \
  --backend openai \
  --base-url http://localhost:8000 \
  --model <model> \
  --dataset-name random \
  --num-prompts 500 \
  --random-input-len 128 \
  --random-output-len 128 \
  --request-rate 10
```

### 비교 벤치마크

```bash
# GPU-only (baseline)
vllm serve <model> --tensor-parallel-size 8

# Hybrid (CPU+GPU)
vllm serve <model> --tensor-parallel-size 8 --hybrid-mode parallel-batch

# 두 설정에서 동일 벤치마크 실행 후 처리량 비교
```

---

## 10. 트러블슈팅

### CPU 프로세스가 시작되지 않음

```bash
# 로그 확인
# "Hybrid mode detected but CPU process failed to start"

# 원인: CUDA_VISIBLE_DEVICES 설정 실패
# 해결: CPU 프로세스는 CUDA_VISIBLE_DEVICES="" 자동 설정됨
```

### NUMA 메모리 바인딩 실패

```bash
# 확인
numactl --hardware
cat /proc/self/numa_maps | head

# libnuma 미설치 시
sudo apt install -y numactl libnuma-dev
```

### IPEX 미설치 경고

```
WARNING: IPEX not available! CPU decode performance will be significantly degraded.
```

IPEX 설치로 해결:
```bash
pip install intel-extension-for-pytorch==2.8.0
```

### AMX 미감지 (Sapphire Rapids)

```bash
# 커널 버전 확인 (5.16+ 필요)
uname -r

# AMX 플래그 확인
grep -E "amx_bf16|amx-bf16" /proc/cpuinfo

# BIOS에서 AMX 활성화 확인
# 컨테이너/VM에서는 호스트가 AMX를 지원해야 함
```

### 스레드 수 불일치

```bash
# 실행 로그에서 확인:
# "Resolved CPU params: ... threads=56"
# "OMP_NUM_THREADS=56"

# 수동 확인
python -c "import torch; print(torch.get_num_threads())"
```

### CMake 빌드 오류

```bash
# NVTX 헤더 오류 → 3.2절 패치 적용
# CPU 확장 빌드 실패 → AVX-512 미지원 시 자동 스킵 (정상)
# 빌드 로그에서 확인:
# "CPU hybrid extension: building _C_cpu_ops (AVX512=ON, VNNI=ON)"
# 또는
# "CPU hybrid extension: skipping (no AVX-512 support)"
```

---

## 부록: 파일 구조

```
vllm_hybrid/
├── CLAUDE.md                          # AI 어시스턴트 컨텍스트
├── README.md                          # 프로젝트 소개
├── Deployment.md                      # 이 문서
├── PLAN.md                            # Option A (moe-hybrid) 계획
│
├── vllm/v1/engine/
│   ├── hybrid_core.py                 # [핵심] 하이브리드 엔진 로직
│   ├── core_client.py                 # [확장] Hybrid MPClient
│   └── core.py                        # [무수정] EngineCore
│
├── vllm/v1/worker/
│   ├── cpu_worker.py                  # CPU 워커 (NUMA/Intel 최적화)
│   └── cpu_model_runner.py            # CPU 모델 러너
│
├── vllm/platforms/
│   ├── intel_cpu_utils.py             # Intel CPU 유틸리티
│   ├── cpu.py                         # CPU 플랫폼
│   └── heterogeneous.py               # 이기종 플랫폼
│
├── vllm/config.py                     # HybridConfig
├── vllm/engine/arg_utils.py           # CLI 인자
│
├── csrc/cpu/                          # AVX-512 C++ 커널
│   ├── gemm_vnni.cpp                  # VNNI INT8 GEMM
│   ├── quant_q8_0.cpp                 # Q8_0 양자화
│   ├── decode_gemv.cpp                # Decode GEMV
│   ├── batch_attention.cpp            # 배치 Attention
│   ├── mem_opt.cpp                    # 메모리 최적화
│   ├── torch_bindings_hybrid.cpp      # 하이브리드 빌드 바인딩
│   └── torch_bindings.cpp             # CPU-only 빌드 바인딩
│
├── cmake/
│   ├── cpu_hybrid_extension.cmake     # _C_cpu_ops 타겟
│   └── cpu_extension.cmake            # CPU-only 빌드
│
├── docs/
│   ├── HYBRID_OPTIONS_IMPLEMENTATION_PLAN.md
│   ├── HETEROGENEOUS_CPU_OPTIMIZATIONS.md
│   ├── AVX512_OPTIMIZATION_IMPLEMENTATION_PLAN.md
│   └── test_method.md
│
└── analysis/                          # 아키텍처 분석 문서
    ├── overview.md
    ├── platform.md
    ├── worker.md
    └── communication.md
```
