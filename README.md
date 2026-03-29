# vLLM Hybrid: CPU+GPU Parallel-Batch Inference

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Status: Experimental](https://img.shields.io/badge/Status-Experimental-orange.svg)]()

**vLLM Hybrid**는 [vLLM](https://github.com/vllm-project/vllm)을 확장하여, GPU와 CPU를 **별도 프로세스**에서 동시 실행하는 하이브리드 추론 엔진입니다.

`total_throughput = GPU_throughput + CPU_throughput`

---

## Architecture

GPU와 CPU가 각각 독립된 `EngineCoreProc` 프로세스로 실행됩니다. 클라이언트 레이어의 `CapacityAwareRouter`가 CPU 용량 기반으로 요청을 분배합니다.

```
                      ┌─────────────────────────────┐
                      │  HybridAsyncMPClient         │
                      │  (CapacityAwareRouter)       │
                      └─────┬───────────┬────────────┘
                    ZMQ ROUTER      ZMQ PULL
                   ┌────┘   └────┐       │
                   ▼              ▼       │
      ┌────────────────┐  ┌────────────────┐
      │ GPU EngineCore │  │ CPU EngineCore │
      │ (Process A)    │  │ (Process B)    │
      │                │  │                │
      │ MultiprocExec  │  │ UniProcExec    │
      │ 8x H100 (TP=8)│  │ CPUWorker      │
      │ KV: GPU VRAM   │  │ KV: DRAM+NUMA  │
      └────────────────┘  └────────────────┘
            │                     │
            └──── ZMQ PUSH ───────┘
                      │
              Async Output Merge
```

### Key Features

| Feature | Description |
| :--- | :--- |
| **Dual-Process** | GPU/CPU가 별도 프로세스에서 완전 병렬 실행 (별도 PID, GIL, busy loop) |
| **CapacityAwareRouter** | CPU 슬롯 여유시 CPU로, 가득차면 GPU로 라우팅 → CPU 활용률 100% |
| **Auto-Detection** | CPU 코어수, 메모리, NUMA 토폴로지 자동 감지 (기본값 0=auto) |
| **Intel Optimized** | AVX-512 VNNI, AMX-BF16/INT8, IPEX, NUMA-aware KV Cache |
| **Zero GPU Impact** | core.py 무수정 — hybrid 코드는 hybrid_core.py/core_client.py에만 존재 |

---

## Quick Start

### Prerequisites

- NVIDIA GPU (CUDA 12.1+)
- Linux OS (NUMA 지원 권장)
- Python 3.10+

### Installation

```bash
git clone git@github.com:mystous/vllm_hybrid.git
cd vllm_hybrid

# 가상환경
uv venv vllm_dev_prj --python 3.12 --seed
source vllm_dev_prj/bin/activate

# 의존성 설치
VLLM_USE_PRECOMPILED=1 uv pip install -U -e . --torch-backend=auto
uv pip install -r requirements/build.txt --torch-backend=auto

# (선택) IPEX 설치 (CPU 성능 최적화)
pip install intel-extension-for-pytorch==2.8.0

# (선택) NUMA 지원
sudo apt install -y numactl libnuma-dev
```

#### Version Compatibility

| PyTorch | torchvision | IPEX | CUDA |
|---------|-------------|------|------|
| **2.8.0** | **0.23.0** | **2.8.0** | 12.1/12.4 |

### Build (from source)

```bash
python tools/generate_cmake_presets.py
cmake --preset release
cmake --build --preset release --target install
```

> CMakeLists.txt NVTX 헤더 패치 필요 시 [Deployment.md](Deployment.md) 참조

### Docker

사전 구성된 Docker 이미지로 빠르게 시작할 수 있습니다. 환경 설정이 완료되어 있으므로 컨테이너 진입 후 빌드부터 진행하면 됩니다.

```bash
docker run -it \
  --gpus all \
  --privileged \
  --network host \
  --ipc host \
  --pid host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.claude:/root/.claude \
  -v ~/.config/claude:/root/.config/claude \
  mystous/vllm_hybrid:v1.4 \
  /bin/bash
```

컨테이너 진입 후:

```bash
# 빌드
python tools/generate_cmake_presets.py
cmake --preset release
cmake --build --preset release --target install

# 서빙 실행
vllm serve <model> \
  --tensor-parallel-size 8 \
  --hybrid-mode parallel-batch
```

> 상세 옵션 및 트러블슈팅은 **[Docker Guide](docs/DOCKER_GUIDE.md)** 참조

### Run

```bash
# 자동 감지 모드 (권장)
vllm serve <model> \
  --tensor-parallel-size 8 \
  --hybrid-mode parallel-batch

# 수동 설정
vllm serve <model> \
  --tensor-parallel-size 8 \
  --hybrid-mode parallel-batch \
  --hybrid-cpu-max-seqs 28 \
  --hybrid-cpu-kvcache-gb 800 \
  --hybrid-cpu-threads 112
```

### Verify

```bash
# CPU 기능 감지 확인
python -c "
from vllm.platforms.intel_cpu_utils import detect_intel_cpu_features
f = detect_intel_cpu_features()
print(f'{f.model_name}: {f.num_sockets}S x {f.cores_per_socket}C')
print(f'AVX-512={f.avx512f}, AMX={f.amx_bf16}, VNNI={f.avx512_vnni}')
"

# 프로세스 확인
ps aux | grep -E "GPU_EngineCore|CPU_EngineCore"
```

### Benchmarking

```bash
python benchmarks/benchmark_serving.py \
  --backend openai --base-url http://localhost:8000 \
  --model <model> --dataset-name random \
  --num-prompts 500 --random-input-len 128 --random-output-len 128
```

---

## CLI Options

| Option | Default | Description |
| :--- | :--- | :--- |
| `--hybrid-mode` | none | `parallel-batch` / `moe-hybrid` / `none` |
| `--hybrid-cpu-max-seqs` | 0 (auto) | CPU 최대 동시 시퀀스 (auto: 물리코어/4) |
| `--hybrid-cpu-kvcache-gb` | 0 (auto) | CPU KV cache GB (auto: 총메모리*0.4) |
| `--hybrid-cpu-threads` | 0 (auto) | CPU 스레드 수 (auto: NUMA 노드 물리코어) |
| `--hybrid-cpu-max-batched-tokens` | 0 (auto) | CPU 배치 토큰 수 (auto: seqs*256) |
| `--hybrid-numa-aware` | True | NUMA 최적화 활성화 |
| `--hybrid-numa-node` | auto | 특정 NUMA 노드 바인딩 |

---

## CPU Optimization Features

| Feature | Intel Xeon (AVX-512) | Desktop CPU (AVX2) |
| :--- | :---: | :---: |
| **SIMD Vectorization** | 512-bit (simdlen=16) | 256-bit (simdlen=8) |
| **NUMA-aware KVCache** | Multi-node binding | Single-node |
| **Intel IPEX** | Supported | Supported |
| **AMX (BF16/INT8)** | Sapphire Rapids+ | Not available |
| **AVX-512 VNNI** | INT8 acceleration | Not available |
| **PyTorch Inductor** | Enabled | Enabled |

---

## Documentation

| Document | Description |
| :--- | :--- |
| **[Deployment.md](Deployment.md)** | 상세 배포 가이드 (빌드/설치/실행/트러블슈팅) |
| **[docs/DOCKER_GUIDE.md](docs/DOCKER_GUIDE.md)** | Docker 실행 가이드 |
| **[CLAUDE.md](CLAUDE.md)** | 프로젝트 컨텍스트 (AI 어시스턴트용) |
| **[docs/HYBRID_OPTIONS_IMPLEMENTATION_PLAN.md](docs/HYBRID_OPTIONS_IMPLEMENTATION_PLAN.md)** | 하이브리드 옵션 상세 설계 |
| **[docs/HETEROGENEOUS_CPU_OPTIMIZATIONS.md](docs/HETEROGENEOUS_CPU_OPTIMIZATIONS.md)** | CPU 최적화 상세 |
| **[docs/AVX512_OPTIMIZATION_IMPLEMENTATION_PLAN.md](docs/AVX512_OPTIMIZATION_IMPLEMENTATION_PLAN.md)** | AVX-512 커널 구현 |
| **[analysis/overview.md](analysis/overview.md)** | 시스템 아키텍처 분석 |

---

## Limitations & Known Issues

- **CPU Performance**: CPU 추론은 GPU 대비 느림. NUMA 미설정 시 성능 저하 가능.
- **IPEX Dependency**: IPEX 미설치 시 CPU decode가 Python loop fallback (성능 저하).
- **moe-hybrid mode**: 아직 미구현 (parallel-batch만 사용 가능).
- **Torch Compile**: CPU 워커에서 `DYNAMO_ONCE` 모드로 자동 활성화.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
