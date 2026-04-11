# Docker를 이용한 vLLM Hybrid 실행 가이드

> **Docker 이미지**: `mystous/vllm_hybrid:cu13_v0.6_h100x4` (CUDA 13.0 + torch 2.9 기반)
> **사전 조건**: Docker, NVIDIA Container Toolkit 설치 필요
> **마지막 업데이트**: 2026-04-11

---

## 목차
1. [컨테이너 실행](#1-컨테이너-실행)
2. [빌드](#2-빌드)
3. [서빙 실행](#3-서빙-실행)
4. [검증](#4-검증)
5. [트러블슈팅](#5-트러블슈팅)

---

## 1. 컨테이너 실행

Docker 이미지에는 CUDA, PyTorch, 의존성이 사전 설치되어 있습니다. 호스트의 HuggingFace 캐시와 Claude 설정을 마운트하고, 모든 하드웨어 자원을 사용합니다.

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
  mystous/vllm_hybrid:cu13_v0.6_h100x4 \
  /bin/bash
```

### 옵션 설명

| 옵션 | 설명 |
| :--- | :--- |
| `--gpus all` | 호스트 GPU 전체 사용 |
| `--privileged` | 모든 디바이스 접근 허용 |
| `--network host` | 호스트 네트워크 공유 (포트 매핑 불필요) |
| `--ipc host` | 공유 메모리 제한 해제 (대규모 모델 로딩 시 필수) |
| `--pid host` | 호스트 프로세스 네임스페이스 공유 |
| `-v ~/.cache/huggingface:...` | HuggingFace 모델 캐시 마운트 (중복 다운로드 방지) |
| `-v ~/.claude:...` | Claude 설정 마운트 |

### 추가 옵션 (선택)

```bash
# HuggingFace 토큰 전달
-e HF_TOKEN=$(cat ~/.cache/huggingface/token)

# 작업 디렉토리 마운트 (결과 파일 등 호스트와 공유)
-v /path/to/workspace:/workspace
```

---

## 2. 빌드

컨테이너 진입 후, 환경 설정은 이미 완료되어 있으므로 빌드부터 진행합니다.

### 2.1 CMake Preset 생성

```bash
cd /workspace/vllm_hybrid  # 또는 소스가 위치한 경로
python tools/generate_cmake_presets.py
```

### 2.2 NVTX 헤더 패치 (필요 시)

`CMakeLists.txt` 73번째 줄 근처에 NVTX 워크어라운드가 없으면 추가합니다. 상세 내용은 [Deployment.md](../Deployment.md#32-nvtx-헤더-패치)를 참조하세요.

### 2.3 빌드 및 설치

```bash
cmake --preset release
cmake --build --preset release --target install
```

### 2.4 빌드 결과 확인

```bash
ls -la vllm/_C.abi3.so vllm/_moe_C.abi3.so vllm/_C_cpu_ops.abi3.so vllm/_C_utils.abi3.so
```

| 모듈 | 파일 | 내용 | 빌드 요구 |
| :--- | :--- | :--- | :--- |
| `_C` | `vllm/_C.abi3.so` | CUDA 메인 ops | CUDA toolkit |
| `_moe_C` | `vllm/_moe_C.abi3.so` | MoE ops | CUDA toolkit |
| `_C_cpu_ops` | `vllm/_C_cpu_ops.abi3.so` | AVX-512 CPU 커널 (VNNI GEMM / Q8_0 / decode GEMV / batched attention / mem_opt) | AVX-512F 이상 |
| `_C_utils` | `vllm/_C_utils.abi3.so` | `init_cpu_threads_env` (OMP 1:1 pin + NUMA strict membind) | OpenMP + libnuma (어떤 x86_64) |

---

## 3. 서빙 실행

### 3.1 자동 감지 모드 (권장)

```bash
vllm serve <model> \
  --tensor-parallel-size 8 \
  --hybrid-mode parallel-batch
```

### 3.2 수동 설정 (override)

기본적으로 모든 CPU 파라미터는 0 (auto) 로 두는 것을 권장한다. 원칙:
`cpu_max_num_seqs = 1` per NUMA engine 고정. 아래 예시처럼 명시 override 시
`cpu_max_num_seqs ≠ 1` 는 경고 로그를 출력한다 (원칙 위반 알림).

```bash
# 디버그용 — 값은 예시이며 auto 가 권장
vllm serve <model> \
  --tensor-parallel-size 8 \
  --hybrid-mode parallel-batch \
  --hybrid-num-cpu-engines 2 \
  --hybrid-cpu-max-seqs 1 \
  --hybrid-cpu-kvcache-gb 400 \
  --hybrid-cpu-threads 56 \
  --hybrid-cpu-max-batched-tokens 256
```

---

## 4. 검증

### CPU 기능 감지

```bash
python -c "
import vllm._custom_ops as ops, torch
print('HAS_CPU_OPS:', ops.HAS_CPU_OPS)
print('HAS_CPU_UTILS:', ops.HAS_CPU_UTILS)
print('init_cpu_threads_env:', torch.ops._C_utils.init_cpu_threads_env)

from vllm.platforms.intel_cpu_utils import detect_intel_cpu_features
f = detect_intel_cpu_features()
print(f'{f.model_name}: {f.num_sockets}S x {f.cores_per_socket}C')
print(f'AVX-512={f.avx512f}, AMX={f.amx_bf16}, VNNI={f.avx512_vnni}')
"
```

### 프로세스 확인

```bash
ps aux | grep -E "GPU_EngineCore|CPU_EngineCore"
```

### GPU 상태

```bash
nvidia-smi
```

---

## 5. 트러블슈팅

### NVIDIA Container Toolkit 미설치

```
docker: Error response from daemon: could not select device driver ""
```

→ [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) 설치 필요

### 공유 메모리 부족

```
RuntimeError: DataLoader worker is killed by signal: Bus error
```

→ `--ipc host` 옵션이 빠져 있는지 확인

### HuggingFace 모델 다운로드 실패

→ `HF_TOKEN` 환경 변수 전달 또는 호스트에서 먼저 `huggingface-cli login` 수행 후 캐시 마운트
