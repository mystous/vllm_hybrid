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

## Ninja Gap 성능 개선 추적 (Applied Features Log)

**Ninja Gap** = `T_hybrid < T_gpu_only` 달성 목표. 현재 H100x8 / 7B / 500×128 기준 hybrid wall 은 GPU-only 대비 26–143× 느림 (batch scaling 실패). 아래는 **소스 코드에 적용된 기법을 누적 기록** 하는 로그. 각 항목 적용 후 주간 측정 (`VLLM_HYBRID_PROFILE=1` + `num_seqs` sweep) 결과와 함께 위에서 아래로 쌓아간다.

기법 상세: [NinjaGap_Todo/README.md](NinjaGap_Todo/README.md) (전체 Gate / flag 테이블 / 진도)
기법 각 문서: [NinjaGap_Todo/](NinjaGap_Todo/) (§01–§22)

**측정 워크플로** (수동 이동):

```bash
# 1. template 복사
cp eval/envs/g0_h100x8_qwen7b.env /tmp/run.env

# 2. 편집
#    HYBRID_TODO_NN=00              # baseline=00, §05 후=05, §06 후=06 ...
#    HYBRID_CPU_MAX_SEQS=1          # sweep: 1/2/4/8/16 각각 재실행

# 3. 두 터미널 실행 — 결과는 eval/results/<ts>_.../ 에
./eval/serve.sh hybrid /tmp/run.env   # 서버 (터미널 1)
./eval/bench.sh hybrid /tmp/run.env   # 벤치 (터미널 2, ready 후)
# PROFILE=1 이므로 eval/results/<ts>_.../ 에 applied_features.json,
# env_snapshot.txt, git_sha.txt 포함 + server log 에 [HYBRID-CPU-PROFILE] 라인

# 4. 사용자가 sweep 정리 (수동 mv)
mv eval/results/<ts1>_...  measurement_results/H100x8/g0_00/seqs1
mv eval/results/<ts2>_...  measurement_results/H100x8/g0_00/seqs2
# ...

# 5. sweep 모두 끝난 후 분석
python3 eval/g0_analyze.py measurement_results/H100x8/g0_00/
```

상세 절차와 실전 주의사항: [NinjaGap_Todo/01_G0_measurement.md](NinjaGap_Todo/01_G0_measurement.md) §실행 방법 / §실전 주의사항

### 적용 순서 (적층 로그)

| # | 일자 (KST) | 기법 | TODO 문서 | 상태 | 주요 커밋 | Gate | 측정 결과 (wall/ratio) |
|---:|---|---|---|:---:|---|:---:|---|
| 0 | 2026-04-15 | **Baseline (H100x8, 7B)** — hybrid dual-process, wave-batch routing, cpu_max_num_seqs=1 (auto), NUMA strict membind, C++ init_cpu_threads_env, feature 기반 ONEDNN ISA | — | ✅ 기존 | — | pre-G0 | wall 394-2003 s, ratio(16/1)=5.3× |
| 1 | 2026-04-17 | **§01 G0 measurement 완료** — `num_seqs` sweep 고정, sublayer profiling, manifest/summary pipeline, `measurement_results/<HW>/g0_*` schema 정착 | [01_G0_measurement.md](NinjaGap_Todo/01_G0_measurement.md) | ✅ | `22afea529` | G0 완료 | RTX3090 `g0_00` + H100x8 `g0_00_*` sweep 확보 |
| 2 | 2026-04-17 | **§02 Tier 0 baseline defense 완료** — baseline 기본값을 `cpu_max_num_seqs=1`, `strategy=capacity`, `priority=cpu-first` 로 고정, wave env 분리, 전략 비교 env 3종 추가 | [02_tier0_baseline_defense.md](NinjaGap_Todo/02_tier0_baseline_defense.md) | ✅ | `22afea529` | G0 baseline 고정 | H100x8 `g0_02_strat_{capacity,length_aware,throughput_adaptive}` |
| 3 | 2026-04-15 | **§05 KMP_BLOCKTIME=0** (auto 기본) — `_setup_cpu_process_env` 에서 `HYBRID_KMP_BLOCKTIME=auto` 시 강제 적용. hybrid dual-process IPC 경합 완화 | [05_omp_env_finalize.md](NinjaGap_Todo/05_omp_env_finalize.md) | ✅ | `869c736eb` | — | 측정 대기 |
| 4 | — | (예정) §03 Huge Pages (2MB THP → 1GB hugetlb) | [03_huge_pages.md](NinjaGap_Todo/03_huge_pages.md) | ⭕ | — | — | — |
| ~~5~~ | 2026-04-19 | ~~§04 IPEX WoQ INT8~~ **기각** — vLLM `QKVParallelLinear` 비호환, §23 CPU Native Quant 편입 | [04_ipex_woq_int8.md](NinjaGap_Todo/04_ipex_woq_int8.md) | ✗ | — | — | — |
| 6 | 2026-04-19 | **§06 Hot Path Wiring (Q8_0 dispatch, Qwen2 MLP)** — `hot_path_wiring.py` + `_Q8_0LinearMethod` 로 `gate_up_proj`/`down_proj` 를 `torch.ops._C_cpu_ops.q8_0_linear` 로 apply-time 치환. `HybridConfig.vnni_hot_path` CLI 경로 + LoRA 이후 hook + `_create_cpu_vllm_config` passthrough 3 버그 fix. H100x8 32B 128 layer 치환 확인 | [06_hot_path_wiring.md](NinjaGap_Todo/06_hot_path_wiring.md) | ✅ | `6f904b39b` | G1 단독 미통과 | `g0_06_qwen2.5_32b/` — wall seqs=1: 80.0→57.6 s (−28%), TPOT 63.6→49.6 ms (−22%). `cost(4)/cost(1)=2.89` (G1 ≤ 2.0 미달), wall ratio 10.7×~357× |
| 7 | — | (예정) §07 ISA Binary Dispatch | [07_isa_binary_dispatch.md](NinjaGap_Todo/07_isa_binary_dispatch.md) | 🔶 | — | — | — |
| 8 | — | (예정) §08 Kernel Fusion | [08_kernel_fusion.md](NinjaGap_Todo/08_kernel_fusion.md) | 🔶 | — | — | — |
| 9 | — | (예정) §11 Batch-aware Decode Attention (v2 동적) | [11_batch_aware_decode_attn.md](NinjaGap_Todo/11_batch_aware_decode_attn.md) | 🔶 | — | G2 경유 | — |
| 10 | — | (예정) §13 T-MAC LUT GEMV INT4 | [13_tmac_lut_gemv_int4.md](NinjaGap_Todo/13_tmac_lut_gemv_int4.md) | ⭕ | — | G3 지점 | — |
| … | | | | | | | |

**운영 규칙**:
1. 기법 구현 + 측정 완료 후에만 한 행 추가 (설계만 된 항목은 "예정" 으로 별도)
2. 각 행은 직전 행 대비 **누적 (적층)** 성능. 독립 이득 아님
3. `측정 결과` 열에는 `(wall, ratio(16/1), batch_scaling_ratio)` 최소 3 수치 — 상세는 `measurement_results/<HW>/g0_<NN>/` 디렉토리 링크
4. Gate 열 변화 (G0 → G1 → G2 → G3) 가 Ninja Gap 진척의 단일 지표
5. 행 추가 시 `applied_features.json` 의 flag state 도 같이 기록 (추적성)

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
