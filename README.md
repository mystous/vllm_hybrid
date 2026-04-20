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

**Ninja Gap** = `T_hybrid < T_gpu_only` 달성 목표.

**대표 workload (2026-04-20 고정)**: Qwen2.5-32B-Instruct × H100x8 (TP=8) × 500 req × 128/128. 7B + RTX3090 는 dev secondary.

**현재 실측 gap**: gpu_only outTP **11,523 tok/s** vs hybrid 최고 (§06-1 v1 seqs=1) **1,196 tok/s** = gpu_only 의 **10.4%**.

**주원인**: CPU engine batch 병렬화 구조적 결함. seqs=1 → 8 에서 §06-1 v1 outTP 1196 → 272 (4.4× 감소). 상세: [Tech_done.md](Tech_done.md) v8 §SSOT-3.

기법 상세: [NinjaGap_Todo/README.md](NinjaGap_Todo/README.md) (전체 Gate / flag 테이블 / 진도)
기법 각 문서: [NinjaGap_Todo/](NinjaGap_Todo/) (§01–§28)

**측정 워크플로** (대표 workload: Qwen2.5-32B × H100x8 TP=8):

```bash
# 1. 32B template 복사
cp eval/envs/g0_h100x8_qwen32b_00_tp8.env /tmp/run.env

# 2. 편집
#    HYBRID_TODO_NN=<기법 번호>     # base=00, §06=06, §06-1 v1=06_1_v1 ...
#    HYBRID_CPU_MAX_SEQS=1          # sweep: 1/2/4/8/16 각각 재실행

# 3. serve + bench (한 줄 체인 — sweep loop)
bash eval/g0_seq_sweep.sh /tmp/run.env 1 2 4 8 16

# 4. 결과는 eval/results/<ts>_.../ 에 저장. 사용자가 수동 mv:
mv eval/results/<ts1>_..._seqs1  measurement_results/H100x8/g0_<NN>_qwen2.5_32b/seqs1
# ...

# 5. sweep 모두 끝난 후 분석
python3 eval/g0_analyze.py measurement_results/H100x8/g0_<NN>_qwen2.5_32b/
```

7B + RTX3090 dev 환경용 예시는 `eval/envs/g0_dev_rtx3090_qwen1.5b.env` 등 별도 template.

상세 절차와 실전 주의사항: [NinjaGap_Todo/01_G0_measurement.md](NinjaGap_Todo/01_G0_measurement.md) §실행 방법 / §실전 주의사항

### 적용 순서 (적층 로그) — Qwen2.5-32B × H100x8 TP=8 기준

| # | 일자 | 기법 | 상태 | 측정 결과 (outTP, 32B TP=8) |
|---:|---|---|:---:|---|
| 0 | 2026-04-20 | **Baseline (32B TP=8, all Ninja Gap off)** — `measurement_results/H100x8/g0_00_qwen2.5_32b_base/` | ✅ | seqs=1: 908.9 tok/s · seqs=16: 637.8 · gpu_only 11,523 |
| 1 | 2026-04-19 | **§06** Q8_0 dispatch (`6f904b39b`) | 🔶 dispatch 완료, kernel 결함 | seqs=1 +18%, seqs≥2 역효과 |
| 2 | 2026-04-20 | **§06-1 v1** M-aware MLP kernel (`0c066f0e7`) | 🔶 v1 최종 (v2 기각 `33361eadc`→`0ca4466b7`) | §06 대비 seqs 2/4/8 +21~34% 회복, base 대비 일부 열세 |
| ~~3~~ | 2026-04-20 | ~~**§11 Phase 1** Batch-aware decode attn~~ (`f14bfad16`…`6604ceaab`) | ✗ 기각 | §06-1 v1 대비 −12~−5% regression. 측정: `g0_11_qwen2.5_32b_phase1(fail)/` |

**다음 단계**: Tier 1 후보 4개 (선행 연구에 실측 수치 + 조건 보고) 중 우선순위 순 — 상세 [NinjaGap_Todo/README.md](NinjaGap_Todo/README.md).

1. **§16 SparAMX** — linear 1.42× / attn 1.14× (Xeon SPR, 우리 HW 동일)
2. **§22 NEO asymmetric** — H100 70B 14.3% (MLSys'25, HW+규모 동일)
3. **§28 xFasterTransformer 이식** — Intel 공식 SPR stack
4. **§13 T-MAC LUT INT4** — 4× (edge ARM, SPR 재검증 필요)

<details>
<summary>Infra 이력 (§01 G0 / §02 baseline defense / §05 OMP env / §04 WoQ 기각 / §03 Huge Pages Phase 2 기각) — 개별 NinjaGap 문서 참조</summary>

- §01 G0 measurement (`22afea529`, 2026-04-17): `num_seqs` sweep + sublayer profiling + manifest pipeline
- §02 Tier 0 baseline defense (`22afea529`, 2026-04-17): `cpu_max_num_seqs=1`, strategy=capacity, priority=cpu-first
- §03 Huge Pages: Phase 1 (2MB THP) host default / Phase 2 (1GB hugetlb) ✗ 기각 2026-04-19 (SPR TLB 역효과)
- §04 IPEX WoQ INT8: ✗ 기각 2026-04-19 (vLLM Linear 비호환, §23 편입)
- §05 OMP env (`869c736eb`, 2026-04-15): `KMP_BLOCKTIME=0` 강제

초기 7B + RTX3090 측정 이력은 `old_doc/` 참조.

</details>

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
