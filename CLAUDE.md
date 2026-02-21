Answer to me always Korean
Write your task on CLAUDE.md file
Don't commit or push without explicit command

# vLLM Hybrid 프로젝트

## 프로젝트 개요
vLLM의 CPU/GPU 하이브리드 추론 최적화 포크. GPU와 CPU를 **별도 프로세스**에서 동시에 실행하여 `total_throughput = GPU + CPU` 달성.

## 타겟 하드웨어
- **GPU**: NVIDIA H100 x8 (TP=8)
- **CPU**: Intel Xeon Platinum 8480+ (Sapphire Rapids), 2소켓 112코어
- **RAM**: 2TB DDR5, NUMA 멀티소켓
- **ISA**: AVX-512, AVX-512 VNNI, AMX-BF16/INT8

---

## 핵심 아키텍처: Dual-Process Parallel-Batch

```
HybridAsyncMPClient (CapacityAwareRouter로 라우팅)
├─ input_socket (ZMQ ROUTER)
│   ├─ GPU: identity=b'\x00\x00' (engine_index=0)
│   └─ CPU: identity=b'\x01\x00' (engine_index=1)
├─ output_socket (ZMQ PULL) ← GPU/CPU 결과 비동기 수집
└─ CapacityAwareRouter: CPU 슬롯 여유시 CPU, 가득차면 GPU

GPU EngineCoreProc [별도 프로세스]
├─ EngineCore → MultiprocExecutor (8x H100)
└─ KV Cache: GPU VRAM

CPU EngineCoreProc [별도 프로세스]
├─ EngineCore → UniProcExecutor (CPUWorker)
└─ KV Cache: NUMA-aware DRAM
```

### 설계 원칙
1. **core.py 무수정** — hybrid 코드는 hybrid_core.py와 core_client.py에만 존재
2. **별도 프로세스** — GPU/CPU 각 독립 PID, GIL, busy loop
3. **CapacityAwareRouter** — CPU 슬롯 기반 라우팅 (RequestRouter 대비 CPU 활용률 극대화)
4. **자동 감지** — cpu_max_num_seqs, kvcache, threads 모두 0(auto) 기본값

---

## 핵심 파일

### Hybrid 엔진
| 파일 | 역할 |
|------|------|
| `vllm/v1/engine/hybrid_core.py` | CapacityAwareRouter, _resolve_cpu_params, _setup_cpu_process_env, run_cpu_engine_core, launch_hybrid_engines |
| `vllm/v1/engine/core_client.py` | HybridAsyncMPClient, HybridSyncMPClient, _HybridEngineLauncherMixin |
| `vllm/v1/engine/core.py` | EngineCore/EngineCoreProc (hybrid 코드 없음) |

### CPU 최적화
| 파일 | 역할 |
|------|------|
| `vllm/platforms/intel_cpu_utils.py` | Intel CPU 감지, NUMA, AMX/AVX-512, OpenMP 설정 |
| `vllm/v1/worker/cpu_worker.py` | CPUWorker (NUMA 스레드 바인딩, IPEX 감지) |
| `vllm/v1/attention/backends/cpu_attn.py` | CPU PagedAttention (IPEX 최적 커널) |
| `vllm/config.py` (HybridConfig) | 하이브리드 설정 (기본값 0=auto) |
| `vllm/engine/arg_utils.py` | CLI 인자 정의 |

### AVX-512 C++ 커널 (하이브리드 빌드)
| 파일 | 역할 |
|------|------|
| `csrc/cpu/gemm_vnni.cpp` | VNNI INT8 GEMM (6x16 마이크로커널) |
| `csrc/cpu/quant_q8_0.cpp` | Q8_0 양자화 |
| `csrc/cpu/decode_gemv.cpp` | BF16/FP32 Decode GEMV |
| `csrc/cpu/batch_attention.cpp` | 16-시퀀스 배치 Attention |
| `csrc/cpu/mem_opt.cpp` | NT memcpy, NUMA 할당, 프리페치 |
| `cmake/cpu_hybrid_extension.cmake` | _C_cpu_ops 타겟 빌드 |

### 기타 컴포넌트
| 파일 | 역할 |
|------|------|
| `vllm/model_executor/layers/fused_moe/expert_offload.py` | MoE Expert Offload (미래용) |
| `vllm/v1/spec_decode/ngram_proposer_dynamic.py` | N-gram Speculative Decode (미래용) |
| `vllm/engine/disaggregated/` | Disaggregated Serving (미래용) |

---

## CLI 옵션

```bash
# 기본 실행 (자동 감지 — 권장)
vllm serve <model> \
  --tensor-parallel-size 8 \
  --hybrid-mode parallel-batch

# 수동 오버라이드
vllm serve <model> \
  --tensor-parallel-size 8 \
  --hybrid-mode parallel-batch \
  --hybrid-cpu-max-seqs 28 \
  --hybrid-cpu-kvcache-gb 800 \
  --hybrid-cpu-threads 112 \
  --hybrid-cpu-max-batched-tokens 7168
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--hybrid-mode` | none | parallel-batch / moe-hybrid / none |
| `--hybrid-cpu-ratio` | auto | CPU 비율 (CapacityAwareRouter 사용 시 무시됨) |
| `--hybrid-cpu-max-seqs` | 0 (auto) | CPU 최대 동시 시퀀스 (auto: 물리코어/4) |
| `--hybrid-cpu-kvcache-gb` | 0 (auto) | CPU KV cache GB (auto: 총메모리*0.4) |
| `--hybrid-cpu-threads` | 0 (auto) | CPU 스레드 수 (auto: NUMA 노드 물리코어) |
| `--hybrid-cpu-max-batched-tokens` | 0 (auto) | CPU 배치 토큰 수 (auto: seqs*256) |
| `--hybrid-numa-aware` / `--no-hybrid-numa-aware` | True | NUMA 최적화 |
| `--hybrid-numa-node` | auto | NUMA 노드 지정 |

---

## 환경 변수 (하이브리드 모드에서 자동 설정됨)

```bash
# _setup_cpu_process_env()에서 자동 설정:
CUDA_VISIBLE_DEVICES=""              # GPU 격리
VLLM_CPU_KVCACHE_SPACE=<auto>        # CPU KV cache GB
OMP_NUM_THREADS=<auto>               # NUMA 노드 물리 코어 수
VLLM_CPU_OMP_THREADS_BIND=auto       # 스레드 자동 바인딩
VLLM_CPU_NUM_OF_RESERVED_CPU=0       # 예약 코어 없음

# configure_intel_optimizations()에서 자동 설정:
KMP_AFFINITY=granularity=fine,compact,1,0
KMP_BLOCKTIME=1
MKL_ENABLE_INSTRUCTIONS=AVX512
ONEDNN_MAX_CPU_ISA=AVX512_CORE_AMX   # AMX 가용 시
```

---

## 빌드

```bash
# CUDA + CPU 하이브리드 빌드
pip install -e . --config-settings="cmake.args=-DVLLM_TARGET_DEVICE=cuda"

# 빌드 결과: _C.abi3.so (CUDA) + _C_cpu_ops.abi3.so (CPU 커널)
```

## 검증

```bash
# CPU 기능 감지
python -c "
from vllm.platforms.intel_cpu_utils import detect_intel_cpu_features
f = detect_intel_cpu_features()
print(f'{f.model_name}: {f.num_sockets}S x {f.cores_per_socket}C x {f.threads_per_core}T')
print(f'AVX-512={f.avx512f}, VNNI={f.avx512_vnni}, AMX={f.amx_bf16}')
"

# 프로세스 확인
ps aux | grep -E "GPU_EngineCore|CPU_EngineCore"
```

---

## 문서 참조

| 문서 | 내용 |
|------|------|
| `Deployment.md` | 배포 가이드 (빌드/설치/실행/트러블슈팅) |
| `docs/HYBRID_OPTIONS_IMPLEMENTATION_PLAN.md` | 하이브리드 옵션 상세 설계 |
| `docs/HETEROGENEOUS_CPU_OPTIMIZATIONS.md` | CPU 최적화 상세 |
| `docs/AVX512_OPTIMIZATION_IMPLEMENTATION_PLAN.md` | AVX-512 커널 구현 계획 |
| `analysis/overview.md` | 시스템 아키텍처 분석 |

---

## 호환성

| 환경 | AVX | NUMA | IPEX | AMX | 상태 |
|------|-----|------|------|-----|------|
| H100 서버 (Xeon 8480+) | AVX-512 | 멀티소켓 | 지원 | 지원 | 최적화 |
| 개발 머신 (i9-12900KF) | AVX2 | 단일 | 미설치 | 없음 | 호환 |
| 일반 x86_64 | AVX2 | - | - | - | 호환 |

모든 기능은 graceful fallback 지원. IPEX/NUMA/AMX 없어도 정상 동작.

---

*마지막 업데이트: 2026-02-21*
