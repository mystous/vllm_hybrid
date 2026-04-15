Answer to me always Korean
Don't commit or push without explicit command

Task 기록은 `Task_down.md`, 남은 작업은 `TODO.md`, 설계 의도의 단일 진실 공급원은 `docs/paper/main.tex` 이다. CLAUDE.md 는 이 세 파일을 묶는 현재 구성 요약만 유지한다.

# vLLM Hybrid 프로젝트

## 프로젝트 개요
vLLM의 CPU/GPU 하이브리드 추론 최적화 포크. GPU와 CPU를 **별도 OS 프로세스**에서 독립 EngineCore 로 실행하여 `T_hybrid = T_GPU + α·T_CPU` 달성. 설계·분석은 논문 `docs/paper/main.tex` 참조.

## 하드웨어 요구 사항

특정 기종에 매이지 않는다. 요구 사항은 **x86_64 + NVIDIA GPU (CUDA)** 뿐이고, 나머지는 모두 런타임 자동 감지 + graceful fallback (아래 "하드웨어 호환성" 섹션 참조).

논문 `docs/paper/main.tex` 의 평가 섹션에 실측 기준 환경이 명시되어 있으며, 구체적 스펙이 필요하면 그쪽이 단일 진실 공급원이다.

---

## 핵심 아키텍처: Dual-Process Parallel-Batch

```
HybridAsyncMPClient  (API Endpoint, 단일 라우터)
│
├─ CapacityAwareRouter  (논문 Algorithm 1/2/3: capacity/length-aware/throughput-adaptive)
│   └─ cpu_in_flight < cpu_max_num_seqs  ?  CPU  :  GPU
│
├─ input socket  (ZMQ ROUTER, identity dispatch)
│   ├─ GPU engine: identity = b'\x00\x00'   (engine_index 0)
│   └─ CPU engine: identity = b'\x01\x00'   (engine_index 1, multi-NUMA 시 ... \x02\x00 ...)
│
└─ output socket  (ZMQ PULL, async completions)

GPU EngineCoreProc  [별도 PID]
├─ EngineCore  (unmodified 상위 V1 scheduler)
└─ MultiprocExecutor → N × (GPUWorker)  on HBM3

CPU EngineCoreProc  [별도 PID, num_cpu_engines = num_numa 개]
├─ EngineCore  (CPU-only vllm_config, mode="none" passthrough)
└─ UniProcExecutor → CPUWorker  on local NUMA DDR5
    ├─ init_cpu_threads_env (C++, _C_utils extension)
    │     OMP thread 1:1 pin  +  numa_set_membind strict
    └─ IPEX → ONEDNN → AVX-512/AMX (가용 시) or AVX2 graceful fallback
```

### 설계 원칙 (논문 §3 Design 과 일치, 예외 없음)
1. **core.py 무수정** — hybrid 코드는 `hybrid_core.py` / `core_client.py` / `cpu_worker.py` / `intel_cpu_utils.py` 에만 존재
2. **프로세스 격리** — GPU/CPU 각 독립 PID, GIL, 스케줄러, 주소공간 (논문 §3.1)
3. **`num_cpu_engines = num_numa_nodes`** — 자동 감지. CPU engine 은 각자 자기 NUMA 노드의 모든 물리 코어와 DRAM 에 strict bind
4. **`cpu_max_num_seqs = 1` per engine (고정)** — 1 시퀀스가 해당 NUMA 노드의 모든 물리 코어를 OMP + BLAS matmul 병렬로 사용. batch 는 만들지 않음. 총 동시 CPU 시퀀스 = num_numa
5. **CapacityAwareRouter** — `C < N` 이면 CPU, 아니면 GPU (논문 Algorithm 1). CPU-first 가 기본, 논문 Property 2
6. **Zero configuration** — `num_cpu_engines`, `cpu_max_num_seqs`, `cpu_num_threads`, `cpu_kvcache_space_gb` 모두 기본값 0 (auto), `NUMAAllocator` / `/proc/cpuinfo` / `lscpu` 에서 자동 유도

---

## 핵심 파일

### Hybrid 엔진 / 라우팅
| 파일 | 역할 |
|------|------|
| `vllm/v1/engine/hybrid_core.py` | `CapacityAwareRouter`, `_resolve_cpu_params` (cpu_max_num_seqs=1 고정), `_resolve_num_cpu_engines` (= NUMA 노드 수), `_setup_cpu_process_env`, `_create_cpu_vllm_config` (HybridConfig passthrough 포함), `run_cpu_engine_core`, `launch_hybrid_engines` |
| `vllm/v1/engine/core_client.py` | `HybridAsyncMPClient`, `HybridSyncMPClient`, `_HybridEngineLauncherMixin`. client 초기화 시 `num_cpu_engines` resolve → `vllm_config` 에 write-back (router / launcher / `_compute_core_engines` 단일 진실 공급원 확보) |
| `vllm/v1/engine/core.py` | `EngineCore` / `EngineCoreProc` — hybrid 코드 없음, 원본 유지 |
| `vllm/config.py` (`HybridConfig`) | 하이브리드 설정 dataclass. `num_cpu_engines = 0` (auto sentinel), `cpu_max_num_seqs = 0` (auto sentinel), `numa_aware = True` |
| `vllm/engine/arg_utils.py` | CLI 인자 정의. `hybrid_num_cpu_engines = 0` (auto sentinel) |

### CPU 최적화 / 진단
| 파일 | 역할 |
|------|------|
| `vllm/platforms/intel_cpu_utils.py` | Intel CPU 감지, NUMA, AMX/AVX-512/VNNI, OpenMP, IPEX 설정 |
| `vllm/v1/worker/cpu_worker.py` | `CPUWorker`. `init_device` 에서 C++ `init_cpu_threads_env` 호출, 실패 시 `_python_init_cpu_threads_env` (sched_setaffinity) fallback. `_get_autobind_cpu_ids` 가 `hybrid_config.numa_bind_node` 우선 사용. `device_type='heterogeneous'` 방어 coerce. `execute_model` per-step trace |
| `vllm/v1/attention/backends/cpu_attn.py` | CPU PagedAttention. decode path counter (`_decode_path_counts`) 로 custom_avx / ipex / sdpa_batched / sdpa_loop 중 어느 경로가 사용되는지 기록 |
| `vllm/worker/worker_base.py` | `init_worker` 의 heterogeneous 휴리스틱에 `is_hybrid_cpu_engine` 우회 조건. `CUDA_VISIBLE_DEVICES=""` + hybrid_config + cpu_worker 조합이면 `device_type="cpu"` 유지 |

### C++ 확장
| 파일 / 타겟 | 역할 |
|------|------|
| `csrc/cpu/utils.cpp` | `init_cpu_threads_env` 구현 (NUMA strict membind + OMP 1:1 sched_setaffinity). SIMD 의존 없음 |
| `csrc/cpu/torch_bindings_utils.cpp` | `_C_utils` namespace 에 `init_cpu_threads_env` 등록. CUDA/ROCm 빌드에서도 항상 컴파일 |
| `cmake/cpu_utils_extension.cmake` | **`_C_utils` 타겟** — OpenMP + libnuma 만 require, AVX-512/AMX 무관 |
| `csrc/cpu/gemm_vnni.cpp` | VNNI INT8 GEMM (6×16 micro-kernel, AVX-512 + VPDPBUSD) |
| `csrc/cpu/quant_q8_0.cpp` | Q8_0 quantization |
| `csrc/cpu/decode_gemv.cpp` | BF16/FP32 decode GEMV (software prefetch) |
| `csrc/cpu/batch_attention.cpp` | BF16 batch-16 paged attention (head-dim SIMD) |
| `csrc/cpu/mem_opt.cpp` | NT memcpy, NUMA local alloc, prefetch |
| `cmake/cpu_hybrid_extension.cmake` | **`_C_cpu_ops` 타겟** — AVX-512F required, VNNI 있으면 gemm/quant 포함 |

빌드 결과:
- `vllm/_C.abi3.so` — CUDA 메인 extension (hybrid 코드 없음)
- `vllm/_C_cpu_ops.abi3.so` — AVX-512 CPU 커널들 (AVX-512F 필요, dev 는 skip 또는 stub)
- `vllm/_C_utils.abi3.so` — `init_cpu_threads_env` 전용 작은 extension (어떤 x86_64 에서도 빌드)

### G0 측정 / Post-processing (2026-04-15 구현)
| 파일 | 역할 |
|------|------|
| `eval/serve.sh` / `eval/bench.sh` | `VLLM_HYBRID_PROFILE=1` 시 `applied_features.json` + `env_snapshot.txt` + `git_sha.txt` 를 `eval/results/<ts>_.../` 에 포함 |
| `eval/envs/g0_dev_rtx3090_qwen1.5b.env` | dev G0 template — `HYBRID_TODO_NN` + `HYBRID_CPU_MAX_SEQS` 만 수정해 사용. 21개 Ninja Gap flag + § 주석 포함 |
| `eval/envs/g0_h100x8_qwen7b.env` | H100x8 G0 template (동일 구조) |
| `eval/g0_analyze.py` | Post-processing: sweep 디렉토리 → sublayer scaling plot + bench summary + markdown. 사용: `python3 eval/g0_analyze.py measurement_results/<HW>/g0_<NN>/` |
| `measurement_results/<HW>/g0_<NN>/seqs<N>/applied_features.json` | serve.sh 가 boot 시 자동 저장하는 활성 flag snapshot (attribution 추적) |

### 기타 컴포넌트 (미래용)
| 파일 | 역할 |
|------|------|
| `vllm/model_executor/layers/fused_moe/expert_offload.py` | MoE Expert Offload (미래) |
| `vllm/v1/spec_decode/ngram_proposer_dynamic.py` | N-gram Speculative Decode (미래) |
| `vllm/engine/disaggregated/` | Disaggregated Serving (미래) |

---

## CLI 옵션 (권장: 전부 auto)

```bash
# 권장 — 모든 CPU 파라미터 auto. 원칙이 자동 적용됨
vllm serve <model> \
  --tensor-parallel-size 8 \
  --hybrid-mode parallel-batch
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--hybrid-mode` | none | `none` / `parallel-batch` / `moe-hybrid` |
| `--hybrid-num-cpu-engines` | 0 (auto) | CPU engine 프로세스 수. auto 는 NUMA 노드 수 |
| `--hybrid-cpu-max-seqs` | 0 (auto) | per-engine 동시 시퀀스 수. **auto 는 1 고정** — NUMA 당 1 시퀀스 원칙 |
| `--hybrid-cpu-kvcache-gb` | 0 (auto) | CPU KV cache GB. auto 는 `clamp(eff_mem × 0.4, 32, 512)` |
| `--hybrid-cpu-threads` | 0 (auto) | CPU OMP thread 수. auto 는 NUMA 노드 물리 코어 전체 |
| `--hybrid-cpu-max-batched-tokens` | 0 (auto) | `cpu_max_num_seqs × 256` |
| `--hybrid-numa-aware` / `--no-hybrid-numa-aware` | True | NUMA 최적화 on/off |
| `--hybrid-numa-node` | auto | 특정 NUMA 노드 강제 바인딩 |
| `--hybrid-routing-strategy` | capacity | `capacity` / `length-aware` / `throughput-adaptive` / `round-robin` |
| `--hybrid-routing-priority` | gpu-first | `gpu-first` / `cpu-first` |
| `--hybrid-cpu-prefill-threshold` | 512 | length-aware / throughput-adaptive 용 |
| `--hybrid-warmup-requests` | 10 | throughput-adaptive EMA 워밍업 |
| `--hybrid-stats-log-interval` | 50 | router 통계 로깅 간격 (완료 요청 수) |

수동 override 는 허용하되 `cpu_max_num_seqs ≠ 1` 은 경고 로그가 출력된다 (원칙 위반 알림).

---

## 진단 환경 변수

```bash
# Trace (coarse per-step elapsed)
VLLM_HYBRID_TRACE=1            # 모든 hybrid marker 매 호출 로깅
VLLM_HYBRID_TRACE_EVERY=50     # N-th 호출마다 로깅

# G0 Profile mode (sublayer hook + manifest, 측정 전용)
VLLM_HYBRID_PROFILE=1          # sublayer forward hook 활성, manifest 기록
VLLM_HYBRID_PROFILE_SUBLAYER=1 # qkv/o/gate_up/down/norm/act 세분화 로그
VLLM_HYBRID_PROFILE_EVERY=1    # N step 마다 출력 (1=매 step)

# Ninja Gap 기법 활성화 tag (측정 결과 경로에 반영)
HYBRID_TODO_NN=00              # 00=baseline / 05=§05 / 06=§06 / ... 누적
HYBRID_KMP_BLOCKTIME=auto      # auto 면 KMP_BLOCKTIME=0 강제 (§05)
# 나머지 21개 HYBRID_* 기법 flag 는 NinjaGap_Todo/README.md 참조
```

PROFILE=1 이면 sublayer hook + manifest 가 활성. **결과는 기존 `eval/results/<ts>_.../` 에 저장**. sweep 단위로 모아 분석하려면 사용자가 `measurement_results/<HW>/g0_<NN>/seqs<N>/` 같은 규칙으로 수동 mv 후 `eval/g0_analyze.py <sweep_dir>` 실행. Template env: `eval/envs/g0_dev_rtx3090_qwen1.5b.env`, `eval/envs/g0_h100x8_qwen7b.env`.

### 진단 로그 marker
| marker | 의미 |
|------|------|
| `[HYBRID-RESOLVE]` | `_resolve_cpu_params` 최종 결과 + user override 여부 |
| `[HYBRID-LAUNCH]` | `launch_hybrid_engines` 가 결정한 `num_cpu_engines` + NUMA 정보 |
| `[HYBRID-APPLIED-FEATURES]` | PROFILE=1 시 activated flag manifest (boot 1회) |
| `[HYBRID-CLIENT]` | 라우팅 dispatch (request_id → engine identity) |
| `[HYBRID-CPU-ENV]` | `_setup_cpu_process_env` 가 설정한 OMP/MKL/OPENBLAS 환경변수 |
| `[HYBRID-CPU-PROC]` | CPU EngineCore 프로세스 초기화 상태 (torch threads, mkldnn) |
| `[HYBRID-CPU-WORKER]` | thread binding 경로 (C++ vs Python fallback), affinity, post-init 상태 |
| `[HYBRID-CPU-EXEC]` | CPU worker `execute_model` per-step trace (reqs, tokens, elapsed) |
| `[HYBRID-CPU-PROFILE]` | PROFILE=1 시 per-step sublayer breakdown (qkv/o/gate_up/...) |
| `[HYBRID-CPU-ATTN]` | decode path counter (`custom_avx` / `ipex` / `sdpa_batched` / `sdpa_loop`) |
| `[HYBRID-ROUTER-INIT]` `[HYBRID-ROUTER-DISPATCH]` `[HYBRID-ROUTER-STATS]` `[HYBRID-WAVE]` | 라우터 내부 상태 |

---

## 하이브리드 모드에서 자동 설정되는 OS 환경 변수

```bash
# _setup_cpu_process_env() 가 CPU engine 프로세스에 강제 적용:
CUDA_VISIBLE_DEVICES=""               # GPU 격리
VLLM_CPU_KVCACHE_SPACE=<auto GB>      # CPU KV cache
OMP_NUM_THREADS=<auto>                # NUMA 노드 물리 코어 수
MKL_NUM_THREADS=<auto>
OPENBLAS_NUM_THREADS=<auto>
NUMEXPR_NUM_THREADS=<auto>
VECLIB_MAXIMUM_THREADS=<auto>
BLIS_NUM_THREADS=<auto>
OMP_DYNAMIC=FALSE
MKL_DYNAMIC=FALSE
OMP_WAIT_POLICY=ACTIVE
VLLM_CPU_OMP_THREADS_BIND=auto        # CPUWorker init_device 에서 참조
VLLM_CPU_NUM_OF_RESERVED_CPU=0

# configure_intel_optimizations() 가 feature 감지 후 setdefault 로 설정:
KMP_AFFINITY=granularity=fine,compact,1,0
KMP_BLOCKTIME=0                         # §05. HYBRID_KMP_BLOCKTIME=auto 시 강제 (dual-process IPC 경합 완화)
KMP_TPAUSE=0
MKL_ENABLE_INSTRUCTIONS=AVX512          # avx512f 감지 시
ONEDNN_MAX_CPU_ISA=AVX512_CORE_AMX      # amx 감지 시
ONEDNN_MAX_CPU_ISA=AVX512_CORE_VNNI     # avx512 있고 amx 없을 때 fallback
```

실제 OMP 1:1 pinning 과 NUMA strict membind 는 `init_cpu_threads_env` (C++) 가 OS 환경변수와 독립적으로 `sched_setaffinity` / `numa_set_membind` 로 직접 수행한다.

---

## 빌드

```bash
# CUDA + CPU hybrid 빌드 (환경 구분 없이 동일)
pip install -e . --config-settings="cmake.args=-DVLLM_TARGET_DEVICE=cuda"

# 빌드 결과:
#   vllm/_C.abi3.so         — CUDA 메인 extension
#   vllm/_C_cpu_ops.abi3.so  — AVX-512 CPU 커널 (AVX-512F 있을 때)
#   vllm/_C_utils.abi3.so    — init_cpu_threads_env (항상)
```

---

## 검증 (dev)

```bash
# 1. extension 로드 + init_cpu_threads_env 등록 확인
python -c "
import vllm._custom_ops, torch
print('HAS_CPU_OPS:', vllm._custom_ops.HAS_CPU_OPS)
print('HAS_CPU_UTILS:', vllm._custom_ops.HAS_CPU_UTILS)
print('init_cpu_threads_env:', torch.ops._C_utils.init_cpu_threads_env)
"

# 2. resolver 동작 확인
python -c "
from vllm.config import HybridConfig
from vllm.v1.engine.hybrid_core import _resolve_cpu_params, _resolve_num_cpu_engines
hc = HybridConfig(mode='parallel-batch')
print('num_cpu_engines auto →', _resolve_num_cpu_engines(hc))
r = _resolve_cpu_params(hc)
print(f'cpu_max_num_seqs={r.cpu_max_num_seqs} threads={r.cpu_num_threads}')
"

# 3. Intel CPU 기능 감지
python -c "
from vllm.platforms.intel_cpu_utils import detect_intel_cpu_features
f = detect_intel_cpu_features()
print(f'{f.model_name}: {f.num_sockets}S × {f.cores_per_socket}C × {f.threads_per_core}T')
print(f'AVX-512={f.avx512f}, VNNI={f.avx512_vnni}, AMX-BF16={f.amx_bf16}')
"

# 4. 프로세스 분리 확인
ps aux | grep -E "GPU_EngineCore|CPU_EngineCore"

# 5. 1 시퀀스 CPU 추론 중 OMP thread pinning 확인
WORKER_PID=$(ps aux | grep "VllmWorker\|spawn_main" | grep -v grep | awk '{print $2}' | tail -1)
ps -L -p $WORKER_PID -o tid,psr,pcpu,comm | awk 'NR==1 || $3 > 30'
# → 모든 OMP thread 가 정확히 1:1 로 다른 core 에 pin 되어 있고 100% 사용 중이어야 함
```

---

## 문서 참조

| 문서 | 내용 |
|------|------|
| `docs/paper/main.tex` | **설계의 단일 진실 공급원** (IEEE 논문 draft) |
| `docs/CUDA13_MIGRATION_STATUS.md` | CUDA 13.0 마이그레이션 진단/fix 상세 |
| `docs/HYBRID_OPTIONS_IMPLEMENTATION_PLAN.md` | 하이브리드 옵션 설계 |
| `docs/HETEROGENEOUS_CPU_OPTIMIZATIONS.md` | CPU 최적화 상세 |
| `docs/AVX512_OPTIMIZATION_IMPLEMENTATION_PLAN.md` | AVX-512 커널 구현 계획 |
| `analysis/overview.md` | 시스템 아키텍처 분석 |
| `Deployment.md` | 배포 가이드 |
| `Task_down.md` | 완료된 작업 이력 |
| `TODO.md` | 남은 작업 |

---

## 하드웨어 호환성

**단일 코드베이스 + 런타임 자동 감지 + graceful fallback.** 환경별 분기 매트릭스는 없다.

- **ISA (AVX-512/AMX/VNNI/BF16)**: CMake 가 빌드 시 `/proc/cpuinfo` 로 `_C_cpu_ops` 포함 여부 결정, 런타임에 `intel_cpu_utils.detect_intel_cpu_features()` 가 재확인하여 `ONEDNN_MAX_CPU_ISA` / `MKL_ENABLE_INSTRUCTIONS` 설정. 가용하면 사용, 없으면 한 단계 낮은 ISA 로 자동 폴백 (AMX → AVX-512 VNNI → AVX-512 → AVX2). 성능만 차이, 동작은 동일.
- **IPEX**: 설치되어 있으면 항상 사용. `_IPEXPagedAttention` 이 decode/prefill 경로의 기본. 미설치 시 `_PagedAttention` (torch SDPA) fallback.
- **NUMA 개수**: `NUMAAllocator.num_nodes` 로 `num_cpu_engines` 자동 결정. 1 node 면 1 engine, N node 면 N engine × (각자 해당 노드에 strict bind).
- **`_C_utils` (init_cpu_threads_env)**: x86_64 + libnuma + OpenMP 있으면 항상 빌드. 미빌드 시 Python `sched_setaffinity` fallback (OMP 1:1 pin 효과는 약화되지만 동작).

실제로 검증된 환경 목록과 각 환경에서 관찰된 실측은 `Tech_done.md` 참조.

---

*마지막 업데이트: 2026-04-11 — CUDA 13.0 마이그레이션 이후 런타임 레이어 재정비*
