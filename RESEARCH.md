# vLLM Hybrid — 코드 구현 현황 (RESEARCH.md)

> 최종 업데이트: 2026-02-27

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [핵심 아키텍처: Dual-Process Parallel-Batch](#2-핵심-아키텍처-dual-process-parallel-batch)
3. [하이브리드 엔진 핵심 구현](#3-하이브리드-엔진-핵심-구현)
4. [설정 및 CLI](#4-설정-및-cli)
5. [Intel CPU 유틸리티 및 워커](#5-intel-cpu-유틸리티-및-워커)
6. [AVX-512 C++ 커널](#6-avx-512-c-커널)
7. [기타 하이브리드 컴포넌트](#7-기타-하이브리드-컴포넌트)
8. [빌드 시스템](#8-빌드-시스템)
9. [파일 의존관계 맵](#9-파일-의존관계-맵)
10. [구현 완성도 요약](#10-구현-완성도-요약)

---

## 1. 프로젝트 개요

vLLM의 CPU/GPU 하이브리드 추론 최적화 포크. GPU와 CPU를 **별도 프로세스**에서 동시에 실행하여 `total_throughput = GPU + CPU` 달성.

**타겟 하드웨어:**
- GPU: NVIDIA H100 x8 (TP=8)
- CPU: Intel Xeon Platinum 8480+ (Sapphire Rapids), 2소켓 112코어
- RAM: 2TB DDR5, NUMA 멀티소켓
- ISA: AVX-512, AVX-512 VNNI, AMX-BF16/INT8

---

## 2. 핵심 아키텍처: Dual-Process Parallel-Batch

```
HybridAsyncMPClient (CapacityAwareRouter로 라우팅)
├─ input_socket (ZMQ ROUTER)
│   ├─ GPU: identity=b'\x00\x00' (engine_index=0)
│   └─ CPU: identity=b'\x01\x00' (engine_index=1)
├─ output_socket (ZMQ PULL) ← GPU/CPU 결과 비동기 수집
└─ CapacityAwareRouter: CPU 슬롯 여유시 CPU, 가득차면 GPU

GPU EngineCoreProc [별도 프로세스, PID 독립]
├─ EngineCore → MultiprocExecutor (8x H100, TP=8)
└─ KV Cache: GPU VRAM

CPU EngineCoreProc [별도 프로세스, PID 독립]
├─ EngineCore → UniProcExecutor (CPUWorker)
└─ KV Cache: NUMA-aware DRAM
```

### 설계 원칙
1. **core.py 무수정** — hybrid 코드는 `hybrid_core.py`와 `core_client.py`에만 존재
2. **별도 프로세스** — GPU/CPU 각 독립 PID, GIL, busy loop
3. **CapacityAwareRouter** — CPU 슬롯 기반 라우팅 (CPU 활용률 극대화)
4. **자동 감지** — cpu_max_num_seqs, kvcache, threads 모두 0(auto) 기본값
5. **Graceful Fallback** — IPEX/NUMA/AMX 없어도 정상 동작

---

## 3. 하이브리드 엔진 핵심 구현

### 3.1 `vllm/v1/engine/hybrid_core.py` (1,001줄)

하이브리드 엔진의 핵심 파일. 라우팅, 자동 감지, 환경 설정, 프로세스 스폰 전반을 담당.

#### A. RequestRouter (90~139줄) — 기존 라운드-로빈 방식

| 항목 | 내용 |
|------|------|
| 역할 | CPU 비율(`cpu_ratio`) 기반 단순 라운드-로빈 라우팅 |
| 메서드 | `route(request_id)` → "gpu" 또는 "cpu" 반환 |
| 상태 | 유지만 되고 있으며, `CapacityAwareRouter`로 대체됨 |

#### B. CapacityAwareRouter (145~425줄) — 핵심 라우터

CPU 용량 기반 요청 분배 + 실시간 처리량 모니터링.

**구성원:**

| 필드 | 설명 |
|------|------|
| `cpu_max_num_seqs` | CPU 최대 동시 시퀀스 수 |
| `routing_strategy` | `"capacity"` / `"length-aware"` / `"throughput-adaptive"` |
| `cpu_prefill_threshold` | CPU로 보낼 최대 프롬프트 토큰 수 (기본 512) |
| `_warmup_requests` | 워밍업 요청 수 (초기 프로파일링, 기본 10) |
| `_stats_log_interval` | 통계 로깅 주기 (기본 50) |
| `_adaptive_cpu_max_seqs` | EMA 기반 동적 CPU 슬롯 수 |

**3가지 라우팅 전략:**

| 전략 | 메서드 | 알고리즘 | HPC 계보 |
|------|--------|---------|---------|
| **capacity** | `_route_capacity()` | CPU 슬롯 여유 시 CPU, 가득차면 GPU | Work Stealing (Blumofe & Leiserson, 1999) |
| **length-aware** | `_route_length_aware()` | 프롬프트 > τ 토큰이면 GPU, 아니면 capacity 방식 | HEFT (Topcuoglu et al., 2002) |
| **throughput-adaptive** | `_route_throughput_adaptive()` | EMA 처리량 기반 동적 슬롯 조정 | StarPU (Augonnet et al., 2011) |

**EMA (Exponential Moving Average) 처리량 추적:**

```
T̂_d^(t) = α · T_d^(obs) + (1-α) · T̂_d^(t-1)    (α = 0.3)
N^(t) = ⌊N_max · T̂_CPU / (T̂_GPU + T̂_CPU)⌋
동적 슬롯 범위: [2, cpu_max_num_seqs × 2]
```

**워밍업 프로파일링:**
- 완료 조건: GPU 10개 + CPU 10개 완료, 또는 GPU 20개 + CPU 1개
- 완료 시 `_finalize_warmup()` → EMA 초기값 설정
- 이후 `_update_adaptive_slots()` 실시간 조정

**주요 메서드:**

| 메서드 | 역할 |
|--------|------|
| `route(request_id, prompt_len)` | 전략별 라우팅 결정 → "gpu" 또는 "cpu" |
| `on_request_finished(request_id, was_cpu, num_tokens)` | 완료 처리: CPU 슬롯 반환, 처리량 측정, EMA 업데이트 |
| `_finalize_warmup()` | 워밍업 완료 → EMA 초기화 |
| `_update_adaptive_slots()` | CPU/GPU 처리량 비율 기반 슬롯 동적 조정 |
| `get_stats()` | 상세 통계 반환 |

#### C. _resolve_cpu_params (441~569줄) — CPU 자동 감지

입력 `HybridConfig`에서 0(auto) 값을 실측 하드웨어 기반으로 자동 해석.

```
NUMA 감지:
  └─ NUMAAllocator → 노드 토폴로지
     └─ 논리CPU → 물리코어 변환 (threads_per_core로 나눔)

effective_cores:
  └─ NUMA 바인딩 시: 해당 노드 물리 코어
  └─ 아니면: 전체 물리 코어 수

자동 감지값:
  cpu_num_threads     = effective_cores (전부 사용)
  cpu_max_num_seqs    = max(4, effective_cores // 4)
  cpu_kvcache_space_gb = max(32, min(512, effective_mem_gb * 0.4))
  cpu_max_num_batched_tokens = cpu_max_num_seqs × 256
```

반환: `ResolvedCpuParams` 데이터클래스

#### D. _setup_cpu_process_env (576~691줄) — CPU 프로세스 환경 설정

CPU 자식 프로세스 시작 전 Intel 최적화 환경을 구성.

```
1. CUDA_VISIBLE_DEVICES="" (CUDA 격리)
2. VLLM_CPU_KVCACHE_SPACE, OMP_NUM_THREADS 설정
3. configure_intel_optimizations() 호출
   ├─ KMP_AFFINITY=granularity=fine,compact,1,0
   ├─ KMP_BLOCKTIME=1
   ├─ MKL_ENABLE_INSTRUCTIONS=AVX512
   └─ ONEDNN_MAX_CPU_ISA=AVX512_CORE_AMX (AMX 가용 시)
4. configure_pytorch_for_intel() 호출 (IPEX, AMX 타일, Inductor)
5. NUMA affinity 설정 (NUMAAllocator 기반)
```

#### E. _create_cpu_vllm_config (698~790줄) — CPU VllmConfig 파생

GPU VllmConfig에서 CPU 전용 설정을 파생.

| 항목 | GPU → CPU 변경 |
|------|---------------|
| DeviceConfig | `device="cpu"` |
| ParallelConfig | TP=1, PP=1, world_size=1 |
| CacheConfig | `cpu_kvcache_space_bytes = resolved.cpu_kvcache_space_gb × 1024³` |
| SchedulerConfig | `max_num_seqs`, `max_num_batched_tokens` = resolved 값 |
| CompilationConfig | CUDA graph 비활성화 |
| HybridConfig | 비활성화 (CPU 엔진 내부에서 hybrid 미사용) |

`CpuPlatform.check_and_update_config()` 호출로 플랫폼 적용.

#### F. run_cpu_engine_core (797~873줄) — CPU 프로세스 진입점

CPU 자식 프로세스의 메인 함수.

```
1. CUDA_VISIBLE_DEVICES="" 설정
2. Signal handler 등록 (SIGTERM, SIGINT)
3. _resolve_cpu_params() 재호출
4. _setup_cpu_process_env() → Intel 환경 설정
5. _create_cpu_vllm_config() → CPU 전용 config
6. CPU 전용 Executor 클래스 결정
7. EngineCoreProc 생성 + run_busy_loop()
```

#### G. launch_hybrid_engines (909~1001줄) — 프로세스 스폰

GPU + CPU 엔진 프로세스를 동시에 스폰.

```
1. ZMQ IPC 주소 설정
2. 핸드셰이크 소켓 바인딩 (ROUTER mode)
3. GPU 프로세스 스폰 (engine_index=0, EngineCoreProc.run_engine_core)
4. CPU 프로세스 스폰 (engine_index=1, run_cpu_engine_core)
5. HybridEngineProcManager 생성
6. 핸드셰이크 대기 (data_parallel_size_local=2)
```

엔진 Identity:
- GPU: `engine_index=0` → `b'\x00\x00'`
- CPU: `engine_index=1` → `b'\x01\x00'`

---

### 3.2 `vllm/v1/engine/core_client.py` (1,500+ 줄)

클라이언트 ↔ GPU/CPU 엔진 양방향 통신 구현.

#### _HybridEngineLauncherMixin (1338~1357줄)

하이브리드 클라이언트의 공통 엔진 시작 로직.

| 메서드 | 역할 |
|--------|------|
| `_create_engine_launcher()` | `launch_hybrid_engines()` 호출 |
| `_compute_core_engines()` | engine_ranks=[0,1], identities 반환 |

#### HybridAsyncMPClient (1360~1469줄)

상속: `_HybridEngineLauncherMixin + AsyncMPClient`

**핵심 구조:**

| 필드 | 타입 | 설명 |
|------|------|------|
| `_hybrid_router` | CapacityAwareRouter | 라우팅 엔진 |
| `_hybrid_reqs_in_flight` | dict[req_id → identity] | 요청별 실행 엔진 추적 |
| `_hybrid_req_token_counts` | dict[req_id → int] | 요청별 생성 토큰 누적 |

**요청 처리 흐름:**

```
1. add_request_async(request)
   ├─ prompt_len 추출
   ├─ router.route(request_id, prompt_len) → "gpu" or "cpu"
   ├─ 적절한 엔진에 ADD 메시지 전송
   └─ _hybrid_reqs_in_flight[request_id] = engine

2. GPU/CPU 엔진에서 독립 처리 (별도 프로세스)

3. 결과 수신 (ZMQ PULL)
   └─ 토큰 수 누적: _hybrid_req_token_counts[request_id] += num_new

4. 요청 완료 시
   ├─ engine = _hybrid_reqs_in_flight.pop(request_id)
   ├─ num_tokens = _hybrid_req_token_counts.pop(request_id)
   └─ router.on_request_finished(request_id, was_cpu, num_tokens)
       ├─ CPU 슬롯 반환
       ├─ 처리량 누적
       ├─ EMA 업데이트 (throughput-adaptive)
       └─ 주기적 통계 로깅
```

**abort 처리:** 요청별 엔진 그룹화 → 각 엔진에 abort 메시지 전송
**utility 호출:** GPU 엔진에만 전송 (quantization, model_config 조회 등)

#### HybridSyncMPClient (1472~1537줄+)

상속: `_HybridEngineLauncherMixin + SyncMPClient`

HybridAsyncMPClient와 동일 로직을 동기 방식으로 구현.

---

## 4. 설정 및 CLI

### 4.1 HybridConfig (`vllm/config.py`, 4590~4690줄)

| 필드 | 기본값 | 설명 |
|------|--------|------|
| `mode` | `"none"` | `"none"` / `"parallel-batch"` / `"moe-hybrid"` |
| `cpu_ratio` | None | CPU 배치 비율 (RequestRouter용, 구식) |
| `cpu_num_threads` | 0 | CPU 스레드 수 (0=auto) |
| `cpu_dtype` | `"bfloat16"` | CPU 모델 dtype |
| `auto_profile` | True | 자동 프로파일링 |
| `numa_aware` | True | NUMA 최적화 활성화 |
| `numa_bind_node` | None | NUMA 노드 명시 바인딩 (None=auto) |
| `cpu_kvcache_space_gb` | 0 | CPU KV cache (GB) (0=auto, 40% DRAM) |
| `cpu_max_num_seqs` | 0 | CPU 최대 동시 시퀀스 (0=auto, cores/4) |
| `cpu_max_num_batched_tokens` | 0 | CPU 배치 토큰 (0=auto, seqs×256) |
| `routing_strategy` | `"capacity"` | `"capacity"` / `"length-aware"` / `"throughput-adaptive"` |
| `cpu_prefill_threshold` | 512 | CPU로 보낼 최대 프롬프트 토큰 |
| `warmup_requests` | 10 | 워밍업 요청 수 (0=비활성화) |
| `stats_log_interval` | 50 | 통계 로깅 주기 (0=비활성화) |
| `moe_num_gpu_experts` | 8 | GPU MoE expert 수 (미래용) |
| `ngram_enabled` | False | N-gram speculative decoding (미래용) |

메서드: `__post_init__()` (검증), `is_enabled()` (mode != "none")

### 4.2 CLI 인자 (`vllm/engine/arg_utils.py`, 408~440줄)

| CLI 옵션 | 기본값 | 매핑 |
|----------|--------|------|
| `--hybrid-mode` | `"none"` | `HybridConfig.mode` |
| `--hybrid-cpu-ratio` | None | `HybridConfig.cpu_ratio` |
| `--hybrid-cpu-threads` | 0 | `HybridConfig.cpu_num_threads` |
| `--hybrid-numa-aware` / `--no-hybrid-numa-aware` | True | `HybridConfig.numa_aware` |
| `--hybrid-numa-node` | None | `HybridConfig.numa_bind_node` |
| `--hybrid-cpu-kvcache-gb` | 0 | `HybridConfig.cpu_kvcache_space_gb` |
| `--hybrid-cpu-max-seqs` | 0 | `HybridConfig.cpu_max_num_seqs` |
| `--hybrid-cpu-max-batched-tokens` | 0 | `HybridConfig.cpu_max_num_batched_tokens` |
| `--hybrid-routing-strategy` | `"capacity"` | `HybridConfig.routing_strategy` |
| `--hybrid-cpu-prefill-threshold` | 512 | `HybridConfig.cpu_prefill_threshold` |
| `--hybrid-warmup-requests` | 10 | `HybridConfig.warmup_requests` |
| `--hybrid-stats-log-interval` | 50 | `HybridConfig.stats_log_interval` |

---

## 5. Intel CPU 유틸리티 및 워커

### 5.1 `vllm/platforms/intel_cpu_utils.py` (550+ 줄)

#### NUMAAllocator (싱글톤)

NUMA 멀티소켓 시스템 메모리 관리 (libnuma 기반).

| 메서드 | 역할 |
|--------|------|
| `get_node_info(node_id)` | NUMA 노드 정보 (cpu_ids, memory) 반환 |
| `allocate_on_node(size, node_id)` | 특정 노드에 메모리 할당 |
| `get_preferred_node_for_rank(rank, world_size)` | rank % num_nodes 기반 노드 선택 |
| `bind_to_node(node_id)` | 현재 스레드 메모리 할당 노드 바인딩 |

#### IntelCPUFeatures (데이터클래스)

`/proc/cpuinfo`, `lscpu` 파싱으로 CPU 기능 감지.

| 카테고리 | 감지 항목 |
|---------|---------|
| AVX | `avx2`, `avx_vnni` |
| AVX-512 | `avx512f`, `avx512_vnni`, `avx512_bf16` |
| AMX | `amx_bf16`, `amx_int8` (Sapphire Rapids+) |
| 토폴로지 | `model_name`, `num_sockets`, `cores_per_socket`, `threads_per_core`, `l3_cache_mb` |

#### configure_intel_optimizations(features)

환경변수 기반 Intel 최적화 설정.

| 환경변수 | 값 | 조건 |
|---------|-----|------|
| `KMP_AFFINITY` | `granularity=fine,compact,1,0` | 항상 |
| `KMP_BLOCKTIME` | `1` | 항상 |
| `KMP_TPAUSE` | `0` | 항상 |
| `MKL_ENABLE_INSTRUCTIONS` | `AVX512` / `AVX2` | ISA 따라 |
| `ONEDNN_MAX_CPU_ISA` | `AVX512_CORE_AMX` / `AVX512_CORE` | AMX 따라 |

#### configure_pytorch_for_intel(features)

PyTorch Inductor + IPEX + AMX 타일 설정. `OMP_NUM_THREADS` 존재 시 해당 값 사용.

### 5.2 `vllm/v1/worker/cpu_worker.py` (530줄)

GPU Worker와 동일 인터페이스의 CPU 워커.

**하이브리드 모드 감지:**

```python
_is_hybrid_cpu_process = (
    CUDA_VISIBLE_DEVICES == ""
    and VLLM_CPU_KVCACHE_SPACE is not None
)
```

**초기화 흐름:**
- CPU-only 모드: `setup_intel_cpu_environment()` 호출 (NUMA, AVX, IPEX 전부)
- Hybrid 모드: env 이미 `_setup_cpu_process_env()`에서 구성됨 → 기능 감지 로깅만

**IPEX 의존성:** IPEX 미지원 시 순수 PyTorch SDPA fallback (성능 대폭 저하)

### 5.3 `vllm/v1/attention/backends/cpu_attn.py` (200+ 줄)

CPU Attention 백엔드.

| 클래스 | 역할 |
|--------|------|
| `TorchSDPABackend` | `torch.nn.functional.scaled_dot_product_attention` 활용 |
| `TorchSDPAMetadata` | 시퀀스 길이, 블록 테이블, 청크 프리필 메타데이터 |

IPEX 조건부 로딩: `intel_extension_for_pytorch.llm.modules` → 미지원 시 PyTorch fallback.

---

## 6. AVX-512 C++ 커널

빌드 타겟: `_C_cpu_ops.abi3.so` (CPU 전용 확장 라이브러리)

### 6.1 `csrc/cpu/batch_attention.cpp` — 배치 Attention

| 항목 | 내용 |
|------|------|
| 핵심 함수 | `batch16_paged_attention_v1()` |
| 배치 크기 | BATCH16 = 16 (16개 시퀀스 동시 처리) |
| 데이터 타입 | BF16, FP32 |
| Query 구조 | `[num_seqs, num_heads, head_size]` |
| KV Cache | Paged Attention: K=`[D/x, block_size, x]`, V=`[D, block_size]` |
| 병렬화 | `#pragma omp parallel for schedule(dynamic, 1)` |
| 정렬 | 64바이트 (AVX-512 캐시라인) |

**AVX-512 활용:**
- `bf16x16_to_fp32()`: 16-bit 좌측 시프트로 BF16→FP32 변환 (AVX512BF16 ISA 불필요)
- `_mm512_fmadd_ps()`: FP32 FMA (Fused Multiply-Add) — Q·K 점곱
- `_mm512_reduce_add_ps()`: 수평 리듀스

**프리페치 최적화 (4개 지점):**
- `_mm_prefetch(..., _MM_HINT_T1)`: 다음 블록의 K/V를 L2로 프리페치
- 4개 캐시라인 × 64바이트씩 미리 로드 (라인 165-173, 257-264, 310-318, 376-384)

### 6.2 `csrc/cpu/gemm_vnni.cpp` — VNNI INT8 GEMM

| 항목 | 내용 |
|------|------|
| 마이크로커널 | `vnni_micro_kernel_6x16()` — MR=6, NR=16 |
| ISA | `_mm512_dpbusd_epi32()` (VNNI INT8 내적) |
| 포맷 | u8s8 + s8s8 보상 (SGL 호환) |
| GEMV 경로 | `int8_gemv_vnni()` — M=1 최적화 |
| GEMM 경로 | `int8_gemm_vnni()` — 3단계 캐시 블로킹 |
| 역양자화 | `dequant_int32_to_output()` — INT32 → BF16/FP32/FP16 |

**3단계 캐시 블로킹:**

| 레벨 | 타일 | 크기 | 캐시 타겟 |
|------|------|------|---------|
| L1 | NC (열) | 256 | L1/L2 |
| L2 | MC (행) | 72 | L2 (OpenMP 병렬화) |
| L3 | KC (K축) | 256 | L3 |

**가중치 패킹:** `[N, K]` row-major → `[N/16, K/4, 16, 4]` VNNI 포맷 + 보상값 계산

### 6.3 `csrc/cpu/quant_q8_0.cpp` — Q8_0 양자화

**Q8_0 블록 구조:**

```
Q8_0Block {
  uint16_t scale_fp16;   // 2 바이트 — FP16 스케일
  int8_t quants[32];     // 32 바이트 — INT8 양자화값
}  // 총 34 바이트/블록
```

| 함수 | 역할 |
|------|------|
| `q8_0_gemv_vnni_impl()` | 블록당 동적 스케일링 GEMV (VNNI 처리) |
| `q8_0_linear_impl()` | 동적 입력 양자화 + AVX-512 최대값 감지 |
| `quantize_to_q8_0_impl()` | 가중치 → Q8_0 변환 |

**ISA 활용:**
- `_mm_cvtph_ps()` / `_mm_cvtps_ph()`: F16C FP16 변환
- `_mm512_cvtepi8_epi16()` / `_mm512_madd_epi16()`: INT8 → INT16 MADD
- `_mm512_reduce_max_ps()`: AVX-512 최대값 감지

### 6.4 `csrc/cpu/decode_gemv.cpp` — Decode GEMV

| 함수 | 역할 |
|------|------|
| `bf16_dot_product_avx512()` | BF16 단일 행 내적 (2×16 언롤) |
| `fp32_dot_product_avx512()` | FP32 단일 행 내적 (2×16 언롤) |
| `bf16_decode_gemv()` | 단일 시퀀스 Decode (M=1) |
| `bf16_batch_gemv()` | 배치 GEMV (B ≤ 32) |
| `decode_gemv()` | PyTorch 엔트리 포인트 (M=1/M>1 분기) |

**최적화 기법:**
- 누적기 2개 (`acc0`, `acc1`): 파이프라인 병렬성 증가
- 루프 언롤링: 32 요소(2×16) — 3-cycle FMA 레이턴시 숨김
- 프리페치: 다음 가중치 행 미리 로드
- OpenMP: `schedule(static)` 병렬화

### 6.5 `csrc/cpu/mem_opt.cpp` — 메모리 최적화

| 함수 | 역할 |
|------|------|
| `alloc_on_node()` | NUMA 노드별 메모리 할당 |
| `alloc_interleaved()` | NUMA 인터리브 할당 |
| `bind_thread_to_node()` | 스레드 바인딩 |
| `nt_memcpy()` | Non-Temporal Memcpy (캐시 우회) |
| `prefetch_kv_blocks()` | KV 캐시 블록 다단계 프리페치 |

**Non-Temporal Memcpy:**
- 임계값: 256KB (L2 캐시 크기 초과 시 적용)
- 루프: 512바이트/반복 (8개 캐시라인)
- `_mm512_stream_si512()`: 캐시 우회 스트림 스토어
- `_mm_sfence()`: 메모리 펜스

**KV 캐시 3단계 프리페치:**

| 단계 | 힌트 | 타겟 | 용도 |
|------|------|------|------|
| T0 | `_MM_HINT_T0` | L1/L2 | 현재 블록 |
| T1 | `_MM_HINT_T1` | L2 | 다음 블록 |
| NTA | `_MM_HINT_NTA` | 비템포럴 | 먼 블록 |

### 6.6 커널 요약

| 커널 | 주요 ISA | 최적화 기법 | 병렬화 |
|------|---------|-----------|--------|
| batch_attention | AVX-512 FP32/BF16 FMA | L2 프리페치, 16-시퀀스 배치 | OpenMP dynamic |
| gemm_vnni | VNNI INT8 (dpbusd) | 3단계 캐시 블로킹, T0 프리페치 | OpenMP dynamic |
| quant_q8_0 | VNNI + F16C | 동적 양자화, 블록 스케일 | OpenMP static |
| decode_gemv | AVX-512 FP32/BF16 FMA | 루프 언롤링 (2×16), 프리페치 | OpenMP static |
| mem_opt | AVX-512 stream_si512 | NT memcpy (256KB), 다단계 프리페치 | — |

---

## 7. 기타 하이브리드 컴포넌트

### 7.1 MoE Expert Offload (`vllm/model_executor/layers/fused_moe/expert_offload.py`)

GPU/CPU 간 MoE Expert 분배 관리. **완성도: ~70%**

| 클래스 | 역할 |
|--------|------|
| `ExpertOffloadConfig` | MoE Expert offloading 설정 |
| `ExpertStats` | Expert 사용 통계 추적 |
| `ExpertOffloadManager` | LRU 기반 GPU/CPU Expert 캐싱 + 비동기 전송 |

**핵심 알고리즘:**
- LRU 기반 Expert 캐싱: 자주 사용되는 Expert → GPU, 드물게 사용 → CPU
- Async CPU-GPU 전송: CUDA Stream으로 비동기 가중치 교환
- INT8 양자화: CPU Expert 메모리 효율성
- ThreadPoolExecutor: CPU Expert 병렬 실행
- Swap 주기: 100 forward pass마다 체크, 빈도 상위 N개를 GPU로 이동

### 7.2 Dynamic N-gram Proposer (`vllm/v1/spec_decode/ngram_proposer_dynamic.py`)

CPU 전용 실시간 N-gram Speculative Decoding. **완성도: ~85%**

| 클래스 | 역할 |
|--------|------|
| `DynamicNgramConfig` | N=2~4, min_frequency, max_table_size 설정 |
| `NgramStats` | 제안 성공률, 수락률, 테이블 크기 통계 |
| `DynamicNgramProposer` | 실시간 학습 + N-gram 테이블 기반 제안 |
| `HybridNgramWorker` | GPU 모델과의 협력 인터페이스 |

**핵심 알고리즘:**
- 제안: N=4,3,2 순서로 테이블 조회 (큰 N부터 구체성 우선)
- 학습: 생성된 토큰 → Queue → 백그라운드 스레드에서 비동기 업데이트
- 테이블 관리: max 100,000 항목, frequency decay (0.99 factor)
- RLock으로 스레드 안전성 보장

### 7.3 Disaggregated Serving (`vllm/engine/disaggregated/`)

Prefill/Decode 분리 서빙 아키텍처. **완성도: ~40% (스켈레톤)**

| 파일 | 주요 클래스 | 역할 |
|------|-----------|------|
| `kv_transfer.py` | `KVCacheSender`, `KVCacheReceiver`, `TCPTransferBackend`, `SharedMemoryBackend` | KV Cache 전송 (TCP/SHM 완성, RDMA 미구현) |
| `coordinator.py` | `PrefillNode`, `DecodeNode`, `DisaggregatedCoordinator`, `LoadBalancer` | 요청 조율 + Prefill/Decode 노드 관리 |

**아키텍처 흐름:**

```
Request → DisaggregatedCoordinator
├─ Load balance → Select Prefill Node
├─ PrefillNode.run_prefill() → KV cache 전송
├─ Load balance → Select Decode Node
└─ DecodeNode.run_decode()
    ├─ KV cache 수신
    ├─ N-gram Speculative Proposals (CPU, 선택)
    ├─ MoE Expert Offload (선택)
    └─ Decode Loop → Output tokens
```

### 7.4 Parallel Batch Executor (`vllm/executor/parallel_batch_executor.py`, 1,033줄)

CPU와 GPU가 서로 다른 배치를 동시 처리하는 실행기. **완성도: ~75%**

| 클래스 | 역할 |
|--------|------|
| `ProfileResult` | GPU/CPU 처리량 및 최적 비율 저장 |
| `ParallelBatchProfiler` | CPU/GPU 처리량 자동 측정 |
| `ParallelBatchScheduler` | 요청 분할 (길이 기반, 비율 기반) |
| `CPUWorkerWrapper` | IPEX + AVX-512 + NUMA 최적화 워커 |
| `ParallelBatchExecutor` | 메인 실행기 (동기/비동기) |
| `MoEHybridExecutor` | 미구현 (스켈레톤만) |

**IPEX 최적화 계층:**

| 타입 | 우선순위 |
|------|---------|
| INT8 | AMX-INT8 > AVX-512 VNNI > PyTorch |
| BF16 | AMX-BF16 > AVX-512 BF16 > PyTorch |
| FP32 | AMX (FP32→BF16) > AVX2 |

---

## 8. 빌드 시스템

### `cmake/cpu_hybrid_extension.cmake`

| 항목 | 내용 |
|------|------|
| 아키텍처 | x86_64만 지원 |
| ISA 감지 | `/proc/cpuinfo` 파싱 (필수: AVX-512F, 선택: AVX-512 VNNI) |
| 컴파일 플래그 | `-mavx512f -mavx512vl -mavx512bw -mavx512dq -mf16c -O3 -funroll-loops -ffast-math` |
| VNNI 활성 시 | `-mavx512vnni` (GCC ≥ 12.3) |
| 라이브러리 | OpenMP (권장), libnuma (선택, 없으면 `-DVLLM_NUMA_DISABLED`) |
| 빌드 타겟 | `_C_cpu_ops` (shared library) → `vllm/` 디렉토리 |
| ABI | abi3 호환성 |

**소스 파일 포함 규칙:**

| 조건 | 포함 파일 |
|------|---------|
| 항상 | `torch_bindings_hybrid.cpp`, `decode_gemv.cpp`, `batch_attention.cpp`, `mem_opt.cpp` |
| VNNI 지원 시 | + `gemm_vnni.cpp`, `quant_q8_0.cpp` |

**빌드 명령:**

```bash
pip install -e . --config-settings="cmake.args=-DVLLM_TARGET_DEVICE=cuda"
# 결과: _C.abi3.so (CUDA) + _C_cpu_ops.abi3.so (CPU 커널)
```

---

## 9. 파일 의존관계 맵

```
┌─────────────────────────────────────────────────────┐
│             Application / EngineArgs                │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  arg_utils.py (CLI 옵션)    │
        │  - hybrid_mode             │
        │  - hybrid_cpu_* 옵션들     │
        │  - routing_strategy 등     │
        └────────────┬───────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  config.py (HybridConfig)  │
        │  - 모든 hybrid 설정값      │
        │  - __post_init__ 검증     │
        └────────────┬───────────────┘
                     │
        ┌────────────┴───────────────────────┐
        │                                    │
        ▼                                    ▼
   ┌────────────────────┐    ┌──────────────────────────┐
   │ core_client.py     │    │ hybrid_core.py (핵심)    │
   │ - HybridAsync      │    │ - CapacityAwareRouter    │
   │   MPClient         │◄──┤ - _resolve_cpu_params    │
   │ - HybridSyncMP     │    │ - _setup_cpu_process_env │
   │   Client           │    │ - run_cpu_engine_core    │
   │ - _HybridEngine    │    │ - launch_hybrid_engines  │
   │   LauncherMixin    │    └────────────┬─────────────┘
   └────────────────────┘                 │
                                          │
                    ┌─────────────────────┤
                    │                     │
                    ▼                     ▼
              ┌──────────────┐    ┌─────────────────┐
              │ intel_cpu_   │    │ cpu_worker.py   │
              │ utils.py     │    │ - CPUWorker     │
              │ - NUMA       │    │ - Intel 감지    │
              │ - CPU 감지   │    │ - Hybrid 인식   │
              │ - 환경 설정  │    └─────────┬───────┘
              └──────────────┘              │
                                           ▼
                                    ┌──────────────────┐
                                    │ cpu_attn.py      │
                                    │ - TorchSDPA      │
                                    │ - IPEX PagedAttn │
                                    └──────────────────┘

  [C++ 커널 레이어]
  ┌──────────────────────────────────────────────────┐
  │ _C_cpu_ops.abi3.so                               │
  │ ├─ batch_attention.cpp (16-시퀀스 배치 Attn)     │
  │ ├─ decode_gemv.cpp (BF16/FP32 GEMV)              │
  │ ├─ mem_opt.cpp (NT memcpy, NUMA, 프리페치)       │
  │ ├─ gemm_vnni.cpp (VNNI INT8 GEMM) [조건부]      │
  │ └─ quant_q8_0.cpp (Q8_0 양자화) [조건부]        │
  └──────────────────────────────────────────────────┘

  [미래용 컴포넌트]
  ┌──────────────────────────────────────────────────┐
  │ ├─ expert_offload.py (MoE Expert GPU/CPU 분배)   │
  │ ├─ ngram_proposer_dynamic.py (N-gram Spec Decode)│
  │ ├─ parallel_batch_executor.py (배치 병렬 실행)   │
  │ └─ disaggregated/ (Prefill/Decode 분리 서빙)     │
  └──────────────────────────────────────────────────┘
```

---

## 10. 구현 완성도 요약

### 핵심 컴포넌트 (프로덕션 준비)

| 컴포넌트 | 파일 | 완성도 | 상태 |
|---------|------|--------|------|
| CapacityAwareRouter | hybrid_core.py | **95%** | 3가지 전략 완성, EMA + 워밍업 |
| CPU 자동 감지 | hybrid_core.py | **95%** | NUMA, 코어, KV cache 자동 결정 |
| HybridAsyncMPClient | core_client.py | **95%** | ZMQ 통신, 요청 추적, 라우터 통합 |
| HybridSyncMPClient | core_client.py | **95%** | 동기 버전 완성 |
| Intel CPU 유틸리티 | intel_cpu_utils.py | **90%** | NUMA, AVX-512, AMX, IPEX 통합 |
| CPU Worker | cpu_worker.py | **90%** | Hybrid 모드 인식, IPEX fallback |
| CPU Attention | cpu_attn.py | **85%** | TorchSDPA + IPEX PagedAttn |
| HybridConfig | config.py | **95%** | 전체 옵션 정의 + 검증 |
| CLI 인자 | arg_utils.py | **95%** | 12개 hybrid 옵션 완성 |

### C++ 커널

| 커널 | 파일 | 완성도 | 상태 |
|------|------|--------|------|
| Batch Attention | batch_attention.cpp | **90%** | 16-시퀀스 배치, L2 프리페치 |
| VNNI INT8 GEMM | gemm_vnni.cpp | **85%** | 6x16 마이크로커널, 3단계 블로킹 |
| Q8_0 양자화 | quant_q8_0.cpp | **85%** | 블록 양자화 + VNNI GEMV |
| Decode GEMV | decode_gemv.cpp | **90%** | BF16/FP32, 2×16 언롤 |
| 메모리 최적화 | mem_opt.cpp | **85%** | NT memcpy, NUMA, 3단계 프리페치 |
| 빌드 시스템 | cpu_hybrid_extension.cmake | **90%** | ISA 감지, 조건부 빌드 |

### 미래용 컴포넌트

| 컴포넌트 | 파일 | 완성도 | 상태 |
|---------|------|--------|------|
| MoE Expert Offload | expert_offload.py | **70%** | LRU 캐싱 완성, 프로덕션 테스트 필요 |
| N-gram Spec Decode | ngram_proposer_dynamic.py | **85%** | 실시간 학습 완성, 메모리 최적화 필요 |
| Parallel Batch Executor | parallel_batch_executor.py | **75%** | 프레임워크 완성, 원본 executor 연결 필요 |
| Disaggregated Serving | disaggregated/ | **40%** | 스켈레톤, 분산 RPC 미구현 |
| KV Transfer | kv_transfer.py | **75%** | TCP/SHM 완성, RDMA 미구현 |

### 설계 패턴

| 패턴 | 위치 | 설명 |
|------|------|------|
| 싱글톤 | NUMAAllocator | NUMA 정보 시스템당 1개 |
| EMA | CapacityAwareRouter | 실시간 처리량 추적 (α=0.3) |
| Work Stealing | capacity 전략 | idle CPU가 GPU 작업 가져감 |
| HEFT | length-aware 전략 | 최소 완료시간 디바이스 선택 |
| StarPU 방식 | throughput-adaptive | 런타임 성능 캘리브레이션 |
| Graceful Fallback | intel_cpu_utils, cpu_worker | IPEX/NUMA/AMX 미지원 시 기본 동작 |
| 프로세스 격리 | hybrid_core | GPU/CPU 별도 PID → GIL 회피 |

---

*이 문서는 코드베이스의 실제 구현 상태를 반영합니다. 논문(`docs/paper/main.tex`)의 이론적 설명과 대응됩니다.*
