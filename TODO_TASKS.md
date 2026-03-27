# vLLM Hybrid - Code Tasks

> 논문/문서 작업 제외, 코드 관련 완료 항목 및 할 일 정리
> 마지막 업데이트: 2026-03-27

---

## 완료된 작업 (Done)

### A. 핵심 아키텍처: Dual-Process Parallel-Batch

| # | 작업 | 파일 | 완료일 |
|---|------|------|--------|
| A1 | GPU/CPU 별도 프로세스 병렬 실행 (GIL 제거) | `hybrid_core.py` | 2026-02-21 |
| A2 | ZMQ IPC 통신 (ROUTER/DEALER + PUSH/PULL) | `hybrid_core.py`, `core_client.py` | 2026-02-21 |
| A3 | CapacityAwareRouter 구현 (3가지 전략) | `hybrid_core.py` | 2026-02-21 |
| A4 | HybridAsyncMPClient / HybridSyncMPClient | `core_client.py` | 2026-02-21 |
| A5 | CPU 파라미터 자동 감지 (`_resolve_cpu_params`) | `hybrid_core.py` | 2026-02-21 |
| A6 | CPU 환경 자동 설정 (`_setup_cpu_process_env`) | `hybrid_core.py` | 2026-02-21 |
| A7 | `vllm serve` hybrid 경로 연결 (`make_client` 분기) | `core_client.py` | 2026-03-10 |
| A8 | NUMA 어피니티 일관성 보장 | `hybrid_core.py`, `cpu_worker.py` | 2026-02-21 |

### B. CPU 스케줄링 고도화

| # | 작업 | 파일 | 완료일 |
|---|------|------|--------|
| B1 | `capacity` 전략: 슬롯 기반 라우팅 | `hybrid_core.py` | 2026-02-26 |
| B2 | `length-aware` 전략: 프롬프트 길이 기반 라우팅 | `hybrid_core.py` | 2026-02-26 |
| B3 | `throughput-adaptive` 전략: EMA 기반 동적 슬롯 조정 | `hybrid_core.py` | 2026-02-26 |
| B4 | CLI: `--hybrid-routing-strategy`, `--hybrid-cpu-prefill-threshold` | `arg_utils.py` | 2026-02-26 |
| B5 | 워밍업 프로파일링 (첫 N개 요청 throughput 수집) | `hybrid_core.py`, `core_client.py` | 2026-02-26 |
| B6 | 주기적 통계 로깅 (GPU/CPU throughput, in_flight) | `hybrid_core.py` | 2026-02-26 |
| B7 | CLI: `--hybrid-warmup-requests`, `--hybrid-stats-log-interval` | `arg_utils.py` | 2026-02-26 |

### C. Intel CPU 최적화

| # | 작업 | 파일 | 완료일 |
|---|------|------|--------|
| C1 | CPU 기능 감지 (AVX-512, VNNI, AMX, BF16) | `intel_cpu_utils.py` | 2026-02-03 |
| C2 | NUMA 토폴로지 감지 및 메모리 할당 | `intel_cpu_utils.py` | 2026-02-03 |
| C3 | OpenMP 스레드 바인딩 (KMP_AFFINITY) | `intel_cpu_utils.py` | 2026-02-03 |
| C4 | ONEDNN/MKL ISA 자동 선택 | `intel_cpu_utils.py` | 2026-02-03 |
| C5 | AMX 타일 활성화 (ARCH_REQ_XCOMP_PERM 시스콜) | `intel_cpu_utils.py` | 2026-02-03 |
| C6 | IPEX 자동 통합 및 에러 핸들링 | `intel_cpu_utils.py`, `_ipex_ops.py` | 2026-02-03 |
| C7 | NUMA-aware KV Cache 할당 | `cpu_worker.py`, `cpu_model_runner.py` | 2026-02-03 |
| C8 | PyTorch Inductor CPU 컴파일 설정 | `cpu_worker.py` | 2026-02-03 |

### D. AVX-512 C++ 커널

| # | 작업 | 파일 | 완료일 |
|---|------|------|--------|
| D1 | Phase 1: VNNI INT8 GEMM (6x16 마이크로커널, 3단계 캐시 블로킹) | `gemm_vnni.cpp/hpp` | 2026-02-19 |
| D2 | Phase 2: Q8_0 양자화/역양자화 (llama.cpp 호환) | `quant_q8_0.cpp` | 2026-02-19 |
| D3 | Phase 3: BF16/FP32 Decode GEMV (32-element 언롤, SW 프리페치) | `decode_gemv.cpp` | 2026-02-19 |
| D4 | Phase 4: Batch Attention (Batch-16, AVX-512 FMA, L2 프리페치) | `batch_attention.cpp` | 2026-02-19 |
| D5 | Phase 5: NT memcpy, NUMA 할당, KV cache 블록 프리페치 | `mem_opt.cpp` | 2026-02-19 |
| D6 | `cpu_attn.py`에 `_C_cpu_ops.batch16_paged_attention_v1` 디스패치 연결 | `cpu_attn.py` | 2026-03-10 |
| D7 | `batch_attention.cpp` inline `_mm_prefetch` (K/V → L2) 삽입 | `batch_attention.cpp` | 2026-02-26 |

### E. 빌드 시스템

| # | 작업 | 파일 | 완료일 |
|---|------|------|--------|
| E1 | CUDA + CPU 동시 빌드 (`_C.abi3.so` + `_C_cpu_ops.abi3.so`) | `cpu_hybrid_extension.cmake` | 2026-02-20 |
| E2 | AVX-512/VNNI 자동 감지 및 조건부 컴파일 | `cpu_hybrid_extension.cmake` | 2026-02-20 |
| E3 | NVTX 헤더 호환성 워크어라운드 (CUDA 12.x) | `CMakeLists.txt` | 2026-02-20 |
| E4 | PyTorch `int64_t`/`double` 디스패치 래퍼 추가 | `torch_bindings_hybrid.cpp` | 2026-02-20 |

### F. Heterogeneous Platform (V0 레거시)

| # | 작업 | 파일 | 완료일 |
|---|------|------|--------|
| F1 | HeterogeneousPlatform 클래스 (Lazy init, CUDA/CPU 위임) | `heterogeneous.py` | 2026-02-03 |
| F2 | Rank 기반 워커 선택 (GPU rank < GPU count → CUDA, else → CPU) | `worker.py`, `worker_base.py` | 2026-02-03 |
| F3 | Hierarchical AllReduce (NCCL → CPU bridge → Gloo) | `parallel_state.py` | 2026-02-03 |
| F4 | CPU 워커 NUMA 바인딩 | `worker.py` | 2026-02-03 |
| F5 | CPU attention stub 및 fallback | `cpu_stubs.py`, `layer.py` | 2026-02-03 |
| F6 | Gloo 백엔드 강제 (Heterogeneous 모드) | `gpu_worker.py`, `parallel_state.py` | 2026-02-03 |

### G. 미래용 컴포넌트 (구현 완료, 미통합)

| # | 작업 | 파일 | 완료일 |
|---|------|------|--------|
| G1 | MoE Expert Offload (LRU 캐싱, INT8 양자화, 비동기 전송) | `expert_offload.py` | 2026-02-04 |
| G2 | 동적 N-gram Speculative Decode (학습 기반 제안자) | `ngram_proposer_dynamic.py` | 2026-02-04 |
| G3 | Disaggregated Serving 코디네이터 (Prefill/Decode 분리) | `coordinator.py` | 2026-02-04 |
| G4 | KV Cache 전송 (TCP/SHM/RDMA 백엔드) | `kv_transfer.py` | 2026-02-04 |
| G5 | Q8_0 양자화 레이어 (CPU 전용, llama.cpp 호환) | `q8_0.py` | 2026-02-04 |

### H. 테스트 및 검증

| # | 작업 | 파일 | 완료일 |
|---|------|------|--------|
| H1 | 단위 테스트 30개 (CapacityAwareRouter, _resolve_cpu_params) | `test_hybrid_core.py` | 2026-03-10 |
| H2 | 하이브리드 병렬 검증 스크립트 | `verify_hybrid_parallel.py` | 2026-02-26 |
| H3 | AMX 감지 확인 스크립트 | `check_amx.sh` | 2026-02-03 |

### I. 버그 수정

| # | 작업 | 파일 | 완료일 |
|---|------|------|--------|
| I1 | CPU 프로세스 종료 시 hang 해결 | `hybrid_core.py` | 2026-02-20 |
| I2 | IPEX AttributeError/Exception 핸들링 | `_ipex_ops.py` | 2026-02-03 |
| I3 | Heterogeneous 모드 NCCL/Gloo 혼합 deadlock 방지 | `parallel_state.py` | 2026-02-03 |
| I4 | CPU 워커에서 Mamba/Triton 초기화 방지 (lazy import) | `gpu_model_runner.py` | 2026-02-20 |
| I5 | `compute_qk_batch16` 미사용 스텁 함수 제거 | `batch_attention.cpp` | 2026-03-10 |
| I6 | ParallelBatchExecutor deprecated 표시 | `parallel_batch_executor.py` | 2026-03-10 |

---

## 해야 할 작업 (TODO)

### 높은 우선순위

| # | 작업 | 상세 | 관련 파일 |
|---|------|------|-----------|
| T1 | **실제 H100 환경 벤치마크** | GPU/CPU throughput 실측, Roofline 검증, 논문 Table 값 실증 | 벤치마크 스크립트 필요 |
| T2 | **GPU throughput 프로파일링 개선** | 현재 워밍업 기반 측정 부정확 → GPU executor 직접 측정 필요 (`HYBRID_OPTIONS_IMPLEMENTATION_PLAN.md` 912-920줄 TODO) | `hybrid_core.py` |
| T3 | **통합 테스트 확대** | 현재 30개 단위 테스트 → E2E 테스트 (실제 모델 로딩, 추론, 결과 검증) 필요 | `tests/` |
| T4 | **디버그 print 문 정리** | `DEBUG_AG` prefix print 다수 존재 → logger 전환 또는 제거 | `multiproc_executor.py`, `parallel_state.py`, `gpu_model_runner.py`, `worker.py` |

### 중간 우선순위

| # | 작업 | 상세 | 관련 파일 |
|---|------|------|-----------|
| T5 | **MoE-Hybrid 엔진 통합** | `expert_offload.py` 구현은 완료되었으나 엔진에 미연결. HybridConfig에 MoE 설정 추가, `--moe-cpu-offload` CLI, hybrid_core에 moe-hybrid 경로 | `hybrid_core.py`, `config.py`, `arg_utils.py` |
| T6 | **N-gram Speculative Decode 통합** | `ngram_proposer_dynamic.py` 구현 완료, 엔진 decode 루프에 연결 필요 | `core.py`, `ngram_proposer_dynamic.py` |
| T7 | **Disaggregated Serving 통합** | `coordinator.py`, `kv_transfer.py` 구현 완료, 실제 서빙 파이프라인 연결 필요 | `coordinator.py`, `kv_transfer.py` |
| T8 | **CPU fault tolerance 강화** | CPU 프로세스 crash 시 GPU-only fallback은 암묵적. Health-check watchdog 필요 (논문 Property 2 관련) | `hybrid_core.py`, `core_client.py` |
| T9 | **RAPL 에너지 측정** | 논문에서 "theoretical estimate"로 명시. 실제 RAPL 기반 전력 측정 구현 필요 | 신규 파일 |

### 낮은 우선순위

| # | 작업 | 상세 | 관련 파일 |
|---|------|------|-----------|
| T10 | **AVX-512 커널 마이크로 최적화** | 캐시 블로킹 파라미터 튜닝, AMX 타일 활용 확대 | `csrc/cpu/*.cpp` |
| T11 | **Continuous batching 상호작용 분석** | 논문 Limitations에 명시된 미분석 사항 | `hybrid_core.py` |
| T12 | **모델 가중치 중복 로딩 최적화** | GPU/CPU 각각 ~70GB 로드 → 공유 메모리 또는 순차 로딩 검토 | `hybrid_core.py` |
| T13 | **EMA 스무딩 계수 sensitivity 분석** | 현재 alpha=0.3 고정, half-life 1.9회. 다양한 워크로드에서 최적값 프로파일링 | `hybrid_core.py` |
| T14 | **Cross-platform 테스트** | AMD CPU, ARM 등 non-Intel 환경에서 graceful fallback 검증 | `intel_cpu_utils.py` |
| T15 | **Upstream vLLM 최신 버전 rebase** | Fork 기점 이후 upstream 변경사항 병합 | 전체 |

---

## 알려진 이슈 / 제한사항

| # | 이슈 | 상태 | 비고 |
|---|------|------|------|
| K1 | IPEX 미설치 시 CPU 성능 저하 (Python fallback) | 문서화 완료 | graceful fallback 동작 |
| K2 | AMX 활성화에 Linux 5.16+ 커널 필요 | 문서화 완료 | 컨테이너 호스트 커널 의존 |
| K3 | CPU TTFT 10-50x 느림 (prefill) | 논문에 명시 | length-aware 라우팅으로 완화 |
| K4 | ZMQ IPC 레이턴시 (고빈도 소형 메시지) | 인지됨 | 현재 수준 실용적 |
| K5 | 모델 가중치 이중 로딩 (~70GB, startup 2배) | 논문에 명시 | T12에서 최적화 계획 |
| K6 | DGX H100 single-socket NUMA에서 DDR5 경합 가능 | 논문에 caveat 추가 | 멀티소켓 환경 권장 |
| K7 | p99 TTFT 바운딩 어려움 (CPU/GPU 혼합) | 논문에 명시 | SLO 민감 워크로드 주의 |

---

## 출처 파일 목록

이 문서는 아래 MD 파일들을 분석하여 작성되었습니다:

- `CHANGES_FROM_UPSTREAM.md`, `RESEARCH.md`, `FUTURE_PLAN.md`, `PLAN.md`, `Deployment.md`
- `docs/HYBRID_OPTIONS_IMPLEMENTATION_PLAN.md`, `docs/HETEROGENEOUS_CPU_OPTIMIZATIONS.md`
- `docs/AVX512_OPTIMIZATION_IMPLEMENTATION_PLAN.md`, `docs/REPORT.md`, `docs/test_method.md`
- `analysis/task.md`, `analysis/implementation_plan.md`, `analysis/heterogeneous_design.md`
- `analysis/heterogeneous_modifications_summary.md`, `analysis/heterogeneous_worker_fix_analysis.md`
- `analysis/kv_cache_numa_analysis.md`, `analysis/overview.md`, `analysis/communication.md`
- `analysis/walkthrough.md`
