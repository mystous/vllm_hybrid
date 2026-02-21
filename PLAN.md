# vLLM Hybrid 구현 현황

## 완료된 작업

### 1. Parallel-Batch Dual-Process 아키텍처 (2026-02-21) ✅
- GPU + CPU 별도 프로세스 병렬 실행
- CapacityAwareRouter (CPU 용량 기반 라우팅)
- CPU 파라미터 자동 감지 (_resolve_cpu_params)
- Intel 최적화 자동 설정 (_setup_cpu_process_env)
- NUMA affinity 일관성 보장

### 2. Intel CPU 최적화 (2026-02-03) ✅
- NUMA-aware KV Cache 할당
- AVX-512/AMX/VNNI 자동 감지 및 활성화
- IPEX 자동 통합
- OpenMP 스레드 바인딩

### 3. AVX-512 C++ 커널 (2026-02-19) ✅
- Phase 1: VNNI INT8 GEMM
- Phase 2: Q8_0 양자화
- Phase 3: Decode GEMV
- Phase 4: 배치 Attention
- Phase 5: 메모리 최적화

### 4. CUDA + CPU 동시 빌드 (2026-02-20) ✅
- _C_cpu_ops 별도 CMake 타겟
- CUDA 빌드 시 CPU 커널 동시 빌드

### 5. Option A 컴포넌트 (2026-02-04) ✅
- MoE Expert Offload (expert_offload.py)
- N-gram Proposer (ngram_proposer_dynamic.py)
- Disaggregated Serving (coordinator.py, kv_transfer.py)

---

## 미구현 (향후 작업)

### Option A (moe-hybrid) 엔진 통합
- HybridConfig에 MoE 세부 설정 추가
- CLI 인자 추가 (--moe-cpu-offload 등)
- moe-hybrid 엔진 코어 통합
- 상세 내용: `docs/HYBRID_OPTIONS_IMPLEMENTATION_PLAN.md`

### 성능 최적화
- GPU 처리량 실제 프로파일링 (현재 휴리스틱 100 tok/s)
- CPU 스케줄링 고도화
- KV cache 비동기 프리페치
