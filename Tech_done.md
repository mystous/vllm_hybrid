# vLLM Hybrid — 기술 검증 / 분석 결론

이 파일은 코드·로그 기반 기술 검증 결과와 설계 분석 결론을 append-only 로 기록한다.
작업 이력은 `Task_done.md`, 남은 작업은 `TODO.md`, 설계 의도는 `docs/paper/main.tex` 참조.

append-only 원칙: 이전 버전 섹션은 절대 수정/삭제하지 않는다. 새 결론은 파일 말미에 `## vN — YYYY-MM-DD 주제` 섹션으로 이어 붙인다.

---

## v1 — 2026-04-11: Hybrid 동작 4대 핵심 질문 검증 (dev, i9-12900KF + RTX 3090)

### 검증 환경
- **dev 하드웨어**: Intel i9-12900KF (16 physical core = 8P + 8E, AVX2 + AVX-VNNI, AVX-512 ❌, AMX ❌, NUMA 1 node) + RTX 3090 × 1
- **모델**: Qwen/Qwen2.5-7B-Instruct
- **서버 설정**: `--hybrid-mode parallel-batch --hybrid-cpu-kvcache-gb 16 --hybrid-routing-priority cpu-first --hybrid-routing-strategy capacity --hybrid-numa-aware --max-model-len 4096`
- **워크로드**: 500 req burst (`dev_rtx3090_qwen7b_500.env`)
- **로그**: `/tmp/hybrid_diag/7b_hybrid.log` (2677 lines)
- **빌드 상태**: `_C_utils.abi3.so` (66 KB, `init_cpu_threads_env` 심볼 등록 확인), `_C_cpu_ops` ❌ (AVX-512 없어 skip), IPEX ✅ 설치됨

### Q1. CPU 가 진짜로 모든 Core 를 하나의 Request 에 사용하는가? → **YES**

**코드 경로**:
- `vllm/v1/engine/hybrid_core.py :: _resolve_cpu_params` — NUMA 노드당 `cpu_max_num_seqs=1` 고정 (원칙)
- `vllm/v1/worker/cpu_worker.py :: init_device` (line 419~) — `VLLM_CPU_OMP_THREADS_BIND='auto'` → `_get_autobind_cpu_ids` 가 x86 SMT-2 에서 `cpus[-1:]` 로 physical core 만 선택
- `torch.ops._C_utils.init_cpu_threads_env(local_omp_cpuid)` 호출
- `csrc/cpu/utils.cpp :: init_cpu_threads_env` (line 34~) — `#pragma omp parallel for schedule(static, 1)` 내에서 각 OMP thread 마다 `sched_setaffinity` 로 **단일 코어에 1:1 pin**, 그리고 `numa_set_membind` + `numa_set_strict(1)` + `numa_migrate_pages`

**dev 실측 로그**:
```
[HYBRID-RESOLVE] max_seqs=1 threads=16 kvcache=16GB batched_tokens=256 |
  effective_cores=16 (physical=16) numa_nodes=1
[HYBRID-CPU-WORKER] init_device: VLLM_CPU_OMP_THREADS_BIND='auto' →
  local_omp_cpuid='1,3,5,7,9,11,13,15,16,17,18,19,20,21,22,23' (rank=0, local_rank=0)
[HYBRID-CPU-WORKER] init_cpu_threads_env (C++) returned:
  OMP tid: <tid0>, core 1 / tid: <tid1>, core 3 / ... / tid: <tid15>, core 23
[HYBRID-CPU-WORKER] thread binding established via: C++ (init_cpu_threads_env)
[HYBRID-CPU-WORKER] thread config: torch_threads=16 torch_interop=2
  (from OMP_NUM_THREADS=16)
```

- `local_omp_cpuid='1,3,5,7,9,11,13,15,16,17,18,19,20,21,22,23'` = i9-12900KF 의 16 physical core (P-core 8 + E-core 8, hyperthread sibling 제외)
- C++ `init_cpu_threads_env` 가 OMP worker pool 16개를 각기 다른 core 에 1:1 pin, 16 OMP tid ↔ 16 core mapping 출력
- `torch_threads=16` — PyTorch matmul/attention 이 16-way OMP 병렬 실행
- `num_seqs=1 num_tokens=1` — 매 decode step 1 seq 만 처리, 전체 16 core 가 해당 seq 에 전용됨

**주의 (설계 의도)**: `post-init: cpu_affinity=1 cores [1]` — **main thread (scheduler loop)** 는 core 1 하나에만 affinity. 이는 C++ `init_cpu_threads_env` 가 OMP parallel region 의 worker thread 에만 1:1 pin 을 적용하고 main thread 는 첫 core 에 남겨두기 때문. Matmul 병렬 실행에는 영향 없음. 의도된 설계.

**결론**: 1 request 가 16 physical core 를 OMP 병렬로 **전부** 사용한다. 설계·코드·실측 일치.

### Q2. Hybrid 동작에서 CPU 가 진짜 의미 있는 일을 하고 작업을 완료하는가? → **YES** (단 dev 에서 느림)

**Router stats 로그** (log line 2198 이후, 500 req burst 완료 직후):
```
Router stats [501 reqs]: GPU=5.8 tok/s (499 reqs), CPU=2.3 tok/s (2 reqs),
  cpu_ratio=0.4%, in_flight=1/1
```

**근거**:
- `[HYBRID-CLIENT] dispatch req=cmpl-7444b99... prompt_len=128 → cpu:0 (engine_identity=b'\x01\x00')` — 첫 요청 CPU 라우팅 확인
- `[HYBRID-CPU-ATTN] decode call#500 / 1000 / 1500 / 2000 / 2500 / 3000 / 3500 path=ipex num_seqs=1 num_tokens=1` — **실제 decode step 3500+ 회 실행**. python fake loop 아님
- Router stats 에 **CPU=2.3 tok/s (2 reqs)** — 500 req 중 CPU 가 2 req **완료** 기록. 즉 slot 이 반납되고 다음 요청이 들어가는 cycle 이 실제로 돌았음
- `in_flight=1/1` — CPU slot 1 이 항상 차 있음. burst 직후 CPU 가 극도로 느려 (2.3 tok/s) 나머지 498 req 는 GPU 로 overflow 됨 (capacity router 정상 작동 증거)

**dev 에서 CPU 가 느린 이유**: i9-12900KF 16 core × DDR5 가 아닌 DDR4/DDR5 dev 메모리 bandwidth 는 Xeon 8480+ 대비 훨씬 낮고, AVX-512/AMX 없어 IPEX 도 AVX2 path 로 동작. 요청당 ~2분 소요. 이건 dev 의 근본 한계이고, **로직 무결성 검증 목적으로는 충분**.

**결론**: CPU engine 이 end-to-end 로 request 를 받고, token 을 생성하고, 완료 신호를 내보내며, slot 을 반납한다. "완료" 이벤트는 Router stats 카운터 증가로 확인됨. 의미 있는 일을 하고 있다. 성능은 dev 환경 제약으로 낮을 뿐.

### Q3. IPEX 를 사용해서 Python 느린 속도를 대체하고 있는가? → **YES**

**코드 경로**:
- `vllm/v1/attention/backends/cpu_attn.py :: _get_paged_attn_impl()` (line 1250) — IPEX 가용 시 `_IPEXPagedAttention` 반환
- `_IPEXPagedAttention.forward_decode` (line 1219) — `ipex_modules.PagedAttention.single_query_cached_kv_attention(...)` 직접 호출 (C++ oneDNN kernel)
- `_IPEXPagedAttention.write_to_paged_cache` — `ipex_modules.PagedAttention.reshape_and_cache` (C++)
- prefill path: `ipex_modules.PagedAttention.flash_attn_varlen_func` (chunked prefill, line 640)

**dev 실측 로그**:
```
[HYBRID-CPU-ATTN] decode call#3500 path=ipex num_seqs=1 num_tokens=1 |
  totals={'custom_avx': 0, 'ipex': 3500, 'sdpa_batched': 0, 'sdpa_loop': 0}
```

- **3500 회 decode 전부 `path=ipex`**
- **`sdpa_loop=0`** — 순수 Python 루프 기반 fallback (`for seq_idx in range(num_seqs): ... scaled_dot_product_attention(...)`) 은 **0 회 호출**
- **`sdpa_batched=0`** — torch SDPA batched fallback 도 0 회
- **`custom_avx=0`** — dev 는 AVX-512 없어 `_C_cpu_ops` 빌드 skip, 정상

**주의 해석**: "Python 을 IPEX 가 대체" 라기보다 "Python 은 graph 관리·sampling·scheduling 만 하고 inner-loop compute 는 IPEX/MKL C++ kernel 에 위임" 이 정확한 구조. 단 사용자 질문 의도 (= CPU inference 가 Python 인터프리터 속도에 갇혀 있는가?) 에 대한 답은 **NO, 갇혀 있지 않다**. Attention 은 IPEX oneDNN, Linear/FFN matmul 은 torch CPU backend (MKL/oneDNN) — 모두 C++ kernel.

**결론**: CPU 경로의 성능 critical path 는 완전히 native code. Python overhead 는 step dispatch 수준에만 존재.

### Q4. AVX-512 / AMX / NUMA 개수에 무관하게 동작하고 각 경우 정상 동작하는가? → **YES** (3차원 독립 fallback chain)

**(a) Attention decode kernel 선택** (`cpu_attn.py :: _PagedAttention.forward_decode` line 911~, `_get_paged_attn_impl()` line 1250~)
```
_USE_CUSTOM_CPU_ATTN (HAS_CPU_OPS, AVX-512F 빌드)   → custom_avx (batch16 AVX-512 paged attn)
      │ 없거나 실패 시
_is_ipex_available()                                  → ipex (IPEX oneDNN C++ kernel)
      │ IPEX 없으면 _PagedAttention 사용:
num_tokens == num_seqs                                → sdpa_batched (torch batched SDPA)
      │ num_tokens != num_seqs (edge case: 재개 등)
순차 Python 루프                                        → sdpa_loop (per-seq SDPA)
```
- 각 단계는 `try/except` 로 감싸져 있어 상위 경로 실패 시 다음으로 graceful degrade
- dev 에서 `_C_cpu_ops` 빌드 skip 확인 → `custom_avx=0`, `ipex` 로 정상 분기 → `ipex=3500`

**(b) Thread binding** (`cpu_worker.py :: init_device` line 441~)
```
torch.ops._C_utils.init_cpu_threads_env (C++)          → 1:1 pin + strict numa membind (primary)
      │ AttributeError: _C_utils 미등록 (old CUDA build)
      │ RuntimeError: VLLM_NUMA_DISABLED 빌드
self._python_init_cpu_threads_env                       → os.sched_setaffinity (process-level)
      │ 실패 시
warning log + 계속 진행 (affinity 없이 동작, 성능 저하 경고)
```
- `csrc/cpu/utils.cpp` 자체도 `#ifdef VLLM_NUMA_DISABLED` guard 로 libnuma 없는 환경에서 warning string 만 반환 → Python fallback 경로로 위임

**(c) Intel ISA 환경 변수** (`intel_cpu_utils :: configure_intel_optimizations`)
```
AMX 감지        → ONEDNN_MAX_CPU_ISA=AVX512_CORE_AMX + MKL_ENABLE_INSTRUCTIONS=AVX512
AVX-512 only   → ONEDNN_MAX_CPU_ISA=AVX512_CORE_VNNI
AVX2 only      → oneDNN 자동 선택 (env 미설정)
KMP_AFFINITY/KMP_BLOCKTIME/KMP_TPAUSE 는 항상 설정
```

**(d) NUMA 개수** (`hybrid_core.py :: _resolve_num_cpu_engines`, `cpu_worker.py :: _get_autobind_cpu_ids`)
- `NUMAAllocator.num_nodes` 로 자동 감지
- 1 NUMA → `num_cpu_engines=1`
- N NUMA → `num_cpu_engines=N`, 각 engine 이 `local_rank → node_idx` 매핑으로 자기 NUMA 의 physical core + DRAM 에 strict bind
- `numa_bind_node` 사용자 override 시 그 값을 우선 사용

**매트릭스 검증**:

| 환경 | AVX-512 | AMX | NUMA | 실제 경로 | 검증 상태 |
|------|---------|-----|------|----------|----------|
| **dev (i9-12900KF + RTX3090)** | ❌ (AVX2+VNNI) | ❌ | 1 node | `ipex` + C++ `init_cpu_threads_env` (NUMA libnuma ✅) + oneDNN 기본 | ✅ **이번 세션 로그로 실증 완료** |
| H100x4 KVM (일부 Xeon, AVX-512 only) | ✅ AVX-512/VNNI | ❌ | 1 node | `custom_avx` or `ipex` + VNNI oneDNN | 코드 경로 존재, 실측 pending (TODO §3) |
| H100x8 + Xeon 8480+ 2S (target) | ✅ AVX-512/VNNI/BF16 | ✅ | 2 node | `custom_avx` + AMX oneDNN + `num_cpu_engines=2` | 코드 경로 존재, 실측 pending (TODO §3) |
| 일반 x86_64 laptop (AVX2, NUMA 없음) | ❌ | ❌ | 0 (no libnuma) | `ipex` or `sdpa_batched` + Python `sched_setaffinity` fallback | 이론상 graceful, 실측 안 함 |

**결론**: 3 차원 (attention kernel / thread bind / NUMA count) 의 **독립적** fallback chain 이 구현되어 있고, 각 분기는 try/except + 사전 존재 확인으로 감싸져 있어 환경에 따라 자동 선택된다. dev 환경 (AVX2 + NUMA 1) 에서 실제로 서버 부팅부터 500 req burst 까지 정상 동작 검증 완료. H100/Xeon 경로는 코드상 구현되어 있으나 실제 측정은 `TODO §3 H100 타겟 검증` 에서 수행해야 함.

### 종합

| Q | 결론 | 검증 방법 |
|---|------|---------|
| 1. 모든 core 사용 | ✅ YES | 코드 (`utils.cpp :: #pragma omp parallel for + sched_setaffinity`) + 로그 (`16 tid ↔ 16 core 1:1 mapping`) |
| 2. 의미있는 완료 | ✅ YES (dev 에서 느림) | 로그 (`CPU=2.3 tok/s (2 reqs)`, `decode call#3500 path=ipex`) |
| 3. IPEX native | ✅ YES | 코드 (`_IPEXPagedAttention.forward_decode → ipex_modules.PagedAttention.single_query_cached_kv_attention`) + 로그 (`ipex=3500, sdpa_loop=0`) |
| 4. fallback chain | ✅ YES | 코드 (3 차원 independent fallback) + dev 매트릭스 실증 (AVX2 + NUMA 1) |

**남은 작업**: Q4 의 다른 매트릭스 셀 (H100x4 AVX-512, H100x8 + Xeon 2S AMX) 은 코드 경로는 확인되었으나 실제 실행 검증 필요 → TODO §3.
