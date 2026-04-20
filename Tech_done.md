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

---

## v2 — 2026-04-11: 기록 원칙 정정 — "실측 완료 환경만 나열"

### 원칙 (신규, 이후 모든 버전에 적용)

Tech_done.md 에 하드웨어/환경 관련 검증 결과를 기록할 때는 **실제로 실측이 완료된 환경만 나열한다**. "코드 경로 존재, 실측 pending" / "이론상 가능" / "다른 환경에서는 작동할 것으로 예상" 같은 셀은 Tech_done.md 에 속하지 않는다. 그런 항목은 `TODO.md` 의 검증 잔여 섹션에 기록한다.

Why: Tech_done.md 는 "검증이 완료된 사실" 의 단일 진실 공급원이다. 미검증 항목과 검증 완료 항목이 한 표에 섞이면 독자가 "실측" 을 "예상" 으로 오해할 수 있고, Tech_done 의 신뢰도가 훼손된다.

### v1 Q4 매트릭스 정정

v1 의 4 행 표에서 **실측 완료된 환경은 1 행뿐**이다:

| 환경 | AVX | AMX | NUMA | 실측 경로 | 검증 출처 |
|------|-----|-----|------|----------|----------|
| dev (i9-12900KF + RTX 3090) | AVX2 + AVX-VNNI | ❌ | 1 node | `ipex` decode + C++ `init_cpu_threads_env` + oneDNN 기본 | Tech_done.md v1 (Qwen2.5-7B, 500 req, 2677 line log) |

v1 표의 나머지 3 행 (H100x4 KVM, H100x8 + Xeon 8480+ 2S, 일반 x86_64 laptop) 은 "코드 경로 존재, 실측 pending" 에 해당하므로 **v2 이후 본 문서에서 제외**한다. 해당 환경의 실측은 `TODO.md` §3 (H100 타겟 검증) 에서 수행하고, 결과가 나오면 그때 각 환경을 새 버전 섹션에 추가한다.

### v1 에 대한 보존 선언

append-only 정책에 따라 v1 섹션 자체는 수정/삭제하지 않는다. v1 의 Q4 매트릭스 표는 **당시 시점의 기록** 으로 그대로 남고, 본 v2 섹션의 정정이 v1 의 해당 표 해석을 덮어쓴다. 이후 독자는 v1 의 4행 표를 볼 때 본 v2 섹션을 함께 참조해야 한다.

---

## v3 — 2026-04-11: 1.5B/7B 재현 + CPU slot cycle 무결성 (abort 경로 포함)

> append-only. v1/v2 섹션 불변. 본 v3 는 당일 후속 세션에서 추가로 검증된 사실만 기록한다.

### 검증 환경
- v1/v2 와 동일 (i9-12900KF + RTX 3090, AVX2+NUMA1, vLLM `0.1.dev8475+g78fa48cb8`, torch `2.9.0+cu130`, CUDA 13.0)
- 세션 시작 시 HEAD `a0d15b3788d40fd85a19e1635bd2d30b08a5bc71` (clean)
- 세션 중 `vllm/v1/engine/core_client.py` 패치 1 건 적용 (abort slot 반납)

### 추가 검증된 사실 (실측 완료분만 — v2 원칙 준수)

#### V3-F1. 1.5B / 7B 양 모델에서 16 core 포화 재현
- **환경**: dev (i9-12900KF + RTX 3090)
- **측정 방법**: monitor.py 1 Hz per-core CPU util sampling, `/tmp/hybrid*psr.log` 로 per-thread PSR 샘플링
- **1.5B Hybrid** (500 req burst, busy window 3.96s → 29.65s, 26 samples):
  - pinned 16 코어 avg = **96.5% (max 100%)**, P-core odd 95.7~97.6% / E-core 97.0% (전원)
  - 비-pin 짝수 P-core SMT sibling = 5~9% (hyperthread contention)
- **7B Hybrid** (500 req burst, busy window 3.96s → 108.13s, 102 samples):
  - pinned 16 코어 avg = **97.7% (max 100%)**, P-core odd 95.8~96.3% / E-core 99.4% (전원)
- **결론**: 모델 크기 4.67× 증가에도 16 core 포화율은 동일 (96.5% / 97.7%). C++ `init_cpu_threads_env` 의 OMP 1:1 pin + `sched_setaffinity` 로직이 모델 크기에 무관하게 안정.

#### V3-F2. 1.5B/7B CPU decode rate 선형 스케일링
- **1.5B**: Router stats `CPU=9.9 tok/s (2 reqs)` → per-req latency ≈ 13s (128 output tokens)
- **7B**: Router stats `CPU=2.3 tok/s (2 reqs)` → per-req latency ≈ 55s — **v1 Q2 (7B, 이전 세션) 의 2.3 tok/s 와 정확히 일치 → 재현성 확인**
- 모델 크기 4.67× → CPU throughput 0.23× (≈ 1/4.3, 선형에 가까움). CPU 경로에 체계적 regression 없음.

#### V3-F3. 60 req 순차 반복 — slot cycle 무결성 (dev)
- **스크립트**: `/tmp/seq_repeat_test.py` (sequential, `max_tokens=16`, 앞 req 완료 후 다음 송출)
- **결과**: 60/60 성공, 평균 1.58s/req, 편차 ±0.05s
- Router stats: `GPU=0.0 tok/s (0 reqs), CPU=10.2 tok/s (50 reqs), cpu_ratio=100.0%, in_flight=0/1`
- 60/60 모두 CPU 로 라우팅 (sequential 이라 매 req 마다 in_flight=0 에서 출발 → cpu-first → CPU)
- **누수 없음**: latency 가 끝까지 일정하고 stats 의 `in_flight=0` 로 복귀 — `length` 종료 경로의 slot 반납은 60 회 연속 정상

#### V3-F4. `length` / `stop` 종료 — slot 반납 정상 (dev)
- **length**: `max_tokens=10` → `finish_reason=length`, 0.91s on CPU, 다음 req 즉시 CPU 로 라우팅
- **stop**: `stop=["."]` → `finish_reason=stop`, 0.21s on CPU, 다음 req 즉시 CPU 로 라우팅
- 두 경로 모두 엔진이 `output.finished=True` 를 emit → `process_engine_outputs` 의 line 1507 경로로 `on_request_finished` 호출 → `cpu_in_flight` 감소
- **결론**: 정상 종료 (length/stop) 경로는 수정 없이 기존 구현으로 정상 동작

#### V3-F5. `abort` (client disconnect) — slot 반납 **버그 재현 → 수정 → 검증**
- **버그 증상 (수정 전)**: CPU request 를 mid-stream abort 하면 `cpu_in_flight` 가 영구 `1` 로 stuck. 이후 모든 요청이 GPU 로 fallback.
- **재현 스크립트**: `/tmp/cpu_abort_test.py` (long CPU req → 2s 후 client close → 10 probe 로 라우팅 확인)
- **수정 전 측정**:
  - long req: dispatched to `cpu:0`, aborted after 2s
  - probe 0~9: 전원 GPU 로 (~0.03-0.05s = GPU latency), CPU 로 가는 probe 0개
- **수정 후 측정**:
  - long req: dispatched to `cpu:0`, aborted after 2s, **`Request finished: ... (cpu_count=1)` 로그 관측 — abort 시점에 slot 반납됨**
  - probe 0~9: 전원 CPU 로 (~0.37-0.47s = 4 tokens / 9.9 tok/s × k), `cpu_count=1→11` 순차 증가
- **Root cause (코드 추적)**:
  - `vllm/v1/engine/core_client.py::HybridAsyncMPClient.abort_requests_async` (수정 전) 이 `_hybrid_reqs_in_flight.get()` 만 호출하고 pop/router 반납 안 함
  - Engine 측 scheduler 는 `finish_requests()` → `_free_request()` 호출해서 request 를 해제하지만
  - `self.finished_req_ids_dict` 는 `vllm/v1/engine/core.py:122` 에서 `include_finished_set=(data_parallel_size > 1)` 로 초기화 → **DP=1 (dev/H100x4 모두 해당) 에서는 `None`**
  - `_free_request` 가 `finished_req_ids_dict` 에 추가하는 코드 (`scheduler.py:1018`) 는 `is not None` 가드 덕분에 no-op
  - Update step 에서 `EngineCoreOutputs.finished_requests` 필드가 empty 로 남음
  - Aborted request 는 새 토큰도 없으므로 `outputs.outputs` 의 `output.finished=True` 도 emit 안 됨
  - `process_engine_outputs` 의 어느 경로로도 slot 반납 신호가 안 옴 → 영구 누수
- **Fix**: `abort_requests_async` / `abort_requests` 두 함수에서 `_hybrid_reqs_in_flight.pop()` + `_hybrid_router.on_request_finished()` 를 명시 호출. Engine 쪽 이중 반납은 `process_engine_outputs` 의 `.pop(req_id, None)` 패턴으로 자동 방지.
- **의의**: H100 운영 환경의 "capacity 에서 멈춤" 증상 (TODO v1 §1 마지막 항목) 의 직접 원인 후보 1 로 **dev 재현 + 원인 확정 + 수정 완료**. client disconnect 는 production 환경에서 자주 발생 (timeout, LB health check, 네트워크 오류) — 이 경로 한 번이면 CPU slot 영구 점유.

#### V3-F6. V1 scheduler `cpu_max_num_seqs=1` 경계 경로 확정 (dev, 코드 분석)
- `vllm/v1/engine/hybrid_core.py::_create_cpu_vllm_config` (line 945~1004):
  - `cpu_sched.max_num_seqs = resolved.cpu_max_num_seqs` (= 1) — 표준 `max_num_running_reqs` 메커니즘으로 enforce
  - `cpu_sched.enable_chunked_prefill = False`, `chunked_prefill_enabled = False` — **CPU 엔진에서 chunked prefill 명시 비활성화** (decode 와 interleave 시 극심하게 느려지기 때문)
  - `cpu_max_model_len = min(gpu_max, cpu_max_batched_tokens × cpu_max_num_seqs)` — chunked prefill 끄면 `max_num_batched_tokens >= max_model_len` 조건 필요
- `vllm/v1/core/sched/scheduler.py::Scheduler.schedule` (line 334):
  - `if len(self.running) == self.max_num_running_reqs: break` — 표준 경계 체크, cpu 엔진도 동일 경로
- 정상 종료 (`length`/`stop`) 경로는 `_free_request()` → `update_from_output` → per-output `finished=True` → `process_engine_outputs` → `on_request_finished` 으로 완결
- Preemption (KV cache exhaustion) 경로는 `num_cpu_engines=1` + `cpu_kvcache=8~16GB` + `cpu_max_model_len` 제한 덕에 실제 발생 가능성 매우 낮음 — 이 경로의 edge case 는 검증 미수행 (dev 에서 트리거 어려움)
- **결론**: CPU 엔진의 slot 경계는 표준 V1 scheduler 메커니즘으로 처리되며 chunked prefill 경계 edge case 없음. 유일한 edge case 였던 abort 경로는 V3-F5 로 수정됨.

### v3 매트릭스 (실측 완료 환경, v2 정정 원칙 준수)

| 환경 | AVX | AMX | NUMA | 실측 경로 | 검증 범위 | 근거 |
|------|-----|-----|------|----------|---------|------|
| dev (i9-12900KF + RTX 3090) | AVX2 + AVX-VNNI | ❌ | 1 node | `ipex` + C++ `init_cpu_threads_env` | 1.5B/7B 벤치 500 req 완결, 60 req 순차 누수 zero, length/stop/abort slot cycle 정상 (**단 abort 는 v3 패치 포함 시**), V1 scheduler 경계 경로 확정 | v1 Q1~Q4 + v3 F1~F6 |

H100x4 KVM / H100x8 + Xeon 2S / 일반 x86_64 laptop 등 기타 환경은 여전히 실측 없음 → TODO §3.

### v3 의 TODO 해소 현황

- TODO v1 §1 "1-시퀀스 라이프사이클 반복 검증" → v3 F3 으로 **완결** (60 req 순차)
- TODO v1 §1 "`output.finished` 감지 확실성 (length/stop/abort)" → v3 F4 + F5 로 **완결** (abort 는 수정 후)
- TODO v1 §1 "CPU scheduler 코드 경로 트레이싱" → v3 F6 으로 **완결** (코드 분석)
- TODO v1 §1 "H100 capacity 멈춤 증상 dev 배제" → v3 F5 로 **재현 + 원인 확정 + 수정 완료**. dev 에서 배제 성공.
- TODO v1 §1 "동시 요청 스트레스 50+ burst 확장" → v3-F1 의 500 req burst 로 이미 커버됨 (1.5B/7B 양쪽)

→ TODO v1 §1 은 본 v3 로 실질적으로 완결. 남은 것은 TODO v1 §2 (논문 정합성) / §3 (H100 실측) / §4 (기타 잠재 이슈) / §5 (문서화 잔여).

---

## v4 — 2026-04-11: stdout fast-path contention / H100 라우팅 Property 2 gate / 1.5B~32B scaling baseline / ISA + AMX brgemm 실측

> append-only. v1/v2/v3 섹션 보존. 본 v4 는 v3 이후 연속 세션에서 **실측으로 확정된 기술 결론만** 기록 (v2 원칙 준수).

### 검증 환경
- dev: i9-12900KF + RTX 3090 (AVX2+NUMA1), vLLM `0.1.dev8475+g78fa48cb8`, torch `2.9.0+cu130`, CUDA 13.0
- H100x4 KVM: Intel Xeon Platinum 8480+ (1S × 96 vCPU, 1 NUMA, 944 GB DDR5), H100 80GB HBM3 × 4, 동일 vLLM / torch / CUDA 스택, IPEX 2.8.0+gitcb81bf2
- 세션 동안 코드 commit 연대: `019c7121a` → `9bccbe651` → `3f528123e` → `3e6e514ca` → `81f46717d`

### 추가 검증된 사실 (실측 완료분만)

#### V4-F1. Stdout fast-path contention — H100x4 1.5B/500 req + `TRACE=1` 에서 hybrid wall ×7.6
- **환경**: H100x4, `h100x4_qwen1.5b_hybrid.env` 의 이전 버전 (NUM_PROMPTS=500, `VLLM_HYBRID_TRACE=1`, `TRACE_EVERY=1`)
- **측정 출처**: `experiment_result/20260411_090942_h100x4_qwen1.5b_capacity_trace_on_500/`
- **결과**:
  - GPU-only: wall 13.96 s, duration 3.68 s, output TP 16,742 tok/s, TPOT 22.36 ms, GPU mean util 42.6%
  - Hybrid (capacity + cpu-first, **TRACE=1**): wall **106.66 s**, duration **49.54 s**, output TP **1,243 tok/s**, TPOT **60.06 ms**, GPU mean util **1.5%**
  - Ratio: wall **×7.64**, output TP **×0.074**, TPOT **×2.69**
  - CPU 96-core busy window avg **93.1% (87/95 samples)** — CPU 는 2 reqs 만 받아서 끝까지 처리 중이었음
  - 모니터 CSV 로 교차 확인: `experiment_result/20260411_121509_.../` 의 rate 추정표

- **직접 증거**: 같은 env 에서 `TRACE=0` + `throughput-adaptive` 로 돌린 `20260411_085801_h100x4_qwen1.5b_thro_adaptive_500/` 는 hybrid wall 25.67 s ≈ gpu_only wall 25.70 s 로 **hybrid penalty 0**. 유일한 변수는 `VLLM_HYBRID_TRACE=1`.

- **메커니즘**: stdout 은 하드웨어 속도와 독립적인 상수 성능 단일 channel (커널 file offset lock + write syscall). H100x4 같이 throughput 이 2~8× 큰 환경에서는 동일 "decode call 마다 1 line" 정책이 초당 emit 되는 log 수를 기하급수적으로 키우고, 여기에 TP=4 로 writer process 수까지 2× 늘어남. `VLLM_HYBRID_TRACE_EVERY=1` 은 dev (`TRACE_EVERY=500`) 대비 500× aggressive → 총 stdout 부담 ~2000×.

- **해결 (silent 패치 적용 후 dev 재검증)**: `experiment_result/20260411_120746_dev_rtx3090_1.5B_silent_stdout_rerun/`
  - dev 1.5B gpu_only / hybrid wall 13.97 / 34.90 s 양쪽 모두 이전 60412 / 60712 run 과 **동일** (TPOT 완전 동일 27.79 ms ↔ 27.79 ms)
  - hybrid server log 라인 수 2701 → 1094 (−59.5%), serving 중 HYBRID per-req/per-call marker 전부 0, 부팅 markers 11 preserved
  - **결론**: dev 1.5B 에서는 stdout 이 원래부터 병목이 아니었음을 직접 증명. H100x4 쪽의 7.6× slowdown 은 fast-path contention (hardware throughput ↑ + TRACE_EVERY ↓ + TP writer multi-process) 조합이 만든 것.

#### V4-F2. H100x4 hybrid 라우팅 회귀 — 근본 원인 + Property 2 expected-finish gate 수정
- **환경**: H100x4, `h100x4_qwen1.5b_hybrid.env` (production config: throughput-adaptive + cpu-first + silent + auto everything)
- **측정 출처**: `experiment_result/20260411_130959_.../` (회귀 발견), `20260411_141500_.../` (근본 원인 + 수정), `20260411_142900_.../` (4-run 검증)
- **회귀 초기 측정** (수정 전 hybrid vs gpu_only 같은 분 run):
  - bench duration **3.64 → 44.64 s (×12.3)**
  - output TP **16,924 → 1,380 tok/s (×0.082)**
  - TPOT **22.10 → 60.34 ms (×2.73)**
  - CPU mean util 6.7% → **84.7%** (76/83 busy samples)
  - GPU mean util 13.7% → **1.4%**
- **5 단계 가설 분리 검증** (`141500` README §2):
  - 가설 A (resolver `max_seqs` 오판): 반증 — `[HYBRID-RESOLVE] max_seqs=1` 정상
  - 가설 B (CPU scheduler 가 max_seqs=1 무시하고 batch): 반증 — `[HYBRID-CPU-EXEC] reqs=0..1 tokens=1`
  - 가설 C (라우터가 다수를 CPU 로): 반증 — `[HYBRID-ROUTER-DISPATCH] cpu=2 gpu=498` (대부분 GPU 임)
  - 가설 D (CPU OMP 96C 점유로 API server starve): 반증 — `HYBRID_CPU_THREADS=1` 로 재실험해도 TPOT 64 ms 그대로
  - 가설 E (라우팅이 "CPU 가 GPU 보다 빠른가" 를 안 봐서 1~2건 long-tail): **확정 ✓**
- **결정적 증거** (GPU `nvidia-smi` 시계열 96 s):
  ```
  [0..51] 0%   ← GPU 완전 idle (51 초)
  [52..55] 54%/29%/27%/32%
  [56..95] 0%  ← GPU 다시 idle (40 초)
  ```
  GPU 실제 work 시간 = **4 초** (gpu_only 의 3.55 s 와 일치). 나머지 92 초는 idle 대기.
- **메커니즘**: `benchmark_serving.py` 가 main bench 전 1건 probe 전송 → 이전 `_route_throughput_adaptive` 는 cpu-first 분기에서 `cpu_in_flight < effective_max * num_cpu_engines` 만 확인 → probe 가 CPU 로 라우팅됨 → 1.5B CPU decode ~47s → probe 혼자 47 s 점유 → main bench 시작 지연 + long tail → wall ≈ max(GPU 4s, CPU 47s) = 47s (+ 부팅).
- **수정** (`vllm/v1/engine/hybrid_core.py::_route_throughput_adaptive`):
  - Cold start gate: `_gpu_ema_throughput <= 0` 이면 항상 GPU (probe 블라인드 회피)
  - Expected-finish 비교: `cpu_finish = (cpu_in_flight+1)·(L_out / tput_cpu_ema)` vs `gpu_finish = ceil((gpu_in_flight+1)/gpu_max_seqs)·(L_out / tput_gpu_ema)`, CPU 가 더 빠를 때만 CPU 선택
  - `cpu-first` / `gpu-first` 에 동일 비교 gate, 우선순위는 "동률 시 CPU 우선" 의미만 남음
  - Paper §3 Property 2 의 직접 구현
- **수정 후 검증 (instrumentation run)**:
  | | 수정 전 | 수정 후 | gpu_only |
  |---|---:|---:|---:|
  | bench dur (s) | 52.94 | **4.02** | 3.55 |
  | TPOT (ms) | 61.09 | **23.71** | 21.48 |
  | output TP (tok/s) | 1,212 | **15,305** | 17,362 |
  | router CPU / total | 2/501 | **0/501** | — |
  | wall (s) | 107.64 | **17.15** | 12.88 |
- **4-run 최종 검증 (production env 무수정)**:
  | run | bench dur (s) | TPOT (ms) | output TP | CPU/GPU disp | hybrid/gpu ratio (TPOT) |
  |---|---:|---:|---:|---|---:|
  | 1.5B gpu_only | 3.94 | 23.56 | 15,640 | — | — |
  | 1.5B hybrid | **3.87** | **23.03** | 15,911 | **0/501** | **0.978×** |
  | 7B gpu_only | 4.02 | 24.73 | 15,492 | — | — |
  | 7B hybrid | **3.93** | **23.04** | 15,984 | **0/501** | **0.932×** |
- **결론**: H100x4 + 1.5B/7B 에서 수정 후 hybrid ≈ gpu_only (±2% 노이즈). Property 2 gate 가 정확히 작동 — GPU 가 압도적으로 빠른 구간에서는 모든 요청을 GPU 로 보낸다. **7B probe hang** 증상 (1차 probe 에서 분 단위 정지) 도 동일 기전이었고 cold start gate 로 해소.
- **부수 발견 (Bug 1)**: `_update_adaptive_slots` (`hybrid_core.py:436-443`) 는 `cpu_max_num_seqs=1` 에서 항상 `new_max=2` 로 고정되는 dead code. Property 2 gate 이후 라우팅 영향은 0 이지만 코드는 정리 필요 → `TODO §3.1`.

#### V4-F3. 1.5B / 7B / 32B scaling baseline — H100x4 에서 32B 까지도 GPU sub-saturation
- **환경**: H100x4, production env (`h100x4_qwen{1.5b,7b,32b}_hybrid.env`), 라우팅 fix 포함 상태
- **측정 출처**: `experiment_result/20260411_142900_.../` (1.5B/7B), `20260411_145900_.../` (32B)
- **결과**:
  | 모델 | 모드 | bench dur (s) | TPOT (ms) | output TP (tok/s) | GPU mean util |
  |---|---|---:|---:|---:|---:|
  | 1.5B | gpu_only | 3.94 | 23.56 | 15,640 | 13.9% |
  | 1.5B | hybrid | 3.87 | 23.03 | 15,911 | 13.1% |
  | 7B | gpu_only | 4.02 | 24.73 | 15,492 | 19.2% |
  | 7B | hybrid | 3.93 | 23.04 | 15,984 | 13.8% |
  | **32B** | gpu_only | **7.01** | **41.82** | **8,774** | **42.6%** |
  | 32B | hybrid | 7.09 | 41.93 | 8,674 | 43.3% |
- **Scaling 관찰**:
  - 1.5B → 7B TPOT 1.05× (거의 동일) — GPU 가 양쪽에서 underutilized, **launch overhead > compute** 영역
  - 7B → 32B TPOT **1.69×** + GPU util **2.22×** — compute 가 launch 를 처음으로 추월
  - 32B GPU mean util 43% 는 여전히 sub-saturation. **H100x4 는 32B 까지도 over-provisioned**
- **의미**: paper 의 1.5B/7B/32B scaling section baseline. "H100x4 는 32B 까지 GPU saturated 가 아니다" 가 핵심. ninja gap 정량 측정을 위해서는 **70B+, batch ↑, 긴 context, 또는 spec decode** 같은 구조적 변경이 필요.
- **32B 부팅 시간 실측**: gpu_only ~100 s / hybrid ~150 s. `SERVER_READY_TIMEOUT=1800` 필요. Weight 64 GB 로드 + TP=4 CUDA graph capture (~2.9 GB) 가 지배.
- **라우터 분배 (3 모델 공통)**: `[HYBRID-ROUTER-STATS] finished=501 GPU=501 CPU=0, cpu_ratio=0.0%` — Property 2 gate 가 모든 모델에서 "GPU always wins" 로 올바르게 결정.

#### V4-F4. ISA / IPEX / oneDNN / AMX BF16 brgemm 실측 — H100x4 + Xeon 8480+ 에서 AMX 가 실제로 dispatch 되고 있음
- **환경**: H100x4 KVM guest, Intel Xeon Platinum 8480+ (1S × 96 vCPU, SPR+)
- **측정 출처**: `experiment_result/20260411_143500_h100x4_isa_verification_and_ninja_gap_strategy/` §1 (5 단계 evidence)
- **결과**:
  | 단계 | 측정 | 결과 |
  |---|---|---|
  | 1. CPU feature detect | `intel_cpu_utils.detect_intel_cpu_features()` | AVX-512 F/VNNI/BF16/FP16 ✓, AVX-VNNI ✓, **AMX-BF16 ✓**, **AMX-INT8 ✓**, 96 cores |
  | 2. 환경변수 | `[HYBRID-CPU-ENV]` 부팅 로그 | `ONEDNN_MAX_CPU_ISA=AVX512_CORE_AMX` 정상 적용 |
  | 3. Extension / module load | `vllm._custom_ops` + `torch.ops._C_utils` + IPEX | `HAS_CPU_OPS=True`, `HAS_CPU_UTILS=True`, `init_cpu_threads_env` 등록 ✓, IPEX 2.8.0+gitcb81bf2 ✓, mkldnn enabled ✓ |
  | 4. Decode path 실측 | 1.5B 회귀 run 의 1900 회 attention 호출 | 100% `[HYBRID-CPU-ATTN] path=ipex`, fallback (custom_avx / sdpa_batched / sdpa_loop) **0 건** |
  | 5. oneDNN 커널 실측 | `ONEDNN_VERBOSE=1` BF16 64×128 × 128×256 matmul | `brg_matmul:avx10_1_512_amx` — **실제 AMX BF16 brgemm 커널 dispatch 확인** |
- **`ONEDNN_VERBOSE` 원문 발췌**:
  ```
  onednn_verbose,v1,info,cpu,isa:Intel AVX-512 with float16, Intel DL Boost
    and bfloat16 support and Intel AMX with bfloat16 and 8-bit integer support
  onednn_verbose,v1,primitive,exec,cpu,matmul,brg_matmul:avx10_1_512_amx,...
    src:bf16::blocked:ab::f0 wei:bf16::blocked:ab::f0 dst:bf16::blocked:ab::f0
  ```
- **결론**: ISA / IPEX / oneDNN / AMX 스택은 H100x4 + SPR 8480+ 에서 **빈틈없이 실행 경로에 들어온다**. 라우팅 fix 이후 H100 + 1.5B/7B/32B 에서 CPU 분배가 0건이 된 것은 **Property 2 가 올바르게 "CPU 가 더 느려서 안 쓰는 것"** 이지, ISA 가 안 도는 것이 아님. AMX 의 **compute 잠재력은 실재**하지만, H100+BF16 dense decode 워크로드에서는 memory-BW bound 에 먼저 도달해 AMX peak 가 사용되지 못하는 것이 현재 천장.

#### V4-F5. Ninja Gap 이론적 천장 — request-level partition 의 상한
- **환경**: 이론 분석 + H100x4 실측 데이터 교차 검증
- **측정 출처**: `20260411_143500_.../` §2
- **결론**: 현재 hybrid 의 구조적 천장은
  ```
  T_hybrid = max(T_gpu_share, T_cpu_share)
  ```
  여기서 CPU per-req latency > GPU per-req latency 인 환경 (H100+1.5B 에서 13×, H100+7B 에서 20×+, H100+32B 에서도 비슷) 에서는 어떤 request-level routing 으로도 `T_hybrid < T_gpu` 불가. Property 2 gate 는 이 상한 **안에서** optimal (hybrid ≤ gpu_only) 이지만 그 이상은 불가능.
- **하한 돌파 조건**:
  - (a) GPU 가 queue-saturated 되어 `gpu_wait > cpu_per_req` — 현 H100 + 1.5B/7B/32B burst 에서 발생 안 함
  - (b) CPU 와 GPU 가 **본질적으로 다른 layer / 다른 step / 다른 precision** 을 처리 — A1/A2/A3 후보. 현재 미구현
- → paper Property 2 를 "complement" 에서 "expected-finish-time gate + 구조적 분해" 로 강화해야 실제 ninja gap 실현 가능. `TODO §1`, `§4.1` 로 이관.

### v4 매트릭스 (실측 완료 환경만 나열, v2 정정 원칙 준수)

| 환경 | CPU | NUMA | ISA 검증 | hybrid routing 검증 | 1.5B | 7B | 32B | 근거 |
|------|-----|------|---------|-------------------|------|----|-----|------|
| **dev** (i9-12900KF + RTX 3090, TP=1) | AVX2 + AVX-VNNI | 1 | v1 Q1~Q4 + v3 F1~F6 | 라우팅 fix 미검증 (TODO §5.1) | wall 34.9s / TPOT 30.0ms (hybrid), ≈ gpu_only wall 14.3s | wall 109.3s / TPOT 116ms (hybrid, CPU tail 지배) | 미수행 | v3 F1~F6 + v4 F1 |
| **H100x4 KVM** (Xeon 8480+, TP=4, 944 GB DDR5) | AVX-512 F/VNNI/BF16, AVX-VNNI, **AMX-BF16/INT8** | 1 (KVM 1 node) | v4 F4 (5단계 + `brg_matmul:avx10_1_512_amx`) | v4 F2 (Property 2 gate 수정 후) | wall 17.13s / TPOT 23.03ms / 0/501 CPU / hybrid ≈ gpu_only | wall 17.15s / TPOT 23.04ms / 0/501 CPU / hybrid ≈ gpu_only | wall 17.21s / TPOT 41.93ms / 0/501 CPU / hybrid ≈ gpu_only | v4 F2 + F3 |

다른 환경 (H100x8 2-socket, 일반 laptop, 70B 모델, 긴 context) 은 여전히 실측 없음 → `TODO §2.1` / `§2.2` / `§2.6`.

### v4 의 TODO 해소 현황
- TODO v1 §3 "H100x4 KVM 재측정" → v4 F2 + F3 로 **완결** (1.5B / 7B / 32B baseline + 라우팅 fix 검증)
- TODO v1 §3 "Exp 1 end-to-end throughput" → v4 F3 으로 **부분 완결** (단일 workload, production bench)
- TODO v4 (이전 버전) §3 "H100 capacity 멈춤" → v4 F2 로 **완결** (dev + H100 양쪽에서 근본 원인 확정 + 수정 + 검증). 대부분은 v3 F5 의 abort slot leak fix 로 해소되었고, 본 세션에서 발견된 routing regression 은 별개의 관련 이슈로 동시에 해결.
- TODO v1 §4 "AMX tile permission 커널 버전 의존성" → v4 F4 로 **완결** (H100x4 KVM 에서 `brg_matmul:avx10_1_512_amx` 실제 dispatch 확인)
- → 남은 것은 `TODO.md` (v4 재작성본, clean file) 의 §1~§7 — ninja gap 구현 (A1~A4), 70B / 긴 context 확장, bug 1 / OMP binding 정리, 논문 정합성, dev 보조 검증.

---

## v5 — 2026-04-14 (H100x8 2-NUMA 실측 검증 + dev 최적화)

### V5-F1. H100x8 2-NUMA hybrid 경로 실동작 증명
- **환경**: H100x8 물리 (violet-h100-023), Xeon Platinum 8480+ 2S × 56C × 2T = 224 logical, 2 NUMA
- **측정 출처**: `eval/basic/H100x8/20260414_{044922,045947,054010}_H_C_*/hybrid_server_boot.log`
- **서버 로그 증거**:
  ```
  [HYBRID-LAUNCH] num_cpu_engines=2 (numa_aware=True, config=2)
  CPU_EngineCore_1 → numa_bind_node=0, local_omp_cpuid='112..167' (56 cores NUMA 0)
  CPU_EngineCore_2 → numa_bind_node=1, local_omp_cpuid='168..223' (56 cores NUMA 1)
  ```
- **결론**: 2 NUMA engine 이 각자 자기 노드 물리 코어 56개에 strict bind 되어 spawn. 이전 0413 run 에서 N_CPU=16 이 나온 것은 `arg_utils.py default=1` / `serve.sh -gt 1` / `copy.replace` (Python 3.13+) / `numa_node` kwarg mismatch 의 4 겹 버그 복합 결과. 본 세션에서 4개 모두 수정 후 실측으로 복원 완료.

### V5-F2. Wave-batch max_seqs=16 vs max_seqs=1 — 2-NUMA 정상 동작 하에도 재앙 재현
- **환경**: 위와 동일, TP=4, Qwen2.5-7B, 500×128/128 burst
- **측정 출처**: `eval/basic/H100x8/`
- **실측**:
  | config | wall | output TP | TPOT med/mean/p99 | TTFT p99 |
  |---|---:|---:|---:|---:|
  | gpu_only baseline | 14.0s | 16,501 | 22.7 / 23.1 / 55.8 | 1,075 |
  | max_seqs=1 threads=32 2NUMA | 394s | 158 | 26 / 37 / 83 | 1,106 |
  | max_seqs=16 threads=32 2NUMA | 2,098s | 30 | 22 / 1,047 / 15,966 | 69,959 |
  | max_seqs=1 threads=56 2NUMA | 408s | 153 | 26 / 38 / 79 | 1,179 |
- **TPOT 역산 (max_seqs=16)**:
  ```
  500 × 1047 = 468 × 21.6 + 32 × X  →  X = 16,043 ms ≈ p99 15,966 ✓
  ```
  **N_CPU=32** 으로 깔끔히 맞음 → 2 engine × max_seqs=16 alternating routing 정상 작동.
- **결론**:
  - 2 NUMA 가 작동하는데도 wave=16 은 여전히 **2098s**, max_seqs=1 대비 5.3× 느림
  - 원인은 **IPEX FD kernel batch>1 KV paged-access penalty** + **CPU prefill 직렬화 (chunked_prefill=False)**. NUMA 분산으로 해결 불가.
  - **max_seqs=1 이 어떤 상황에서도 맞는 답** — wave-batch 의 "batch" 자체가 독이므로 본질적으로 `throughput-adaptive cpu-first` 와 동치.
  - threads=32 vs 56 (394 vs 408s) 는 -3.6% 차이. profile Section C.5 peak 32 재현 (BW-bound). NUMA 내 56 core 풀가동은 GPU worker / scheduler core contention 으로 오히려 손해.

### V5-F3. dev (i9-12900KF) CPU thread sweep — 16 peak
- **환경**: i9-12900KF (8P+8E, 16 physical / 24 logical, 1 NUMA), AVX2+VNNI (AVX-512 없음)
- **측정 출처**: `eval/analysis_log/20260414_050715_cpu_profile_dev/vllm_thread_sweep.json`
- **실측 (Qwen2.5-1.5B CPU solo decode, 3-run avg, 64 tokens)**:
  ```
  threads= 4 : 10.56 tps
  threads= 8 : 10.64 tps (+0.7% vs 4)
  threads=12 : 11.35 tps (+7.5%)
  threads=16 : 11.62 tps (+10.0%) ⭐ peak
  threads=24 : 11.20 tps (-3.6% SMT 오버서브)
  ```
- **GEMM 층별 peak (Section 2)**:
  - decode_qkv (16×3584²): threads=16 peak (427 GFLOPS)
  - decode_ffn_up/dn (16×3584×9728): **threads=8 peak** (301/316 GFLOPS)
  - decode_single (1×3584×9728): threads=4-8 plateau
- **Layer breakdown (Section 5, 16 threads)**:
  - MLP **76%** / Attn 11% / Other 13% — AVX-512 없어 MLP GEMM 이 압도적 병목
- **결론**: 8P+8E 물리 16 코어 전부 사용이 최적. E-core barrier stall 우려는 실측에서 무효화. MLP GEMM peak 가 8 이고 전체 peak 가 16 인 것은 attention GEMM (peak=16) + MLP + overhead 가 섞여 종합 최적이 16 으로 밀린 결과. dev env 3개 `HYBRID_CPU_THREADS=16` 업데이트 완료.

### V5-F4. 서버 로그 캡처 인프라 완성
- **구조**:
  - `serve.sh`: `tee` 로 stdout/stderr 를 `eval/serve_logs/server_YYYYMMDD_HHMMSS_MODE.log` 에 복제, `server_latest.log` 심링크 갱신
  - `bench.sh`: 시작 시 `wc -c` 로 byte offset 기록 → 종료 시 `tail -c +START` 로 run-window slice 추출 + `grep` 으로 boot markers 추출
- **산출물 (per RUN_DIR)**:
  - `MODE_server_boot.log` — grep 기반 boot markers (LAUNCH/RESOLVE/CPU-ENV/CPU-PROC/CPU-WORKER/ROUTER-INIT)
  - `MODE_server_run.log` — bench 구간 byte-offset slice (dispatch/profile/stats markers)
- **검증**: `eval/basic/H100x8/20260414_*/` 의 `hybrid_server_boot.log` / `hybrid_server_run.log` 에 실제로 2-NUMA 증거 (`[HYBRID-LAUNCH] num_cpu_engines=2` 등) 포함 확인.
- **새 diagnostic markers (2026-04-14 추가, 모두 `VLLM_HYBRID_PROFILE=1` 게이트, 기본 silent)**:
  - `[HYBRID-CPU-PROFILE] step=N reqs=R tokens=T threads=K total=Xms attn=Y ms (N layers) mlp=Z ms (N layers) other=W ms`
    → `cpu_worker.py _install_hybrid_profile_hooks` 가 `model.model.layers[i].self_attn` / `.mlp` 에 forward pre/post hook 부착. per-step 시간 누적. 처음 1회 `hooks installed on N layers` info.
  - `[HYBRID-CPU-ATTN-IPEX] call=N num_seqs=K elapsed=X.XX ms avg=Y.YY ms batch_hist={1: N1, 4: N4, ...}`
    → `cpu_attn.py` IPEX `single_query_cached_kv_attention` 호출 timing + batch-size histogram. 실제 dispatch batch 분포 확인.
  - `[HYBRID-WAVE-DISPATCH] req=REQ_ID → cpu:ENGINE_IDX accepted=K max=M`
    → `_route_wave_batch` 의 engine 선택 + wave lifecycle (open → accepted 증가 → closed → drain → reset) 가시화.
- **오버헤드**: hook 실행 + logger.info 포맷팅으로 per-step 수십 μs 수준 예상. `_EVERY=N` 간격 조절 가능 (기본 10). 진단 run 전용, production run 은 silent.
- **결론**: 과거 session 에서 추측으로만 진단하던 num_cpu_engines / NUMA binding / max_seqs resolve / IPEX batch 분포 / layer 병목 전부 **log 증거 기반 확정 가능**.

### V5-F5. Python 3.12 / 3.13 API 호환성
- **증상**: `copy.replace()` 사용 코드가 H100 환경 (Python 3.12.13) 에서 `AttributeError: module 'copy' has no attribute 'replace'`
- **원인**: `copy.replace()` 는 **Python 3.13+ 전용** 신규 API. `dataclasses.replace()` 는 3.7+ 이고 동일 동작.
- **수정**: `hybrid_core.py:1263-1274` — `dataclasses.replace()` 로 대체. HybridConfig / VllmConfig 둘 다 dataclass 이므로 동치.
- **결론**: 향후 신규 API 사용 시 `import sys; sys.version_info >= (3, 13)` 로 게이트하거나, `dataclasses` / `typing` / `functools` 의 안정 API 선호.

### v5 매트릭스 (실측 완료 환경만)

| 환경 | CPU | NUMA | 2-NUMA 실측 | 1.5B | 7B | 32B |
|------|-----|------|-----------|------|----|-----|
| **dev** (i9-12900KF + RTX 3090, TP=1) | AVX2+VNNI | 1 | N/A | cpu_profile peak 16t = 11.62 tps | 없음 (TP=1 에서 KV 압박) | 없음 |
| **H100x4 KVM** (Xeon 8480+, TP=4) | AVX-512 F/VNNI + AMX | 1 (KVM 통합) | N/A | v4 F2 | v4 F2 | v4 F2 |
| **H100x8 물리** (Xeon 8480+ 2S, TP=4) | AVX-512 + AMX | **2** | **V5-F1 ✓** | 없음 | **V5-F2 ✓** | 없음 |

다음 우선순위 (TODO §2):
- H100x8 70B TP=8 (GPU 포화 조건에서 hybrid 이득 실측)
- H100x8 long-context (16K input, GPU KV 압박)
- H100x8 rate-limited burst 2000+ req (GPU queue-saturation)

---

## v6 — 2026-04-19: §06 Q8_0 Hot Path Wiring — H100x8 + Qwen2.5-32B 실측 기술 결론

### 검증 환경

- **하드웨어**: H100×8 (HBM3 80GB, TP=8) + Xeon 8480+ 2-socket (NUMA 2 nodes, 56 core/socket, AMX + AVX-512 VNNI)
- **모델**: Qwen/Qwen2.5-32B-Instruct
- **워크로드**: 500 req × 128 input / 128 output, `request_rate=inf`
- **측정 설정**: `VLLM_HYBRID_PROFILE=0` (production 수치), `HYBRID_CPU_THREADS=48`, `num_cpu_engines=2`, `cpu_max_num_seqs` sweep = {1,2,4,8,16,32,64}
- **Git**: `538276073` 이후 (CLI arg / LoRA 순서 / passthrough 3 fix 반영)
- **결과 위치**: `measurement_results/H100x8/g0_06_qwen2.5_32b/`

### Q1. Patch 가 실제로 걸렸나 → YES

Boot log:
```
[HYBRID-KERNEL] §06 patched=128 skipped=0 (filter=0, error=0)
arch=Qwen2ForCausalLM lora=False
scope=Qwen2_MLP(.mlp.gate_up_proj,.mlp.down_proj)
quantize=load-time-1x repack=0 non_patched_layers=ipex_unchanged
```

CPU engine 2개 각각 128 layer (64 × 2 proj) 치환 확인. `applied_features.json.hybrid_config.vnni_hot_path = True`. Filter 로 MoE / vision / audio 계열은 제외되며 이번 32B 모델에는 해당 없음 (skipped=0).

### Q2. seqs=1 에서 유의미한 이득이 있는가 → YES (−28% wall, −22% TPOT)

동일 git sha / 동일 env 에서 `HYBRID_VNNI_HOT_PATH=0` vs `1` 비교 (각각 `090849` / `092458`):

| 지표 | off | on | 변화 |
|---|---:|---:|---:|
| duration | 80.0 s | **57.6 s** | **−28%** |
| output_throughput | 770.7 tok/s | **1070 tok/s** | **+39%** |
| median TPOT | 63.6 ms | **49.6 ms** | **−22%** |
| mean TPOT | 66.3 ms | 47.5 ms | −28% |
| mean TTFT | 1517 ms | 1510 ms | ≈ 0 |
| p99 TPOT | 68.4 ms | 72.9 ms | +7% (outlier) |

Q8_0 (INT8 weight + fp16 per-block scale, activation BF16 유지) 로 MLP DDR read 절반화 → decode memory-bound 구간 명확 개선. TTFT 불변은 prefill 경로 무관 (예상). p99 미세 증가는 일부 tail outlier, 전체 분포 하락.

### Q3. Batch scaling 이 나오는가 → NO (선형 확장, G1 미달)

`per_req_cost(N) = duration(N) / completed(N) × 1000` (ms):

| seqs | per_req_cost (ms) | ratio vs seqs=1 |
|---:|---:|---:|
| 1 | 115.3 | 1.00 |
| 2 | 188.4 | 1.63 |
| 4 | **333.4** | **2.89** (G1 ≤ 2.0 미달) |
| 8 | 584.0 | 5.07 |
| 16 | 1047.8 | 9.09 |
| 32 | 1939.6 | 16.82 |
| 64 | 3836.5 | **33.27** (~선형) |

`cost(N)/cost(1) ≈ N × 0.52 + constant`. 즉 batch amortization 이 전혀 안 되고 per-req cost 가 seqs 에 선형 증가. G1 Batch scaling 조건 `cost(4)/cost(1) ≤ 2.0` **실패**.

### Q4. Wall ratio (hybrid / gpu_only) → G1 미달 확정

gpu_only TP=8 baseline duration 5.4 s (output_throughput 11,473 tok/s) 기준:

| seqs | wall ratio |
|---:|---:|
| 1 | 10.7× |
| 4 | 31.0× |
| 16 | 97.5× |
| 64 | 356.9× |

G1 조건 `< 8×` 모든 seqs 실패. seqs=1 에서도 여전히 GPU-only 대비 10.7× 느림 (단 seqs=1 은 CPU engine 에 1 req × 2 NUMA 만 가는 구조라 대부분 GPU 가 처리, 이 이득은 CPU 쪽 tail 감소에서 나온 것).

### G1 판정 (축별)

| 축 | 조건 | 실측 | 결과 |
|---|---|---|:---:|
| Batch scaling | `cost(4)/cost(1) ≤ 2.0` | 2.89 | ✗ |
| Tail | `< 100 s` | seqs=4 에서 333 ms×500req=167 s 초과 시작, seqs=64 1918 s | ✗ |
| Wall ratio | `< 8×` | seqs=1 에서도 10.7× | ✗ |
| CPU contribution | 증가 | baseline 대비 증가 없음 (routing 동일) | — |

> **정정 마커 (2026-04-20, v8 SSOT 이후)**: 아래 v6 해석의 `§11/§25 필수 전제`, `§18 G3 핵심` 문구는 당시 시점의 해석이다. 최신 단일 진실 공급원은 본 파일의 v8 `SSOT-2/3/4` 이다.

**§06 단독 G1 미통과 확정**. 당시에는 §01 Stop/Go Case 3 ("단일 req 만 빨라지고 batch scaling 없음") 에 부분 해당하나, §06 의 scope 자체가 MLP 만 치환이라 batch scaling 해소는 처음부터 §11/§25 의 역할로 설계되어 있었다고 해석했다. **현재 기준으로는 이 결론은 obsolete** 이며, 최신 원인 트리는 v8 SSOT 참조.

### 원인 분석 (batch scaling 실패)

1. **Attention 경로 IPEX 유지** — §06 의 scope 가 MLP 만 (`*.mlp.gate_up_proj`, `*.mlp.down_proj`). Attention 은 `_IPEXPagedAttention` → `ipex_modules.PagedAttention.single_query_cached_kv_attention`. batch>1 에서 per-seq attention 이 선형 확장하며 전체 step 을 지배.
2. **Activation BF16** — Q8_0 는 weight-only quantization. kernel 내부에서 per-row INT8 dynamic quant 을 하지만, Python 쪽 tensor 는 BF16 유지 → activation 메모리 read/write 는 여전히 BF16. W8A8 (§24) 이 들어가야 activation bandwidth 도 절반화.
3. **GQA-aware 구조 부재** — Qwen2.5-32B 는 GQA 40:8 (Q 40 head, KV 8 head, ratio 5:1). CPU attention 은 per-query-head 로 KV 를 5번 재로드 (중복). §25 의 GQA-aware batched paged attention 이 이걸 1회 로드로 재구성해야 batch 친화적 scaling 가능.

### 결론

**§06 hot path wiring 은 기법 자체는 완결**. seqs=1 단독 decode 이득 −28% 확인. 하지만 아래의 "`§11/§25`가 G1/G2의 필수 전제" 결론은 **현재 기준으로 obsolete** 이다.

### 후속 측정 필요 항목 (historical, 이후 v8 SSOT 로 대체)

- 당시에는 `§11`, `§24` 측정이 다음 단계 후보로 적혔으나, 이후 `§11 Phase 1` 기각과 v8 SSOT 정리로 우선순위가 바뀌었다.
- 최신 다음 단계는 Tier 1 후보 (`§22 → §28 → §13`) 순차 검증이다. `§16 SparAMX` 는 2026-04-20 후반 기각.

---

## v7 — 2026-04-20: §06 batch 역효과 실측 + 원인 확정 + §06-1 분리

### 검증 환경

- 하드웨어: H100×8 (HBM3 80GB, TP=8) + Xeon 8480+ 2-socket (NUMA 2, 56 core/socket, AMX + AVX-512 VNNI)
- 모델: Qwen/Qwen2.5-32B-Instruct
- 워크로드: 500 req × 128/128, `request_rate=inf`, `HYBRID_CPU_THREADS=48`, `num_cpu_engines=2`
- 비교 쌍:
  - `g0_00_qwen2.5_32b_base` — `HYBRID_VNNI_HOT_PATH=0` (§06 off, baseline)
  - `g0_06_qwen2.5_32b` — `HYBRID_VNNI_HOT_PATH=1` (§06 on)
  - 나머지 env 완전 동일, **단일 flag 차이**
- Git: main `6375665e2` (§06 코드 merge 완료)

### Q1. §06 on 이 seqs=1 에서 이득인가 → YES (+18% outTP, +17% duration)

| 지표 (seqs=1) | base (§06 off) | §06 on | Δ |
|---|---:|---:|---:|
| duration | 67.8 s | 57.6 s | **−15%** |
| outTP | 908.9 tok/s | **1069.7** | **+18%** |
| medTPOT | 58.5 ms | 49.6 ms | −15% |
| reqTP | 7.37 | 8.67 | +18% |

Q8_0 kernel 이 M=1 GEMV 로 돌 때 weight BW 절반화 이득이 그대로 나타남. 가설 (decode memory-bound) 과 일치.

### Q2. §06 on 이 seqs≥2 에서 어떤가 → 대규모 역효과 (seqs=64 에서 outTP −90%)

| seqs | base dur | §06 dur | base outTP | §06 outTP | Δ outTP |
|---:|---:|---:|---:|---:|---:|
| 1 | 67.8 | 57.6 | 908.9 | **1069.7** | **+18%** |
| 2 | 68.8 | 94.2 | 895.9 | 654.6 | −27% |
| 4 | 103.6 | 166.7 | 595.3 | 370.0 | −38% |
| 8 | 107.2 | 292.0 | 575.2 | 211.2 | −63% |
| 16 | 96.7 | 523.9 | 637.8 | 118.2 | −81% |
| 32 | 145.7 | 969.8 | 423.1 | 63.7 | −85% |
| 64 | 181.7 | 1918.3 | 339.7 | 32.2 | **−90%** |

wall / reqTP / outTP 모두 같은 방향. routing 정책 동일하므로 CPU 처리 속도 자체의 차이.

### Q3. 원인은? → §06 kernel 이 batch-oblivious (`q8_0_linear_impl` 의 M 번 GEMV 반복)

`csrc/cpu/quant_q8_0.cpp::q8_0_linear_impl` (241-247):
```cpp
for (int m = 0; m < M; ++m) {
    q8_0_gemv_vnni_impl(xq_ptr + m*K, ..., out_f32 + m*N, N, K);
}
// q8_0_gemv_vnni_impl 내부:
#pragma omp parallel for schedule(static)
for (int n = 0; n < N; ++n) { ... }       // N 축만 병렬
```

M 축을 GEMM 차원으로 활용하지 않고 GEMV 를 M 번 순차 호출. IPEX AMX (baseline) 는 동일 M 을 GEMM 차원으로 써서 amortize. 그래서 §06 의 역효과는 AMX 가 VNNI 보다 빠른 게 아니라 §06 kernel 구현 결함.

### 이전 분석의 오류 정정

> **정정 마커 (2026-04-20, v8 SSOT 이후)**: 이 v7 섹션은 v6 오해석을 바로잡는 중간 단계 기록이다. 여기서 남아 있는 "`§11/§25 착수 여부 결정`" 류의 서술도 이후 §11 Phase 1 실패로 더 축소됐다.

v6 에서 "§06 단독 G1 미통과 → §11/§25 (attention batch scaling) 필요" 로 결론내렸던 것은 **절반만 맞다**:
- 정확한 부분: §06 on 상태에서 batch scaling 이 실패하는 것 자체는 사실
- 오해석: 그 원인을 "attention 이 선형 확장하며 병목" 으로 지목했으나, 실제 주된 원인은 **§06 MLP kernel 의 batch-oblivious 구현**. Attention 경로는 §06 off/on 양쪽 동일 (IPEX `_IPEXPagedAttention`) 이므로 §06 의 역효과를 attention 탓으로 돌린 건 잘못.

Baseline (§06 off) 의 batch scaling 은 실제로 나쁘지 않음:

| seqs | base per_req_cost | ratio vs 1 |
|---:|---:|---:|
| 1 | 135.6 ms | 1.00 |
| 4 | 207.2 ms | **1.53** (G1 조건 ≤ 2.0 충족) |
| 16 | 193.4 ms | 1.43 |
| 64 | 363.4 ms | 2.68 |

즉 IPEX + IPEX attention 조합이 이미 seqs 1–16 에서 cost/1 ≈ 1.0–1.6 로 amortize. **G1 조건 `cost(4)/cost(1) ≤ 2.0` 은 baseline 상태에서 이미 통과**. §11/§25 가 추가로 들어갈 개선 여지는 처음 생각한 것보다 작다.

### Q4. AMX vs VNNI 우열은? → 본 측정으로는 확정 불가

"AMX peak > VNNI peak 이므로 large M 에서 AMX 가 유리" 는 일반론으로 맞지만, 본 대조는 "AMX 써진 BF16 kernel (IPEX) vs batch-oblivious VNNI kernel (§06)" 비교라 **AMX 자체의 이득을 분리해 말할 수 없다**. §06-1 에서 batch-aware VNNI GEMM 을 먼저 완성한 뒤, 그게 large M 에서도 IPEX AMX BF16 과 비교해서 어떤지 측정해야 한다. Phase 2 (AMX-INT8) 는 그 결과를 보고 결정.

### Routing 해석의 한계

두 측정 모두 `hybrid_server_run.log` 에 `Router stats` 로그가 없었다 (`VLLM_HYBRID_PROFILE=0` 이라 stats log interval 이 다운). 따라서 "CPU 가 몇 req 를 처리했는가" 의 직접 증거는 없다. 간접적으로:
- wall time 의 대부분이 CPU tail 이라는 건 `wall − gpu_only_wall` 이 거의 그대로 CPU 처리 시간이라는 점에서 확인 가능 (seqs=64 기준 base 176 s, §06 on 1913 s → 10× 차이가 CPU 처리 속도에서 발생)
- routing 정책은 양쪽 동일하므로 CPU 할당 req 비율은 같다고 간주

완전한 routing 분리를 위해선 `VLLM_HYBRID_PROFILE=1` + `HYBRID_STATS_LOG_INTERVAL` 강제 로깅 설정 필요. §06-1 측정 전에 이 설정도 baseline 수준에서 다시 한 번 찍어야 attribution 이 깔끔.

### §06-1 로 분리한 근거

초기 §06 문서에 "Phase A 구현 완료, Phase B 추후" 라는 placeholder 를 남긴 뒤, 이번 측정 결과에 맞춰 kernel 수정 작업을 Phase B 에 사후 끼워 넣는 것은 기록 품질이 나빴다. Phase 용어를 철회하고 **정식 § 번호 06-1** 로 정리. §06 은 "Q8_0 dispatch 경로 구축 (seqs=1 이득 확인)" 까지로 범위를 한정, §06-1 에서 kernel 의 batch 결함 수정.

### 다음 측정 계획

- §06-1 (A) VNNI INT8 GEMM path 구현 → `g0_06_1_qwen2.5_32b/` sweep 측정 → base/§06 on 대비 outTP 방향 확인
- 기준: seqs=1 기존 이득 (+18%) 유지 + seqs 4/8/16 에서 base 대비 손실 없음
- 충족 시 G1 gate 재판정 → 당시에는 `§11/§25` 착수 여부를 결정하려 했으나, 이후 `§11`은 Phase 1 기각 상태가 됨
- 불충족 시 (seqs≥16 에서 여전히 base 대비 열세) Phase 2 (AMX-INT8) 또는 §06 전면 제거 판단

---

## 2026-04-20: §11 Phase 1 기각 + Tier 1 근거 기준 정립

### §11 Phase 1 실패 기술 분석

**조건**: §06 + §06-1 v1 스택 위에 §11 Option A (IPEX 우회 + 기존 `batch16_paged_attention_v1` dispatch 활성) 적용. C++ 변경 0 줄, Python 변경만.

**측정 결과** (500 req × 128/128, TP=8, cpu_max_num_seqs sweep):

| seqs | §06-1 v1 outTP | §11 Phase 1 outTP | Δ |
|---:|---:|---:|---:|
| 1 | 1,196.3 | 1,056.5 | −11.7% |
| 2 | 794.0 | 735.3 | −7.4% |
| 4 | 496.2 | 501.1 | +1.0% |
| 8 | 272.3 | 258.4 | −5.1% |

gpu_only 대조: 11,522.95 tok/s (hybrid 는 어느 설정에서도 gpu_only 의 10% 수준).

**실패 원인** (구현 전 예측했어야 했던 구조적 이슈):

1. `batch16_paged_attention_v1` 의 dispatch 로직 상 `num_seqs >= 16` 일 때만 full-batch path (`num_full_batches >= 1`) 활성. seqs=2/4/8 은 **remainder loop** 로 떨어지며, 이 loop 는 per-seq OMP-parallel over heads 구조 — IPEX `single_query_cached_kv_attention` 와 memory access / 병렬화 패턴 사실상 동일. 이득 구조적으로 불가능
2. layout 일관성 위해 `_PagedAttention.split_kv_cache` layout (`[blocks, kv_heads, head_size/x, block_size, x]`) 를 강제, 덕분에 prefill 경로도 IPEX `flash_attn_varlen_func` → pure SDPA fallback 으로 전환. IPEX prefill 이 SDPA 보다 빠른 구간이 있으면 순손실
3. seqs=1 에서도 −11.7% 관측 — attention kernel 과 무관한 M=1 GEMV path. 원인 추정: (a) 측정 노이즈 (§06-1 v2 도 seqs=1 에서 −10% 동일 패턴), (b) `_PagedAttention` 경로 강제 전환이 KV cache allocator / TLB 에 간접 영향

**구조적 발견** (§11 과 무관, 상위 문제):

§06-1 v1 자체가 seqs=1 (1196) → seqs=8 (272) 로 **4.4× 감소**, seqs 증가에 따라 warmup 시간 기하급수 증가. M>1 에서 kernel 이 **serialize 하고 있음** (weight reuse 이론 실효 없음). `q8_0_gemm_vnni_impl` 의 per-M 비용이 super-linear → batch 는 cost 를 amortize 해야 하는데 거꾸로 증가. 이 문제는 attention kernel 변경으로 해결 불가능.

### Tier 1 근거 기준 정립

2회 연속 kernel 실패 (§06-1 v2, §11 Phase 1) 의 공통 원인: "원리/인용 수준" 인 Tier 2 기법을 "검증된" Tier 1 확신으로 추진했음. 기준 재정의:

**Tier 1 (근거 단단, 우선 검토)** — 선행 연구에 **실측 수치 + 조건** 이 보고된 기법만:

| § | 보고 수치 | 측정 조건 | 출처 |
|---|---|---|---|
| §13 T-MAC LUT INT4 | 4× (22 tok/s > NPU 10.4) | edge CPU (ARM Snapdragon) | Microsoft T-MAC arXiv 2407.00088 + GitHub 공식 |
| ~~§16 SparAMX~~ | ~~linear 1.42×, attention 1.14×~~ | ~~Xeon SPR~~ | **기각 2026-04-20** (unstructured pruning 은 GPU tensor core sparse 미지원, 2:4 로는 SparAMX 수치 근거 깨짐) |
| §22 NEO asymmetric | throughput 14.3% | **H100 + 70B** (우리 HW/규모 동일) | Jiang et al. MLSys'25 + GitHub |
| §28 xFasterTransformer 이식 | Intel SPR 실측 (블로그) | SPR production | Intel 공식 maintained |

**Tier 2 (원리만, 추가 검증 필요)**: §10, §11, §14, §15, §17, §24, §25. 인용 논문은 있으나 "이 조건에서 X% 이득" 같은 보고 수치 없거나, 보고 조건이 우리 환경과 괴리.

**Tier 3 (infra / 기각)**: §01~§09, §04 (기각), §06/§06-1 (infra 구현).

### 적용 규칙

1. 기법 제안/착수 시 Tier 1 → Tier 2 → Tier 3 순
2. Tier 2 를 Tier 1 확신으로 취급 금지
3. 각 제안마다 **선행 연구의 보고 수치 + 조건** 명시. 우리 환경과 차이 있을 시 재검증 예산 별도 책정

### 교훈

1. 구현 전에 kernel 의 **실제 호출 경로 분석 필수**. §11 의 경우 "seqs<16 은 batch16 이 아니라 remainder path" 가 설계 문서에 이미 명시돼 있었으나 "scope 는 seqs 2/4/8 개선" 이라 착오
2. 측정 중단 판정은 조기에 — §11 은 seqs=8 까지 regression 확인된 시점에서 seqs=16 진행 보류 판단 정상
3. CPU batch 병렬화의 근본 결함은 MLP GEMM 축 문제. attention 만 건드려 해결 불가. 다음 시도는 Tier 1 후보 (§22/§28/§13) — §16 2026-04-20 후반 기각 중 선택

---

## v8 SSOT — 2026-04-20: Gate 재정의 + 원인 트리 통합 + 대표 workload 고정

문서 자기모순 / 가설 잔존 지적을 받아 아래 5개를 **단일 진실 공급원 (SSOT)** 으로 고정. 타 문서 (TODO.md, NinjaGap_Todo/README.md) 는 이 섹션을 참조해야 한다.

### SSOT-1. 대표 workload 고정

- **Primary**: Qwen2.5-32B-Instruct × H100x8 (TP=8) × Xeon 8480+ 2S SPR × 500 req × 128/128
- 7B + RTX3090 는 dev 검증용 secondary. **의사결정은 32B 기준만**
- 이전 문서 서두에 남아있는 "7B 26~143× 느림" 서술은 legacy. 32B 기준 최신 gap 은 아래 SSOT-3

### SSOT-2. Gate 재정의 (baseline-relative)

이전 Gate 정의 (`cost(4)/cost(1) ≤ 2×`) 는 baseline §06 off 측정에서 이미 통과됨이 확인됨 (v7 §06 분석, base seqs=4 ratio=1.53). gate 조건으로 유지 불가. 폐지 + 재정의:

| Gate | 이전 조건 | **재정의 조건** (baseline-relative) |
|---|---|---|
| G0 | sublayer breakdown 확보 | (유지) |
| G1 | ~~4req cost ≤ 2×, tail < 100s, wall ratio < 8×~~ | **hybrid outTP ≥ base outTP × 1.0 at seqs 1/2/4/8/16** — CPU engine 이 baseline 대비 순손실 아님을 증명 |
| G2 | ~~4req cost ≤ 1.5×, tail < 10s, wall ratio < 1.5×~~ | **hybrid outTP ≥ gpu_only outTP × 0.30 at any seqs** — hybrid 가 gpu_only 대비 30% 이상 처리. α > 0 의 실효적 증거 |
| G3 (Ninja Gap) | CPU req↑ + wall ≤ gpu_only | **hybrid outTP ≥ gpu_only outTP at any seqs** — 동일 |

현재 실측 (2026-04-20 기준):
- G0: ✅ 통과
- G1: ✗ 미통과 (§06-1 v1 가 §06 off base 대비 일부 seqs 에서 우위 + 일부 seqs 에서 열세)
- G2/G3: ✗ 미통과 (hybrid 최고치 1196 vs gpu_only 11,523 = 10.4%)

### SSOT-3. §06/§06-1/§11 실패 원인 트리 (단일 버전)

"문서마다 다른 서사" 지적 해결. 아래 트리 1장으로 고정.

**주원인 (primary root cause)**:  
**CPU engine 의 batch 병렬화 자체가 구조적으로 불작동.** seqs=1 → seqs=8 까지 §06-1 v1 에서 outTP 가 1196 → 272 (4.4× 감소). warmup 시간이 seqs 에 기하급수 증가. 즉 현재 CPU kernel 군 (`q8_0_gemv_vnni_impl`, `q8_0_gemm_vnni_impl`, `batch16_paged_attention_v1`) 어느 것도 M>1 에서 amortize 를 못함.

**부원인 (contributing)**:
1. `q8_0_linear_impl` 의 M>1 경로가 원래 GEMV 를 M 번 순차 호출하는 batch-oblivious 구현이었음. §06-1 v1 에서 weight reuse GEMM 으로 교체했으나 실측상 여전히 super-linear cost. 이론상 weight reuse 이득이 실제로 발현되지 않음
2. IPEX baseline attention (`single_query_cached_kv_attention`) 이 이미 batch 를 어느 정도 amortize 하고 있어, 우리 `batch16_paged_attention_v1` 의 remainder path 가 이득 창출 공간을 가지지 못함
3. §11 Phase 1 의 prefill IPEX → SDPA fallback 오버헤드는 순손실. layout 일관성 강제의 부작용

**반증된 가설** (이전 문서에 남아있었으나 실측으로 기각):
- ~~"§06 의 seqs≥2 역효과는 attention 이 선형 확장하기 때문"~~ — attention 은 §06 on/off 양쪽에서 동일한 IPEX 경로. §06 의 역효과 원인이 될 수 없음. v7 에서 이미 정정했으나 일부 문서에 잔존
- ~~"§11 이 G2 핵심축 (batch-aware attention 이 scaling 돌파구)"~~ — §11 Phase 1 전 seqs (1/2/4/8) regression 으로 기각. MLP 축이 주원인인데 attention 만 건드려 해결 불가
- ~~"§06-1 v2 의 VNNI `vpdpbusd` 직접 사용이 kernel 속도 개선"~~ — half-tile waste + s8s8 compensation overhead 로 v1 대비 −7~−13% regression

### SSOT-4. §11 지위

- 이전: G2 핵심축 / scenario 에 "batch-aware attn 12× scaling"
- **현재**: 실패 가설 (failed hypothesis, pending redesign). Phase 2 (v2 신규 kernel) 재시도는 Tier 1 후보 (§22/§28/§13) — §16 2026-04-20 후반 기각 선행 검증 후에만 재평가

### SSOT-5. 이론 상한 표 무효화

`NinjaGap_Todo/README.md` 의 "경로 1 누적 이론 상한" 표 (+35×, +70× 누적) 는:
- 출처가 "논문 수치 직접 곱하기" — TODO.md 의 Guardrails 중 "외부 논문 speedup 수치를 우리 코드에 직접 곱하기" 금지 조항 자기 위반
- §06-1 v1 실측 이후 전제 자체가 무너짐 (그 표는 baseline=1× 기준인데 실측 1× 자체가 역효과 구간 가짐)
- 조치: **"HISTORICAL FANTASY — INVALIDATED (2026-04-20)"** 로 강등. 삭제보다 이력 유지 가치.

### SSOT-6. Spec Decode (§18) 시나리오 확률

이전: "경로 1 + Spec Decode 50% 권장". 현재 CPU kernel 자체가 baseline 대비 순손실인 상태에서 Spec Decode 2× 를 공식 권장 시나리오로 두는 건 근거 빈약.

- 조치: 시나리오 확률을 50% → **근거 불충분. CPU baseline 통과 후 재평가** 로 강등
- §18 은 Tier 2 (원리만) 로 분류 유지. Leviathan / EAGLE / DuoDecoding 논문 자체는 GPU 실측 위주

### SSOT 반영 대상 파일

- `TODO.md` — workload 헤더, Gate 조건, §06-1 상태, §11 지위, 이론 상한, §18 시나리오
- `NinjaGap_Todo/README.md` — 위 동일 + flag 표 (HYBRID_BATCH_AWARE_ATTN ✗)
- 기타 문서는 위 두 개를 참조. 원인 서술은 본 SSOT 의 원인 트리 인용
