# vLLM Hybrid — 작업 이력

이 파일은 완료된 작업 내역만 기록한다. 남은 작업은 `TODO.md`, 프로젝트 설계/구성은 `CLAUDE.md`, 설계 의도는 `docs/paper/main.tex` 를 참조.

---

## 2026-04-10 ~ 2026-04-11: CUDA 13.0 마이그레이션 + Hybrid 동작 검증 (이번 세션)

### 개요
CUDA 12.8 → 13.0 빌드 전환은 별도 4 commits 로 이미 완료된 상태에서 시작. 이번 세션에서는 마이그레이션 후 드러난 **런타임 레이어의 치명적 이슈들** 을 진단·수정하고 dev 환경에서 end-to-end 검증을 수행했다.

### 선행 commit (CUDA 13.0 빌드 마이그레이션 — 이전 세션에서 완료)

| commit | 내용 |
|------|------|
| `88da7ffd` feat | CUDA 12.8 → 13.0 전환 · torch 2.7.1 → 2.9.0 · `cuda_compat_cub.cuh` shim (`cub::Sum/Max` 제거 대응) · marlin `__global__` hidden visibility fix · flash-attn CUTLASS deprecated warning 억제 · Dockerfile/CI 업데이트 |
| `f8d51872` fix | Marlin device linking: `CUDA_SEPARABLE_COMPILATION`, `CUDA_RESOLVE_DEVICE_SYMBOLS`, `gptq_marlin.cu` 에 `-rdc=true` · `DeepseekVLV2Config` dataclass field order (transformers 5.x) · `all_special_tokens_extended` 제거 대응 |
| `9f2b932b` fix | `_custom_ops.py` 에 CPU fallback (`rms_norm`, `fused_add_rms_norm`, `rotary_embedding`, `apply_repetition_penalties`) · `process_engine_outputs` 가 `output.finished` 속성으로 완료 감지 · `on_request_finished` docstring 버그 · `torch.compile` CPU 비활성 · `flashmla` CCCL 헤더 경로 |
| `484b8901` fix | `_resolve_cpu_params` + `_setup_cpu_process_env` 를 torch import 전에 호출 · `intel_cpu_utils` torch lazy import · flashmla CCCL 재수정 · eval env 파일들 정정 |

### 이번 세션에서 진단·수정한 런타임 이슈 (uncommitted)

#### Issue #1 — `_C_utils.init_cpu_threads_env` CUDA 빌드에서 미등록 (가장 치명적)
**증상**: H100x4 KVM 이전 실험에서 96 logical CPU 중 6-8개만 100%, 평균 CPU util 6.4%. CPU worker 부팅 로그에 `init_cpu_threads_env not found` warning 은 있으나 조용히 무시됨.

**원인 체인**:
1. `CMakeLists.txt:128` 은 `VLLM_TARGET_DEVICE` 가 `cuda`/`rocm` 이 아닐 때만 `cpu_extension.cmake` 를 include 함.
2. `csrc/cpu/torch_bindings.cpp` 가 `init_cpu_threads_env` 를 `_C_utils` namespace 에 등록하는 유일한 지점인데 이 파일은 `cpu_extension.cmake` sources 에만 있음.
3. CUDA 빌드에서는 `csrc/cpu/torch_bindings.cpp` 가 컴파일되지 않아 `init_cpu_threads_env` 가 **어떤 .so 에도 등록되지 않음**.
4. `CPUWorker.init_device` 는 `torch.ops._C_utils.init_cpu_threads_env(...)` 를 호출하나 `AttributeError` 가 `except` 에 잡혀 warning 만 찍고 넘어감.
5. 결과: `csrc/cpu/utils.cpp` 의 핵심 경로가 **전혀 수행되지 않음** — `omp_set_num_threads(N)`, `#pragma omp parallel for` 로 각 OMP thread 를 단일 코어에 1:1 `sched_setaffinity`, `numa_set_membind` + `numa_set_strict(1)` + `numa_migrate_pages` 모두 건너뜀.

**Fix (정도)**: `_C_utils` standalone extension 신규 생성
- `cmake/cpu_utils_extension.cmake` (신규): OpenMP + libnuma 만 require, AVX-512/AMX 무관, x86_64 어느 환경에서든 빌드. OpenMP 없으면 graceful skip, libnuma 없으면 `VLLM_NUMA_DISABLED` stub.
- `csrc/cpu/torch_bindings_utils.cpp` (신규): `_C_cpu_ops` 와 동일 include 패턴 (`<torch/all.h>`, `<torch/library.h>`, `core/registration.h`, SABI 호환) 으로 `init_cpu_threads_env` forward declare + TORCH_LIBRARY_EXPAND 등록.
- `csrc/cpu/utils.cpp` 수정: 불필요한 `cpu_types.hpp` include 제거 (SIMD 의존 0). `<torch/all.h>`, `<omp.h>`, `<vector>`, `<sstream>` 명시 include. AVX-512/AMX 없는 dev 에서도 컴파일.
- `CMakeLists.txt`: `cpu_hybrid_extension.cmake` include 바로 아래에 `cpu_utils_extension.cmake` include. CUDA/ROCm build 경로 내 (CPU-only build 와 자동 mutually exclusive).
- `setup.py`: `CMakeExtension("vllm._C_utils", optional=True)` 등록.
- `vllm/_custom_ops.py`: `contextlib.suppress(ImportError)` 안에서 `import vllm._C_utils` → `HAS_CPU_UTILS` 플래그.

**Fix (방어)**: `CPUWorker._python_init_cpu_threads_env` Python fallback (`vllm/v1/worker/cpu_worker.py`)
- C++ binding 이 AttributeError / RuntimeError 로 실패 시 `os.sched_setaffinity` + `torch.set_num_threads` + NUMA membind 로 graceful degrade.
- process-level affinity 만 설정하므로 1:1 pinning 효과는 없지만 완전 무효화보다 나음.

**검증 (dev, i9-12900KF + RTX 3090, AVX2)**:
- `_C_utils.abi3.so` (66 KB) 빌드 성공, `nm -D` 로 `init_cpu_threads_env` 심볼 확인.
- 서버 부팅 로그: `[HYBRID-CPU-WORKER] init_cpu_threads_env (C++) returned: OMP tid 2504590→core 1, 2504312→core 3, ... 2504326→core 23` — 16 OMP thread 가 16 코어에 정확히 1:1 pin.
- 1 시퀀스 추론 중 per-thread sampling 6 샘플: **TID-PSR 고정 매핑, 전원 100% CPU**, OS migration 0건.

#### Issue #2 — `Failed to update config for CPU:` (빈 AssertionError)
**증상**: CPU EngineCore 부팅 시 항상 출력되는 ERROR. 메시지가 빈 문자열이라 원인 진단 불가.

**원인 체인**:
1. `run_cpu_engine_core` 가 `os.environ["CUDA_VISIBLE_DEVICES"] = ""` 로 CUDA 격리.
2. CPU 프로세스 내부에서 `torch.cuda.device_count() == 0`.
3. `WorkerBase.init_worker` (`vllm/worker/worker_base.py:577`) 휴리스틱: `if world_size > num_gpus: device_type = "heterogeneous"`. `world_size=1, num_gpus=0` 이라 항상 발동.
4. `CPUWorker.__init__` 가 `CpuPlatform.check_and_update_config(vllm_config)` 호출.
5. `vllm/platforms/cpu.py:264` 의 `assert vllm_config.device_config.device_type == "cpu"` 가 fail → **빈 메시지 AssertionError**.
6. CPUWorker 의 `except Exception as e: logger.error(f"...: {str(e)}")` 가 빈 문자열만 출력.

**Fix (정도)**: `vllm/worker/worker_base.py` 의 heterogeneous 휴리스틱에 `is_hybrid_cpu_engine` 우회 조건 추가. `CUDA_VISIBLE_DEVICES=""` + `hybrid_config` 존재 + `worker_cls` 에 `cpu_worker` 포함이면 휴리스틱 skip.

**Fix (방어)**: `vllm/v1/worker/cpu_worker.py` 의 `CpuPlatform.check_and_update_config` 호출 직전에 `device_type` 이 `"cpu"` 가 아니면 명시적으로 강제 set. 다른 경로로 heterogeneous 가 들어와도 안전.

**진단 개선**: `logger.error(f"...: {str(e)}")` → `logger.exception("..., type=%s, msg=%r", type(e).__name__, str(e))`. 이제 assertion 처럼 empty string 예외도 traceback + exception type 으로 추적 가능.

**검증**: dev 7B 서버 재부팅 로그에서 `[HYBRID-CPU-WORKER] Forcing device_config.device_type 'heterogeneous' → 'cpu'` 관측 후 `Updated vllm_config for CPU platform via check_and_update_config (device_type='cpu')` 로 정상 통과. `Failed to update config` 에러 0건.

#### Issue #3 — `num_cpu_engines` auto 감지 미구현 + `cpu_max_num_seqs` 공식 오류
**증상**: 설계 원칙은 `num_cpu_engines = num_numa, cpu_max_num_seqs = 1 per engine` 인데 코드는 따르지 않고 있었음.

| 항목 | 설계 원칙 | 이전 코드 |
|------|---------|---------|
| `num_cpu_engines` auto | NUMA 노드 수 | 자동 감지 없음, `HybridConfig.num_cpu_engines=1` 고정 |
| `cpu_max_num_seqs` auto | **1 고정** (per engine) | `max(4, effective_cores // 4)` |

H100 Xeon 8480+ 2 socket 환경에서 auto 로 두면 `num_cpu_engines=1` (NUMA 2 무시), `cpu_max_num_seqs=14` (= 56/4) 가 되어 원칙과 정반대. 이전에는 env 파일에서 `HYBRID_CPU_MAX_SEQS=1` 을 수동 override 로 증상만 가림.

**Fix**:
- `vllm/v1/engine/hybrid_core.py`:
  - `_resolve_cpu_params` 의 `cpu_max_num_seqs` 자동값을 `max(4, eff/4)` → **1 고정** 으로 변경. 사용자 수동 override 는 허용하되 경고 로그.
  - `_resolve_num_cpu_engines()` 공용 resolver 신규: `numa_aware=True` 이면 `NUMAAllocator.num_nodes` 반환, 아니면 1.
  - `launch_hybrid_engines` 가 공용 resolver 호출하도록 교체.
- `vllm/v1/engine/core_client.py`:
  - `HybridAsyncMPClient.__init__` / `HybridSyncMPClient.__init__` 에서 resolve 후 `vllm_config.hybrid_config.num_cpu_engines` 에 write-back. client (router + `_compute_core_engines`) 와 launcher 가 모두 같은 값을 보도록 단일 진실 공급원 확보.
- `vllm/config.py`: `HybridConfig.num_cpu_engines` 기본값 `1` → `0` (auto sentinel).
- `vllm/engine/arg_utils.py`: `hybrid_num_cpu_engines` 기본값 `1` → `0` (auto sentinel).

**검증 (dev single NUMA)**:
- 부팅 로그: `[HYBRID-RESOLVE] max_seqs=1 threads=16 ... user_overrides: max_seqs=auto`
- `[HYBRID-LAUNCH] num_cpu_engines=1 (numa_aware=True, config=1)` — resolver write-back 이 정상 전파.
- `CapacityAwareRouter initialized: cpu_max_num_seqs=1 (per-engine), num_cpu_engines=1`.
- env 파일에서 `HYBRID_CPU_MAX_SEQS`, `HYBRID_NUM_CPU_ENGINES` 를 **둘 다 설정하지 않고** auto 경로로 7B 서버가 정상 부팅.

#### Issue #4 — `num_cpu_engines > 1` 환경에서 `_get_autobind_cpu_ids` 가 `local_rank` 로만 NUMA 선택
**증상 (H100x8 2-NUMA 에서만 표면화)**: `num_cpu_engines=2` 일 때 두 CPU 엔진 프로세스 모두 `local_rank=0` 이므로 `_get_autobind_cpu_ids` 가 둘 다 NUMA 노드 0 을 선택 → 같은 56 core 에 두 프로세스가 1:1 pin 시도 → contention 으로 성능 1/2.

**원인**:
1. `launch_hybrid_engines` 가 multi-engine 시 각 엔진에 `numa_node=0, 1, ...` 을 proc kwargs 로 전달.
2. `run_cpu_engine_core` 가 `numa_node_override` 를 받아 **로컬 변수** `hybrid_cfg` 에만 `numa_bind_node` inject. `vllm_config` 는 원본 유지.
3. `_create_cpu_vllm_config` 가 `cpu_config.hybrid_config = HybridConfig()` (빈 객체) 로 reset → `numa_bind_node` 정보 손실.
4. CPUWorker 는 `self.vllm_config.hybrid_config.numa_bind_node = None` 을 봄.
5. `_get_autobind_cpu_ids` 는 fallback path (`local_rank=0`) 로 떨어져 모든 CPU 엔진이 NUMA 0 만 선택.

**Fix**:
- `vllm/v1/engine/hybrid_core.py`:
  - `run_cpu_engine_core` 에서 `numa_node_override` 를 `hybrid_cfg` 뿐 아니라 `vllm_config` 에도 `replace` 로 반영 + `kwargs["vllm_config"] = vllm_config` 로 쓰기.
  - `_create_cpu_vllm_config` 의 `HybridConfig()` 빈 객체 reset 을 `HybridConfig(mode="none", numa_aware=..., numa_bind_node=...)` passthrough 로 교체. `mode="none"` 은 hybrid 무한 재귀 방지용.
- `vllm/v1/worker/cpu_worker.py` `_get_autobind_cpu_ids`:
  - `self.vllm_config.hybrid_config.numa_bind_node` 가 set 되어 있으면 **우선 사용**, 없으면 기존 `local_rank` 기반 fallback.

**검증 (dev single NUMA)**:
- 단일 NUMA 환경에서 fallback 경로가 깨지지 않음: `[HYBRID-CPU-WORKER] _get_autobind_cpu_ids: local_rank=0 → node_idx=0 → NUMA node 0`.
- H100x8 2-NUMA 에서의 실제 검증은 타겟 환경 이관 시 수행 예정.

#### 진단 로그 체계 7종 신규 추가
모든 핵심 경로에 trace marker:
- `[HYBRID-RESOLVE]` — `_resolve_cpu_params` 의 최종 결과값과 user override 여부
- `[HYBRID-CPU-ENV]` — `_setup_cpu_process_env` 가 설정한 OMP/MKL/OPENBLAS 환경변수
- `[HYBRID-CPU-PROC]` — CPU EngineCore 프로세스 초기화 직후 torch threads / mkldnn 상태
- `[HYBRID-CPU-WORKER]` — thread binding 경로 (C++ vs Python fallback), init_device 결정, post-init affinity
- `[HYBRID-CLIENT]` — 라우팅 dispatch (request → engine identity)
- `[HYBRID-CPU-EXEC]` — CPU worker `execute_model` per-step trace (reqs, tokens, elapsed)
- `[HYBRID-CPU-ATTN]` — decode path counter (`custom_avx`, `ipex`, `sdpa_batched`, `sdpa_loop`)

`VLLM_HYBRID_TRACE=1` 또는 `VLLM_HYBRID_TRACE_EVERY=N` 으로 빈도 제어. `eval/serve.sh` 에 export 추가.

### dev 환경 end-to-end 검증 결과

#### 1.5B (Qwen2.5-1.5B-Instruct), 500 prompts
| 지표 | GPU only | Hybrid (cpu-first) |
|------|---------|--------------------|
| Wall time | 14.3 s | 34.9 s |
| Duration | 8.2 s | 14.6 s |
| Req TP | 60.77 req/s | 34.30 req/s |
| Output TP | 7,485 tok/s | 4,225 tok/s |
| GPU util avg | 57.2% | 21.7% |
| **CPU util avg** | **6.2%** | **74.8%** |

CPU util 6.2% → 74.8% (**12배 증가**). 16 OMP thread 가 16 코어에 1:1 pinned, per-thread sampling 에서 PSR 고정 관측. Hybrid throughput 은 −44% — 1.5B + RTX 3090 환경의 구조적 미스매치 (CPU per-req 11 tok/s vs GPU 누적 7,485 tok/s, 비율 0.15%).

#### 7B (Qwen2.5-7B-Instruct) end-to-end 검증
- 서버 부팅: C++ `init_cpu_threads_env` 활성, 16 코어 1:1 pin (`TID 2507392→core 1, 2507548→core 3, ...`).
- 단일 요청 추론 중 per-thread CPU sampling 6 샘플: **16 TID 전원 100%, PSR 6 샘플 내내 고정**.
- Decode path: **532/532 calls `path=ipex`** (fallback 0건, `totals={custom_avx:0, ipex:532, sdpa_batched:0, sdpa_loop:0}`).
- CPU execute_model 타이밍: prefill 1 step 705 ms (9 tokens), decode ~367 ms/token → CPU ~2.7 tok/s.
- 3-req burst: cpu-first + cpu_max_num_seqs=1 이 정확히 동작. 1 req → `cpu:0` (cpu_in_flight=1/1), 2-3 req → `gpu (cpu full, gpu_in_flight=1,2)`. 6.06 s 에 모두 정상 응답.

#### 7B 500 prompts 벤치 (성능 비교가 아닌 경로 재확인)
| 지표 | GPU only | Hybrid |
|------|---------|--------|
| Wall time | 38.8 s | 111.4 s |
| Duration | 30.6 s | 49.0 s |
| Output TP | 2,039 tok/s | 1,270 tok/s |
| GPU util avg | 85% | 26.8% |
| CPU util avg | 5.5% | **75.2%** |

Hybrid wall time 이 −187% 로 악화된 것은 **dev 환경의 구조적 제약**: CPU 1 req 완료 시간 (~47 s) 이 GPU 전체 wall time (~30 s) 보다 길어 CPU tail 이 벤치 전체를 끌고 감. 논문 §Discussion Limitations 의 "작은 GPU 환경에선 hybrid gain 없음" 과 일치. **dev 에서 성능 비교는 무의미**, 경로 검증이 목적.

### 신규/수정 파일 목록

**신규**:
- `cmake/cpu_utils_extension.cmake`
- `csrc/cpu/torch_bindings_utils.cpp`
- `eval/envs/dev_rtx3090_hybrid_smoke.env`
- `eval/envs/dev_rtx3090_500.env`
- `eval/envs/dev_rtx3090_qwen7b_hybrid_verify.env`
- `eval/envs/dev_rtx3090_qwen7b_500.env`
- `docs/CUDA13_MIGRATION_STATUS.md`

**수정**:
- `CMakeLists.txt` (cpu_utils_extension include)
- `setup.py` (`_C_utils` extension)
- `csrc/cpu/utils.cpp` (cpu_types.hpp 제거, header 명시)
- `vllm/_custom_ops.py` (`_C_utils` import, `HAS_CPU_UTILS`)
- `vllm/worker/worker_base.py` (heterogeneous 휴리스틱 is_hybrid_cpu_engine 우회)
- `vllm/v1/engine/hybrid_core.py` (OMP env, `_resolve_num_cpu_engines`, `_resolve_cpu_params` 수정, HybridConfig passthrough, `run_cpu_engine_core` vllm_config replace, `launch_hybrid_engines` resolver 사용, 진단 로그)
- `vllm/v1/engine/core_client.py` (Hybrid async/sync client: resolver write-back, dispatch 로그)
- `vllm/v1/worker/cpu_worker.py` (Python fallback, `_get_autobind_cpu_ids` numa_bind_node 우선, device_type coerce, `execute_model` step trace, 진단 로그, `logger.exception`)
- `vllm/v1/attention/backends/cpu_attn.py` (decode path counter/trace)
- `vllm/config.py` (`HybridConfig.num_cpu_engines` 기본값 `1 → 0`)
- `vllm/engine/arg_utils.py` (`hybrid_num_cpu_engines` 기본값 `1 → 0`)
- `eval/serve.sh` (`VLLM_HYBRID_TRACE` export)

### 이번 세션 검증 미흡 / 이후 과제
남은 작업은 `TODO.md` 참조.

---

## 2026-03-10: Abstract 교체 및 본문 교차 검증 (세션 6)
- 사용자 제공 장문 Abstract로 교체 (6줄 → 14줄)
- 본문 우선 원칙에 따라 3건 수정: "pure energy waste"→"significant energy inefficiency", 중복 "cleanly eliminates" 통합, "all sources"→"software-level" (Claim 1 condition ii 반영)
- Introduction L111 "every source"→"software-level sources" 동일 수정
- 11개 주장 교차 검증 완료: Major 불일치 0건
- 수정 파일: `docs/paper/main.tex`

## 2026-03-10: 3인 페르소나 피어 리뷰 반복 (세션 5)
- 3명 리뷰어 페르소나: A(시스템/OSDI), B(HPC/SC), C(에너지/MLSys)
- 3회 반복으로 Major 0건 달성 (1회차 8건 → 2회차 1건 → 3회차 0건)
- 핵심 수정: Abstract 축약, Theorem→Claim 격하, HPC 유추 톤다운, Roofline KV cache 확장
- 독립 2-인스턴스 baseline 비교 섹션 추가, EMA α→γ 기호 변경
- Table 예측값 Caution 경고, 73% 프레이밍 균형화, Granite/Blackwell 비율 분석
- BSP 형식주의 → "Why separate processes?" 실용적 근거로 교체
- 수정 파일: `docs/paper/main.tex`

## 2026-03-10: 반복 검증 루프로 논문-코드 정합성 완전 검증 (세션 4)
- 4회 반복 (2 에이전트 × 4회 = 8회 검증), 총 16건 수정으로 Major/Critical 불일치 0 달성
- Critical: TP=8 시스템 기여도 9.2% → 1.15% (B_CPU/(k×B_GPU))
- Major: H100 TF32→BF16 dense 정정 (989=BF16 dense, 495=TF32 dense, 1979=BF16 sparse)
- Major: hwloc "defined by" → "conceptual model" (실제 /proc/cpuinfo, lscpu, libnuma 사용)
- Major: FP32 peak 표기 명확화 (2 FMA × 16 elements × 2 ops/FMA = 64 FP32 ops/cycle)
- Major: P_GPU 명시적 정의 추가 (Energy Efficiency Corollary)
- Major: Theorem 1 → "Proof sketch" + 분해 가정의 실험 검증 위임 명시
- Major: Low-load starvation caveat 추가 (CPU-first 시 λ≤N이면 GPU 미사용)
- Major: ZMQ PULL 출력 경로 경합 분석 추가
- Major: GPU tok/s 출처(vLLM 벤치마크) 명시
- Major: Abstract "pure waste" → "significant energy inefficiency"
- Major: 프로세스 시작 방법 → get_mp_context() 동적 선택
- Major: FP32 attention → "per-sequence without batch-16 grouping"
- Major: N_max → 사용자 오버라이드 + 2N_max 초과 설명
- Major: ZMQ identity 표기 일관성 통일 (b'\x00\x00' / b'\x01\x00')
- Major: Q8_0 커널도 VNNI 의존성 필요 → "(1)과 (2) 모두 VNNI 필요"
- Major: Figure 3 mutually exclusive 명확화 + 본문 설명 보강
- 수정 파일: `docs/paper/main.tex`

## 2026-03-10: 논문 비판점 12건 대응 수정 (세션 3)
- **N1(엔지니어링 기여)**: Contribution #1에 "systems integration"으로 재포지셔닝, practical value 강조
- **N3(Theorem 자명)**: Theorem 1에 비간섭 조건 (i)프로세스 격리 (ii)HW 경쟁 없음 명시 + 조건(ii) 논증 단락 추가 (NUMA 분리, HBM/DDR 독립, PCIe 미미)
- **N2(HPC 과장)**: BSP→"Relation to parallel models"로 톤다운, 라우팅 전략 설명을 "inspired by" + 차이점 명시 (work stealing은 proactive dispatch, HEFT는 single threshold, StarPU는 single parameter)
- **T1(메모리 경쟁)**: Theorem proof 뒤에 hardware contention 분석 단락 추가 (NUMA 분리, HBM/DDR 독립 경로, PCIe 미미, 실험 검증 계획)
- **T2(Prefill latency)**: Limitations에 CPU TTFT 10-50x 느림 정량화, length-aware 완화, 실험 우선 계획 명시
- **T4(EMA α=0.3)**: EMA half-life 계산(1.9회), 높은/낮은 α 트레이드오프, sensitivity analysis 계획
- **T5(모델 정확도)**: Limitations에 수치 일관성 논의 추가 (동일 precision, 독립 요청이므로 누적 없음)
- **T6(에너지 미측정)**: Corollary 2 + Discussion에 "theoretical estimate" 명시, RAPL 검증 계획
- **T7(LogGP 불필요)**: LogGP 수식 3개 제거 → 실용적 1문장 결론으로 간소화
- **P1(과잉 형식화)**: "formalize" → "model", "strictly additive" → "additive" 톤다운
- **P3(인용 불균형)**: Related Work에 PowerInfer/HeteGen 정량 비교 추가 + 트레이드오프 명시
- 수정 파일: `docs/paper/main.tex`

## 2026-03-10: 코드 완성도 문제 4건 해결 (세션 3)
- **C++ 커널 연결**: `cpu_attn.py`의 CPU decode 경로에 `_C_cpu_ops.batch16_paged_attention_v1` 디스패치 추가. `HAS_CPU_OPS=True` + BF16/FP32 + num_tokens==num_seqs 조건 충족 시 커스텀 AVX-512 커널 사용, 실패 시 PyTorch SDPA fallback
- **compute_qk_batch16 스텁 제거**: `batch_attention.cpp`에서 미사용 더미 함수 삭제, 파일 헤더 주석을 실제 구현에 맞게 수정 (시퀀스 순차 + head dim SIMD)
- **ParallelBatchExecutor 폐기 표시**: V0 레거시 코드에 deprecated 경고 + 활성 경로(hybrid_core.py) 안내 docstring 추가
- **단위 테스트 30개 작성**: `tests/v1/engine/test_hybrid_core.py` 신규 생성
  - CapacityAwareRouter: capacity(6), length-aware(5), throughput-adaptive(5), warmup(3), fault-tolerance(2)
  - _resolve_cpu_params: auto-detection 공식(5), manual override(1), ResolvedCpuParams(1)
  - 전체 30/30 통과
- 수정 파일: `cpu_attn.py`, `batch_attention.cpp`, `parallel_batch_executor.py`, `test_hybrid_core.py`

## 2026-03-10: 논문 ↔ 코드 교차 검증 및 불일치 수정 (세션 3)
- 논문(main.tex)과 실제 코드 구현을 다각도로 교차 검증
- **불일치 4건 발견 및 논문 수정**:
  1. EMA 스무딩 계수: 논문 β=0.9 → 코드 α=0.3에 맞춰 수정 (수식/표기 전체)
  2. KV cache 상한: 논문 "800GB" → 코드 `max(32, min(512, mem×0.4))`에 맞춰 수정 (Table + 본문)
  3. 동적 슬롯 수식: 논문 `N_max × ratio` → 코드 `clamp(N_max×(1+ratio), 2, 2N_max)`에 맞춰 수정
  4. ONEDNN fallback: non-AMX 시 `AVX512_CORE_VNNI` fallback 추가 기술
- **ParallelBatchExecutor 존재 확인**: 코드 완전 구현(1033줄)이나 실제 호출 경로 없음 (V0 호환용, 미사용)
- **`vllm serve` hybrid 경로 연결 수정**: `make_async_mp_client()`에 `is_hybrid_mode()` 분기 추가 → `HybridAsyncMPClient` 도달 가능하도록 수정. 이전에는 AsyncLLM이 hybrid 분기를 우회하여 GPU-only로 동작했음
- RESEARCH.md 신규 생성 (코드 구현 현황 종합 문서)
- **2차 교차 검증 (소스→논문, 논문→소스 양방향) → 불일치 12건 발견 및 논문 수정**:
  - [높음] 라우팅 우선순위: 논문 "GPU default" → 코드 "CPU-first". Definition 1, Theorem proof, Property 2 전면 수정
  - [높음] NUMA 노드: 논문 "CPU→NUMANode 1" → 코드 "rank=0→NUMANode 0". Fig.4 caption/label, Package 설명 수정
  - [높음] Batch Attention SIMD: 논문 "16 시퀀스×16 lanes 동시" → 코드 "시퀀스 순차, SIMD는 head dim". 설명 수정
  - [중간] 307 GB/s "aggregate" → "per-socket" 수정 (CPU 엔진은 단일 NUMA 노드 사용)
  - [중간] KV cache total_mem → eff_mem (NUMA 노드 메모리) 명시, 예시 수치 수정
  - [중간] 워밍업 강제 완료 메커니즘 (GPU 2W + CPU 1개) 추가 기술
  - [중간] CPU 크래시 장애 허용: "built-in" → "implicit (부수적 효과)" 수정
  - [낮음] ZMQ identity: 0x00/0x01 → 0x0000/0x0100 (2바이트 LE)
  - [낮음] max_seqs: 최소값 4 추가 (Table + 본문)
  - [낮음] cpu_threads: NUMA 비활성화 시 전체 물리 코어 사용 명시
  - [낮음] ISA 감지: "빌드타임만" → "빌드타임(CMake)+런타임(Python)" 수정
- 수정 파일: `docs/paper/main.tex`, `core_client.py`, `CLAUDE.md`, `RESEARCH.md`
- **3차 교차 검증 → 불일치 10건 발견 및 논문 수정**:
  - Fig.4 NUMANode 대역폭: 153.6 → 307 GB/s (8채널)
  - Fig.4 색상 반전 수정: CPU=orange, GPU=green 일관성 복원
  - Fig.3 caption: "layered refinements" → 전략은 mutually exclusive 명시
  - §2.3 전력비: $158K → $18.4K (계산식 명시)
  - §5 프로세스: "spawn" → "fork" 기본값, spawn은 CLI 옵션
  - §5 VNNI: "fused multiply-add" → "VPDPBUSD dot-product accumulation"
  - §6.1 전력비: $15K → $18.4K
  - §1 전력량: 1,500 MWh → 153 MWh (계산식 추가)
  - Fig.1 Xeon peak: 2.0 → 3.6 TFLOP/s (ridge point 6.5→11.7)
  - Table 1 BW: "307 GB/s" → "307 GB/s (per-socket)"
- **4차 교차 검증 (3에이전트: 소스→논문, 논문→소스, 비판적 지도교수) → 7건 수정**:
  - [높음] Property 2 fault tolerance: crash 시 C<N 가능성 명시, health-check watchdog 필요성 추가
  - [높음] Batch Attention: "16 시퀀스 SIMD 동시처리" → "시퀀스 순차 + head dim SIMD" 정확히 수정
  - [중간] throughput-adaptive에 length threshold 적용 사실 누락 → 명시 추가
  - [중간] Fig.3 설명: 시각적 계층 표현이 mutually exclusive와 모순 → 설명문 보강
  - [중간] Roofline 73% 오해 유발 → per-GPU vs system-wide 명확 구분
  - [낮음] "CLI configuration" → 환경변수 VLLM_WORKER_MULTIPROC_METHOD 정정
  - [낮음] NUMA fallback (라이브러리 미가용 시 전체 코어) 명시

## 2026-02-27: 논문 그림 크기 수정 및 논리 흐름 개선 (세션 2)
- §4.3/§4.4 상세 설명 작성 → `docs/DETAIL_INFO_4_3_4_4.md` 저장
- 상세 설명을 논문(main.tex)에 반영: §4.3 HPC 계보 확장, §4.4 NUMA/SMT/hwloc 상세화
- 새 BibTeX 4개 추가: lameter2013numa, eyerman2010smt, vaswani2017attention, hunter1986ema
- TikZ 그림 5개 생성 (fig_script/ 독립 파일 + main.tex 인라인)
  - fig1: Dual-Process Architecture, fig2: Roofline Model, fig3: Routing Flow
  - fig4: hwloc Topology, fig5: Throughput Bar Chart
- Abstract~§4 논리 흐름 검토 → 8개 문제 식별 및 전부 수정
  - §2.4/§2.5 삭제 (중복), Table을 §3.2로 이동, Intro 단축, Roofline 전방/후방 참조
  - Introduction 수식 번호 제거, §4.2 전환 보강, 자기참조 수정
- 그림 크기 수정: Fig.2 resizebox, Fig.4 figure* 두컬럼, Fig.5 annotation axis cs 좌표
- 수정 파일: `docs/paper/main.tex`, `docs/paper/references.bib`, `docs/DETAIL_INFO_4_3_4_4.md`, `docs/paper/fig_script/*.tex`

## 2026-03-10: 6차 검증 결과 논문 수정 (15건)
- H100 989 TFLOP/s가 TF32임을 각주로 명시 (BF16 1,979와 구분, 결론 불변 설명)
- "fork" 기본 → spawn 자동 전환 가능성 명시
- Conclusion "proven" → "analyzed...to be validated empirically"로 톤다운
- Property 2 "CPU is first target" → capacity 전략 한정, 다른 전략 언급
- EMA 수식: per-request throughput 명시, N 변동 범위(0.02-0.10) 및 하향 조정 역할 설명
- Limitations: 모델 가중치 중복 로딩 (~70GB, KV cache 감소, startup 2배) 상세화
- Limitations: continuous batching 상호작용 미분석 사항 추가
- Abstract "zero interference" → "software-level interference + subject to HW contention" 톤다운
- batch-16 attention → BF16 경로 한정 명시 (FP32는 비배치 구현)
- Roofline Corollary에 KV cache 접근 비용 미포함 한계 명시
- TTFT 불일치에 SLO 관점 추가 (latency inconsistency, p99 TTFT 바운딩 어려움)
- Property 2 fault tolerance → "incidental, not designed" 명시
- Fig.2에 Q8_0 Decode 점(OI=2) 추가 (GPU/CPU 양쪽)
- AVX-512 커널 조건부 빌드: "subset selection" → AVX-512F 필수, VNNI 추가 조건 명시
- 에너지 분석: 실 TDP 60-70% 미달 + DRAM 전력 증가 언급
- BibTeX 검증: neo2025/apex2025/dovetail2024 저자 교체, zhao2024hetegen 제목, patel2024splitwise Best Paper Nominee
- cite key 완전 일치 확인 (0 mismatch)
- 수정 파일: `docs/paper/main.tex`, `docs/paper/references.bib`

## 2026-03-10: 5차 검증 결과 논문 수정 (코드 무수정, 논리/자료 기반)
- Fig.2 Prefill(CPU) 좌표 오류 수정: (128, 2.0) → (128, 7.2) (CPU compute ceiling에 일치)
- Xeon 8480+ peak FP32 수정: 3.6 → 7.2 TFLOP/s (2 AVX-512 FMA units, 56C × 2.0GHz × 64 ops)
- Ridge point 수정: 11.7 → 23 FLOP/byte (7.2/0.307)
- Q8_0/BF16 Roofline 불일치 해소: Corollary에 BF16(OI=1)과 Q8_0(OI=2) 모두 명시
- Fig.2 caption에 Q8_0 OI 추가
- "five kernels" → "up to five" + 조건부 빌드(ISA 가용성) 명시
- 350W 전력 명확화: 1소켓 TDP + 1소켓 idle로 분리 설명
- DGX H100 NUMA 토폴로지 caveat 추가: single-socket에서의 DDR5 contention 경고
- Decode-only 분석 한계 명시: end-to-end throughput은 prefill 포함 필요
- Fig.3 caption 개선: capacity 전략이 length/adaptive 단계를 건너뛰는 것을 명시
- Corollary의 throughput ratio에 양자화 형식 독립성 강조
- 수정 파일: `docs/paper/main.tex`

## 2026-02-27: §4 System Design에 HPC 기법 적용 Refinement (Draft v4)
- §4.1: Heterogeneous BSP 형식화 — Dual-Process를 Relaxed HBSP로 위치 지정 [Valiant, CACM 1990]
- §4.2: Roofline 기반 α 상한 Corollary 추가 (B_CPU/B_GPU = 9.2%, TP=8시 73% GPU-equivalent) [Williams et al., CACM 2009]
- §4.2: Corollary 1(GPU Latency Preservation)을 LogGP 모델로 강화 (T_route ≈ 11μs, < 0.04%) [Culler/Alexandrov]
- §4.3: 3가지 라우팅 전략의 HPC 스케줄링 계보 확립 (Work Stealing → HEFT → StarPU)
- §4.3: Algorithm 2 (Length-Aware Strategy) 의사코드 추가, HEFT 원리 연결
- §4.4: hwloc 토폴로지 계층(Machine→NUMANode→Package→L3Cache→Core→PU) 기반 자동 설정 서술
- references.bib에 8개 HPC 참고문헌 추가
- 수정 파일: `docs/paper/main.tex`, `docs/paper/references.bib`

## 2026-02-27: HPC 기법 문헌 조사 (CPU-GPU 이기종 추론 시스템 적용)
- 10개 HPC 주제 영역에 걸친 학술 문헌 및 기술 조사 수행
- Topic 1: Work Stealing & Task Scheduling (Cilk, TBB, StarPU, OmpSs, XKaapi, CoreTSAR)
- Topic 2: HPC Communication Patterns (BSP, HBSP, MPI Asynchronous Progress)
- Topic 3: Roofline Model (Williams et al., Heterogeneous Roofline for MI300A/GH200)
- Topic 4: NUMA-aware Scheduling (hwloc, libnuma, thread/memory affinity)
- Topic 5: Pipeline Parallelism & Overlap (computation-communication overlap, non-blocking MPI)
- Topic 6: Lock-free Data Structures (LCRQ, Michael-Scott Queue)
- Topic 7: Heterogeneous Load Balancing (HEFT, PEFT, E-HEFT)
- Topic 8: Memory Bandwidth Optimization (non-temporal stores, prefetch, Intel CAT cache partitioning)
- Topic 9: Batch Scheduling (SLURM backfill, bin packing)
- Topic 10: Performance Modeling (LogP, LogGP, HLogGP/mHLogGP, PRAM)
- 각 기법에 대해 핵심 개념, CPU-GPU LLM 서빙 매핑, 주요 인용문헌 제공

## 2026-02-26: IEEE 논문 Related Work 섹션 문헌 조사
- 6개 카테고리에 걸친 CPU-GPU 하이브리드 LLM 추론 관련 문헌 조사 수행
- Category 1: LLM Serving Systems (vLLM, FlexGen, DeepSpeed, Orca, AlpaServe, TGI, LightLLM)
- Category 2: Heterogeneous CPU-GPU Inference (PowerInfer, PowerInfer-2, LLM in a Flash, Dovetail, APEX, HGCA, HeteGen)
- Category 3: Speculative Decoding (Leviathan et al., Medusa, EAGLE, EAGLE-2)
- Category 4: Disaggregated/Distributed Serving (Mooncake, DistServe, Splitwise, TetriInfer)
- Category 5: Green AI / Energy-Efficient Computing (Patterson et al., Strubell et al., Wu et al.)
- Category 6: Memory Offloading (DeepSpeed-Inference, ZeRO-Inference, FlexGen)
- 각 논문에 대해 full citation, BibTeX entry, relevance summary 제공

## 2026-02-26: GPU 처리량 실제 프로파일링
- CapacityAwareRouter에 워밍업 프로파일링 페이즈 추가 (첫 N개 요청으로 실측 처리량 수집)
- 워밍업 완료 시 throughput-adaptive EMA 초기화 → 즉시 adaptive 슬롯 조정
- 주기적 통계 로깅 (N 요청마다 GPU/CPU 처리량, cpu_ratio, in_flight 출력)
- CLI 옵션 `--hybrid-warmup-requests`, `--hybrid-stats-log-interval` 추가
- 수정 파일: `config.py`, `arg_utils.py`, `hybrid_core.py`, `core_client.py`

## 2026-02-26: CPU 스케줄링 고도화 + KV Cache 인라인 프리페치
- CapacityAwareRouter에 3가지 라우팅 전략 추가: `capacity` (기존), `length-aware`, `throughput-adaptive`
- CLI 옵션 `--hybrid-routing-strategy`, `--hybrid-cpu-prefill-threshold` 추가
- EMA 처리량 기반 동적 CPU 슬롯 조정 (`throughput-adaptive` 전략)
- `batch_attention.cpp` 6개 블록 루프에 `_mm_prefetch` 인라인 프리페치 삽입 (K/V cache → L2)
- 수정 파일: `config.py`, `arg_utils.py`, `hybrid_core.py`, `core_client.py`, `batch_attention.cpp`

## 2026-02-24: 학술 논문 및 기술 참고문헌 조사
- vLLM Hybrid 프로젝트 관련 15개 주제에 대한 학술 논문 및 기술 참고문헌 웹 검색 수행
- 결과를 `docs/REFERENCES.md`에 정리

---

## v2 — 2026-04-11: Hybrid 동작 코드/로그 검증 세션

> append-only 정책 시작. 이전 세션 섹션은 보존, 신규 작업만 파일 말미에 버전 섹션으로 append.
> 이 세션은 코드 변경 없음 — 기존 구현의 동작 검증 + 문서화만 수행.

### 작업 개요
앞 세션에서 CUDA 13.0 마이그레이션 + hybrid 런타임 이슈들이 수정된 상태에서, 현재 구현이 실제로 설계 원칙대로 동작하는지 4가지 핵심 질문에 대해 코드 추적 + dev 실측 로그로 검증. 검증 결과는 `Tech_done.md v1` 에 저장.

### 구체적 작업
1. **TODO.md + `docs/paper/main.tex` 구조 분석**
   - 논문 1033 lines 전체 리뷰 (§1 Intro ~ §7 Conclusion)
   - 논문 ↔ 현재 코드 불일치 4건 식별:
     - Table 2 `max_seqs` auto rule: 논문 `max(4, ⌊cores/4⌋)` ↔ 실제 `1 per NUMA engine`
     - `num_cpu_engines = num_numa` auto 규칙 논문 누락
     - Algorithm 1 의 `N` 표기 모호 (실제 `N = 1 per engine × num_numa`)
     - §5 Implementation 에 `_C_utils` standalone extension 언급 누락
   - → TODO §2 "논문 ↔ 코드 재정합" 으로 유지

2. **돌고 있던 dev 서버 정상 종료**
   - PID 2508784 (`python -m vllm.entrypoints.openai.api_server`, Qwen2.5-7B hybrid, cpu-first/capacity) SIGTERM 종료
   - 자식 프로세스 (`VLLM::EngineCore` 2508919, `VLLM::CPU_EngineCore` 2508920, spawn worker 2509042) 모두 정상 종료 확인

3. **4대 핵심 질문 검증** — 상세 결론은 `Tech_done.md v1` 참조
   - Q1. 1 request 가 16 physical core 를 전부 쓰는가? → **YES** (C++ `init_cpu_threads_env` OMP 1:1 pin, 16 tid ↔ 16 core mapping 로그 확인)
   - Q2. CPU 가 의미있는 일을 하고 완료하는가? → **YES** (500 req burst 중 CPU 2 req 완료, decode call#3500+ path=ipex, slot 반납 cycle 확인. dev 환경에서 2.3 tok/s 로 느린 건 h/w 한계)
   - Q3. IPEX native kernel 로 Python 인터프리터 속도 극복 중인가? → **YES** (`ipex=3500, sdpa_loop=0, sdpa_batched=0`. attention 경로 100% IPEX oneDNN C++ kernel)
   - Q4. AVX-512/AMX/NUMA 매트릭스 무관 동작? → **YES** (3차원 독립 fallback chain. dev 의 AVX2+NUMA1 경로 실증, H100/Xeon 경로는 코드상 존재하나 실측 pending)

4. **작업 기록 파일 3종 운용 규칙 확립**
   - `TODO.md` / `Task_done.md` / `Tech_done.md` — 모두 append-only
   - 업데이트 타이밍: 사용자가 **"현재 작업 기록하고 commit & push"** 명령할 때만
   - 버전 섹션 (`## vN — YYYY-MM-DD`) 으로 파일 말미에 append, 이전 섹션 절대 수정·삭제 금지
   - 규칙을 `.claude/memory/feedback_work_log_files.md` 에 저장

### 새로 생성된 파일
- `Tech_done.md` (신규) — 기술 검증/분석 결론 단일 진실 공급원. v1 섹션에 4대 질문 검증 결론

### 수정된 파일
- `Task_done.md` — v2 섹션 append (본 섹션)
- `TODO.md` — v2 섹션 append (현재 상태 스냅샷)

### 코드 변경
**없음**. 이 세션은 순수 검증·문서화 세션이다. 기존 uncommitted 변경 (앞 세션의 `_C_utils`/CPUWorker/NUMA 자동 감지 등) 은 여전히 uncommitted 상태이고, TODO §0 에서 논리 그룹 commit 대기 중이다.

