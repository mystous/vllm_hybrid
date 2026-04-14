# vLLM Hybrid — 남은 작업

작업 이력: `Task_done.md` / 기술 검증 결론: `Tech_done.md` / 설계 단일 진실 공급원: `docs/paper/main.tex` / 프로젝트 구성: `CLAUDE.md`.

**본 파일 운용 규칙 (2026-04-11 변경)**: 기존의 append-only 버전 히스토리 방식에서 **"남은 작업만 담는 clean 파일"** 로 정책 변경. 이전 버전은 `old_doc/TODO_v4_20260411.md` 에 스냅샷으로 보존. 완료된 항목은 이 파일에서 제거하고 `Task_done.md` 에 append 한다.

**상태 요약 (2026-04-14 기준)**:
- dev 로직 검증 (v1 §1) **완결** — 순차 반복 / finish variety / scheduler 경계 / capacity 멈춤 dev 배제 전부 완료. abort slot leak 버그 발견 + 수정 + 검증.
- H100x4 KVM bring-up 및 1.5B/7B/32B baseline **완결** — hybrid routing regression 근본 원인 수정, 3개 모델에서 hybrid ≈ gpu_only 확인.
- stdout fast-path contention 이슈 **완결** — per-req/per-call 전부 silent, boot 만 emit.
- **H100x8 물리 2-NUMA 경로 실동작 증명 완결 (2026-04-14, v5)** — 4겹 버그 (arg_utils default / serve.sh -gt 1 / copy.replace / numa_node kwarg) 전원 수정 + server log 로 2 engine × NUMA 0/1 strict bind 확인 + wave=16 재앙 재현.
- **dev CPU thread 최적 16 확정 (2026-04-14)** — cpu_profile_dev.sh Section 6 vLLM sweep 기반.
- **서버 로그 캡처 인프라 완성 (2026-04-14)** — serve.sh tee + bench.sh byte offset slice + boot marker grep.
- **남은 큰 방향**: ninja gap 구현 (A-series architectural changes), 70B / long-context / rate-saturated workload 실측, 논문 정합성.

---

## 1. Ninja Gap 구현 — hybrid 가 실제로 gpu_only 를 이기게 만들기

현재 H100x4 + 1.5B/7B/32B 에서 hybrid ≈ gpu_only (±2% 노이즈). 원인은 **request-level partition** 구조 자체의 천장 — `T_hybrid = max(T_gpu, T_cpu)` 에서 CPU per-req 가 더 느리면 어떤 라우팅 전략으로도 gain 불가. 구조 변경이 필요.

상세 분석: `experiment_result/20260411_143500_h100x4_isa_verification_and_ninja_gap_strategy/README.md`

### 1.1 [A1] Speculative decode with CPU drafter ⭐⭐⭐⭐⭐ — **1순위**
- [ ] Implementation plan 문서화 (`docs/SPEC_DECODE_CPU_DRAFTER_PLAN.md`) — `HybridConfig` 확장 필드, `launch_hybrid_engines` third engine spawn, `_route_speculative` fanout, accept/reject 로직
- [ ] `HybridConfig.spec_decode_draft_model: str | None` 필드 추가
- [ ] Third engine 프로세스 spawn (CPU EngineCore 와 동일 패턴, ZMQ identity `b'\x02\x00'` 예정)
- [ ] `_route_speculative` 라우터 구현 (모든 요청 → GPU + draft 양쪽 fanout)
- [ ] `process_engine_outputs` 에서 GPU verify result + CPU draft tokens combine + accept/reject
- [ ] V0 `vllm.spec_decode` 참조하여 accept/reject 로직 차용
- [ ] 32B + Qwen2.5-0.5B draft 조합으로 측정 — 목표: **TPOT 41.82 ms → ~22~28 ms** (1.5~2×)
- [ ] Accept rate 측정 + 로깅 (`[HYBRID-SPEC-STATS] accept=N/M rate=0.xx`)
- [ ] 1.5B/7B 에서도 동일 fix 의 non-regression 확인

**왜 1순위**: 기존 dual-process 인프라 재사용, AMX/IPEX 검증 완료, 효과 정량 측정 가능, 본 라우팅 fix 와 직교, H100+1.5B/7B 같이 GPU 가 fast 한 환경에서도 작동.

### 1.2 [A4] AMX-INT8 dispatch path 활성화 ⭐⭐⭐ — A1 의 곱셈 인자
- [ ] `csrc/cpu/gemm_vnni.cpp` (VNNI INT8 6×16 micro-kernel) 를 실제 런타임에 호출하는 경로 연결
- [ ] `cpu_attn.py` 에 INT8 dispatch 분기 추가 (IPEX BF16 path 옆)
- [ ] `_C_cpu_ops` 의 INT8 GEMM 엔트리포인트를 Python 쪽에서 호출 가능하게
- [ ] Int8 quant 모델 (예: `Qwen2.5-0.5B-Instruct-W8A8`) 로 CPU per-req throughput 측정 — 목표 **2× 가속**
- [ ] A1 의 draft 모델을 INT8 로 돌려 draft throughput 2× 추가 확인

**왜 A1 과 동시**: 단독 효과 0 이지만 A1 draft 속도를 2× 가속하면 ninja gap 이 그만큼 더 커짐. 작은 PR 로 시작 가능.

### 1.3 [A2] KV cache CPU tier offload ⭐⭐⭐⭐ — 70B 데모 이후
- [ ] PagedAttention `block_table` 에 tier 필드 추가 (hot=HBM / cold=DRAM)
- [ ] Eviction policy 구현 (LRU / recency-based, 설정 가능)
- [ ] DMA path (`cudaMemcpyAsync` + pinned host memory, `torch.cuda.Stream` 분리)
- [ ] Attention kernel 의 cold block swap-in trigger + prefetch hint
- [ ] Demo workload: 70B + batch 1500+ 에서 GPU saturated 확인
- [ ] 측정 목표: 동시 시퀀스 ~3×, total throughput **2~3×**

**전제**: 70B baseline (§3.1) 이 있어야 KV 한계가 실제로 드러남.

### 1.4 [A3] Long-context Prefill/Decode disaggregation ⭐⭐⭐⭐ — 32K+ workload 필요
- [ ] `vllm/engine/disaggregated/` stub 구조 파악 + hybrid process isolation 과 조합 설계
- [ ] Prefill 전용 CPU EngineCore (AMX BF16 prefill 특화, decode 안 함)
- [ ] Decode 전용 GPU EngineCore (기존 구조)
- [ ] Request 가 prefill 단계는 CPU, decode 단계는 GPU 로 넘어가는 hand-off 메커니즘
- [ ] KV cache transfer path (CPU → GPU via DMA)
- [ ] Demo workload: 32K+ input length, 100 reqs, GPU TPOT p99 개선 측정

---

## 2. H100 확장 실험 — 더 큰 모델 / 더 큰 workload

### 2.1 70B baseline
- [ ] `h100x4_Llama-3.3-70B-Instruct_hybrid.env` 생성 (production config 기준)
- [ ] 70B gpu_only + hybrid 500 req 128/128 측정
- [ ] HBM 압력 관찰: weight 140 GB / TP=4 → 35 GB per GPU. KV cache 한계 batch 예측
- [ ] 부팅 시간 측정 (weight 로드 + CUDA graph capture ~ 수 분 예상)
- [ ] SERVER_READY_TIMEOUT 조정 (3600s ?)
- [ ] GPU mean util 이 32B 의 43% 에서 얼마나 상승하는지 관찰
- [ ] KV offload (A2) demo 후보로 확정 가능한지 판단

### 2.2 Long-context (32B + 16K/32K input)
- [ ] 환경 파일: `h100x4_qwen32b_hybrid_longctx.env` (INPUT_LEN=16384, OUTPUT_LEN=512)
- [ ] GPU TPOT p99 측정: decode 가 long prefill 동안 stall 되는 증상 재현
- [ ] A3 P/D disaggregation 효과의 기준선 확정
- [ ] 64K 시도 (KV 한계 워크로드)

### 2.3 Routing strategy 비교 실험 (paper §3 Exp 3)
- [ ] 동일 모델 (32B 권장) / 동일 shape 에서 4 전략 비교: `capacity` / `round-robin` / `length-aware` / `throughput-adaptive`
- [ ] 주의: random 고정 길이 dataset 에선 length-aware / throughput-adaptive prefill_threshold 의 효과 0 → **ShareGPT 길이 분포 dataset 사용**
- [ ] 각 전략에서 CPU/GPU 분배 비율, TPOT, P99 TTFT 측정
- [ ] 본 fix 이후 throughput-adaptive 는 항상 GPU 로 수렴할 것 → 전략 비교 의미가 흐려질 수 있으므로 GPU가 saturated 되는 설정 (큰 모델 + 긴 input) 필수

### 2.4 Ablation — NUMA binding / IPEX / auto config
- [ ] `HYBRID_NUMA_AWARE=false` vs `true` (H100 KVM 은 1 NUMA 이라 차이 적음)
- [ ] IPEX 비활성화 경로 (`_PagedAttention` + `sdpa_batched`) fallback throughput 측정
- [ ] `cpu_max_num_seqs=1` (원칙) vs 수동 override 2~4 의 throughput 차이 (BW reuse 효과 측정)
- [ ] 본 ablation 은 CPU 에 실제 요청이 갈 때만 의미 있음 → spec decode (A1) 구현 이후 수행

### 2.5 에너지 효율 (paper §6 Exp 6)
- [ ] Intel RAPL counters 활용 (`/sys/class/powercap/intel-rapl/`)
- [ ] `eval/monitor.py` 에 RAPL 주기 샘플링 추가
- [ ] gpu_only vs hybrid 의 perf/watt 비교 (H100 + Xeon 의 power 모니터링)
- [ ] spec decode (A1) 구현 후 재측정 — draft 가 GPU TPOT 을 줄이므로 per-token 에너지 개선 예상

### 2.6 H100x8 + Xeon 2-socket 환경 — **2-NUMA 경로 검증 완결 (v5)**
- [x] 2-NUMA auto 감지: `num_cpu_engines=2` resolve 확인 → **Tech_done v5 F1**
- [x] 각 CPU engine 이 자기 NUMA 의 코어에 1:1 pin (`HYBRID-CPU-WORKER` 로그) → **확인**
- [x] `_get_autobind_cpu_ids` 의 `numa_bind_node` 우선 경로 실측 → **확인**
- [x] 7B 기본 bench (500×128/128) → `eval/basic/H100x8/` 4 runs 보존
- [ ] **1.5B / 32B H100x8 bench 는 미수행** — 필요 시 추가
- [ ] **wave=16 재앙이 2-NUMA 상태에서도 재현** (Tech_done v5 F2) → max_seqs=1 가 고정 답, max_seqs 실험은 종료

### 2.7 GPU 포화 workload 탐색 — hybrid 이득 실제 검증
현재 모든 실측에서 `T_hybrid ≥ T_gpu_only` — GPU 가 항상 여유 있어 CPU 경로는 overhead. 이득을 보려면 GPU 가 먼저 saturate 되는 조건이 필요.
- [ ] **70B TP=8 H100x8** — weight 140GB → GPU HBM 압박, batch slot 감소
- [ ] **long-context 16K+ input** — GPU prefill bottleneck, CPU prefill 분담 효과 측정 (§1.4 A3 으로 이관)
- [ ] **rate-limited burst 2000+ req** — GPU queue 가 saturated 되는 조건 재현
- [ ] 이 3 조건 중 어느 것도 충족 안 되면 request-level hybrid 의 이득은 구조적 불가 (§1.1 A1 spec decode 필요)

---

## 3. 라우팅/코드 품질 — 별도 정리 PR

### 3.1 Bug 1 — `_update_adaptive_slots` 상수 2 고정
- [ ] `hybrid_core.py:436-443` 의 `_update_adaptive_slots` 정리 — `cpu_max_num_seqs=1` 에 대해 `new_max = max(2, min(2, 1)) = 2` 로 항상 고정되는 dead code
- [ ] Property 2 expected-finish gate 이후 이 함수의 라우팅 영향은 0 이지만 코드 자체가 잘못됨
- [ ] 원칙대로 "CPU slot 수는 고정 1 per engine × num_cpu_engines" 을 유지하고 `adaptive_cpu_max_seqs` 필드 자체를 제거 또는 의미 재정의

### 3.2 OMP binding defaults
- [ ] `_setup_cpu_process_env` 에 다음 추가:
  ```python
  os.environ.setdefault("OMP_PROC_BIND", "close")
  os.environ.setdefault("OMP_PLACES", "cores")
  ```
- [ ] 현재 C++ `init_cpu_threads_env` 가 `sched_setaffinity` 로 1:1 pin 하므로 동작 무결하지만, 향후 large CPU workload (spec decode drafter, KV offload) 시나리오에 대비 OpenMP hint 도 같이 주는 게 안전

### 3.3 Router instrumentation 정리
- [ ] `[HYBRID-ROUTER-INIT]`, `[HYBRID-ROUTER-DISPATCH]`, `[HYBRID-ROUTER-STATS]` 마커가 현재 `hybrid_core.py` 에 영구 보존됨
- [ ] silent 정책 기준으로 DISPATCH 는 debug (대다수 요청마다), STATS 는 info 유지 (periodic), INIT 는 info (boot)
- [ ] 현재 코드 상태 재확인 필요

---

## 4. 논문 ↔ 코드 재정합 (`docs/paper/main.tex`)

현재 논문과 실제 코드가 어긋나는 항목 5건. H100 bring-up 과정에서 Property 2 구현 세부가 추가되어 이 항목이 가장 큼.

### 4.1 Property 2 expected-finish 정량식 본문 추가
- [ ] 현재 paper §3 은 "CPU is complement" 라고만 적혀 있고 정량 식 없음
- [ ] 실제 구현 (`_route_throughput_adaptive`) 의 공식 본문화:
  ```
  cpu_finish = (cpu_in_flight + 1) · (L_out / tput_cpu_ema)
  gpu_finish = ceil((gpu_in_flight + 1) / gpu_max_seqs) · (L_out / tput_gpu_ema)
  route_to_cpu  iff  cpu_finish ≤ gpu_finish
  ```
- [ ] cold start gate (`gpu_ema == 0 → GPU`) 정당화 — probe blind 회피
- [ ] cpu-first / gpu-first 의 semantics: "동률 시 누가 우선" 으로 축소된 것 명시

### 4.2 Table 2 `max_seqs` auto rule 수정
- [ ] 논문 현재: `max(4, ⌊cores / 4⌋)` + "4 threads/sequence" rationale
- [ ] 실제: **`1` per NUMA engine** + "1 sequence saturates whole NUMA node via OMP" rationale
- [ ] Table 2 + §3.4 "Maximum concurrent sequences" 단락 재작성

### 4.3 `num_cpu_engines = num_numa` auto rule 추가
- [ ] 현재: CLI 옵션으로만 언급, auto 감지 설명 없음
- [ ] 실제: `_resolve_num_cpu_engines` 가 `NUMAAllocator.num_nodes` 로 자동 결정
- [ ] §3.4 Table 2 에 auto rule 행 추가
- [ ] Figure 4 (hwloc topology) 캡션 업데이트

### 4.4 §3.3 Algorithm 1 의 `N` 표기 명확화
- [ ] 현재: `N = cpu_max_num_seqs` 로 추상화
- [ ] 설계 원칙: `N = 1 per engine × num_numa` 임을 명시

### 4.5 §5 Implementation 에 `_C_utils` standalone extension 언급
- [ ] 현재: `_C_cpu_ops` 만 언급
- [ ] 추가: `_C_utils` 는 `init_cpu_threads_env` 전용, AVX-512/AMX 무관, CUDA/ROCm 빌드에서도 항상 빌드

---

## 5. Dev 환경 보조 검증

### 5.1 Dev RTX 3090 + 1.5B/7B 에서 routing fix 재측정
- [ ] 본 `_route_throughput_adaptive` fix 가 dev 환경의 "weak GPU + 상대적으로 비슷한 CPU" 영역을 망치지 않는지 확인
- [ ] dev 1.5B: 이전 hybrid (capacity + cpu-first) 가 wall 34.9s 였음. 본 fix 후 thro-adaptive 로 돌리면 모든 요청 GPU 로 수렴하여 wall ≈ 14s (gpu_only) 로 감소 예상
- [ ] 만약 dev 에서도 "CPU 가 실제로 도움이 되는 영역" 이 발견되면 fix 의 cold-start gate 나 EMA 초기값 조정 필요 여부 판단

### 5.2 dev 의 60 req 순차 / finish variety test 재실행 (fix 포함 상태)
- [ ] `/tmp/seq_repeat_test.py`, `/tmp/finish_variety_test.py`, `/tmp/cpu_abort_test.py` 를 routing fix 적용 상태에서 재실행하여 regression 없음 확인
- [ ] 특히 abort slot leak 재현 안 되는지 재확인

---

## 6. 기타 관찰된 잠재 이슈 (우선순위 낮음)

- [ ] **`set_num_interop_threads` 타이밍**: 첫 op 실행 후에는 `RuntimeError`. 현재 `try/except RuntimeError` 로 감쌌지만 H100 실측에서 interop thread 가 원하는 값으로 설정되는지 로그 확인
- [ ] **`numa_migrate_pages` × 2TB 지연**: 부팅 지연만 영향, 런타임 성능 무관. H100x8 2-socket 환경 접근 시 측정
- [ ] **70B weight 중복 로딩 실측**: CPU engine 은 별도 프로세스이므로 weight 독립 로드. 70B 환경에서 부팅 가능성 / startup time 측정 (A2 KV offload 전제조건)
- [ ] **`post-init: cpu_affinity=1 cores [1]` 문서화**: `Tech_done v1 Q1` 에서 이미 상세 설명됨. `CLAUDE.md` 진단 섹션에 1줄 주석 추가 (혼란 방지)
- [ ] **CPU prefill 직렬화 (chunked_prefill=False) 영향 재점검** (v5 신규): max_seqs=1 하에서는 비문제이지만 batch 크기 변동 또는 다중 workload 도입 시 TTFT P99 급증 가능. 현재 `_create_cpu_vllm_config` 가 강제 `False` 인데, chunked prefill 활성화 시 per-step 이 어떻게 변하는지 실험적 확인
- [ ] **Python 3.12/3.13 API 호환 정책** (v5 신규): `copy.replace` 와 같은 3.13+ 전용 API 피하기. CI 에서 3.12 로 import 테스트 추가 고려

---

## 7. 문서화 잔여

- [ ] `docs/CUDA13_MIGRATION_STATUS.md` 에 "H100x4 실측 결과" 섹션 추가 (1.5B/7B/32B baseline + 라우팅 fix 언급)
- [ ] `docs/HYBRID_OPTIONS_IMPLEMENTATION_PLAN.md` 가 현재 코드 + 본 세션 fix 와 맞는지 재확인
- [ ] `CLAUDE.md` 의 "진단 로그 marker 7종" 표에 `[HYBRID-ROUTER-INIT]`, `[HYBRID-ROUTER-DISPATCH]`, `[HYBRID-ROUTER-STATS]`, `[HYBRID-WAVE-DISPATCH]`, `[HYBRID-CPU-PROFILE]`, `[HYBRID-CPU-ATTN-IPEX]` 추가 (v5 PROFILE logging 반영)
- [ ] `CLAUDE.md` 에 서버 로그 캡처 메커니즘 (serve.sh tee + bench.sh slice/grep) 섹션 추가
- [ ] 본 TODO.md 재작성 + old_doc 백업 규칙을 `CLAUDE.md` 의 "3 파일 운용 규칙" 섹션에 반영
- [ ] `eval/basic/H100x8/README.md` — 4 runs (gpu_only + max_seqs=1/16, threads=32/56) 비교 요약 (Tech_done v5 F2 기반)
- [ ] `eval/basic/RTX3090/README.md` — 6 runs 요약 (1.5B/7B × gpu_only/hybrid × max_seqs=4)

---

## 8. 다음 작업 세션 시작 시 체크

- [ ] 이 파일의 stale 여부 재확인 — `git log --since=<last_modified>` 로 commit 이력 + `experiment_result/` 최신 추가분 교차 확인
- [ ] Stale 항목은 완료 표시 후 `Task_done.md` 에 append 하고 본 파일에서 제거
- [ ] 새로 발견된 작업은 본 파일의 적절한 섹션에 추가
