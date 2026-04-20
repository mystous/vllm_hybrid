# vLLM Hybrid — Ninja Gap 성능 향상 작업 (v6, 2026-04-15)

작업 이력: `Task_done.md` / 기술 검증 결론: `Tech_done.md` / 설계 단일 진실 공급원: `docs/paper/main.tex` / 프로젝트 구성: `CLAUDE.md`.

**운용 규칙**: "남은 성능 향상 작업만" 유지. 이전 버전 `old_doc/TODO_v5_20260415.md` 에 스냅샷 보존. 완료는 `Task_done.md` 에 append.

**v6 변경**: Ninja Gap 성능 향상 항목만 남김. 착수 시점·기간 추정 전부 제거. 기법별 메커니즘 / 예상 이득 / 구현 상태 / 스택 호환성 / 성공 조건 만 유지.

---

## 핵심 메시지

> 현재 H100x8 에서 hybrid wall 은 GPU-only 대비 **26-143×** 느리다 (7B, 500×128/128). 실패 원인은 "CPU 가 느리다" 가 아니라 **`num_seqs` 증가에도 per-req cost 가 안 내려감 — batch scaling 실패**. `cpu_max_num_seqs` 확대는 throughput 이 아닌 **tail amplification**.
>
> **순서가 전부다**. 계측(G0) → batch scaling 커널(G1–G2) → big wins(G3) → routing 재활성. 이 순서를 건너뛰면 어떤 기법도 tail 만 만든다.

---

## 하지 말아야 할 것 (Guardrails)

1. batch scaling 확인 전 `cpu_max_num_seqs` 확대
2. `wave-batch` 를 기본 전략으로 유지
3. NUMA/pinning bring-up 재증명 (이미 실측 검증 완료)
4. 이미 구현된 기능 (chunked prefill off, NUMA membind, ISA 감지) 을 **신규 gain 항목으로 재계산**
5. VNNI/AMX 토대 존재만으로 "INT8/AMX 사용 중" 판단 — **hot path 연결 여부는 별개**
6. CPU handled request 수만 늘어도 wall 악화 시 성공으로 판단
7. 외부 논문 speedup 수치를 우리 코드에 직접 곱하기

---

## 1. 성공 판정 4축

Stage 종료 시마다 **동시 확인**:

| 축 | 측정 | 방향 |
|---|---|---|
| CPU scaling | `cost(batch=N) / cost(batch=1)` | N 보다 훨씬 작아야 |
| CPU throughput | CPU-only tok/s, req/s | batch↑ 에 따라 ↑ |
| Tail | GPU bulk 완료 후 CPU drain | ↓ |
| Wall ratio | `hybrid wall / gpu_only wall` | ↓ |
| CPU contribution | handled req | ↑ |

**4축 동시 개선 아니면 성공 아님**.

### Stop/Go Rules

| Case | 관측 | 판정 | 조치 |
|---|---|---|---|
| 1 | CPU handled↑ but wall↓ | 실패 | `cpu_max_num_seqs` 확대 중단, kernel 단계 복귀 |
| 2 | CPU tok/s↑ but tail 그대로 | 부분 실패 | routing/gate/prefill 재검토 |
| 3 | 단일 req 만 빨라짐, batch scaling 없음 | 실패 | single-req 최적화로만 분류 |
| 4 | Kernel 수정 후 metric 변화 없음 | hot path 미타격 | marker 로 실제 호출 확인, 다음 kernel 금지 |

---

## 2. Gate 정의

| Gate | 통과 조건 | 실패 시 |
|---|---|---|
| G0 | seq=1/2/4/8/16 CPU-only scaling + sublayer breakdown 확보 | 계측 보강 |
| G1 | 4req cost ≤ 2× single / tail < 100s / wall ratio < 8× | hot path 미타격 |
| G2 | 4req cost ≤ 1.5× / tail < 10s / wall ratio < 1.5× | routing/gate 재설계 |
| G3 | CPU req↑ + tail 제거 + wall ≤ GPU-only | **Ninja Gap 달성** |

Gate 숫자는 방향성. G0 에서 기준선 재측정으로 조정.

---

## 3. G0 — 계측 재정의 (모든 후속 기법의 전제) ✅ 완료 (2026-04-17, 22afea529)

- [x] `eval/cpu_profile*.sh` 에 `num_seqs=1/2/4/8/16` sweep 고정
- [x] CPU-only 와 hybrid CPU engine 동일 shape 비교 harness
- [x] `cpu_worker.py` attn/mlp coarse hook → QKV/O/Gate/Up/SiLU/Down/Norm 세분화
- [x] per-step barrier/sync time, memory wait, packing/repacking marker
- [x] H100x8 + dev (RTX3090) 결과 동일 CSV schema 로 저장
- [x] 산출물: `batch_scaling_ratio`, `per_req_cost`, sublayer top bottleneck, `num_seqs` 증가 시 폭증 sublayer

---

## 4. 경로 1 — CPU 자체 가속 스택

구현 상태 태그: ✅ 이미 구현 / 🔶 부분 구현 / ⭕ 미구현

### 4.1 [Tier 0] 기준선 방어 ✅ 완료 (2026-04-17, 22afea529)
- [x] 기본 실험 `cpu_max_num_seqs=1` 고정
- [x] `wave-batch` 는 비교 대상으로만 유지
- [x] `throughput-adaptive` vs strict continuous baseline 동일 workload 비교

### 4.2 Huge Pages (2MB THP → 1GB hugetlb) ⭕
- **메커니즘**:
  1. `4KB → 2MB` THP 로 TLB miss / page walk 를 먼저 줄인다
  2. 그 다음 `2MB → 1GB` hugetlb 로 large resident weight / KV 구간을 더 크게 묶는다
- **예상 이득**:
  - **Phase 1 (2MB THP)**: 3–10% (7B), 큰 모델일수록 상한↑
  - **Phase 2 (1GB hugetlb 추가)**: 2MB 대비 추가 3–10%
- **변경 위치**:
  - **Phase 1**: 호스트 THP 설정 (`transparent_hugepage=always`), 코드 수정 없음
  - **Phase 2**: grub/runtime hugepage reservation + vLLM explicit `MAP_HUGETLB`
- **위험**:
  - 2MB 는 비교적 낮음
  - 1GB 는 메모리 단편화, NUMA별 reserve, 컨테이너/cgroup, 운영 승인 리스크 큼
- **주의**: 여기서 말하는 2MB/1GB 는 **캐시 크기**가 아니라 **페이지 크기**다. 표준적으로 의미 있는 중간 단계는 2MB THP 이며, 1GB 는 그 다음 explicit hugetlb 단계다.
- **스택 호환성**: 모든 후속 기법과 독립. 다만 **실험 순서는 반드시 2MB 먼저**, 1GB 는 추가 이득이 확인될 때만 진행

### 4.3 ~~IPEX WoQ INT8~~ ✗ **기각** (2026-04-19)
- **기각 사유**: IPEX `ipex.llm.optimize` + `WoqWeightDtype` 경로가 vLLM 의 `QKVParallelLinear` custom structure 와 비호환 (IPEX `replace_module` 이 vLLM Linear 클래스를 인식 못 함)
- **대체 경로**: §23 CPU Native Quantization (llama.cpp Q8_0/Q4_K 자체 dispatch) — 이득 2× decode 동일, vLLM Linear 에 직접 torch custom op dispatch 로 구조 호환 확보
- **Post-mortem 문서**: `NinjaGap_Todo/04_ipex_woq_int8.md`

### 4.4 OMP env 마무리 ✅ **완료** (2026-04-15)
- **현재**: `csrc/cpu/utils.cpp` 3종 + `_setup_cpu_process_env` 에서 `HYBRID_KMP_BLOCKTIME=auto` (기본) 시 `KMP_BLOCKTIME=0` 강제
- **주의**: `OMP_PROC_BIND=close` 는 **의도적 미설정** (Intel OMP master-thread pin bug → `hybrid_core.py` 에서 pop)

### 4.5 Hot Path 연결 증명 (G1 진입 필수) 🔶 Dispatch 완료 (2026-04-19, `6f904b39b`), kernel 미완 → §06-1 로 후속

> **정정 공지 (2026-04-20)**: 이 섹션은 최초 "✅ 완료" 로 표기됐으나, 2026-04-20 TP=8 baseline 대조 측정 (`g0_00_qwen2.5_32b_base`) 에서 §06 이 seqs≥2 부터 outTP 역효과 (seqs=64 에서 −90%) 임이 확인됐다. 원인은 `quant_q8_0.cpp::q8_0_linear_impl` 이 M>1 에서 GEMV 를 M 번 순차 반복하는 batch-oblivious 구현. kernel 수정은 [§06-1 M-aware MLP kernel](NinjaGap_Todo/06-1_m_aware_mlp_kernel.md) 로 분리, §06 scope 는 "Q8_0 dispatch 경로 구축" 까지로 한정. G1 gate 판정은 §06-1 완료 후 재실행.

- [x] Q8_0 kernel + torch op 을 Qwen2.5 MLP hot path 에 연결 (`vllm/v1/worker/hot_path_wiring.py`)
- [x] load-time Q8_0 변환 hook (`cpu_model_runner.py::load_model`, LoRA 이후)
- [x] shape 별 dispatch log marker (`VLLM_HYBRID_KERNEL_TRACE=1`)
- [x] H100x8 32B sweep 측정 (`measurement_results/H100x8/g0_06_qwen2.5_32b/seqs{1,2,4,8,16,32,64}` + `gpu_only_baseline`)
- [x] 구조 일관성 재작업: `--hybrid-vnni-hot-path` CLI arg + `HybridConfig.vnni_hot_path` + `_create_cpu_vllm_config` passthrough (세 단계의 버그 전부 fix)
- [ ] **§06-1 (kernel M-aware 화)** — `q8_0_linear_impl` 의 M>1 경로를 VNNI INT8 GEMM 으로 교체. 별도 문서 참조

**실측 결과** (500 req × 128/128, TP=8):
- **seqs=1**: §06 단독 이득 확인 — duration 80.0→57.6 s (−28%), outTP 908.9→1069.7 tok/s (+18%)
- **seqs≥2 역효과 (2026-04-20 baseline 대조)**: outTP seqs=2 −27%, seqs=16 −81%, seqs=64 −90%. kernel 결함
- **per_req_cost(4)/per_req_cost(1) = 2.89** (§06 on), baseline (§06 off) 은 1.53 으로 이미 G1 통과 → §06 이 batch 영역에서 baseline 악화시킴

**G1 gate**: §06 단독은 kernel 결함으로 미통과. §06-1 완료 후 재판정. §11/§25/§24/§18 단계는 §06-1 결과 본 뒤 우선순위 재조정.

상세 분석 및 PNG: `measurement_results/H100x8/g0_06_qwen2.5_32b/analysis_g0.ipynb` + `NinjaGap_Todo/06_hot_path_wiring.md`.

### 4.5-1 Q8_0 Kernel M-aware 화 (§06-1, G1 재판정 전제)
- **문서**: [NinjaGap_Todo/06-1_m_aware_mlp_kernel.md](NinjaGap_Todo/06-1_m_aware_mlp_kernel.md)
- **원인**: §06 kernel (`quant_q8_0.cpp::q8_0_linear_impl`) 이 M>1 에서 GEMV 를 M 번 순차 반복. batch-oblivious
- **해결**: M>1 경로를 VNNI INT8 GEMM (기존 `gemm_vnni.cpp::int8_gemm_vnni` 재활용) 으로 교체. M=1 GEMV 유지
- **선택 Phase 2**: SPR AMX-INT8 tile op 추가 — compute-bound 구간 커버. (A) 측정 후 필요시 진행
- **성공 조건**: seqs 4/8/16 에서 baseline outTP 이상. seqs=1 기존 이득 (+18%) 유지. token identical (또는 fallback 허용)
- **상태**: ⭕ 설계 완료, 구현 전

### 4.6 ISA Binary Dispatch 🔶
- **현재**: `cpu_attn.py` decode 경로에 `custom_avx → ipex → sdpa_batched → sdpa_loop` fallback chain. batch size 기반 명시적 dispatch 없음 (IPEX 내부 dispatcher 의존)
- **메커니즘**: batch>4 → AMX, else → AVX-512 VNNI (KTransformers 방식)
- **예상 이득**: decode 1.5–2.22× (KTransformers 실측)
- **변경 위치**: `cpu_worker.execute_model` pre-dispatch + csrc kernel 등록
- **위험**: IPEX 내부 dispatcher 와 충돌. Bypass 필요
- **스택 호환성**: §4.11 cascade 전제

### 4.7 Kernel Fusion 🔶
- **현재**: GPU 경로에 `gate_up_proj`, `qkv_proj` merged linear. CPU 전용 fused kernel 없음
- **메커니즘**: sublayer 8개 독립 kernel → 4개 묶음. 중간 DDR write 제거, 입력 x 단일 로드
- **예상 이득**: 1.5–2× (SGLang SiLU+up 12% × 4 sublayer 누적)
- **변경 위치**: `csrc/cpu/fused_qkv.cpp`, `fused_gate_up_silu_down.cpp`, `fused_add_rmsnorm.cpp`
- **위험**: IPEX 기존 fused kernel 과 충돌 검증 필요
- **주의**: 이미 fused 된 모델 구조인지 먼저 확인 후 착수
- **스택 호환성**: LUT / cascade 와 독립, 항상 병행

### 4.8 Softmax + SiLU LUT 대체 ⭕
- **메커니즘**: `exp()` 20 cycles → `vpshufb` LUT 1 cycle. SiLU "hot range" 선형 근사 + LUT
- **예상 이득**: Softmax 2.2×, SiLU 1.2× (TARDIS vLLM 1.6×)
- **변경 위치**: `csrc/cpu/lut_ops.cpp`. 32B/512B LUT register 상주
- **위험**: 근사 정확도 열화 (<2% 추정, 측정 필요)
- **조건**: scalar transcendental 이 프로파일 top bottleneck 일 때만
- **스택 호환성**: §4.12 GEMV LUT 와 동일 인프라, 같이 개발

### 4.9 Head Folding (GEMV → GEMM) ⭕
- **메커니즘**: decode M=1 GEMV → batch fold M=16 GEMM. AMX tile full 활용
- **예상 이득**: decode attention 1.5–2× (SGLang blog)
- **변경 위치**: `csrc/cpu/fold_attention.cpp` + IPEX `single_query_cached_kv_attention` 대체
- **위험**: MLA 에서는 직접 적용. GQA (Qwen) 에서는 batch fold 변형 필요
- **스택 호환성**: §4.13 batch-aware attention 의 한 방식

### 4.10 Batch-aware Decode Attention ✗ **Phase 1 기각 (2026-04-20)**
- **현재**: Phase 1 (Option A, IPEX 우회 + 기존 `batch16_paged_attention_v1` dispatch 활성) 측정 완료 → §06-1 v1 대비 −12~−5% regression. 데이터: `measurement_results/H100x8/g0_11_qwen2.5_32b_phase1(fail)/`
- **실패 원인**: (1) 측정 구간 (seqs 2/4/8) 이 kernel 의 remainder path — IPEX 와 구조적 동일, (2) prefill IPEX → SDPA fallback 오버헤드, (3) CPU batch 자체의 근본 결함은 attention 로 해결되지 않음
- **Phase 2 (v2 신규 kernel)**: 재시도 여부 보류. Tier 1 후보 (§13/§16/§22/§28) 검토 이후 재평가
- **스택 호환성**: §4.9 Head Folding 과 중복 영역

### 4.11 Barrier/Sync 감소 ⭕ (미계측)
- **메커니즘**: OMP parallel region 이 sublayer 마다 재진입하는지 확인. thread team 재사용 / chunk scheduling / layer·block 단위 persistent region
- **예상 이득**: 프로파일 의존 (G0 산출물 기반)
- **스택 호환성**: 독립

### 4.12 T-MAC LUT GEMV INT4 ⭕
- **메커니즘**: INT4 weight 16 값 × input → LUT 32B precompute. 곱셈 + 역양자화 `vpshufb` 1-cycle
- **예상 이득**: 4× (T-MAC 실측, CPU 22 tok/s > NPU 10.4 tok/s)
- **변경 위치**: `csrc/cpu/lut_gemv.cpp` 전용 kernel. IPEX bypass
- **위험**: ⚠ T-MAC 은 edge CPU 검증 (Snapdragon). **SPR+AMX 재검증 필수**. 강한 가설
- **스택 호환성**: §4.3 WoQ INT8 기각 후 §23 CPU Native Quant 대체. §4.8 LUT Softmax 와 동일 인프라

### 4.13 AVX/AMX Cascade Pipeline ⭕
- **메커니즘**: tile k+2 load (prefetch/DSA) / k+1 dequant·pack (AVX-512) / k matmul (AMX) 3-stage 동시 실행
- **예상 이득**: 1.5–3× (T-MAN NPU 3.1× decode, CPU 이식 보수적 1.5–2×)
- **변경 위치**: 타일 버퍼 설계 + cache-fit 검증
- **위험**: ⚠ Staging overhead 가 이득 상쇄 가능. AVX `zmm` ↔ AMX tile 별도라 **중간 버퍼 L2 상주 필수**. cache-fit 실패 시 pipeline 이 아니라 DDR 왕복 증가
- **구체 항목**:
  - tile size 별 buffer footprint 계산
  - L2 fit staging layout 설계
  - `batch=1` → AVX, `batch>=N` → AMX/cascade shape-aware dispatch
  - 전환 비용과 tile config 비용 profile marker 로 분리
- **스택 호환성**: §4.6 binary dispatch 의 발전형. §4.14 pre-pack 과 조합 필수

### 4.14 AMX Weight Pre-pack 🔶 (IPEX 내부 자동, 독자 없음)
- **현재**: IPEX `ipex.llm.optimize(weights_prepack=True)` 기본값 암묵 활성. KTransformers 스타일 독자 pre-pack (AMX tile layout 직접 제어) 없음
- **메커니즘**: 모델 로드 시 1회 weight → AMX tile layout (16×64 byte) 재배치. tileloadd 가 연속 16 cache line 로드
- **예상 이득**: 1.1–1.2× (KTransformers 10–20%)
- **변경 위치**: CPUWorker `load_model` 후 hook
- **위험**: 낮음. 메모리 부담 2× (원본 + 재배치)
- **스택 호환성**: §4.13 cascade 전제. §4.12 LUT path 에도 유사 pre-pack 필요 (T-MAC group layout)

### 4.15 AVX-512 Bitmask Sparse (SparAMX) 🔶
- **현재**: `csrc/cpu/gemm_vnni.cpp` 의 `int8_gemm_vnni` (dense). sparse 경로 없음
- **메커니즘**: Unstructured sparsity `K` 레지스터 64-bit mask. `_mm512_mask_fmadd_ps` 로 유효 원소만 계산
- **예상 이득**: linear 1.42×, attention 1.14× (SparAMX 실측, Xeon SPR)
- **변경 위치**: `csrc/cpu/sparse_amx.cpp`. 가중치 50% pruning 선행 필요
- **위험**: 프루닝 후 PPL 열화 검증. MLP 희소성은 batch↑ 시 소멸 — **attention head sparsity 만 batch-invariant**
- **스택 호환성**: LUT 과 별개 경로, 대체가 아닌 추가

### 4.16 Core Group Pipeline (Systolic) ⭕
- **메커니즘**: 56 core 를 4 group 분할. A:QKV / B:Attn / C:MLP / D:next layer QKV 파이프라인. L3 로 inter-group 전달
- **예상 이득**: 2–3× latency (4 layer 동시 실행). GPU SM cluster 원리를 CPU L3 로
- **위험**: 매우 높음. L3 BW 가 DDR 보다 높지만 coherence 비용 큼
- **전제**: §4.5~§4.15 완료로 기반 kernel 이 fast 해야 이득

---

## 5. 경로 2 — 역할 재정의 (구조 변경)

### 5.1 Spec Decode CPU Drafter 🔶 (GPU-only spec decode 프레임워크만)
- **현재**: `vllm/v1/spec_decode/` 에 ngram/eagle/medusa (GPU-on-GPU). CPU drafter + GPU verifier DuoDecoding 스타일 미구현
- **메커니즘**: CPU drafter (Qwen2.5-0.5B) k 토큰 생성 → GPU verifier (7B) 한 번에 검증. accept rate ~70% 면 k-1 토큰 free
- **예상 이득**: TPOT 2.1–2.61× (DuoDecoding 실측). TTFT 17% 감소
- **변경 위치**: 3rd EngineCore spawn + `_route_speculative` fanout + accept/reject 로직 + `HybridConfig.spec_decode_draft_model`
- **위험**: ⚠ CPU drafter 속도가 GPU verifier 와 balance 조건 충족 — **경로 1 batch scaling 필수 선행**
- **Ninja Gap 기여도**: 매우 큼. wall 공식 변경 — `max` 의 CPU term 이 "전체 처리" → "draft 만" 이 되어 tail 소멸
- **세부 구현 항목**:
  - Implementation plan 문서화 (`docs/SPEC_DECODE_CPU_DRAFTER_PLAN.md`)
  - `HybridConfig.spec_decode_draft_model: str | None` 필드
  - Third CPU EngineCore 프로세스 (ZMQ identity `b'\x02\x00'`)
  - `_route_speculative` 라우터 (요청 → GPU verifier + CPU draft fanout)
  - `process_engine_outputs` 에서 verify + accept/reject
  - Accept rate 로깅 (`[HYBRID-SPEC-STATS] accept=N/M`)
  - 32B + Qwen2.5-0.5B 조합 측정 / 1.5B/7B non-regression

### 5.2 P/D Disaggregation 🔶 (stub)
- **현재**: `vllm/engine/disaggregated/` stub. hybrid 엔진과 통합 안 됨
- **메커니즘**: prefill CPU (AMX BF16), decode GPU. long-context 16K+ GPU prefill bottleneck 해소
- **예상 이득**: 16K input 에서 GPU TPOT p99 개선
- **Ninja Gap 기여도**: 현 workload (128/128) 에서 0. long-ctx 전용

### 5.3 KV Cache CPU Tier Offload 🔶
- **현재**: `--cpu-offload-gb` CLI flag 존재 (용량 기반만). InfiniGen predictive prefetching / tier-aware block_table / LMCache prefix reuse 미구현
- **메커니즘**: PagedAttention block_table 에 tier 필드. hot → HBM / cold → CPU DRAM. LRU eviction + DMA prefetch
- **예상 이득**: 동시 시퀀스 3×, throughput 2–3× (70B/batch 1500+)
- **변경 위치**: `vllm/v1/core/kv_cache_manager.py` + DMA stream 분리
- **위험**: PCIe 지연. predictive prefetch 필요
- **Ninja Gap 기여도**: 7B 현 workload 0. 70B 에서 큼

### 5.4 ScoutAttention Layer-Ahead ⭕
- **메커니즘**: CPU 가 1 layer 앞서 Q 예측 (Q_{i+1} ≈ Q_i, cos sim 0.93+) → top-k KV block 선별 → partial attention. GPU 는 hot block 만. 결과 합산
- **예상 이득**: decoding 5.1× (장문). GPU idle 57% → <5%
- **위험**: 근사 attention PPL 열화 <2.1%. vLLM 포팅 대규모
- **Ninja Gap 기여도**: 현 workload 제한적. 8K+ context 에서 큼

### 5.5 NEO Asymmetric Batch Split ⭕
- **메커니즘**: 매 decode step batch 를 Batch-0 (GPU attn) + Batch-1 (CPU attn) 분할. GPU linear 실행 중 CPU attn overlap
- **예상 이득**: H100 70B 14.3% (MLSys'25). 작은 workload 축소
- **변경 위치**: `hybrid_core.py` 에 `_split_batch_asymmetric` + CPU worker 가 "attention 전용 워커" 로 역할 재정의
- **Ninja Gap 기여도**: 7B 제한적

---

## 6. 경로 1 스택 누적 이론 상한

순차 적용 (diminishing returns 50% 가정):

| 기법 | 단독 이득 | 누적 | Gate |
|---|---:|---:|---|
| Baseline | 1× | 1× | |
| + Huge Pages (§4.2) | 1.1× | 1.1× | |
| + WoQ INT8 (§4.3 기각 → §23 편입) | 2.0× | 2.1× | §23 CPU Native Quant |
| + OMP env (§4.4) | 1.05× | 2.2× | |
| + ISA binary (§4.6) | 2.0× | **3.3×** | G1 |
| + Fusion (§4.7) | 1.7× | 4.7× | |
| + LUT ops (§4.8) | 1.3× | 5.7× | |
| + Head Folding (§4.9) | 1.5× | **7.4×** | |
| + LUT GEMV INT4 (§4.12) | 3.0× | 13× | (WoQ 대체) |
| + Cascade (§4.13) | 1.7× | **19×** | G2 |
| + Pre-pack (§4.14) | 1.15× | 21× | |
| + Sparse (§4.15) | 1.35× | 27× | |
| + Batch-aware Attn (§4.10) | 1.5× | **35×** | G3 Ninja Gap |
| + Systolic (§4.16) | 2× | 70× | (overshoot) |

**현재 cost_cpu / cost_gpu ≈ 28×**. 경로 1 단독 역전 이론상 가능, 실제는 30% 효율 가정 시 10–20× 구간 예상 → **경로 2 조합 필요**.

---

## 7. 구현 상태 Audit (2026-04-15)

| # | 기법 | 상태 |
|---|---|:---:|
| 4.2 | Huge Pages 1GB | ⭕ |
| 4.3 | ~~IPEX WoQ INT8~~ | ✗ 기각 (§23 편입) |
| 4.4 | OMP + NUMA memory + KMP_BLOCKTIME | ✅ 완료 (§05) |
| 4.6 | ISA binary dispatch | 🔶 fallback chain, 명시적 batch-based 없음 |
| 4.7 | Sublayer fusion | 🔶 GPU 경로 only, CPU 전용 없음 |
| 4.8 | Softmax/SiLU LUT | ⭕ |
| 4.9 | Head Folding | ⭕ |
| 4.10 | Batch-aware decode attn | ✗ **Phase 1 기각** (2026-04-20) |
| 4.11 | Barrier/Sync 감소 | ⭕ (미계측) |
| 4.12 | T-MAC LUT GEMV | ⭕ |
| 4.13 | AVX/AMX cascade | ⭕ |
| 4.14 | AMX pre-pack | 🔶 IPEX 내부 자동 |
| 4.15 | SparAMX bitmask sparse | 🔶 dense int8_gemm_vnni 있음 |
| 4.16 | Core group pipeline | ⭕ |
| 5.1 | Spec decode CPU drafter | 🔶 GPU-only spec decode 만 |
| 5.2 | P/D disaggregation | 🔶 stub |
| 5.3 | KV offload | 🔶 용량 기반만 |
| 5.4 | ScoutAttention | ⭕ |
| 5.5 | NEO asymmetric | ⭕ |

**19 기법**: ✅ 완전 구현 1, 🔶 부분 구현 9, ⭕ 미구현 9

---

## 8. Ninja Gap 달성 시나리오

| 시나리오 | 조건 | 확률 |
|---|---|---:|
| 경로 1 단독 승리 | T-MAC LUT GEMV 4× 실현 + cascade 1.7× + batch-aware attn 12× scaling | 30% |
| 경로 1 + Spec Decode 조합 ★ | Stage C 후 15–20× + DuoDecoding 2× 추가 | 50% |

---

## 9. 실행 순서

1. **G0 profiler**: `num_seqs=1/2/4/8/16` CPU-only/hybrid 동일 shape 에서 `batch_scaling_ratio`, `per_req_cost`, sublayer breakdown 확보
2. **Hot path 증명**: VNNI/pre-pack/oneDNN dispatch 가 실제 CPU linear/attention 경로에 걸리는지 marker 로 확인
3. **Batch scaling kernel**: profiler top bottleneck 기준으로 batch-aware attention, head folding, fusion, barrier 감소 중 먼저 착수
4. **Big wins prototype**: LUT INT4, AVX/AMX cascade, AMX pre-pack, SparAMX 는 cache-fit/정확도/dispatch 조건을 만족하는 shape 에 한해 승격
5. **Routing 재활성**: `num_seqs=4` cost 가 single 대비 2× 이하로 내려간 뒤에만 `cpu_max_num_seqs` knee point 탐색
6. **Spec decode**: 경로 1 이 CPU drafter balance 조건을 만족할 때 본격화

---

## 10. 근거 등급

- **A** (로컬 실측): H100x8 wall 394/2098/14s, RTX3090 wall 23/90/8.1/6.5s
- **B** (유사 HW 논문): SparAMX 1.42× (Xeon SPR), KTransformers ISA batch>4 경계
- **C** (edge/NPU/MoE 논문, 이식 시 재검증): T-MAC 48 tok/s, T-MAN 3.1× decode, DuoDecoding 2.61×
- **D** (강한 가설, 환경 미검증): AVX/AMX cascade, LUT GEMV on SPR+AMX, staging cache-fit

D 에 머무는 기법이 Stage 3개 연속 실패 시 드롭. 각 단계 종료 시 D → B/A 승격 시도.

---

## 11. Tier 1 후보 (선행 연구 실측 수치 보유) — 우선 검토 (2026-04-20 정리)

§11 Phase 1 실패 후 방향 재정립. 자체 kernel 시도 (§06-1 v2, §11 Phase 1) 연속 실패 교훈 반영. **제안/착수 시 이 4개 내에서 시작**.

| § | 기법 | 보고된 실측 | 측정 HW | 근거 |
|---|---|---|---|---|
| **§13** | T-MAC LUT GEMV INT4 | INT4 4× | edge CPU (ARM) — SPR 재검증 필요 | C (Microsoft T-MAC, arXiv 2407.00088, GitHub 공식) |
| **§16** | SparAMX bitmask sparse | linear 1.42×, attention 1.14× | **Xeon SPR** (동일 HW) | B (AbouElhamayed et al. HF 2502.12444) |
| **§22** | NEO asymmetric | H100 70B 14.3% | **H100 + 70B** (동일 규모) | B (Jiang et al. MLSys'25) |
| **§28** | xFasterTransformer 이식 | Intel SPR 실측 (블로그) | SPR production | B (Intel 공식 유지) |

**상호 관계**:
- §13 은 §06 의 Q8_0 kernel 을 MLP 에서 INT4 LUT 로 교체 (상호 배타). §06 infra (torch op, patch wrapper) 재사용
- §16 은 §06/§13 위에 sparsity + AMX 추가
- §22 는 routing 축. kernel layer 와 독립
- §28 은 §23 + §24 + §14 + §08 의 alternative — 자체 kernel 포기하고 Intel 검증된 것 이식

## 12. 실패 기록 (재시도 시 참조)

| 기법 | 측정일 | 결과 | 원인 요약 | 데이터 |
|---|---|---|---|---|
| §04 IPEX WoQ INT8 | 2026-04-19 | 기각 | IPEX Q8 GEMM 연결 실패 | — (§23 로 편입) |
| §03 Huge Pages Phase 2 | 2026-04-19 | 기각 | SPR TLB 역효과 +22% | — |
| §06-1 v2 (VNNI `vpdpbusd`) | 2026-04-20 | 기각 | half-tile waste + compensation 오버헤드, v1 대비 −7~−13% | `g0_06_1_qwen2.5_32b_v2(fail)/` |
| **§11 Phase 1** | **2026-04-20** | **기각** | remainder path 이득 없음 + prefill SDPA 오버헤드, §06-1 v1 대비 −12~−5% | `g0_11_qwen2.5_32b_phase1(fail)/` |

**교훈**: Tier 2 (원리만, 실측 수치 없음) 을 Tier 1 확신으로 추진 금지. 선행 연구의 보고된 수치 + 조건 이 우리 환경에 부합하는지 선검토.

---

## 11. 코드 수정 위치 총괄

- **계측**: `vllm/v1/worker/cpu_worker.py` (sublayer hook, barrier marker), `eval/cpu_profile*.sh` (num_seqs sweep), `eval/basic/H100x8/analysis_h100.ipynb`
- **라우팅**: `vllm/v1/engine/hybrid_core.py` (default strategy, cpu_max_num_seqs, wave-batch, throughput-adaptive), `vllm/v1/engine/core_client.py` (dispatch/finished accounting, throughput feedback)
- **CPU hot path**: `vllm/v1/attention/backends/cpu_attn.py` (batch-aware, IPEX vs custom 분기), `csrc/cpu/*` (VNNI pre-pack, fusion, LUT, AVX/AMX dispatch, cascade), `vllm/v1/worker/cpu_model_runner.py` (load-time optimize/pre-pack hook, NUMA allocator)
