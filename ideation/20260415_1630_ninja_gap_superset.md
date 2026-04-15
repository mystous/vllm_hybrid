# Ninja Gap SuperSet — 종합 수정 방안 + 구현 Playbook 통합본

**작성**: 2026-04-15 KST  
**통합 대상**:
- `20260415_094130_claude_ninja_gap_comprehensive_plan.md` (Claude 종합 수정 방안)
- `20260415_094148_codex_ninja_gap_modification_playbook.md` (Codex 구현 Playbook)
- 외부 논문/프레임워크 (WebSearch 검증) + 실측 (H100x8 + RTX3090)

**이 문서의 목적**: 두 문서의 모든 분석, 기법, 로드맵, 근거, 규칙을 **하나의 참조 문서**로 통합한다. 중복은 병합하고, 상충은 명시적으로 해소하며, 어느 한쪽에만 있는 내용은 빠짐없이 포함한다.

---

## Part 0 — Ninja Gap 정의와 현재 상태

### 0-1. Ninja Gap 정의

**Ninja Gap**: `T_hybrid < T_gpu_only` 달성. 즉 **GPU 단독보다 CPU 추가가 빠름**.

### 0-2. 구조적 방정식

```
T_hybrid = max(T_gpu_completion, T_cpu_tail)
         = max(N_gpu × cost_gpu / batch_gpu,
               N_cpu × cost_cpu × f_scaling(N_cpu))
```

Ninja Gap 조건:
```
T_hybrid < T_gpu_only = (N_gpu + N_cpu) × cost_gpu / batch_gpu
```

`N_cpu > 0` 이려면 **CPU 경로 추가가 GPU 에서 뺀 양보다 wall 을 더 줄여야**.

### 0-3. 현재 실측 (H100x8 기준선)

| 구분 | CPU req | GPU req | wall | GPU-only 대비 |
|---|---:|---:|---:|---:|
| GPU-only | 0 | 500 | 14.01s | 1.0× |
| Hybrid seq=1 best | 2 | 498 | 364.41s | 26.0× |
| Hybrid seq=1 worst | 2 | 498 | 417.67s | 29.8× |
| Hybrid seq=16 best | 32 | 468 | 1993.89s | 142.3× |
| Hybrid seq=16 worst | 32 | 468 | 2003.19s | 143.0× |

추가 실측 (Claude 문서):
- `T_gpu_only` = **14s** (3.77s duration + 오버헤드)
- `T_hybrid_best` = **394s** (max_seqs=1, threads=32, 2 NUMA)
- **격차 28×**

### 0-4. 현재 사실 요약 (Codex 확인)

- CPU가 `2 req`만 가져가도 wall이 ~6분으로 늘어남
- CPU가 `32 req` 가져가면 wall이 ~33분으로 늘어남
- `cpu_max_num_seqs` 확대는 현재 상태에서 throughput gain이 아니라 **tail amplification**
- pinning 변경 (`112..167/168..223` → `0..55/56..111`)으로도 실패 구조 유지

### 0-5. Ninja Gap 실질 목표 (Codex 정의)

> CPU가 가져가는 request 수를 늘리면서도 `max(T_gpu_bulk, T_cpu_work)`에서 `T_cpu_work`가 wall을 지배하지 않게 만드는 것.

단일 CPU request를 조금 빠르게 하는 것만으로는 부족하다. **여러 request를 같이 처리할 때 per-request cost가 실제로 낮아져야** 한다.

### 0-6. 두 가지 달성 경로 (Claude)

**경로 1**: `cost_cpu` 를 GPU 에 가깝게 줄임 (CPU 자체 가속)
- 목표: cost_cpu 를 현재의 1/30 로
- 수단: LUT, fusion, cascade, pre-pack, sparse, WoQ

**경로 2**: CPU 를 동일 request 에서 "다른 일" 을 하게 — batch scaling 곡선을 바꿈
- 목표: 공식 자체 변경
- 수단: spec decode (draft), P/D disagg (prefill-only), KV offload, ScoutAttention

두 경로는 **배타적 아님**. 실제 Ninja Gap 은 경로 1 + 경로 2 의 누적 효과로 올 가능성.

### 0-7. 핵심 관찰 (Codex — 설계 순서를 뒤바꾸는 통찰)

> 현재 구조적 실패 모드는 "CPU 가 느리다" 가 아니라 **"num_seqs 증가에도 per-req cost 가 안 내려감 (batch scaling 실패)"**.

이 관찰이 설계 순서를 뒤바꿈: **Ninja Gap 전에 batch scaling 먼저**. batch scaling 없으면 어떤 가속도 tail 로만 남음.

### 0-8. 핵심 전제 (양 문서 공통)

- **request-level hybrid 유지**
- 목표: CPU가 더 많은 request를 처리하면서 total wall time을 줄이는 것
- 본선은 routing이 아닌 **CPU batch가 진짜 batch scaling을 보이도록 kernel/dataflow를 바꾸는 것**

---

## Part 1 — 실패 모델 분석

### 1-1. 3겹 실패 (Claude 분해)

실측 H100x8 H2 (max_seqs=16): wall 2098s. 분해:

```
실패 1 (Batch scaling 제로):
  per-step batch=1 → 3079ms
  per-step batch=16 → 16,390ms (5.3×, 선형 기대 16× 대비 3× 실패)
  원인: L3 scatter + AMX tile layout 불일치 + sublayer DDR 왕복

실패 2 (ISA 경직):
  AMX 고정 → batch=1 에서 AVX-512 대비 2.22× 손해 (KTransformers 실측)
  원인: AMX tile 초기화 + clock down

실패 3 (Dataflow 미설계):
  sublayer 8개 체인 = 독립 kernel = DDR 왕복 8회 × batch
  LUT 없음, fusion 없음, cascade 없음
  원인: IPEX/oneDNN 의 범용성 우선 설계
```

### 1-2. Codex의 실패 구조 정밀 정의

현재 실패는 "CPU가 느리다"가 아니라 더 구체적으로:

> CPU scheduler가 여러 request를 한 batch로 잡아도, 실제 hot path가 그 request들을 GPU처럼 효율적인 larger-M 연산으로 바꾸지 못한다.

그 결과:
- request 수만 늘어남
- memory traffic은 req별로 반복
- runtime packing/repacking이 반복
- OMP barrier/sync 비용 누적
- GPU가 끝난 뒤 CPU wave drain이 wall을 결정

### 1-3. 잘못된 개선 방향 vs 올바른 방향 (Codex)

**잘못된 방향**:
- `cpu_max_num_seqs`만 올리기
- `wave-batch`를 더 크게 만들기
- pinning/NUMA bring-up을 다시 증명하기
- CPU handled request 수만 보고 성공 판단하기

**올바른 방향**:
- 여러 request를 함께 처리할 때 input read, weight read, KV scan, activation, projection 비용이 amortize되도록 hot path를 바꿈
- batch scaling이 생긴 뒤에만 CPU inflight/routing을 키움

### 1-4. 현재 달성 가능 이론 상한 (Claude)

dev 에서 T-MAC INT4 포팅 시 이론상 (선형 누적 가정):
- Baseline 현재: 3079ms/step
- × WoQ INT8 (2×) × Huge Pages (1.1×) × ISA cascade (2.22×) × LUT GEMV (4×) × Fusion (1.5×) × Pre-pack (1.15×) × Sparse (1.42×) = **~60×** → **~50ms/step**

50ms/step × 128 tokens = 6.4s CPU tail. gpu_only 14s 보다 짧음 → **Ninja Gap 이론상 도달 가능**.

단 diminishing returns, scatter/sync 오버헤드 미반영. 실제는 10-20× 구간에서 정체 예상 → CPU tail 150-300ms 정도로 감소 → **여전히 Ninja Gap**.

**결론**: 경로 1 단독으로도 이론상 Ninja Gap 가능. 단 **batch scaling 이 전제**.

---

## Part 2 — 이미 구현된 항목 (제외 규칙)

> Codex 원칙: 이미 구현된 기능은 **추가 성능 향상분으로 산출하지 않는다**.

### 2-0. 구현 완료 항목 — gain 계산 제외

| 항목 | 상태 | 근거 | gain 계산 |
|---|---|---|---|
| CPU engine launch | 구현됨 | `hybrid_core.py`, `core_client.py` | 제외 |
| `wave-batch` routing | 구현됨 | `CapacityAwareRouter` | 제외 |
| `throughput-adaptive` routing | 구현됨 | `CapacityAwareRouter` | 제외 |
| `cpu_max_num_seqs=1` auto baseline | 구현됨 | `_resolve_cpu_params()` | 제외 |
| CPU `chunked_prefill=False` | 구현됨 | `_create_cpu_vllm_config()` | 제외 |
| CPU core pinning | 구현됨 | `CPUWorker.init_device()` + `_C_utils.init_cpu_threads_env` | 제외 |
| NUMA node 기반 CPU 선택 | 구현됨 | `hybrid_config.numa_bind_node`, `_get_autobind_cpu_ids()` | 제외 |
| NUMA memory bind | 구현됨 | `numa_set_membind()`, `NUMAAllocator.bind_to_node()` | 제외 |
| affinity reset after fork | 구현됨 | `_setup_cpu_process_env()` | 제외 |
| feature 기반 ONEDNN ISA 설정 | 구현됨 | `intel_cpu_utils.py` | 제외 |
| VNNI INT8 GEMM 토대 | 부분구현 | `csrc/cpu/gemm_vnni.*`, `torch_bindings_hybrid.cpp` | hot path 연결분만 계산 |
| attn/mlp coarse profiling | 부분구현 | `cpu_worker.py` forward hook | sublayer 확장분만 계산 |

**핵심 해석** (Codex):
- "NUMA locality를 해야 한다"는 신규 gain 항목이 아님 — 골격 이미 존재
- "chunked prefill을 꺼야 한다"도 신규 gain 아님 — 이미 꺼져 있음
- "VNNI가 있다"도 신규 gain 아님 — hot path에 실제 연결되어 batch scaling을 만든 경우만 gain
- `wave-batch`, `throughput-adaptive` 존재 자체는 개선 아님 — routing은 kernel/dataflow 개선 뒤 재평가

---

## Part 3 — 경로 1: CPU 자체 가속 스택 (전체 기법 카탈로그)

### 구현 상태 태그 (2026-04-15 audit)
- ✅ **이미 구현** — 추가 작업 불요 또는 미미
- 🔶 **부분 구현** — 일부 경로만 / 기능 축소형 존재
- ⭕ **미구현** — 새로 개발

---

### 3-1. [Tier 0] Huge Pages 1GB ⭕ 미구현

| | |
|---|---|
| **구현 상태** | ⭕ grep `MAP_HUGETLB\|hugepagesz` 결과 없음 |
| **메커니즘** | 4KB → 1GB 페이지. TLB 엔트리 70B INT4 기준 900만 → 35개. TLB miss 해소 |
| **예상 이득** | **5-15%** (논문 수치). decode 전반에 균등 영향 |
| **비용** | 0.5일. grub `hugepagesz=1G hugepages=40` + vLLM mmap flag `MAP_HUGETLB` |
| **위험** | 컨테이너 cgroup 설정 필요. dev 머신에서 먼저 검증 |
| **스택 호환성** | 모든 후속 기법과 독립. 항상 깔아야 |
| **근거** | 70B INT4 TLB 분석 (ideation 0950_cpu_llm_optimization_techniques §3.2) |
| **Codex 분류** | Tier 0 기준선 방어의 low-risk 별도 실험 |

### 3-2. [Tier 0] IPEX WoQ INT8 ⭕ 미구현

| | |
|---|---|
| **구현 상태** | ⭕ grep `WoqWeightDtype\|weight_only_quant` 결과 없음. `cpu_worker.py`의 `ipex.llm.optimize` 호출에 quantization_config 미전달 |
| **메커니즘** | BF16 weight → INT8 저장, BF16 연산. weight memory 2× 절감 |
| **예상 이득** | **2× decode throughput** (memory-bound). PPL 열화 <0.5 |
| **비용** | 2-3일. `cpu_worker.py`의 `ipex.llm.optimize`에 `quantization_config=qconfig` 추가 |
| **위험** | IPEX WoQ가 vLLM Hybrid 모델 로딩 경로와 호환되는지 미검증 |
| **스택 호환성** | LUT INT4 (§3-8)로 넘어가면 **대체됨**. Tier 0 임시 |
| **근거** | Intel ICML'24 workshop (arXiv 2407.07304) |
| **Codex 분류** | Tier 0 기준선 방어의 low-risk 별도 실험 |

### 3-3. [Tier 0] OMP 환경 + Memory Pinning ✅ 구현됨 (KMP_BLOCKTIME 제외)

| | |
|---|---|
| **구현 상태** | ✅ `csrc/cpu/utils.cpp:70-71` — `numa_set_membind(mask)` + `numa_set_strict(1)` + `sched_setaffinity` (line 90). 메모리 strict membind + core pinning + page migration 3종 수행. **KMP_BLOCKTIME=0만 H100 env에 누락** |
| **메커니즘** | `OMP_PROC_BIND=close`, `OMP_PLACES=cores`, `KMP_BLOCKTIME=0`, `numactl --membind=strict` |
| **예상 이득** | **<5%** (90% 이미 적용). H100 env에 `KMP_BLOCKTIME=0` 추가만 남음 |
| **비용** | 10분 (env 파일 편집) |
| **위험** | 없음 |
| **스택 호환성** | 독립 |

### 3-4. [Tier 1] ISA Binary Dispatch 🔶 부분 구현

| | |
|---|---|
| **구현 상태** | 🔶 `cpu_attn.py` decode 경로에 `custom_avx → ipex → sdpa_batched → sdpa_loop` fallback chain 존재 (`_decode_path_counts` 카운터). 그러나 **batch size 기반 명시적 dispatch 가 아님** — IPEX 내부 dispatcher가 대부분 처리. KTransformers 방식 (batch>4 → AMX, else → AVX-512 강제) 미구현 |
| **메커니즘** | batch size > 4 → AMX, else → AVX-512 VNNI. KTransformers 방식 |
| **예상 이득** | decode **1.5-2.22×** (KTransformers 실측) |
| **비용** | 1주. `cpu_worker.execute_model` pre-dispatch + csrc kernel 등록 |
| **위험** | IPEX 내부 dispatcher와 충돌 가능. Bypass 필요 |
| **스택 호환성** | §3-9 cascade의 전제 |
| **근거** | [KTransformers AMX doc](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/AMX.md) |
| **Codex 연결** | Tier 1 mainline hot path 연결. VNNI/pre-pack wiring + shape별 dispatch log |

### 3-5. [Tier 1] Kernel Fusion — QKV concat + Gate/Up interleave + Residual+Norm 🔶 부분 구현

| | |
|---|---|
| **구현 상태** | 🔶 vLLM **GPU 경로**에 `gate_up_proj`, `qkv_proj` merged linear 구현. **CPU 전용 fused kernel (`csrc/cpu/`)** 없음. Residual+RMSNorm fusion은 IPEX 내부에 있음 |
| **메커니즘** | sublayer 8개 독립 kernel → 4개 묶음. 중간 DDR write 제거, 입력 x 단일 로드 |
| **예상 이득** | **1.5-2×** (SGLang SiLU+up 12% × 4 sublayer 누적) |
| **비용** | 2주. `csrc/cpu/fused_qkv.cpp`, `fused_gate_up_silu_down.cpp`, `fused_add_rmsnorm.cpp` |
| **위험** | IPEX 기존 fused kernel과 충돌 검증 필요 |
| **스택 호환성** | LUT / cascade와 독립. 항상 병행 |
| **근거** | SGLang CPU 백엔드 블로그 (12% 실측), T-MAC Gate+Up interleave 설계 |
| **Codex 세분화** | QKV fusion (같은 hidden state 반복 read 제거) + Gate+Up fusion (SwiGLU input read amortization) — 이미 fused된 모델 구조인지 먼저 확인 후 착수 |

### 3-6. [Tier 1] Softmax + SiLU LUT 대체 ⭕ 미구현

| | |
|---|---|
| **구현 상태** | ⭕ `csrc/cpu/`에 `lut*` 없음. `vpshufb` 기반 LUT kernel 없음 |
| **메커니즘** | `exp()` 20 cycles → `vpshufb` LUT 1 cycle. SiLU는 "hot range" 선형 근사 + LUT |
| **예상 이득** | Softmax **2.2×**, SiLU **1.2×** (TARDIS 로는 vLLM 1.6× 보고) |
| **비용** | 2주. `csrc/cpu/lut_ops.cpp`. 32B/512B LUT register 상주 |
| **위험** | 근사로 인한 정확도 열화 미검증 (< 2% 추정) |
| **스택 호환성** | §3-8 GEMV LUT와 동일 인프라. 같이 개발 |
| **근거** | T-MAN (arXiv 2511.11248), TARDIS (arXiv 2501.10054) |
| **Codex 조건** | scalar transcendental overhead가 프로파일에서 확인된 경우에만 적용. top bottleneck 아니면 후순위 |

### 3-7. [Tier 1] Head Folding (GEMV → GEMM) ⭕ 미구현

| | |
|---|---|
| **구현 상태** | ⭕ grep `head_fold\|fold_head` 결과 없음 |
| **메커니즘** | decode attention의 M=1 GEMV를 batch fold해서 M=16 GEMM으로. AMX tile full 활용 |
| **예상 이득** | decode attention **1.5-2×** (SGLang blog) |
| **비용** | 2주. `csrc/cpu/fold_attention.cpp` + IPEX single_query 대체. GQA 구조 반영 |
| **위험** | MLA (DeepSeek)에서는 직접 적용. GQA (Qwen)에서는 batch fold 변형 필요 |
| **스택 호환성** | §3-12 batch-aware attention의 한 방식 |
| **근거** | SGLang Head Folding blog |
| **Codex 연결** | Tier 2 "진짜 batch scaling" 핵심 후보 1번. decode batch와 head dimension 재배치로 request 수 증가가 kernel efficiency 증가로 연결 |

### 3-8. [Tier 2] T-MAC LUT-Based GEMV (INT4 핵심) ⭕ 미구현

| | |
|---|---|
| **구현 상태** | ⭕ `csrc/cpu/`에 `lut_gemv`/`tmac` 없음. INT4 LUT 기반 GEMV 경로 없음 |
| **메커니즘** | INT4 weight 16 값 × input을 LUT 32B에 precompute. 곱셈 + 역양자화를 `vpshufb` 1-cycle로 |
| **예상 이득** | INT4 **4×** (T-MAC 실측, CPU 22 tok/s > NPU 10.4 tok/s). bit↓ 선형 가속 |
| **비용** | 3-4주. `csrc/cpu/lut_gemv.cpp` 전용 kernel. IPEX bypass |
| **위험** | ⚠ T-MAC은 edge CPU 검증 (Snapdragon). SPR+AMX 조합 재검증 필요. **강한 가설** |
| **스택 호환성** | §3-2 WoQ INT8 대체. §3-6 LUT Softmax와 동일 인프라 |
| **근거** | [T-MAC EuroSys'25](https://arxiv.org/pdf/2407.00088) |
| **Codex 분류** | 구현 난도 높음. 강한 후보지만 즉시 gain 아님. LUT/low-bit native path/dataflow 재설계 근거 |

### 3-9. [Tier 2] AVX/AMX Cascade Pipeline ⭕ 미구현

| | |
|---|---|
| **구현 상태** | ⭕ grep `enqcmd\|dsa\|3_stage\|cascade` CPU쪽 결과 없음. Intel DSA 사용 경로 없음 |
| **메커니즘** | tile k+2 load (prefetch/DSA) / tile k+1 dequant·pack (AVX-512) / tile k matmul (AMX) 3-stage 동시 실행 |
| **예상 이득** | **1.5-3×** (T-MAN NPU 실측 decode 3.1×. CPU 이식 시 보수적 1.5-2×) |
| **비용** | 4주. 타일 버퍼 설계 + cache-fit 검증. AVX `zmm` ↔ AMX tile 별도라 **중간 버퍼 L2 상주 설계 필수** |
| **위험** | ⚠ Staging overhead가 이득을 상쇄할 수 있음. Shape별 측정 필수 |
| **스택 호환성** | §3-4 binary dispatch의 발전형. §3-10 pre-pack과 조합 필수 |
| **근거** | T-MAN (3-stage 원리 증명) |
| **Codex 주의사항** | AVX `zmm`와 AMX `tile register`는 직접 연결되지 않음. 중간 tile buffer가 L1/L2/L3에 닫혀야 함. **cache-fit 실패 시 pipeline이 아니라 DDR 왕복 증가**. 강한 가설로 취급. prototype 검증 필수 |

**Codex 구체적 수정 항목**:
- tile size별 buffer footprint 계산
- L2 fit 가능한 staging layout 설계
- `batch=1`은 AVX path, `batch>=N`은 AMX/cascade path — shape-aware dispatch
- 전환 비용과 tile config 비용을 profile marker로 분리

### 3-10. [Tier 2] AMX Weight Pre-pack 🔶 부분 구현

| | |
|---|---|
| **구현 상태** | 🔶 IPEX의 `ipex.llm.optimize(..., weights_prepack=True)` 기본값이 암묵적 활성. **KTransformers 스타일 독자 pre-pack (AMX tile layout 직접 제어)** 없음 |
| **메커니즘** | 모델 로드 시 1회 weight를 AMX tile layout (16×64 byte)로 재배치. 런타임 tileloadd가 연속 16 cache line 로드 |
| **예상 이득** | **1.1-1.2×** (KTransformers 실측 10-20%) |
| **비용** | 1주. CPUWorker `load_model` 후 hook |
| **위험** | 낮음. 메모리 부담 2× (원본 + 재배치) |
| **스택 호환성** | cascade 전제. LUT path에도 유사 pre-pack 필요 (T-MAC group layout) |
| **Codex 연결** | Tier 1 mainline — load-time pre-pack cache 추가, runtime repack이 step마다 발생하는지 계측 후 제거 |

### 3-11. [Tier 2] AVX-512 Bitmask Sparse (SparAMX 기반) 🔶 부분 구현

| | |
|---|---|
| **구현 상태** | 🔶 `csrc/cpu/gemm_vnni.cpp`에 `int8_gemm_vnni` kernel 존재 (INT8 VNNI dense GEMM). **sparse 경로 없음** |
| **메커니즘** | Unstructured sparsity를 `K` 레지스터 64-bit mask로 표현. `_mm512_mask_fmadd_ps`로 유효 원소만 계산 |
| **예상 이득** | linear **1.42×**, attention **1.14×** (SparAMX 실측, Xeon SPR) |
| **비용** | 4주. `csrc/cpu/sparse_amx.cpp`. 가중치 50% 프루닝 필요 |
| **위험** | 프루닝 후 PPL 열화 검증 필요. MLP 희소성은 batch↑ 시 소멸 (Polar Sparsity) — attention head sparsity만 batch-invariant |
| **스택 호환성** | LUT과 별개 경로. 대체가 아닌 추가 |
| **근거** | [SparAMX](https://huggingface.co/papers/2502.12444) |

### 3-12. [Tier 2] Batch-aware Decode Attention 🔶 부분 구현 (batch=16 한정)

| | |
|---|---|
| **구현 상태** | 🔶 `csrc/cpu/batch_attention.cpp` + `torch_bindings.cpp:91`에 `batch16_paged_attention_v1` kernel 구현됨. 단 **batch=16 hardcoded**. 동적 batch size 미지원 |
| **메커니즘** | per-seq KV paged access 구조를 batch 단위로 재구성. head-parallel + page-coalesced |
| **예상 이득** | batch=16 scaling을 5.3× → **10-12×** 개선 (목표) |
| **비용** | 4주. `cpu_attn.py`의 IPEX call 대체 + 새 kernel. 가장 복잡 |
| **위험** | 높음. IPEX 내부 FD kernel 재구현에 해당 |
| **스택 호환성** | §3-7 Head Folding과 중복 영역. 하나 선택 or 통합 |
| **근거** | H100x8 H2 실측 재앙 (2098s) |
| **Codex 연결** | Tier 2 "진짜 batch scaling" 핵심 후보 2번. per-seq loop 제거. KV scan/score/softmax/value accumulation req별 반복 제거 |

### 3-13. [Tier 2] Barrier/Sync 감소 (Codex 고유)

| | |
|---|---|
| **구현 상태** | 미확인 (계측 필요) |
| **메커니즘** | OMP parallel region을 sublayer마다 새로 열고 닫는지 확인. thread team 재사용, chunk scheduling, layer/block 단위 persistent region |
| **예상 이득** | 미정 (프로파일 의존) |
| **비용** | 1-2주 (계측 후 판단) |
| **위험** | 낮음 |
| **스택 호환성** | 독립 |

### 3-14. [Tier 3] Core Group Pipeline (Systolic) ⭕ 미구현

| | |
|---|---|
| **구현 상태** | ⭕ CPU core group 분할 / L3 inter-group 전달 구조 없음 |
| **메커니즘** | 56 core를 4 group으로 분할. A: QKV, B: Attn, C: MLP, D: next layer QKV 파이프라인 |
| **예상 이득** | **2-3× latency** (4 layer 동시 실행) |
| **비용** | 6주+. scheduler 재설계, worker 분리, L3 버퍼 설계. 매우 복잡 |
| **위험** | 매우 높음. L3 BW가 DDR보다 높지만 coherence 비용 큼 |
| **스택 호환성** | Tier 2 완료 후. 기반 kernel이 fast해야 이득 |

---

## Part 4 — 경로 2: 역할 재정의 (구조 변경)

### 4-1. Spec Decode CPU Drafter (DuoDecoding 방식) 🔶 부분 구현

| | |
|---|---|
| **구현 상태** | 🔶 `vllm/v1/spec_decode/`에 GPU-on-GPU spec decode 프레임워크 존재. **CPU drafter + GPU verifier DuoDecoding 스타일 미구현** |
| **메커니즘** | CPU가 drafter (Qwen2.5-0.5B)로 k 토큰 생성 → GPU verifier (7B)가 한 번에 검증. accept rate ~70%이면 k-1 토큰 free |
| **예상 이득** | TPOT **2.1-2.61×** (DuoDecoding 실측). TTFT **17% 감소** |
| **비용** | 6주. 3rd EngineCore (drafter) + verifier 동기화 + accept/reject 로직 |
| **위험** | ⚠ **CPU drafter 속도가 GPU verifier와 balance 조건 충족 여부** — 경로 1 필수 선행 |
| **Ninja Gap 기여도** | 매우 큼. wall 공식 변경 — `max`의 CPU term이 "전체 처리"가 아닌 "draft만"이 되어 tail 소멸 |
| **Codex 분류** | 병행 트랙. request-level mainline과 분리. CPU batch scaling 실패를 숨기는 도피처로 쓰면 안 됨 |

### 4-2. P/D Disaggregation 🔶 부분 구현 (stub 수준)

| | |
|---|---|
| **구현 상태** | 🔶 `vllm/engine/disaggregated/` stub 존재. hybrid 엔진과 통합 안 됨 |
| **메커니즘** | prefill은 CPU (AMX BF16), decode는 GPU. long-context 16K+에서 GPU prefill bottleneck 해소 |
| **예상 이득** | 16K input에서 GPU TPOT p99 개선 |
| **비용** | 8주 |
| **Ninja Gap 기여도** | 현 workload (128/128)에서 0. long-ctx 전용 |

### 4-3. KV Cache CPU Tier Offload 🔶 부분 구현

| | |
|---|---|
| **구현 상태** | 🔶 `--cpu-offload-gb` CLI flag 존재. 단순 용량 기반. InfiniGen 스타일 predictive prefetching/tier-aware block_table/LMCache prefix reuse 미구현 |
| **메커니즘** | PagedAttention block_table에 tier 필드. hot → HBM, cold → CPU DRAM |
| **예상 이득** | 동시 시퀀스 **3×**, throughput **2-3×** (70B/batch 1500+에서) |
| **비용** | 6주 |
| **Ninja Gap 기여도** | 7B 현 workload에서 0. 70B에서 큼 |

### 4-4. ScoutAttention Layer-Ahead ⭕ 미구현

| | |
|---|---|
| **구현 상태** | ⭕ 전무 |
| **메커니즘** | CPU가 1 layer 앞서 Q 예측 → top-k KV block 선별 → partial attention |
| **예상 이득** | decoding **5.1×** (장문). GPU idle 57% → <5% |
| **비용** | 11주. 가장 복잡 |
| **Ninja Gap 기여도** | 현 workload 제한적. 8K+ context에서 의미 큼 |

### 4-5. NEO Asymmetric Batch Split ⭕ 미구현

| | |
|---|---|
| **구현 상태** | ⭕ 전무 |
| **메커니즘** | 매 decode step batch를 Batch-0 (GPU attn) + Batch-1 (CPU attn) 분할 |
| **예상 이득** | H100 70B **14.3%** (MLSys'25 실측). 작은 workload에서 축소 |
| **비용** | 8주 |
| **Ninja Gap 기여도** | 7B에서 제한적 |

---

## Part 5 — 구현 상태 Audit 종합 (2026-04-15)

| # | 기법 | Tier | 상태 |
|---|---|:---:|:---:|
| 3-1 | Huge Pages 1GB | 0 | ⭕ 미구현 |
| 3-2 | IPEX WoQ INT8 | 0 | ⭕ 미구현 |
| 3-3 | OMP + NUMA memory | 0 | ✅ **구현됨** (KMP_BLOCKTIME만 누락) |
| 3-4 | ISA binary dispatch | 1 | 🔶 fallback chain만, 명시적 batch-based 없음 |
| 3-5 | Sublayer fusion | 1 | 🔶 GPU 경로 only, CPU 전용 fused kernel 없음 |
| 3-6 | Softmax/SiLU LUT | 1 | ⭕ 미구현 |
| 3-7 | Head Folding | 1 | ⭕ 미구현 |
| 3-8 | T-MAC LUT GEMV | 2 | ⭕ 미구현 |
| 3-9 | AVX/AMX cascade | 2 | ⭕ 미구현 |
| 3-10 | AMX pre-pack | 2 | 🔶 IPEX 내부 자동 (독자 없음) |
| 3-11 | AVX-512 bitmask sparse | 2 | 🔶 dense int8_gemm_vnni 있음, sparse 없음 |
| 3-12 | Batch-aware decode attn | 2 | 🔶 batch16 hardcoded |
| 3-13 | Barrier/Sync 감소 | 2 | 미확인 |
| 3-14 | Core group pipeline | 3 | ⭕ 미구현 |
| 4-1 | Spec decode CPU drafter | 병행 | 🔶 GPU-only spec decode만 |
| 4-2 | P/D disaggregation | 장거리 | 🔶 stub |
| 4-3 | KV offload | 장거리 | 🔶 용량 기반만 |
| 4-4 | ScoutAttention | 장거리 | ⭕ 미구현 |
| 4-5 | NEO asymmetric | 장거리 | ⭕ 미구현 |

**통계**: 19 기법 중 ✅ 완전 구현 1, 🔶 부분 구현 9, ⭕ 미구현 9

---

## Part 6 — 경로 1 스택 누적 예상 이득 (이론 상한)

순차 적용 시 (diminishing returns 50% 가정):

| 기법 | 단독 이득 | 누적 (50% eff.) | Gate |
|---|---:|---:|---|
| Baseline | 1× | 1× | |
| + Huge Pages (3-1) | 1.1× | 1.1× | |
| + WoQ INT8 (3-2) | 2.0× | 2.1× | |
| + OMP env (3-3) | 1.05× | 2.2× | |
| + ISA binary (3-4) | 2.0× | **3.3×** | G1 진입 |
| + Fusion (3-5) | 1.7× | 4.7× | |
| + LUT ops (3-6) | 1.3× | 5.7× | |
| + Head Folding (3-7) | 1.5× | **7.4×** | |
| + LUT GEMV INT4 (3-8) | 3.0× | 13× | (WoQ 대체) |
| + Cascade (3-9) | 1.7× | **19×** | G2 진입 |
| + Pre-pack (3-10) | 1.15× | 21× | |
| + Sparse (3-11) | 1.35× | 27× | |
| + Batch-aware Attn (3-12) | 1.5× | **35×** | G3 Ninja Gap |
| + Systolic (3-14) | 2× | 70× | (overshoot) |

**현재 cost_cpu / cost_gpu ≈ 28×**. 경로 1 단독으로 28× 역전 이론상 가능. 실제는 30% 효율 가정 시 10-20× 구간 예상 → **경로 2와 조합 필요**.

---

## Part 7 — 성공 지표와 경유지 (Gate 시스템)

### 7-1. 4축 동시 확인 (양 문서 합의)

| 축 | 봐야 할 값 | 성공 방향 |
|---|---|---|
| CPU scaling | `cost(batch=N) / cost(batch=1)` | N보다 훨씬 작아야 함 |
| CPU throughput | CPU-only tok/s, req/s | batch 증가와 함께 증가 |
| Tail | GPU bulk 이후 CPU-only drain | 감소 |
| Wall ratio | `hybrid wall / gpu_only wall` | 감소 |
| CPU contribution | CPU handled req | 증가 |

**성공 판정**: 최소 4가지가 동시에 좋아져야 함.
- CPU가 더 많은 request 처리
- CPU batch tok/s 증가
- CPU tail 감소
- hybrid wall ratio 개선

**부분 성공 판별** (Codex):
- CPU handled request만 늘고 wall 악화 → **실패**
- 1만 (속도) → routing 문제 의심
- 2만 (batch scaling) → cache-fit 재검토
- 3만 (tail) → 단순 운 좋은 routing, 본질 아님

### 7-2. Gate 정의

| Gate | 목표 | 통과 조건 | 실패 시 |
|---|---|---|---|
| G0 | 기준선 분해 | seq=1/2/4/8/16 CPU-only scaling, sublayer breakdown 확보 | 계측부터 보강 |
| G1 | scaling 징후 | 4req cost ≤ 2× single, tail < 100s, wall ratio < 8× | hot path 못 건드린 것 |
| G2 | routing 재평가 | 4req cost ≤ 1.5× single, tail < 10s, wall ratio < 1.5× | routing/gate 재설계 |
| G3 | Ninja Gap | CPU req 증가 + tail 제거 + wall ≤ GPU-only | 목표 달성 |

> Codex 주의: Claude 문서의 G1/G2/G3 숫자는 방향성으로 좋지만 현재 코드에서 확정 수치 아님. **실험 통과 기준**으로만 사용.

---

## Part 8 — 통합 실행 로드맵

### 8-1. Stage A: 계측 + Quick Wins (0-2주)

**Tier -1: 계측 재정의** (Codex 핵심 — 여기서 "어디를 고칠지" 확정)

수정:
- `eval/cpu_profile*.sh`에 `num_seqs=1/2/4/8/16` sweep 고정
- CPU-only와 hybrid CPU engine의 동일 shape 비교
- `cpu_worker.py` coarse `attn/mlp` hook을 QKV/O/Gate/Up/SiLU/Down/Norm 수준으로 확장
- per-step barrier/sync time, memory wait, packing/repacking time marker 추가
- H100x8와 dev 머신의 결과를 같은 CSV schema로 저장

산출물:
- `batch_scaling_ratio = step_ms(batch=N) / step_ms(batch=1)`
- `per_req_cost = step_ms / active_reqs`
- sublayer별 top bottleneck
- `num_seqs` 증가 시 어떤 sublayer가 폭증하는지

**Tier 0: Quick Wins** (Claude ROI):
- 3-1 Huge Pages (0.5일)
- 3-3 OMP env KMP_BLOCKTIME (10분)
- 3-2 WoQ INT8 (3일)

**Tier 0: 기준선 방어** (Codex):
- 기본 실험은 `cpu_max_num_seqs=1`로 고정
- `wave-batch`는 기본 전략에서 내리고 비교 대상으로만 유지
- Huge Pages와 WoQ INT8은 별도 low-risk 실험으로 분리

**기대**: baseline × 2-3×. **G1 진입 (ratio <15×)** 확률 높음.

### 8-2. Stage B: Kernel 투자 — Mainline Hot Path (2-8주)

**Tier 1: Hot Path 연결** (Codex mainline):
- VNNI INT8 GEMM 경로를 실제 Qwen2.5 CPU linear hot path에 연결
- load-time weight pre-pack cache 추가
- runtime repack step마다 발생 여부 계측 후 제거
- `batch=1/2/4/8/16` shape별 AVX/VNNI/oneDNN path dispatch 기록
- `ONEDNN_MAX_CPU_ISA` 설정 존재가 아닌 실제 primitive dispatch를 프로파일로 확인

Tier 1 성공 조건:
- hot path가 실제로 바뀐 로그/marker
- `num_seqs=4`에서 per-request cost 감소
- **단일 req만 빨라지고 batch scaling 없으면 다음 Tier 불가**

**Tier 2: 진짜 Batch Scaling** (양 문서 합의):
1. 3-4 ISA Binary Dispatch (1주)
2. 3-5 Kernel Fusion (2주)
3. 3-6 Softmax/SiLU LUT (2주, 조건부)
4. 3-7 Head Folding (2주)
5. 3-12 Batch-aware Decode Attention (4주, 병행)
6. 3-13 Barrier/Sync 감소 (1-2주, 계측 의존)

Tier 2 성공 조건 (Codex):
- `num_seqs=4`가 `1 req` 대비 cost 2× 이하
- CPU handled req 늘렸을 때 wall 악화 안 됨

**기대**: Stage A 위에 × 3-5× 더. **G2 진입 (ratio <1.5×)** 조건부 가능.

### 8-3. Stage C: Big Wins (8-12주)

1. 3-8 T-MAC LUT GEMV INT4 (3-4주) — 가장 큰 단일 이득
2. 3-10 AMX Pre-pack (1주, 병행)
3. 3-9 AVX/AMX Cascade (4주, LUT 경로와 통합)
4. 3-11 SparAMX Sparse (4주, 병행)

**AVX/AMX Cascade 성공 조건** (Codex):
- cascade path가 특정 shape에서 standalone AVX/AMX보다 빠름
- memory wait 비중이 증가하지 않음

**기대**: G2 → G3 돌파 가능. **이 단계에서 Ninja Gap 첫 달성** 기대.

**실패 조건** (Stop/Go):
- 속도 개선 없음 → Stack 재검토, cache-fit 원인 추적
- CPU handled↑ but wall↓ → batch scaling 재검토

### 8-4. Stage D: Routing 재평가 + Spec Decode (Stage B/C 이후)

**Tier 4: Routing 재평가** (Codex — Tier 1-3에서 batch scaling 확인 후에만 진입):
- `cpu_max_num_seqs` knee point 탐색: `1 → 2 → 4 → 8 → 16`
- `wave-batch` vs `throughput-adaptive` vs strict continuous 비교
- CPU engine별 observed throughput을 gate에 반영

Routing 성공 조건:
- CPU req 증가와 wall ratio 개선 동시
- CPU wave가 tail로 고착되면 즉시 이전 Tier로 되돌림

**Spec Decode 병행** (Claude Stage D + Codex Tier 5 병행 트랙):
- Stage C 결과가 CPU draft throughput 충분하면 가장 빠른 Ninja Gap 2차 레버
- 경로 1만으로 실패 시 Plan B

**상충 해소**: Codex "역할 재정의 mainline 앞세우지 말라" vs Claude "Stage D 병행 추진".
→ Stage D는 **설계만 병행**, 구현 리소스는 Stage C에 집중. Stage C 종료 후 Stage D 구현 본격.

### 8-5. Stage E: 장거리 트랙 (별도)

70B / long-context workload가 next target일 때:
- 4-2 P/D disagg
- 4-3 KV offload
- 4-4 ScoutAttention
- 3-14 Core Group Pipeline
- 4-5 NEO asymmetric

현 workload에서 우선순위 최하.

**Codex 경고**: 이들은 request-level CPU batch scaling 실패를 숨기는 도피처로 쓰면 안 됨. mainline이 일정 기간 실패할 때 병행 비중을 높임.

---

## Part 9 — PR 단위 작업 순서 (Codex)

### PR 1. Batch Scaling Profiler
- CPU-only `num_seqs=1/2/4/8/16` sweep
- hybrid CPU engine 동일 metric logging
- CSV/JSON schema 고정
- **완료**: H100x8에서 seq별 step time과 per-request cost 표

### PR 2. Fine-grained Sublayer Profiler
- attn/mlp coarse hook → QKV/O/Gate/Up/SiLU/Down/Norm 수준
- packing/repacking marker 추가
- **완료**: batch scaling 실패 top-2 원인이 sublayer 단위로 보임

### PR 3. VNNI/Pre-pack Hot Path Wiring
- 기존 VNNI 토대를 Qwen CPU linear hot path에 실제 연결
- load-time pre-pack cache
- shape별 dispatch log
- **완료**: 로그로 VNNI/pre-pack path 사용 확인 + seq=4 per-request cost 개선

### PR 4. Batch-aware Attention Prototype
- CPU decode attention에서 req 간 결합 가능 부분 분리
- per-seq loop 제거 prototype
- **완료**: seq=4/8에서 attention cost sublinear 증가

### PR 5. Fusion Prototype
- QKV 또는 Gate+Up 중 profiler상 더 큰 쪽부터
- input read amortization 확인
- **완료**: memory-read dominated 구간 감소

### PR 6. Routing Re-enable
- batch scaling 확인된 shape에서만 `cpu_max_num_seqs` 확대
- routing strategy 비교
- **완료**: CPU handled req 증가 + wall ratio 개선 동시

---

## Part 10 — Stop/Go 규칙 (Codex 4 Cases + Claude 통합)

### Case 1. CPU handled req 늘었지만 wall 악화
- **판정**: 실패
- **조치**: `cpu_max_num_seqs` 확대 중단. kernel/dataflow 단계로 되돌아감

### Case 2. CPU tok/s 올랐지만 tail 그대로
- **판정**: 부분 실패
- **가능 원인**: routing이 wave tail 생성 / prefill/decode 경계 직렬화 / finished/inflight accounting 지연
- **조치**: routing/gate/prefill 상태 전이 재검토

### Case 3. 단일 req만 빨라지고 batch scaling 없음
- **판정**: Ninja Gap 관점 실패
- **조치**: single-request 최적화로만 분류. CPU inflight 확대 근거로 사용 불가

### Case 4. Kernel 수정 후 metric 변화 없음
- **판정**: hot path 미타격
- **조치**: profile marker로 실제 호출 여부 확인. 다음 kernel로 넘어가지 말고 계측 복귀

---

## Part 11 — 하지 말아야 할 것 (Codex 명시 + Claude 보완)

1. batch scaling 확인 전 `cpu_max_num_seqs` 확대
2. `wave-batch`를 기본 전략으로 유지
3. NUMA/pinning bring-up 재증명에 시간 소모
4. 이미 구현된 chunked prefill off를 신규 gain으로 계산
5. VNNI 토대 존재만 보고 INT8 성능 향상으로 계산
6. CPU request 수 증가만 보고 성공 판단
7. H100x8 short burst 문제를 못 푼 상태에서 long-context 전용 해법을 mainline으로 올리기
8. 외부 논문 speedup 수치를 현재 vLLM hybrid에 직접 곱하기 (Codex 근거 사용 원칙)

---

## Part 12 — Ninja Gap 달성 시나리오 3가지 (Claude)

### 시나리오 1: "경로 1 단독 승리" (낙관적, 확률 30%)

Stage A+B+C 완료 시 cost_cpu가 현재 1/30에 근접. Ninja Gap 직접 달성.

조건:
- T-MAC LUT GEMV가 SPR에서 4× 실현
- AVX/AMX cascade가 staging overhead 없이 1.7× 실현
- batch-aware attention이 5.3× → 12× scaling 개선

위험: 하나라도 "강한 가설"이 깨지면 실패.

### 시나리오 2: "경로 1 + Spec Decode 조합" (중립, 확률 50%) ★ 권장 경로

Stage C 결과 cost_cpu × 15-20× 개선 (직접 달성엔 부족). Stage D Spec Decode가 wall 2× 추가 단축.

조건:
- Stage C가 num_seqs=4 scaling ≤1.5× 달성 (DuoDecoding balance 전제)
- Spec decode accept rate ≥70%

### 시나리오 3: "구조 변경 필요" (보수적, 확률 20%)

Stage C 후에도 cost_cpu가 GPU의 5× 이상. request-level hybrid의 구조적 상한.

조치:
- Ninja Gap 목표를 "현 workload에서" 포기
- **70B/long-context workload로 전환** — Path 2 (KV offload, ScoutAttention)가 유리한 영역
- Paper는 "negative result + 조건부 positive result"로

---

## Part 13 — 위험 관리 (Claude)

### 13-1. 기술적 위험

- **T-MAC SPR 재검증 실패** (C→D 강등): WoQ INT8 경로로 fallback, 이득 2×만 수용
- **Cascade staging overhead** (D 미승격): Cascade 포기, binary dispatch만. 이득 1.5×
- **Batch-aware attention 실패**: Head Folding으로 대체. scaling 이득 축소
- **Spec decode accept rate 낮음**: Draft model 교체 (0.3B) 또는 경로 2 포기

### 13-2. 측정 실패 대응

Tier -1 계측에서 sublayer 분해 안 되면:
- 더 세밀한 PROFILE marker 추가 (step-level → op-level)
- Intel VTune / perf 기반 profiling 별도 실험

---

## Part 14 — 추정/근거 등급

### A등급: 로컬 실측
- H100x8 wall 394/2098/14s
- RTX3090 wall 23/90/8.1/6.5s
- num_seqs별 per-req cost (Tier -1 이후 확정)

### B등급: 외부 논문 + 유사 HW 검증
- SparAMX 1.42× (Xeon SPR, 가장 가까움)
- KTransformers 21.3 TFLOPS, ISA batch>4 경계 (Xeon)

### C등급: 외부 논문 (edge/MoE/NPU 기반, 이식 시 재검증)
- T-MAC 48 tok/s (Snapdragon edge)
- T-MAN 3.1× decode (NPU)
- DuoDecoding 2.61× (벤치 공개)

### D등급: 강한 가설 (원리만 확인, 우리 환경 미검증)
- AVX/AMX cascade pipeline
- Staging cache-fit 조건
- LUT GEMV on Xeon + AMX 조합

각 단계 종료 시 D → B/A로 승격 시도. D에 머무는 기법이 Stage 3개 이상 연속 실패 시 드롭.

---

## Part 15 — 근거 자료 종합 (Codex 근거 테이블 통합)

### 15-1. 근거별 적용 수준

| 자료 | 핵심 내용 | 적용 수준 |
|---|---|---|
| H100x8 local analysis notebook | GPU-only 14.01s, hybrid seq=1 364~418s, seq=16 1994~2003s | **직접 근거** |
| H100x8 log analysis | CPU tail 구조, batch scaling 부재 확인 | **직접 근거** |
| T-MAC (EuroSys'25) | LUT 기반 low-bit CPU mpGEMM, dequant 회피 | **강한 후보** |
| T-MAN | 3-stage pipeline, fused dequant, decode/prefill dataflow | **가설적 차용** |
| KTransformers (SOSP'25) | CPU compute limit, AMX-specialized kernels, layout 최적화 | **구조적 근거** |
| SGLang + KTransformers | AMX optimized kernels, dynamic ISA switching, coordination 감소 | **구조적 근거** |
| SparAMX | AMX + unstructured sparsity, linear/attention CPU 경로 개선 | **후보** |
| NEO (MLSys'25) | CPU offload, asymmetric pipelining, load-aware scheduling | **병행/참조** |
| DuoDecoding | CPU draft + GPU target heterogeneous spec decode | **병행 트랙** |
| LUT Tensor Core (ISCA'25) | LUT 기반 tensor core 가속 | **참조** |

### 15-2. 근거별 연결되는 수정 항목

| 수정 항목 | 직접 근거 | 보조 근거 | 주의 |
|---|---|---|---|
| Batch scaling profiler | H100x8 local analysis | KTransformers 문제 정의 | 계측은 gain 아님, 방향 확정 |
| VNNI/pre-pack hot path | KTransformers, SGLang+KTransformers | 현재 `gemm_vnni.*` 부분구현 | 토대 자체는 gain 제외 |
| Head Folding | H100x8 CPU batch failure | T-MAC dataflow | 프로젝트 병목 기반 후보 |
| Batch-aware decode attn | H100x8 tail, SparAMX attn speedup | NEO attention offload | CPU attn hot path 계측 후 착수 |
| QKV / Gate-Up fusion | KTransformers layout/fusion | T-MAN fused dataflow | fused 모델 구조인지 먼저 확인 |
| LUT low-bit path | T-MAC | T-MAN | 구현 난도 높음, 강한 후보 |
| AVX/AMX dispatch | KTransformers, SGLang+KTransformers | Intel feature availability | 설정과 shape별 dispatch는 다름 |
| AVX/AMX cascade | T-MAN pipeline idea | Codex analysis | x86 직접 근거 약함, prototype 필수 |
| Routing 재평가 | H100x8 wave failure | NEO load-aware scheduling | kernel scaling 이후에만 |
| Spec decode CPU drafter | DuoDecoding | TODO A1 | mainline과 분리 |

### 15-3. 근거 사용 원칙 (Codex)

- 외부 논문의 speedup 수치를 현재 vLLM hybrid에 직접 곱하지 않음
- 이미 구현된 기능을 다시 개선 항목으로 세지 않음
- `구조적 근거`와 `직접 구현 근거` 구분
- H100x8 short burst의 직접 근거는 로컬 로그/JSON
- T-MAC/T-MAN/KTransformers/SparAMX는 "방향 가능성" 근거
- NEO/DuoDecoding은 병행 트랙 근거

---

## Part 16 — 코드 수정 위치 총괄 (Codex)

### 계측
- `vllm/v1/worker/cpu_worker.py` — sublayer hook 확장, barrier/sync marker, thread/req/token logging
- `eval/cpu_profile*.sh` — num_seqs sweep, CPU-only/hybrid 비교
- `eval/basic/H100x8/analysis_h100.ipynb` — GPU-only 대비 wall/dispatch/pinning 비교

### 라우팅
- `vllm/v1/engine/hybrid_core.py` — default strategy, cpu_max_num_seqs, wave-batch, throughput-adaptive, CPU config
- `vllm/v1/engine/core_client.py` — dispatch/finished accounting, throughput feedback

### CPU hot path
- `vllm/v1/attention/backends/cpu_attn.py` — batch-aware decode attention, IPEX vs custom 분기
- `csrc/cpu/*` — VNNI pre-pack, fusion kernel, LUT path, AVX/AMX dispatch, cascade prototype
- `vllm/v1/worker/cpu_model_runner.py` — model load-time optimize/pre-pack hook, NUMA-local allocator

---

## Part 17 — 최종 명제

### Claude 결론:

> Ninja Gap은 달성 가능하다. 단 **경로 1 (CPU 자체 가속)의 핵심 5개 기법 (ISA + Fusion + LUT + Cascade + Pre-pack)** 이 누적 15-20× 개선을 만들고, **경로 2 (Spec Decode)** 가 추가로 wall 2×를 가져오는 조합이 가장 현실적. 시작은 계측, 승패는 Stage C (Big Wins), 보험은 Stage D (Spec Decode).
>
> 경로 1 단독 승리 시나리오는 낙관적이지만 불가능 아님. 경로 2 단독은 Spec Decode의 balance 조건 때문에 경로 1 진전 없이는 불가.
>
> 현 workload (7B × 128/128)에서 Ninja Gap 불가판정 시 **70B/long-context workload로 전환**하여 구조적 우위 영역에서 프로젝트 성과 확보.

### Codex 결론:

> 현재 문제는 CPU에 request를 너무 적게 준 것이 아니라, 많이 줬을 때 계산이 싸지지 않는 것입니다. Ninja Gap은 CPU inflight 확대가 아니라, inflight 확대가 실제 data reuse와 kernel efficiency로 바뀌는 순간에 생깁니다.

### SuperSet 통합 결론:

> **순서가 전부다.** 계측(Tier -1) → batch scaling 확인(Tier 1-2) → kernel 투자(Stage B/C) → routing 재활성(Stage D). 이 순서를 건너뛰면 어떤 기법도 tail amplification만 만든다. 경로 1 + 경로 2 조합이 가장 현실적(50% 확률)이며, 경로 1의 batch scaling이 경로 2의 전제조건이다.

---

## 참고 문서

- claude 원칙 v4: `20260415_092738_claude_HPC_breakthrough_principles_v4.md`
- codex 통합: `20260415_085858_codex_hybrid_improvement_integrated_rewrite.md`
- 실측: `eval/basic/H100x8/`, `eval/basic/RTX3090/`
- T-MAC EuroSys'25: https://arxiv.org/pdf/2407.00088
- KTransformers SOSP'25: https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf
- DuoDecoding 2025-03: https://arxiv.org/abs/2503.00784
- SparAMX: https://huggingface.co/papers/2502.12444
- T-MAN: https://arxiv.org/html/2511.11248v1
- NEO MLSys'25: https://openreview.net/forum?id=umgy9tWBLA
- LUT Tensor Core ISCA'25: https://dl.acm.org/doi/10.1145/3695053.3731057
