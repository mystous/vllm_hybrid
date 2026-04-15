# Ninja Gap 달성을 위한 종합 수정 방안

**작성**: 2026-04-15 09:41 KST (Claude)
**종합 대상**:
- `20260415_092738_claude_HPC_breakthrough_principles_v4.md` (claude 원칙 v4)
- `20260415_085858_codex_hybrid_improvement_integrated_rewrite.md` (codex 통합)
- 외부 논문/프레임워크 (WebSearch 검증)
- 실측 (H100x8 + RTX3090)

---

## Part 0 — Ninja Gap 재정의

### 0-1. 정의

**Ninja Gap**: `T_hybrid < T_gpu_only` 달성. 즉 **GPU 단독보다 CPU 추가가 빠름**.

현재 상태 (H100x8, 7B, 500×128/128):
- `T_gpu_only` = **14s** (3.77s duration + 오버헤드)
- `T_hybrid_best` = **394s** (max_seqs=1, threads=32, 2 NUMA)
- **격차 28×** — Ninja Gap 도달 위해 CPU tail 이 GPU bulk 완료 시점 이전에 끝나야

### 0-2. 구조적 방정식

```
T_hybrid = max(T_gpu_completion, T_cpu_tail)
         = max(N_gpu × cost_gpu / batch_gpu ,
               N_cpu × cost_cpu × f_scaling(N_cpu))
```

Ninja Gap 조건:
```
T_hybrid < T_gpu_only = (N_gpu + N_cpu) × cost_gpu / batch_gpu
```

`N_cpu > 0` 이려면 **CPU 경로 추가가 GPU 에서 뺀 양보다 wall 을 더 줄여야**.

### 0-3. 두 가지 달성 경로

**경로 1**: `cost_cpu` 를 GPU 에 가깝게 줄임 (CPU 자체 가속)
- 목표: cost_cpu 를 현재의 1/30 로
- 수단: LUT, fusion, cascade, pre-pack, sparse, WoQ

**경로 2**: CPU 를 동일 request 에서 "다른 일" 을 하게 — batch scaling 곡선을 바꿈
- 목표: 공식 자체 변경
- 수단: spec decode (draft), P/D disagg (prefill-only), KV offload, ScoutAttention

두 경로는 **배타적 아님**. 실제 Ninja Gap 은 **경로 1 + 경로 2 의 누적 효과** 로 올 가능성.

### 0-4. codex 의 핵심 관찰

> 현재 구조적 실패 모드는 "CPU 가 느리다" 가 아니라 **"num_seqs 증가에도 per-req cost 가 안 내려감 (batch scaling 실패)"**.

이 관찰이 설계 순서를 뒤바꿈: **Ninja Gap 전에 batch scaling 먼저**. batch scaling 없으면 어떤 가속도 tail 로만 남음.

---

## Part 1 — 실패 모델 (왜 현재 구조가 Ninja Gap 에 도달 못 하나)

### 1-1. 3겹 실패

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

### 1-2. 현재 달성 가능 이론 상한

dev 에서 T-MAC INT4 포팅 시 이론상 (선형 누적 가정):
- Baseline 현재: 3079ms/step
- × WoQ INT8 (2×) × Huge Pages (1.1×) × ISA cascade (2.22×) × LUT GEMV (4×) × Fusion (1.5×) × Pre-pack (1.15×) × Sparse (1.42×) = **~60×** → **~50ms/step**

50ms/step × 128 tokens = 6.4s CPU tail. gpu_only 14s 보다 짧음 → **Ninja Gap 이론상 도달 가능**.

**단 diminishing returns, scatter/sync 오버헤드 미반영**. 실제는 10-20× 구간에서 정체 예상 → CPU tail 150-300ms 정도로 감소 → **여전히 Ninja Gap**.

**결론**: 경로 1 단독으로도 이론상 Ninja Gap 가능. 단 **batch scaling 이 전제**.

---

## Part 2 — 경로 1. CPU 자체 가속 스택 (ROI 순 정렬)

각 기법: **메커니즘 / 예상 이득 / 구현 비용 / 위험 / 스택 호환성**.

### 2-1. [Tier 0] Huge Pages 1GB

| | |
|---|---|
| **메커니즘** | 4KB 페이지 → 1GB 페이지. TLB 엔트리 70B INT4 기준 900만 → 35개. TLB miss 해소. |
| **예상 이득** | **5-15%** (논문 수치). decode 전반에 균등 영향. |
| **비용** | 0.5일. grub `hugepagesz=1G hugepages=40` + vLLM mmap flag `MAP_HUGETLB`. |
| **위험** | 컨테이너 cgroup 설정 필요. dev 머신에서 먼저 검증. |
| **스택 호환성** | 모든 후속 기법과 독립. 항상 깔아야. |
| **근거** | 70B INT4 TLB 분석 (ideation 0950_cpu_llm_optimization_techniques §3.2) |

### 2-2. [Tier 0] IPEX WoQ INT8

| | |
|---|---|
| **메커니즘** | BF16 weight → INT8 저장, BF16 연산. weight memory 2× 절감. |
| **예상 이득** | **2× decode throughput** (memory-bound). PPL 열화 <0.5 |
| **비용** | 2-3일. `cpu_worker.py` 의 `ipex.llm.optimize` 에 `quantization_config=qconfig` 추가. |
| **위험** | IPEX WoQ 가 vLLM Hybrid 의 모델 로딩 경로와 호환되는지 미검증. |
| **스택 호환성** | LUT INT4 (§2-8) 로 넘어가면 **대체됨**. Tier 0 임시. |
| **근거** | Intel ICML'24 workshop (arXiv 2407.07304) |

### 2-3. [Tier 0] OMP 환경 + Memory Pinning

| | |
|---|---|
| **메커니즘** | `OMP_PROC_BIND=close`, `OMP_PLACES=cores`, `KMP_BLOCKTIME=0`, `numactl --membind=strict`. OS scheduler migration 금지. |
| **예상 이득** | **5-10%** (이미 대부분 설정돼 있음). |
| **비용** | 1일. 현재 env 검증 + 누락분 추가. |
| **위험** | 낮음. |
| **스택 호환성** | 독립. 모든 Tier 기반. |

### 2-4. [Tier 1] ISA Binary Dispatch

| | |
|---|---|
| **메커니즘** | batch size > 4 → AMX, else → AVX-512 VNNI. KTransformers 방식. |
| **예상 이득** | decode **1.5-2.22×** (KTransformers 실측). |
| **비용** | 1주. `cpu_worker.execute_model` pre-dispatch + csrc kernel 등록. |
| **위험** | IPEX 내부 dispatcher 와 충돌 가능. Bypass 필요. |
| **스택 호환성** | § 2-9 cascade 의 전제. |
| **근거** | [KTransformers AMX doc](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/AMX.md) |

### 2-5. [Tier 1] Kernel Fusion — QKV concat + Gate/Up interleave + Residual+Norm

| | |
|---|---|
| **메커니즘** | sublayer 8개 독립 kernel → 4개 묶음 kernel. 중간 DDR write 제거, 입력 x 단일 로드. |
| **예상 이득** | **1.5-2×** (SGLang SiLU+up 12% × 4 sublayer 누적). |
| **비용** | 2주. `csrc/cpu/fused_qkv.cpp`, `fused_gate_up_silu_down.cpp`, `fused_add_rmsnorm.cpp`. |
| **위험** | IPEX 의 기존 fused kernel 과 충돌 검증 필요. |
| **스택 호환성** | § 2-8 LUT / § 2-9 cascade 와 독립. 항상 병행. |
| **근거** | SGLang CPU 백엔드 블로그 (12% 실측), T-MAC 의 Gate+Up interleave 설계 |

### 2-6. [Tier 1] Softmax + SiLU LUT 대체

| | |
|---|---|
| **메커니즘** | `exp()` 20 cycles → `vpshufb` LUT 1 cycle. SiLU는 "hot range" 선형 근사 + LUT. |
| **예상 이득** | Softmax **2.2×**, SiLU **1.2×** (TARDIS 로는 vLLM 1.6× 보고). |
| **비용** | 2주. `csrc/cpu/lut_ops.cpp`. 32B/512B LUT register 상주. |
| **위험** | 근사로 인한 정확도 열화 미검증 (< 2% 추정). |
| **스택 호환성** | § 2-8 GEMV LUT 와 동일 인프라. 같이 개발. |
| **근거** | T-MAN (arXiv 2511.11248), TARDIS (arXiv 2501.10054) |

### 2-7. [Tier 1] Head Folding (GEMV → GEMM)

| | |
|---|---|
| **메커니즘** | decode attention 의 M=1 GEMV 를 batch fold 해서 M=16 GEMM 으로. AMX tile full 활용. |
| **예상 이득** | decode attention **1.5-2×** (SGLang blog). |
| **비용** | 2주. `csrc/cpu/fold_attention.cpp` + IPEX single_query 대체. GQA 구조 반영. |
| **위험** | MLA (DeepSeek) 에서는 직접 적용. GQA (Qwen) 에서는 batch fold 변형 필요. |
| **스택 호환성** | § 2-12 batch-aware attention 의 한 방식. |
| **근거** | SGLang Head Folding blog |

### 2-8. [Tier 2] T-MAC LUT-Based GEMV (INT4 핵심)

| | |
|---|---|
| **메커니즘** | INT4 weight 16 값 × input 을 LUT 32B 에 precompute. 곱셈 + 역양자화를 `vpshufb` 1-cycle 로. |
| **예상 이득** | INT4 **4×** (T-MAC 실측, CPU 22 tok/s > NPU 10.4 tok/s). bit↓ 선형 가속. |
| **비용** | 3-4주. `csrc/cpu/lut_gemv.cpp` 전용 kernel. IPEX bypass. |
| **위험** | ⚠ T-MAC 은 edge CPU 검증 (Snapdragon). SPR+AMX 조합 재검증 필요. 강한 가설. |
| **스택 호환성** | § 2-2 WoQ INT8 대체. § 2-6 LUT Softmax 와 동일 인프라. |
| **근거** | [T-MAC EuroSys'25](https://arxiv.org/pdf/2407.00088), [github](https://github.com/microsoft/T-MAC/) |

### 2-9. [Tier 2] AVX/AMX Cascade Pipeline (codex §2-5)

| | |
|---|---|
| **메커니즘** | tile k+2 load (prefetch/DSA) / tile k+1 dequant·pack (AVX-512) / tile k matmul (AMX) 3-stage 동시 실행. NPU T-MAN 의 DMA+Vec+Mat 패턴. |
| **예상 이득** | **1.5-3×** (T-MAN NPU 실측 decode 3.1×. CPU 이식 시 보수적 추정 1.5-2×). |
| **비용** | 4주. 타일 버퍼 설계 + cache-fit 검증. AVX `zmm` ↔ AMX tile 별도 파일이라 **중간 버퍼 L2 상주 설계** 필수. |
| **위험** | ⚠ Staging overhead 가 이득을 상쇄할 수 있음. Shape 별 측정 필수. codex §2-5 경고: "잘못 설계하면 중간 write/read 비용만 늘 수 있다". |
| **스택 호환성** | § 2-4 binary dispatch 의 발전형. § 2-10 pre-pack 과 조합 필수. |
| **근거** | codex §2-5, T-MAN (3-stage 원리 증명) |

### 2-10. [Tier 2] AMX Weight Pre-pack

| | |
|---|---|
| **메커니즘** | 모델 로드 시 1회 weight 를 AMX tile layout (16×64 byte) 로 재배치. 런타임 tileloadd 가 연속 16 cache line 로드. |
| **예상 이득** | **1.1-1.2×** (KTransformers 실측 10-20%). |
| **비용** | 1주. CPUWorker `load_model` 후 hook. layer 단위 변환. |
| **위험** | 낮음. 메모리 부담 2× (원본 + 재배치). |
| **스택 호환성** | § 2-9 cascade 의 전제. LUT path 에도 유사 pre-pack 필요 (T-MAC 의 group layout). |

### 2-11. [Tier 2] AVX-512 Bitmask Sparse (SparAMX 기반)

| | |
|---|---|
| **메커니즘** | Unstructured sparsity 를 zero-pad 없이 `K` 레지스터 64-bit mask 로 표현. `_mm512_mask_fmadd_ps` 로 유효 원소만 계산. AMX 도 mask 확장해 희소 GEMM. |
| **예상 이득** | linear **1.42×**, attention **1.14×** (SparAMX 실측, Xeon SPR). |
| **비용** | 4주. `csrc/cpu/sparse_amx.cpp`. 가중치 50% 프루닝 필요 (사전 작업). |
| **위험** | 중간. 프루닝 후 PPL 열화 검증 필요. MLP 희소성은 batch↑ 시 소멸 (Polar Sparsity) — attention head sparsity 만 batch-invariant. |
| **스택 호환성** | LUT 과 별개 경로. 대체가 아닌 추가. |
| **근거** | [SparAMX](https://huggingface.co/papers/2502.12444) |

### 2-12. [Tier 2] Batch-aware Decode Attention

| | |
|---|---|
| **메커니즘** | IPEX `single_query_cached_kv_attention` 의 per-seq KV paged access 구조를 batch 단위로 재구성. head-parallel + page-coalesced. |
| **예상 이득** | batch=16 scaling을 5.3× → **10-12×** 로 개선 (목표). |
| **비용** | 4주. `cpu_attn.py` 의 IPEX call 대체 + 새 kernel. 가장 복잡. |
| **위험** | 높음. IPEX 내부 FD kernel 의 재구현에 해당. |
| **스택 호환성** | § 2-7 Head Folding 과 중복 영역 있음. 하나 선택 or 통합. |
| **근거** | H100x8 H2 실측 재앙 (2098s), codex 4-2 #3 |

### 2-13. [Tier 3] Core Group Pipeline (Systolic)

| | |
|---|---|
| **메커니즘** | 56 core 를 4 group 으로 분할. A: QKV, B: Attn, C: MLP, D: next layer QKV 파이프라인. L3 로 inter-group 전달. |
| **예상 이득** | **2-3× latency** (4 layer 동시 실행). GPU SM cluster 원리를 CPU L3 로. |
| **비용** | 6주+. scheduler 재설계, worker 분리, L3 버퍼 설계. 매우 복잡. |
| **위험** | 매우 높음. L3 BW 가 DDR 보다 높지만 coherence 비용 큼. |
| **스택 호환성** | Tier 2 완료 후. 기반 kernel 이 fast 해야 이득 보임. |
| **근거** | Eyeriss systolic (원리), NPU multi-core pipeline |

### 2-14. 경로 1 스택 누적 예상 이득 (이론 상한)

순차 적용 시 (diminishing returns 50% 가정):

| 기법 | 단독 이득 | 누적 (50% efficiency) |
|---|---:|---:|
| Baseline | 1× | 1× |
| + Huge Pages (2-1) | 1.1× | 1.1× |
| + WoQ INT8 (2-2) | 2.0× | 2.1× |
| + OMP env (2-3) | 1.05× | 2.2× |
| + ISA binary (2-4) | 2.0× | **3.3×** (G1 진입) |
| + Fusion (2-5) | 1.7× | 4.7× |
| + LUT ops (2-6) | 1.3× | 5.7× |
| + Head Folding (2-7) | 1.5× | **7.4×** |
| + LUT GEMV INT4 (2-8) | 3.0× | 13× (WoQ 대체) |
| + Cascade (2-9) | 1.7× | **19×** (G2 진입) |
| + Pre-pack (2-10) | 1.15× | 21× |
| + Sparse (2-11) | 1.35× | 27× |
| + Batch-aware Attn (2-12) | 1.5× | **35×** (G3 Ninja Gap) |
| + Systolic (2-13) | 2× | 70× (overshoot) |

**현재 cost_cpu / cost_gpu ≈ 28×**. 경로 1 단독으로 **28× 역전 이론상 가능**. 실제는 30% 효율 가정 시 10-20× 구간 예상 → **Ninja Gap 아슬아슬 or 미달**.

→ **경로 2 와 조합 필요**.

---

## Part 3 — 경로 2. 역할 재정의 (구조 변경)

### 3-1. Spec Decode CPU Drafter (DuoDecoding 방식)

| | |
|---|---|
| **메커니즘** | CPU 가 작은 drafter (Qwen2.5-0.5B) 로 k 토큰 생성 → GPU verifier (7B) 가 한 번에 검증. accept rate ~70% 이면 k-1 토큰 free. |
| **예상 이득** | TPOT **2.1-2.61×** (DuoDecoding 실측). TTFT **17% 감소**. |
| **비용** | 6주. 세 번째 EngineCore (drafter) + verifier 동기화 + accept/reject 로직. |
| **위험** | ⚠ **CPU drafter 속도가 GPU verifier 속도와 balance 조건 충족 여부** (DuoDecoding 전제). 경로 1 필수 선행. |
| **Ninja Gap 기여도** | 매우 큼. wall 공식 변경 — `max` 의 CPU term 이 "전체 처리" 가 아닌 "draft 만" 이 되므로 tail 소멸. |
| **근거** | [DuoDecoding 2503.00784](https://arxiv.org/abs/2503.00784) |

### 3-2. P/D Disaggregation (장거리 context 한정)

| | |
|---|---|
| **메커니즘** | prefill 은 CPU (AMX BF16 compute-bound), decode 는 GPU. long-context 16K+ 에서 GPU prefill bottleneck 해소. |
| **예상 이득** | 16K input 에서 GPU TPOT p99 개선 (KV handoff 비용 제외). |
| **비용** | 8주. `vllm/engine/disaggregated/` stub 활용 + hybrid 구조 병합 + KV DMA. |
| **위험** | 현재 workload (128/128) 에 무효. 16K+ 에서만 의미. |
| **Ninja Gap 기여도** | 현 workload 에서 0. long-ctx 전용. |

### 3-3. KV Cache CPU Tier Offload

| | |
|---|---|
| **메커니즘** | PagedAttention block_table 에 tier 필드. hot → HBM, cold → CPU DRAM. Eviction LRU + DMA prefetch. |
| **예상 이득** | 동시 시퀀스 **3×**, throughput **2-3×** (70B / batch 1500+ 에서). |
| **비용** | 6주. `vllm/v1/core/kv_cache_manager.py` + DMA stream 분리. |
| **위험** | PCIe 지연. InfiniGen 스타일 predictive prefetch 필요. |
| **Ninja Gap 기여도** | 7B 현 workload 에서 0. 70B 에서 큼. |

### 3-4. ScoutAttention Layer-Ahead

| | |
|---|---|
| **메커니즘** | CPU 가 1 layer 앞서 Q 예측 (Q_{i+1} ≈ Q_i, cos sim 0.93+) → top-k KV block 선별 → partial attention. GPU 는 hot block 만. 결과 합산. |
| **예상 이득** | decoding **5.1×** (ScoutAttention 실측, 장문). GPU idle 57% → <5%. |
| **비용** | 11주. 가장 복잡. cos sim 검증 먼저 (1일). |
| **위험** | 근사 attention PPL 열화 <2.1%. vLLM 포팅 대규모. |
| **Ninja Gap 기여도** | 현 workload 제한적. 8K+ context 에서 의미 큼. |

### 3-5. NEO Asymmetric Batch Split

| | |
|---|---|
| **메커니즘** | 매 decode step batch 를 Batch-0 (GPU attn) + Batch-1 (CPU attn) 분할. GPU linear 실행 중 CPU attn overlap. |
| **예상 이득** | H100 70B **14.3%** (MLSys'25 실측). 작은 workload 에서는 축소. |
| **비용** | 8주. `hybrid_core.py` 에 `_split_batch_asymmetric` + CPU worker 가 "attention 전용 워커" 로 역할 재정의. |
| **위험** | GPU HBM 여유 있는 환경 (현재 우리) 에선 이득 제한. NEO 논문도 70B+ 전제. |
| **Ninja Gap 기여도** | 7B 에선 제한적. |

---

## Part 4 — 합리적 실행 순서 (Ninja Gap 달성 조건부 로드맵)

### 4-1. Stage A: 계측 + Quick Wins (1-2주)

**필수**: Tier -1 계측 (num_seqs sweep + sublayer 분해 + cache hit ratio)

**Quick Wins (ROI 가장 높음)**:
- 2-1 Huge Pages (0.5일)
- 2-3 OMP env (1일)
- 2-2 WoQ INT8 (3일)

**기대**: baseline × 2-3×. **G1 진입 (ratio <15×)** 확률 높음.

### 4-2. Stage B: Kernel 투자 (4-6주)

**목표**: 경로 1 에서 Ninja Gap 이론 상한 25-35× 에 도달하기 위한 기법들.

**우선순위**:
1. 2-4 ISA Binary Dispatch (1주) — AVX-512 decode path 기반
2. 2-5 Kernel Fusion (2주) — sublayer 묶음
3. 2-6 Softmax/SiLU LUT (2주) — T-MAC LUT 인프라 1차
4. 2-7 Head Folding (2주) — attention batch 문제 부분 해소

**기대**: Stage A 위에 × 3-5× 더. **G2 진입 (ratio <1.5×)** 조건부 가능.

### 4-3. Stage C: Big Wins (3-4주)

**목표**: 이론 상한 돌파.

**핵심**:
1. 2-8 T-MAC LUT GEMV INT4 (3-4주) — 가장 큰 단일 이득
2. 2-10 AMX Pre-pack (1주, 병행)
3. 2-9 AVX/AMX Cascade (4주, LUT 경로와 통합)

**기대**: G2 → G3 돌파 가능. **이 단계에서 Ninja Gap 첫 달성** 기대.

**실패 조건** (codex §4-5 Stop/Go 3 cases):
- 경우 2 (속도 개선 없음) → Stack 재검토, cache-fit 원인 추적
- 경우 3 (CPU handled↑ but wall↓) → batch scaling 재검토

### 4-4. Stage D: Path 2 결합 (6-12주, 병행 트랙)

Stage C 와 병행 (별도 팀/시간).

**선순위**:
1. 3-1 Spec Decode CPU Drafter — Stage C 결과가 CPU draft throughput 충분하면 가장 빠른 Ninja Gap 2차 레버
2. 경로 1 만으로 Ninja Gap 실패 시 Plan B

**병행 이유**: 경로 1 만으로 Ninja Gap 실패 시 즉시 전환 가능해야. Stage C 종료 후 시작하면 2개월 지연.

### 4-5. Stage E: 장거리 트랙 (별도)

**조건**: 70B / long-context workload 가 본 프로젝트의 next target 이 될 때.

- 3-2 P/D disagg
- 3-3 KV offload
- 3-4 ScoutAttention
- 2-13 Core Group Pipeline
- 3-5 NEO asymmetric

현 workload 에선 우선순위 최하.

---

## Part 5 — Ninja Gap 달성 시나리오 3가지

### 시나리오 1: "경로 1 단독 승리" (낙관적, 확률 30%)

Stage A+B+C 완료 시 cost_cpu 가 현재 1/30 에 근접. Ninja Gap 직접 달성.

**조건**:
- T-MAC LUT GEMV 가 SPR 에서 4× 실현 (Snapdragon edge 에서 그랬듯)
- AVX/AMX cascade 가 staging overhead 없이 1.7× 실현
- batch-aware attention 이 5.3× → 12× scaling 개선

**위험**: 하나라도 "강한 가설" 이 깨지면 실패. 실측 전엔 확률 낮음.

### 시나리오 2: "경로 1 + Spec Decode 조합" (중립, 확률 50%)

Stage C 결과 cost_cpu × 15-20× 개선 (Ninja Gap 직접 달성엔 부족).
Stage D 의 Spec Decode 가 추가로 wall 2× 단축 → Ninja Gap 달성.

**조건**:
- Stage C 가 num_seqs=4 scaling ≤1.5× 달성 (DuoDecoding 의 balance 전제 충족)
- Spec decode accept rate ≥70% (공통 workload 에서 보고된 수치)

**가장 현실적**. 본 문서 권장 경로.

### 시나리오 3: "구조 변경 필요" (보수적, 확률 20%)

Stage C 후에도 cost_cpu 가 현 workload 에서 GPU 의 5× 이상. request-level hybrid 의 구조적 상한.

**조치**:
- Ninja Gap 목표를 "현 workload 에서" 포기
- **70B / long-context workload 로 전환** — 거기서 Path 2 (3-3 KV offload, 3-4 ScoutAttention) 가 Ninja Gap 달성
- Paper 는 "현 workload 의 negative result + 조건부 positive result" 로

---

## Part 6 — 성공 검증 기준 (codex §4-5 Stop/Go 통합)

Stage 종료 시마다 4축 동시 확인:

1. CPU handled requests 증가 (router dispatch 분포)
2. CPU batch tok/s 증가 (num_seqs sweep)
3. Tail 감소 (GPU bulk 완료 후 CPU 마지막까지 걸린 시간)
4. Wall ratio 개선 (hybrid wall / gpu_only wall)

**4 모두 개선 시** 다음 Stage. **1-3 개선 시 성공 아님** (codex 경고):
- 1 만 (속도): routing 문제 의심
- 2 만 (batch scaling): 기법은 돌았으나 wall 영향 없음, cache-fit 재검토
- 3 만 (tail): 단순 운 좋은 routing, 본질 해결 아님

---

## Part 7 — 추정/근거 등급 (Part H 통합)

### A등급: 로컬 실측
- H100x8 wall 394/2098/14s
- RTX3090 wall 23/90/8.1/6.5s
- num_seqs 별 per-req cost (Tier -1 이후 확정)

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

각 단계 종료 시 D → B/A 로 승격 시도. D 에 머무는 기법이 Stage 3개 이상 연속 실패 시 드롭.

---

## Part 8 — 실행 권고 요약

**0-2주**: Stage A (Tier -1 계측 + Quick Wins)
- 필수 측정 5종 + Huge Pages + WoQ INT8 + OMP env
- 결과로 G0 baseline 확정, G1 도달 여부 확인

**2-8주**: Stage B (Kernel 투자)
- ISA binary + Fusion + Softmax LUT + Head Folding
- G2 도달 여부 확인. 도달 시 Stage C 직행.

**8-12주**: Stage C (Big Wins)
- T-MAC LUT GEMV + Pre-pack + AVX/AMX Cascade
- **Ninja Gap 1차 도전** 지점

**병행 (2-8주)**: Stage D (Spec Decode) 설계 시작
- Stage C 결과에 따라 즉시 실험 가능

**12주+**: Stage C/D 결과 평가
- 시나리오 1/2/3 중 어느 경로인지 판정
- 시나리오 3 라면 long-context workload 로 전환

---

## Part 9 — 위험 관리

### 9-1. 기술적 위험

- **T-MAC SPR 재검증 실패** (C→D 강등): WoQ INT8 경로로 fallback, 이득 2× 만 수용
- **Cascade staging overhead** (D 미승격): Cascade 포기하고 binary dispatch 만. 이득 1.5× 만
- **Batch-aware attention 실패**: Head Folding 으로 대체. scaling 이득 축소
- **Spec decode accept rate 낮음**: Draft model 교체 (더 작은 0.3B) 또는 경로 2 포기

### 9-2. 우선순위 충돌

codex 권고 "역할 재정의 메인라인 앞세우지 말라" vs 본 문서 "Stage D 병행 추진".
- 해석: codex 는 **Stage C 결과 확인 전 Spec Decode 에 리소스 집중 금지** 로 읽음
- 본 문서 준수: Stage D 는 **설계만 병행**, 구현 리소스는 Stage C 에 집중. Stage C 종료 후 Stage D 구현 본격.

### 9-3. 측정 실패 대응

Tier -1 계측에서 sublayer 분해가 안 되면:
- 더 세밀한 PROFILE marker 추가 (step-level → op-level)
- Intel VTune / perf 기반 profiling 별도 실험

---

## Part 10 — 최종 명제

본 문서의 결론:

> Ninja Gap 은 달성 가능하다. 단 **경로 1 (CPU 자체 가속) 의 핵심 5개 기법 (ISA + Fusion + LUT + Cascade + Pre-pack)** 이 누적 15-20× 개선을 만들고, **경로 2 (Spec Decode)** 가 추가로 wall 2× 를 가져오는 조합이 가장 현실적. 시작은 계측, 승패는 Stage C (Big Wins), 보험은 Stage D (Spec Decode).
>
> 경로 1 단독 승리 시나리오는 낙관적이지만 불가능 아님. 경로 2 단독은 Spec Decode 의 balance 조건 때문에 경로 1 진전 없이는 불가.
>
> 현 workload (7B × 128/128) 에서 Ninja Gap 이 불가판정되면 **70B/long-context workload 로 전환**하여 구조적 우위 (path 2 의 KV offload, ScoutAttention) 가 유리한 영역 에서 프로젝트 성과 확보.

---

## 참고 (v4, codex, 실측, 외부)

- claude 원칙 v4: `20260415_092738_claude_HPC_breakthrough_principles_v4.md`
- codex 통합: `20260415_085858_codex_hybrid_improvement_integrated_rewrite.md`
- 실측: `eval/basic/H100x8/20260414_213434_claude_*.md`, `eval/basic/RTX3090/20260414_220000_claude_*.md`
- 외부 (WebSearch 검증):
  - T-MAC EuroSys'25: https://arxiv.org/pdf/2407.00088
  - KTransformers SOSP'25: https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf
  - DuoDecoding 2025-03: https://arxiv.org/abs/2503.00784
  - SparAMX: https://huggingface.co/papers/2502.12444
  - T-MAN: https://arxiv.org/html/2511.11248v1
  - NEO MLSys'25: https://openreview.net/forum?id=umgy9tWBLA
  - LUT Tensor Core ISCA'25: https://dl.acm.org/doi/10.1145/3695053.3731057
