# N — cdec leftover 제거 외부 idea 정리 (논문 + GitHub source)

> 작성: 2026-05-21 KST (turn 10).
> 사용자 명시: "cdec leftover 를 제거 해야 할 것 같다. 이걸 어떻게 해결해야 할지 아이디어를 다양한 논문, GitHub source 들을 키워드로 찾아서 제시해봐".
> 자매 doc: [`M_sub015_phase3_hpc_optimization.md`](M_sub015_phase3_hpc_optimization.md), [`../measurements/timeline_neo_amx_apply_20260520/README.md`](../measurements/timeline_neo_amx_apply_20260520/README.md) §A (AMX 최적화 plan).

---

## 0. Problem definition (recap)

NEO architecture sub-batch 분할:
- **b0** = GPU full forward (preproj + attn (flash_attn) + postproj + MLP + AllReduce)
- **b1** = CPU cdec dispatch (preproj + **attn SKIP → paged_attention_cpu** + postproj + MLP + AllReduce)

**cdec leftover (+18 ms wall)** 의 발생 원인:
- main thread 가 `paged_attention_cpu` direct call (S5)
- main C++ blocking 중 GIL release → GPU stream queue 자동 진행
- 단 long context (500p × 8192) 영역에서 per-layer CPU pacpu (~2.3 ms) > GPU concurrent work (~0.4 ms) → 차이 누적
- 80 layer 누적 → step end 영역에서 **GPU IDLE 의 wall path 누적 +18 ms**

**cdec leftover 내부 cycle 분해** (perf record 2026-05-17):

| Layer | cycle % | wall 환산 (② 의 비율) |
|---|---:|---:|
| libgomp (OMP barrier wait) | **43.75%** | **+11.2 ms (62% of ②) ★★★** |
| libpacpu (ISPC compute) | 26.38% | +6.8 ms (38% of ②) |
| ⊳ softmax | 9.73% | +2.5 ms |
| ⊳ qk_product | 8.75% | +2.3 ms |
| ⊳ av_product | 7.90% | +2.0 ms |

**기존 시도 (모두 net loss / 회귀)**:
- P4 async cdec (depth=1 deque) → OOB race, 이전 unstable. 현재 OOB stability 영역 적용 후 stable 단 +0.05% noise
- P3 K BF16 + AMX (M=8 partial tile 50% occupancy) → -2.5% 회귀 (이전), 현재 net zero
- F6 OMP dynamic schedule → -1.4% 회귀 (atomic counter overhead)
- Step 5 AMX optimization (B+A+vec K) → -2.35% 회귀

---

## 1. CPU-GPU asymmetric pipelining (NEO 외 architecture) — 8 idea

### 1.1 OmniServe — CPU-GPU Attention Piggybacking

| 항목 | 값 |
|---|---|
| 출처 | Mo et al., *"Serving Hybrid LLM Loads with SLO Guarantees Using CPU-GPU Attention Piggybacking"*, [arXiv 2603.12831](https://arxiv.org/html/2603.12831v2) (2026-03, U Macau) |
| mechanism | BE service 의 Attention 만 CPU offload, **CPU↔GPU stream 간 비동기 통신 + late merge** 로 partial result aggregate. GPU stream 이 CPU result 대기로 blocking 되지 않음. |
| NEO 적용 영역 | 현재 NEO 는 main thread direct call 이라 CPU 완료까지 wall 위 누적. **non-blocking CPU completion + late merge** 채용 시 step tail (18 ms) 직접 단축 |
| effort | medium (async + LSE merge 구조 + race-safe completion) |
| risk | LSE merge 정확도 / step boundary cross-stream sync overhead |

### 1.2 APEX — Asynchronous Parallel CPU-GPU Execution

| 항목 | 값 |
|---|---|
| 출처 | *"APEX: Asynchronous Parallel CPU-GPU Execution for Online LLM Inference"*, [arXiv 2506.03296](https://arxiv.org/abs/2506.03296) (2025-06, rev. 2026-01) |
| mechanism | **batch splitting 없이** CPU-offloaded decode 와 GPU 작업 overlap. CPU/GPU execution time predictor 로 dynamic dispatch. |
| NEO 적용 영역 | NEO 의 b0/b1 split 영역과 다른 paradigm — **predictor 만 NEO load-aware scheduler 에 통합** 시 cdec tail = 0 수렴 |
| effort | medium-large (predictor + scheduler 통합) |
| risk | prediction error → wrong dispatch / NEO load-aware 와 conflict |

### 1.3 KTransformers — Async CPU-GPU task scheduling (AMX-specialized)

| 항목 | 값 |
|---|---|
| 출처 | Chen et al., SOSP 2025 ([PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)). GitHub [`kvcache-ai/ktransformers`](https://github.com/kvcache-ai/ktransformers). SGLang 통합 ([LMSYS blog](https://lmsys.org/blog/2025-10-22-KTransformers/)) |
| mechanism | **AMX-specialized kernel + asynchronous CPU-GPU task scheduling** 으로 sync overhead 최소화. DeepSpeed-V3/R1 671B 에서 prefill 4.62-19.74×, decode 1.25-4.09× 가속. |
| NEO 적용 영역 | (a) AMX kernel reference (P3 회귀 복구) (b) async scheduler 구조를 cdec dispatch 에 적용 |
| effort | medium (kernel 차용) / large (scheduler 통합) |
| risk | NEO 의 per-head paged decode vs KTransformers 의 MoE expert — ARI pattern 차이 |

### 1.4 HGCA — Hybrid GPU-CPU Attention (per-head sparse on CPU)

| 항목 | 값 |
|---|---|
| 출처 | Deng & Yang, [arXiv 2507.03153](https://arxiv.org/abs/2507.03153) (UT Arlington, 2025-07) |
| mechanism | 최근 token 은 GPU dense, 그 외 salient KV 만 CPU **sparse parallel attention** + **log-sum-exp fusion** merge. fine-grained per-head sparsification. |
| NEO 적용 영역 | NEO b1 의 CPU pacpu compute 38% 영역 자체를 **sparse compute 로 축소** → 6.8 ms → 더 작게. + LSE fusion (#1.1 와 동일 패턴) |
| effort | large (sparsity selection + per-head dispatch + LSE fusion) |
| risk | **★ accuracy gate** — sparse approx 가 분포 동등성 (CLAUDE.md 게이트) 깨면 NG. HGCA 자체는 near-full quality 주장이나 NEO config 재검증 필요 |

### 1.5 SpecOffload — Speculative decoding embedded into offloading

| 항목 | 값 |
|---|---|
| 출처 | [arXiv 2505.10259](https://arxiv.org/abs/2505.10259) (2025-05) |
| mechanism | Offload pipeline 안에 speculative decoding embed. CPU 가 draft, GPU 가 verify. GPU core util 4.49×, throughput 2.54× |
| NEO 적용 영역 | **CPU 가 1-step speculative ahead** 채용 시 GPU IDLE wall 을 다른 일로 채움 → cdec leftover 의미 자체 소멸 |
| effort | large (draft model + verifier 통합) |
| risk | draft model 학습 / accuracy 정의 변경 |

### 1.6 Pie — Pooling CPU Memory for LLM Inference

| 항목 | 값 |
|---|---|
| 출처 | [arXiv 2411.09317](https://arxiv.org/abs/2411.09317) (2024-11) — vLLM 기반 |
| mechanism | vLLM 위 **performance-transparent swapping + adaptive expansion** |
| NEO 적용 영역 | **block-level adaptive expansion** 만 차용 시 b1 batch size 동적 조절 → cdec tail 균등화 |
| effort | medium |
| risk | NEO load-aware scheduler 와 정책 충돌 |

### 1.7 Helix — Max-flow scheduling on heterogeneous GPUs

| 항목 | 값 |
|---|---|
| 출처 | [arXiv 2406.01566](https://arxiv.org/pdf/2406.01566), ASPLOS 2025 |
| mechanism | LLM inference 를 directed weighted graph 의 max-flow formulation. IWRR scheduler. throughput 3.3× / decode latency -24% |
| NEO 적용 영역 | NEO 의 b0/b1 work distribution 을 **max-flow optimal split** 으로 계산 → leftover 정의상 minimized |
| effort | large (NEO scheduler 전면 재설계) |
| risk | online overhead / scheduler complexity |

### 1.8 Hetis — Per-head attention dispatch on heterogeneous

| 항목 | 값 |
|---|---|
| 출처 | SC25 / [arXiv 2509.08309](https://arxiv.org/abs/2509.08309) |
| mechanism | Attention 을 **head granularity** 로 dispatch. Dispatcher online head 분배, Hauler 가 KV cache 동적 migrate |
| NEO 적용 영역 | head granularity CPU↔GPU 분배 → 80 layer × N head fine partition → b1 의 CPU 시간 = b0 의 GPU 시간 매칭 |
| effort | large (head-level KV cache 재설계) |
| risk | NEO paged KV layout 과 충돌 |

---

## 2. CPU attention kernel 가속 (NEO pacpu 보다 빠른 영역) — 6 idea

### 2.1 KTransformers AMX kernel (kernel 측면)

| 항목 | 값 |
|---|---|
| 출처 | #1.3 와 동일 + ggml AMX backend (llama.cpp PR #10570) |
| mechanism | **runtime weight repacking + tile-aware threading** |
| NEO 적용 영역 | P3 회귀 원인 (M=8 partial tile 50% + thread imbalance) 영역의 **tile fill 전략** 차용 → tile occupancy ≥ 80%, pacpu compute (6.8 ms) 단축 |
| effort | medium |
| risk | NEO paged layout vs KTransformers contiguous block 가정 차이 |

### 2.2 SPARAMX — Compressed LLM token gen on AMX

| 항목 | 값 |
|---|---|
| 출처 | [arXiv 2502.12444](https://arxiv.org/pdf/2502.12444) (2025-02) |
| mechanism | AMX 는 dense compute 강하나 **decode low-ARI 영역에 부적합**. AMX ↔ AVX-512 kernel 을 ARI 기반 동적 교체. |
| NEO 적용 영역 | P3 회귀 원인 (AMX paged decode 영역 ARI 낮음) → **ARI-aware dynamic switch** 채용 시 회귀 없이 가속 |
| effort | medium |
| risk | runtime branch overhead / 전환 trigger 정확도 |

### 2.3 Intel xFasterTransformer

| 항목 | 값 |
|---|---|
| 출처 | GitHub [`intel/xFasterTransformer`](https://github.com/intel/xFasterTransformer). [arXiv 2407.07304](https://arxiv.org/pdf/2407.07304) |
| mechanism | prefill 에서 AMX 3.14×, decode (low ARI) 영역에서 AVX-512 우수. KV cache 축소 + oneCCL |
| NEO 적용 영역 | NEO pacpu 의 **AVX-512 kernel reference** — small batch decode 의 vec K, av kernel 영역 (현재 NEO 7.90% av) |
| effort | medium |
| risk | Intel 종속 라이선스 확인 |

### 2.4 IPEX — PagedAttention CPU module

| 항목 | 값 |
|---|---|
| 출처 | GitHub [`intel/intel-extension-for-pytorch`](https://github.com/intel/intel-extension-for-pytorch/tree/release/2.3/examples/cpu/inference/python/llm-modeling) |
| mechanism | `single_query_cached_kv_attention` API 로 PagedAttention MHA CPU 구현. ROPE fusion, BF16/WOQ |
| NEO 적용 영역 | NEO pacpu 와 직접 alternative. IPEX 의 **thread schedule / barrier 영역 source** 가 NEO 의 libgomp 62% barrier 영역에 reference |
| effort | medium |
| risk | IPEX 가 NEO 의 paged layout / block-table 가정과 동일 영역 확인 |

### 2.5 libxsmm + PyTorch-TPP

| 항목 | 값 |
|---|---|
| 출처 | GitHub [`libxsmm/libxsmm`](https://github.com/libxsmm/libxsmm). [arXiv 2304.12576](https://arxiv.org/pdf/2304.12576) |
| mechanism | JIT-compiled small-GEMM/BRGEMM. AVX/AVX-512/AMX target. TPP 가 attention block 영역에서 SOTA 초과 |
| NEO 적용 영역 | NEO pacpu 의 qk (8.75%) / av (7.90%) = small GEMM. libxsmm JIT kernel 교체 시 paged-block-shape-specific kernel 생성 가능 |
| effort | medium (JIT integration + cache) |
| risk | JIT compile latency / kernel cache 관리 |

### 2.6 llama.cpp ggml CPU backend

| 항목 | 값 |
|---|---|
| 출처 | GitHub [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) PR #10570 (AMX → CPU backend merge). [DeepWiki 4.2](https://deepwiki.com/ggml-org/llama.cpp/4.2-cpu-backend-and-optimization) |
| mechanism | OpenMP-based tensor op parallel, hand-tuned AVX/AVX2/AVX-512/AMX assembly. AMX matmul 40% 빠름 |
| NEO 적용 영역 | kernel 부분보다 **threadpool 구현** 이 reference. **custom threadpool assign API** (libgomp barrier 우회 단서) |
| effort | small-medium (threadpool 영역 차용) |
| risk | custom threadpool vs NEO 기존 OMP path 충돌 |

---

## 3. OMP barrier 영역 제거 / 우회 — 5 idea (★ libgomp 43.75% 직접 영역)

### 3.1 ★★★ Tournament / Dissemination barrier algorithm 교체

| 항목 | 값 |
|---|---|
| 출처 | [*"Scalability Evaluation of Barrier Algorithms for OpenMP"* — Stony Brook IWOMP](https://bpb-us-e1.wpmucdn.com/you.stonybrook.edu/dist/6/1671/files/2016/06/iwomp-barrier-1v0xkaw.pdf). GitHub `jedivind/barriersync` |
| mechanism | centralized blocking → **tournament barrier** 교체 시 32 thread 영역에서 wall clock -35%. Dissemination 16 thread 이하 우수, tournament 16 thread 이상 우수 |
| NEO 적용 영역 | **libgomp barrier wait 62% (+11.2 ms)** 영역 직접 단축. NEO 의 thread count 16-56 영역 = tournament 영역 sweet spot |
| effort | **small (env var)** — `KMP_FORCE_REDUCTION_BARRIER_PATTERN`, `KMP_FORK_BARRIER_PATTERN` |
| risk | libgomp default 시 LLVM iomp 교체 필요 / NEO build flag 충돌 |

### 3.2 ★★ KMP_BLOCKTIME 조정 + spin-wait policy

| 항목 | 값 |
|---|---|
| 출처 | [Intel libomp manual](https://www.openmprtl.org/sites/default/files/resources/libomp_20160808_manual.pdf), [LLVM/OpenMP doc](https://openmp.llvm.org/design/Runtimes.html) |
| mechanism | iomp default KMP_BLOCKTIME=200 ms → spin → sleep. **KMP_BLOCKTIME=infinite + OMP_WAIT_POLICY=active** → 영구 spin → wake-up latency = 0 |
| NEO 적용 영역 | NEO cdec 영역 layer 당 2.3 ms — thread 가 sleep 빠지면 매 layer wake-up 100µs+ overhead. **80 layer × wake-up 제거**. libgomp 사용 시 `GOMP_SPINCOUNT=INFINITE` + `OMP_WAIT_POLICY=ACTIVE` |
| effort | **small (env var 만)** |
| risk | 다른 process 와 CPU 경합 / 전력 / E-core spin 효율 |

### 3.3 ★ nowait clause + redundant barrier elimination

| 항목 | 값 |
|---|---|
| 출처 | [LLVM remark OMP190 — redundant barrier elimination](https://openmp.llvm.org/remarks/OMP190.html). IEEE 5286621 |
| mechanism | data dependence 없는 loop 사이 barrier 는 `nowait` 로 제거. LLVM OpenMP device 단 redundant barrier 자동 elimination |
| NEO 적용 영역 | NEO pacpu 의 OMP barrier #1/#2 (softmax + qk/av 사이) 의 data dependence 분석 → barrier 1개로 통합 또는 nowait 가능 영역 식별 |
| effort | small-medium |
| risk | **silent race** / 정확도 회귀 가능. fence 점검 필수 |

### 3.4 LLVM vectorized barrier + reduction

| 항목 | 값 |
|---|---|
| 출처 | *"Vectorized Barrier and Reduction in LLVM OpenMP Runtime"*, IWOMP 2021 ([Springer](https://link.springer.com/chapter/10.1007/978-3-030-85262-7_2)) |
| mechanism | barrier sense flag 를 SIMD batch check, reduction 도 vector reduction. high thread count (32+) 영역에서 barrier overhead 감소 |
| NEO 적용 영역 | NEO Xeon SPR 56-core 영역 build 시 OMP runtime 을 LLVM 의 vectorized barrier 영역으로 link → 11.2 ms 일부 직접 단축 |
| effort | small (build / link 변경) |
| risk | NEO build chain (libgomp default 면 교체 필요) |

### 3.5 Thread imbalance elimination — larger chunk + guided

| 항목 | 값 |
|---|---|
| 출처 | [Intel VTune cookbook OpenMP imbalance](https://www.intel.com/content/www/us/en/docs/vtune-profiler/cookbook/2023-0/openmp-imbalance-and-scheduling-overhead.html) |
| mechanism | implicit barrier origin = thread imbalance. `schedule(dynamic, chunk)` 또는 guided 로 분배 (NEO F6 dynamic 시도 -1.4% 회귀 → chunk size 영역 부적합 가능) |
| NEO 적용 영역 | F6 atomic counter overhead → **larger chunk + guided** 재시도. 또는 static even partition 으로 imbalance 직접 해소 |
| effort | small |
| risk | 이미 시도 후 회귀 — chunk size sweep 필요 |

---

## 4. Async / deferred CDEC pattern (race-safe) — 4 idea

### 4.1 OmniServe Async aggregate (race-safe 측면)

| 항목 | 값 |
|---|---|
| 출처 | #1.1 와 동일 |
| mechanism | CPU 결과 aggregation 에서 GPU stream blocking 회피 |
| NEO 적용 영역 | NEO P4 async (depth=1 deque) 가 OOB race 였으나 OmniServe stream pattern 결합 시 race 영역 완화 가능 |
| effort | medium-large |
| risk | 분포 동등성 검증 |

### 4.2 ★★ FlashDecoding++ asynchronized softmax with unified max

| 항목 | 값 |
|---|---|
| 출처 | [MLSys 2024 PDF](https://proceedings.mlsys.org/paper_files/paper/2024/file/5321b1dabcd2be188d796c21b733e8c7-Paper-Conference.pdf) |
| mechanism | softmax 의 partial result sync 영역에서 **unified max value 로 synchronized update 자체 제거**. 20% overhead 직접 단축 |
| NEO 적용 영역 | NEO pacpu 의 **softmax 9.73% 영역 = partial softmax sync 영역과 정확 매핑**. unified max 채용 시 **OMP barrier #1 자체 제거** → 11.2 ms 의 일부 회수 |
| effort | medium (softmax kernel 재작성) |
| risk | NaN / overflow / numerical stability. 정확도 게이트 검증 필수 |

### 4.3 AsyncTLS — Async Two-Level Sparse attention

| 항목 | 값 |
|---|---|
| 출처 | [arXiv 2604.07815](https://arxiv.org/pdf/2604.07815) (2026) |
| mechanism | two-level sparse + asynchronous offloading. KV transfer 와 compute overlap (temporal locality) |
| NEO 적용 영역 | NEO 는 dense paged decode 인데 sparse 도입 시 PCIe transfer 와 compute overlap. cdec leftover 의 일부 sparse 로 단축 |
| effort | large (sparse selector + KV migration) |
| risk | 분포 동등성 회귀 |

### 4.4 Async KV Cache Prefetching (vLLM)

| 항목 | 값 |
|---|---|
| 출처 | [arXiv 2504.06319](https://arxiv.org/pdf/2504.06319) |
| mechanism | KV cache 를 L2 cache 로 idle bandwidth 영역에서 prefetch → attention kernel 2.15× / e2e 1.97× |
| NEO 적용 영역 | NEO cdec 의 다음 layer KV block 영역을 미리 L2 prefetch → 다음 layer actual compute (6.8 ms) 단축 |
| effort | small-medium (prefetch hint 추가) |
| risk | L2 thrashing / 다른 stream working set 침해 |

---

## 5. Layer-level pipelining (cross-layer overlap) — 3 idea

### 5.1 PipeSpec — Stage dependency breaking in hierarchical decoding

| 항목 | 값 |
|---|---|
| 출처 | [arXiv 2505.01572](https://arxiv.org/pdf/2505.01572) (2025-05) |
| mechanism | pipeline 단계 의존성 제거. 2.54× 가속, pipeline 효율이 model depth 와 함께 증가 |
| NEO 적용 영역 | NEO 80 layer 에서 layer L cdec 와 layer L+1 GPU pre-attention overlap. **이전 TSK_005 의 Q dependency 영역으로 기각된 영역의 대안** mechanism |
| effort | large (layer dependency 분석 + KV double-buffer) |
| risk | Q dependency. PipeSpec 이 speculative 영역 기반이라 NEO 의 exact-equivalence 영역과 충돌 가능 |

### 5.2 CLAA — Cross-Layer Attention Aggregation

| 항목 | 값 |
|---|---|
| 출처 | [arXiv 2602.16054](https://arxiv.org/pdf/2602.16054) |
| mechanism | prefill TTFT -39%. answer-informed oracle 로 token rank aggregate |
| NEO 적용 영역 | prefill 위주 — NEO decode cdec 직접 적용 어려움. 단 prefill 단계에서 cdec 단축 시 prefill→decode 전환 latency 감소 |
| effort | large (model 수정) |
| risk | 정확도 / model retraining [추정] |

### 5.3 LISA / Shared Attention — Layer-shared attention

| 항목 | 값 |
|---|---|
| 출처 | [arXiv 2408.01890](https://arxiv.org/pdf/2408.01890) / [arXiv 2407.12866](https://arxiv.org/pdf/2407.12866) |
| mechanism | LISA = tiny FFN + low-rank approx 로 layer 간 attention head align. 6× Q/K compression, throughput +19-32% |
| NEO 적용 영역 | 80 layer 중 일부 layer 간 attention 공유 → CPU pacpu 호출 횟수 자체 감소. **80 layer × 2.3 ms 의 layer 수 감축** |
| effort | large (model 수정 + fine-tune) |
| risk | model accuracy / NEO constraint (GPU-only 동등) 영역과 충돌. 분포 유사성 영역에서 시도 가능 |

---

## 6. GPU compute 영역으로 cdec 이전 (CPU 제거) — 5 idea

### 6.1 POD-Attention — Prefill-Decode 동시 같은 SM

| 항목 | 값 |
|---|---|
| 출처 | Kamath et al., ASPLOS 2025 ([arXiv 2410.18038](https://arxiv.org/pdf/2410.18038)) |
| mechanism | 같은 SM 에서 prefill (compute-bound) + decode (memory-bound) 동시 처리. attention +59%, 평균 +28% |
| NEO 적용 영역 | NEO 가 b1 을 CPU 보내는 이유 = GPU memory 부족. **POD 가 GPU 에서 둘 다 동시 수행 가능하면 CPU offload 자체 불필요** → cdec leftover 자체 소멸. 단 KV cache memory 는 여전히 GPU |
| effort | large (kernel 교체 + scheduling) |
| risk | KV memory (NEO 의 주요 raison d'être) trade-off |

### 6.2 ★ FlashAttention-3 — Hopper async + warp specialization

| 항목 | 값 |
|---|---|
| 출처 | Dao et al., [arXiv 2407.08608](https://tridao.me/publications/flash3/flash3.pdf) (NeurIPS 2024) |
| mechanism | producer/consumer warp 분리, TMA + Tensor Core 비동기 overlap, FP8. H100 BF16 840 TFLOPs/s (85% util), FP8 1.3 PFLOPs/s |
| NEO 적용 영역 | NEO H100 prod 환경 (CLAUDE.md hardware target) 에서 b0 GPU 의 flash_attn 을 FA3 로 교체 → GPU 속도 ↑. b0/b1 imbalance (cdec leftover 18 ms) 영역에서 **b0 단축은 leftover 영역 가속** |
| effort | small (FA3 binding swap) |
| risk | H100 only / dev 머신 (3090=Ampere) 미적용 |

### 6.3 FlexAttention + PagedAttention (PyTorch)

| 항목 | 값 |
|---|---|
| 출처 | [arXiv 2506.07311](https://arxiv.org/abs/2506.07311) / PyTorch FlexAttention API |
| mechanism | `mask_mod` / `score_mod` hook 으로 paged attention JIT-fuse. kernel overhead <2%, 16K-64K context peak GPU mem -30% |
| NEO 적용 영역 | b1 의 GPU 부분 (preproj/postproj) 영역이 paged layout 에서 FlexAttention 으로 fuse 시 GPU work 시간 ↓, b0/b1 imbalance 매칭 가능 |
| effort | medium (PyTorch 2.x API 마이그레이션) |
| risk | PyTorch 호환성 / paged block size 호환 |

### 6.4 ★ Sequence-Aware Split Heuristic (FA3 low-head-count decode)

| 항목 | 값 |
|---|---|
| 출처 | [arXiv 2604.00028](https://arxiv.org/abs/2604.00028) (Barcelona Supercomputing Center, 2026) |
| mechanism | low occupancy hardware condition 식별 → sequence-level split 허용 → SM occupancy ↑. 21-24% decoder kernel 효율 ↑ |
| NEO 적용 영역 | NEO 의 long context (500p × 8192) = 정확히 **low-head decode 영역**. b0 GPU 시간 ↓ → cdec leftover 매칭 개선 |
| effort | small (heuristic 적용) |
| risk | Hopper only / FA3 의존 |

### 6.5 Mirage Persistent Kernel (mega-kernel)

| 항목 | 값 |
|---|---|
| 출처 | [arXiv 2512.22219](https://arxiv.org/html/2512.22219v1) (2025-12) |
| mechanism | 모든 SM 에서 thread block graph 실행하는 in-kernel parallel runtime. kernel launch overhead 자체 제거, cross-task pipelining |
| NEO 적용 영역 | NEO decode 의 80 layer × N op launch overhead (µs 단위) 누적 제거. b0 wall ↓ |
| effort | large (compiler 통합) |
| risk | MPK 의 NEO 의존성 / debugging (single mega-kernel) |

---

## 7. Recent NEO-like (2024-2026) — 6 idea

### 7.1 NEO 원본 paper 재확인

| 항목 | 값 |
|---|---|
| 출처 | Zhou et al., [arXiv 2411.01142](https://arxiv.org/pdf/2411.01142) / MLSys 2025. GitHub [`NEO-MLSys25/NEO`](https://github.com/NEO-MLSys25/NEO) |
| 본 작업 본체 | 2026-05 시점 follow-up 영역은 검색 결과 OmniServe / APEX / HGCA / KTransformers 같은 인접 영역이 위치 |

### 7.2 Sandwich — CPU LLM serving (prefill-decode 분리 컴파일)

| 항목 | 값 |
|---|---|
| 출처 | [arXiv 2507.18454](https://arxiv.org/pdf/2507.18454) (2025-07, joint config search & hot-switching) |
| mechanism | per-NUMA static partition 의 한계 해소. prefill/decode 다른 execution plan + hot-switching. throughput +2.01×, TTFT/TPOT -3.40× |
| NEO 적용 영역 | NEO 의 pacpu 만 영역 적용 가능. **NUMA-aware partition + hot-switch** 채용 시 pacpu thread efficiency ↑ |
| effort | medium-large |
| risk | NEO single-NUMA assumption 과 정합 확인 |

### 7.3 Fiddler — Expert-wise CPU/GPU dispatch (MoE)

| 항목 | 값 |
|---|---|
| 출처 | Kamahori et al., ICLR 2025 / [arXiv 2402.07033](https://arxiv.org/pdf/2402.07033) |
| mechanism | expert weight 전송 비용 > CPU 직접 compute 비용 insight. expert 별 static 분배 |
| NEO 적용 영역 | NEO 의 head 별 / layer 별 분배에 같은 insight. 단 dense model 영역에서 부분적 일치만 |
| effort | medium |
| risk | dense model 동일 insight 검증 필요 |

### 7.4 MoE-Lightning — CGOPipe

| 항목 | 값 |
|---|---|
| 출처 | Cao et al., ASPLOS 2025 / [arXiv 2411.11217](https://arxiv.org/pdf/2411.11217) |
| mechanism | CGOPipe (CPU-GPU-I/O pipelining) 3-stage 동시 overlap. T4 10.3× |
| NEO 적용 영역 | NEO 는 2-stage (CPU+GPU) — CGOPipe **3-stage 일반화 (PCIe KV transfer 추가)** 시 PCIe idle 활용 |
| effort | large |
| risk | HRM calibration / NEO load-aware 충돌 |

### 7.5 LM-Offload / MLP-Offload (training reference)

| 항목 | 값 |
|---|---|
| 출처 | LM-Offload (PASA labs 2024), MLP-Offload ([arXiv 2509.02480](https://arxiv.org/abs/2509.02480)) |
| mechanism | performance model-guided offload. multi-level multi-path (RAM + NVMe) |
| NEO 적용 영역 | training reference — performance model framework 채용 시 b0/b1 매칭 예측 정확도 ↑ |
| effort | medium |
| risk | training-inference 적합도 |

### 7.6 ArcLight — Many-core CPU NUMA-aware

| 항목 | 값 |
|---|---|
| 출처 | [arXiv 2603.07770](https://arxiv.org/abs/2603.07770) (2026-03). GitHub [`OpenBMB/ArcLight`](https://github.com/OpenBMB/ArcLight) |
| mechanism | node-local 메모리 할당 + tensor parallel cross-NUMA break-down. throughput +46% |
| NEO 적용 영역 | NEO pacpu 의 NUMA-aware allocation 채용 시 cross-NUMA latency (Xeon SPR dual socket) 제거 → CPU compute (6.8 ms) 단축 |
| effort | medium (memory allocator 수정) |
| risk | NEO KV cache layout 과 NUMA 정합 |

---

## 8. ★ 우선순위 영역 (NEO cdec leftover 직접 단축 기준)

### 영역 A — Quick win, low risk (즉시 시도 권고)

| ID | Idea | 출처 | effort | 예상 wall 단축 |
|---|---|---|:-:|---:|
| **A1** | **#3.1 Tournament barrier** (env var) | OpenMP IWOMP | env-only | **libgomp 62% (+11.2 ms) 의 일부** |
| **A2** | **#3.2 KMP_BLOCKTIME=infinite + active spin** | Intel libomp manual | env-only | wake-up overhead 제거 (~0.5-1 ms) |
| **A3** | **#4.4 Async KV prefetch (L2)** | arXiv 2504.06319 | small | compute 38% 단축 (~0.5 ms) |
| **A4** | **#7.6 ArcLight NUMA allocator** | arXiv 2603.07770 | small-medium | cross-NUMA 영역 제거 (~1-2 ms) |
| **A5** | **#6.4 Sequence-Aware Split Heuristic (FA3)** | arXiv 2604.00028 | small | b0 단축 → b0/b1 매칭 (~1-2 ms) |

→ **합산 예상 단축**: **3-6 ms** (effort 합 1-3 일).

### 영역 B — Medium effort, high impact

| ID | Idea | 출처 | effort | 예상 wall 단축 |
|---|---|---|:-:|---:|
| **B1** | **#1.1 OmniServe LSE async merge** | arXiv 2603.12831 | medium | cdec tail 자체 단축 (~5-10 ms) |
| **B2** | **#1.2 APEX async overlap predictor** | arXiv 2506.03296 | medium-large | b0/b1 매칭 자동화 (~3-8 ms) |
| **B3** | **#4.2 FlashDecoding++ unified-max softmax** | MLSys 2024 | medium | **OMP barrier #1 자체 제거** (~2-5 ms) |
| **B4** | **#2.2 SPARAMX ARI-aware AMX↔AVX-512 switch** | arXiv 2502.12444 | medium | P3 회귀 복구 + compute 단축 (~1-3 ms) |
| **B5** | **#2.5 libxsmm JIT kernel** | arXiv 2304.12576 | medium | qk/av kernel 단축 (~1-2 ms) |
| **B6** | **#2.4 IPEX PagedAttention CPU + threadpool reference** | GitHub IPEX | medium | thread schedule 영역 개선 (~1-3 ms) |

→ **합산 예상 단축**: **5-15 ms** (effort 합 2-4 주).

### 영역 C — Large effort, paradigm shift

| ID | Idea | 출처 | effort | 예상 wall 단축 |
|---|---|---|:-:|---:|
| **C1** | **#1.4 HGCA per-head sparse CPU + LSE fusion** | arXiv 2507.03153 | large | CPU compute 절대 축소 (~3-6 ms) |
| **C2** | **#6.1 POD-Attention** (KV memory trade-off) | ASPLOS 2025 | large | **cdec 자체 제거** (~18 ms 전부) |
| **C3** | **#1.3 KTransformers 전면 통합** | SOSP 2025 | large | scheduler + kernel 통합 (~5-10 ms) |
| **C4** | **#5.1 PipeSpec cross-layer pipelining** | arXiv 2505.01572 | large | 80 layer × overlap (~5-15 ms, Q dep 영역 회피 필요) |
| **C5** | **#5.3 LISA layer-shared attention** | arXiv 2408.01890 | large (model 수정) | layer 수 축소 → CPU 호출 횟수 ↓ |

→ **합산 예상 단축**: **10-25 ms** (effort 합 1-3 개월), 단 NEO 의 raison d'être 영역 변경 가능 (KV memory / accuracy gate).

---

## 9. NEO 기존 시도 매핑 (idea backing)

| NEO 시도 | 결과 | 외부 idea backing |
|---|---|---|
| P3 (K BF16 + AMX, M=8 partial tile) | -2.5% 회귀 (이전) / net zero (현재) | **#2.2 SPARAMX ARI-aware switch** 로 복구 가능 |
| F6 (OMP dynamic schedule) | -1.4% 회귀 (atomic counter overhead) | **#3.5 chunk size + #3.1 barrier 교체** 결합 재시도 |
| P4 (async cdec depth=1) | unstable (이전, OOB race) / stable (현재, OOB fix) | **#1.1 OmniServe LSE async merge** 영역으로 race 회피 |
| Step 5 AMX (B+A+vec K) | -2.35% 회귀 | **#2.5 libxsmm JIT** kernel 영역으로 shape-specific 최적화 |
| swap_out async (SUB_025/026) | ✓ 100% async 영역 도달 | (이미 완료 — 추가 영역 없음) |

---

## 10. 다음 turn 권고 영역

| 우선순위 | 작업 | 영역 | effort |
|---|---|---|:-:|
| **★★★** | **A1 Tournament barrier env var sweep** | KMP_FORCE_REDUCTION_BARRIER_PATTERN sweep + measurement | 2-3 시간 |
| **★★★** | **A2 KMP_BLOCKTIME=infinite + GOMP_SPINCOUNT=INFINITE** | env-only sweep | 1 시간 |
| **★★** | **A4 ArcLight NUMA allocator** | numactl + node-local malloc 영역 | 1 일 |
| **★★** | **B3 FlashDecoding++ unified-max softmax** | pacpu.ispc 의 softmax 영역 재작성 | 3-5 일 |
| **★** | **B1 OmniServe LSE async pattern** | P4 (async cdec) 영역 race-safe 영역 재설계 | 1-2 주 |
| **★** | **A5 FA3 Sequence-Aware Split** (H100 only) | FA3 binding 영역 / FlashAttn-3 install | 1-2 일 |
| ⚪ | **C1 HGCA sparse CPU** | sparse selection + LSE fusion + 정확도 검증 | 2-4 주 |

→ **권장 첫 step**: **A1 + A2 env var sweep + measurement** (1 일) — env var 만으로 libgomp 62% 영역의 일부 단축 가능, 회귀 없음.

---

## 11. References (전체)

### NEO 원본 + 인접 영역
- [NEO MLSys 2025 PDF](http://minlanyu.seas.harvard.edu/writeup/mlsys25.pdf) / [arXiv 2411.01142](https://arxiv.org/pdf/2411.01142) / [GitHub](https://github.com/NEO-MLSys25/NEO)
- [OmniServe — arXiv 2603.12831](https://arxiv.org/html/2603.12831v2)
- [APEX — arXiv 2506.03296](https://arxiv.org/abs/2506.03296)
- [KTransformers — SOSP 2025 PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf) / [GitHub](https://github.com/kvcache-ai/ktransformers) / [LMSYS blog](https://lmsys.org/blog/2025-10-22-KTransformers/)
- [HGCA — arXiv 2507.03153](https://arxiv.org/abs/2507.03153)
- [SpecOffload — arXiv 2505.10259](https://arxiv.org/abs/2505.10259)
- [Pie — arXiv 2411.09317](https://arxiv.org/abs/2411.09317)
- [Helix — arXiv 2406.01566](https://arxiv.org/pdf/2406.01566)
- [Hetis — arXiv 2509.08309](https://arxiv.org/abs/2509.08309)

### CPU attention kernel
- [SPARAMX — arXiv 2502.12444](https://arxiv.org/pdf/2502.12444)
- [Intel xFasterTransformer — GitHub](https://github.com/intel/xFasterTransformer) / [arXiv 2407.07304](https://arxiv.org/pdf/2407.07304)
- [IPEX — GitHub](https://github.com/intel/intel-extension-for-pytorch/tree/release/2.3/examples/cpu/inference/python/llm-modeling)
- [libxsmm — GitHub](https://github.com/libxsmm/libxsmm) / [TPP arXiv 2304.12576](https://arxiv.org/pdf/2304.12576) / [TPP doc](https://libxsmm.readthedocs.io/en/latest/libxsmm_tpp/)
- [llama.cpp PR #10570 (AMX → CPU backend)](https://github.com/ggml-org/llama.cpp/pull/10570) / [DeepWiki 4.2](https://deepwiki.com/ggml-org/llama.cpp/4.2-cpu-backend-and-optimization)

### OMP barrier
- [Stony Brook IWOMP — Barrier Algorithms PDF](https://bpb-us-e1.wpmucdn.com/you.stonybrook.edu/dist/6/1671/files/2016/06/iwomp-barrier-1v0xkaw.pdf)
- [LLVM/OpenMP Runtimes](https://openmp.llvm.org/design/Runtimes.html) / [Remark OMP190](https://openmp.llvm.org/remarks/OMP190.html)
- [Vectorized Barrier IWOMP 2021](https://link.springer.com/chapter/10.1007/978-3-030-85262-7_2)
- [Intel libomp manual](https://www.openmprtl.org/sites/default/files/resources/libomp_20160808_manual.pdf)
- [EPCC OpenMP Microbenchmarks](https://github.com/EPCCed/epcc-openmp-microbenchmarks)
- [Intel VTune — OpenMP Imbalance](https://www.intel.com/content/www/us/en/docs/vtune-profiler/cookbook/2023-0/openmp-imbalance-and-scheduling-overhead.html)

### Async pattern
- [FlashDecoding++ MLSys 2024 PDF](https://proceedings.mlsys.org/paper_files/paper/2024/file/5321b1dabcd2be188d796c21b733e8c7-Paper-Conference.pdf)
- [AsyncTLS — arXiv 2604.07815](https://arxiv.org/pdf/2604.07815)
- [Async KV Prefetching — arXiv 2504.06319](https://arxiv.org/pdf/2504.06319)

### Cross-layer pipelining
- [PipeSpec — arXiv 2505.01572](https://arxiv.org/pdf/2505.01572)
- [CLAA — arXiv 2602.16054](https://arxiv.org/pdf/2602.16054)
- [LISA — arXiv 2408.01890](https://arxiv.org/pdf/2408.01890)
- [Shared Attention — arXiv 2407.12866](https://arxiv.org/pdf/2407.12866)

### GPU-only path
- [POD-Attention ASPLOS 2025 — arXiv 2410.18038](https://arxiv.org/pdf/2410.18038)
- [FlashAttention-3 — arXiv 2407.08608](https://tridao.me/publications/flash3/flash3.pdf)
- [FlexAttention + PagedAttention — arXiv 2506.07311](https://arxiv.org/abs/2506.07311)
- [Sequence-Aware Split Heuristic — arXiv 2604.00028](https://arxiv.org/abs/2604.00028)
- [Mirage Persistent Kernel — arXiv 2512.22219](https://arxiv.org/html/2512.22219v1)

### Recent NEO-like
- [Sandwich — arXiv 2507.18454](https://arxiv.org/pdf/2507.18454)
- [Fiddler — arXiv 2402.07033](https://arxiv.org/pdf/2402.07033)
- [MoE-Lightning — arXiv 2411.11217](https://arxiv.org/pdf/2411.11217)
- [LM-Offload (PASA labs)](https://github.com/PASA-Labs)
- [MLP-Offload — arXiv 2509.02480](https://arxiv.org/abs/2509.02480)
- [ArcLight — arXiv 2603.07770](https://arxiv.org/abs/2603.07770) / [GitHub](https://github.com/OpenBMB/ArcLight)

### Curated lists
- [Awesome LLM Inference](https://github.com/xlite-dev/Awesome-LLM-Inference)
- [LLM Inference Optimization Paper List](https://github.com/chenhongyu2048/LLM-inference-optimization-paper)
- [MLSys 2025 Serving Session Review](https://medium.com/byte-sized-ai/quick-review-of-mlsys-2025-llm-model-serving-session-77d2c36d3467)

---

## 12. Change Log

| 일자 (KST) | 변경 |
|---|---|
| **2026-05-21** | 신설. 22 idea (7 영역) 외부 조사 — 사용자 명시 "cdec leftover 제거 아이디어 다양한 논문/GitHub 영역 키워드 조사". 우선순위 영역 A (quick win) / B (medium impact) / C (paradigm shift) 분류. NEO 기존 시도 영역 (P3/F6/P4/Step 5) backing idea 매핑. |
