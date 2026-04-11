# Hybrid CPU/GPU LLM Inference 최적화 — 논문·프로젝트 종합 서베이

**Timestamp (KST)**: 2026-04-11 15:45:23
**Commit**: `63b4c353d` 기준 (h100_cu13 / main 동기화 완료)
**형식**: 실험 run 이 아닌 **research / ideation 노트**. 5 개 주제의 병렬 리서치 에이전트 결과를 종합하여 다음 구조 개선의 로드맵을 도출한다.

---

## 1. 리서치 동기 — 지금 왜 논문을 뒤지는가

### 1.1 현재 상태 (2026-04-11)

- **Dev (RTX 3090 + i9-12900KF, TP=1)**: hybrid 는 CPU tail 지배로 ×2.4~×2.8 penalty. 단 로직은 무결.
- **H100x4 + Xeon 8480+ KVM (TP=4)**: `_route_throughput_adaptive` Property 2 expected-finish gate 수정 후 1.5B / 7B / 32B 모두 **hybrid ≈ gpu_only** (±2% 노이즈). 라우터가 CPU 에 0/501 보냄.
- **GPU 활용률**: 1.5B 14%, 7B 19%, **32B 43%** — H100x4 는 32B 까지도 sub-saturation. 모든 실측에서 "GPU 가 바빠서 CPU 의 도움이 필요" 한 순간이 없음.

### 1.2 정확한 천장 분석

현재 hybrid 구조의 **request-level partition** 은 `T_hybrid = max(T_gpu_share, T_cpu_share)` 를 이론 천장으로 가진다. 실측에서:

- H100 1.5B decode: GPU ~22 ms/tok, CPU ~400 ms/tok (18×)
- H100 7B decode: GPU ~25 ms/tok, CPU ~400 ms/tok (16×)
- H100 32B decode: GPU ~42 ms/tok, CPU 추정 ~500 ms/tok (12×)

CPU per-req latency 가 GPU 보다 10~20× 느린 구간에서는 **어떤 request-level routing 으로도** `T_hybrid < T_gpu` 는 불가능하다. Property 2 gate 는 이 상한 **안에서** optimal (hybrid ≤ gpu_only) 이지만 그 이상은 불가능.

### 1.3 돌파구는 두 방향뿐

**(a) 같은 일을 다른 자원에 분산**: GPU 가 queue-saturated 되어 `gpu_wait > cpu_per_req` 가 되는 영역. 현재 H100 + 1.5B/7B/32B burst 에서는 발생 안 함. 70B / long-context / 큰 batch 가 필요.

**(b) 다른 일을 다른 자원에서**: CPU 와 GPU 가 **같은 요청을 처리하지 않고** 본질적으로 다른 layer / step / precision 을 처리. 진짜 ninja gap.

→ 본 서베이의 목적: **(b) 를 구현할 수 있는 논문·프로젝트 후보를 전 분야에 걸쳐 파악.**

### 1.4 리서치 방법

5 개 agent 가 동시에 서로 다른 주제에 대해 arxiv / ACM / GitHub 검색 수행. 각 agent 는:
- 해당 주제의 Top 5-10 핵심 논문 조사
- Technique 매트릭스 작성
- 우리 H100x4 + SPR 8480+ stack 에 대한 적용성 평가 (5점 척도)
- 구현 난이도 + 예상 효과
- 1순위 구현 경로 제시

5 에이전트: **[A] Speculative decoding**, **[B] KV cache offload / hierarchy**, **[C] P/D disaggregation / scheduling**, **[D] CPU inference / AMX / quant**, **[E] MoE expert offload / sparse activation**.

---

## 2. 각 영역 1순위 (agent 결과)

| 영역 | 1순위 기법 | 핵심 논문 | 예상 효과 |
|---|---|---|---|
| [A] Spec decode | **0.5-1B drafter on CPU + GPU verify** (DuoDecoding-style) | DuoDecoding (2503.00784), OmniDraft (2507.02659, Llama-68m→Qwen2.5-**32B** 2.05×) | 32B TPOT 41→22ms (1.8-2.5×) |
| [B] KV offload | **InfiniGen speculative top-k + layer prefetch** | InfiniGen (OSDI'24 2406.19707), Quest (ICML'24 2406.10774), KIVI (2402.02750) | batch 1500+, throughput 2× |
| [C] P/D disagg | **Sarathi chunked prefill** (이미 vLLM 내장), CPU overflow 는 16K+ 전용 | Sarathi-Serve (OSDI'24 2403.02310), Splitwise (ISCA'24 2311.18677) | 16K 이상 P99 TTFT 1.2-1.5× |
| [D] CPU quant/AMX | **IPEX `WoqWeightDtype.INT8`** 활성화 (한 줄) | llama.cpp AMX PR #6341, IPEX LLM WOQ | 7B 2.3→6-12 tok/s (2-4×) |
| [E] MoE offload | **Fiddler-style Mixtral-8x7B + CPU expert compute** | Fiddler (arXiv 2402.07033), Mixtral-offload (2312.17238) | dense 32B → Mixtral throughput 2×+ |

---

## 3. 가설 수정 — 리서치로 드러난 3가지

### 3.1 "BW bound 2.3 tok/s 천장" 가설이 의심스럽다 ([D])

수치 재검토:
- 7B × BF16 = **14 GB weight**
- SPR 8480+ DDR5-4800 × 8ch 이론 = **~300 GB/s**
- 실측 attain 60-70% ≈ **~200 GB/s**
- 이론 최대 decode rate = **14 tok/s**
- **실측 2.3 tok/s 는 이론의 16%**

→ BW 가 실제 천장이라면 2.3 tok/s 는 너무 낮다. **Attention KV scan + IPEX Python overhead + NUMA remote access + 동기화 비용** 이 지배 중일 가능성.

결정적 증거 후보: **llama.cpp SPR 실측 Q4_0 7B = 12-18 tok/s** 보고치 (github.com/ggerganov/llama.cpp issue 들). 만약 BW 가 천장이라면 INT8/INT4 quant 가 이론상 2×/4× 넘을 수 없어야 함. 실측은 6-8× 개선.

→ **우리가 "BW 가 천장" 이라고 확정하고 AMX 를 포기한 것은 성급했다**. Phase 0 진단으로 되돌려봐야 함.

### 3.2 우리의 A2 (hot HBM / cold DRAM) 가설이 잘못됐다 ([B])

- PCIe Gen5 ×16 = 64 GB/s unidirectional (실측 50)
- H100 decode step 간격 ~20-40 ms 에 "miss-on-demand" 로 swap 하면 **수백 μs stall 이 쌓여 latency 붕괴**
- Uniform hot/cold 이분법 자체가 부정확 — **H2O/Quest 는 attention score 가 극도로 sparse (상위 10% 가 95% 기여) 임을 실증**

수정된 **A2'** = **InfiniGen-style speculative top-k prefetch**:
- Layer i−1 의 partial Q·K score 로 layer i 에서 필요할 **상위 k% block 예측**
- i−1 → i 사이 시간에 PCIe 전송을 **정확히 hide**
- 기본 저장소는 DRAM, HBM 은 sliding window (StreamingLLM attention sink + recent) + prefetched top-k 만
- 직교적으로 **KIVI INT4** 를 먼저 적용하면 KV footprint 4× 축소 → offload 부담 자체 감소

### 3.3 Spec decode 는 32B+ 에만 natural fit ([A])

- **H100 + 1.5B/7B**: verifier step ~3 ms vs CPU 1B drafter step ~15 ms → drafter 가 step 을 묶음. γ=c 매칭해도 효과 ≤ 1.3×
- **H100 + 32B**: verifier step ~40 ms vs CPU drafter step ~15 ms → drafter step 이 verify step 안에 **숨음**. 실효 1.8-2.5×

→ **32B+ 에서만 spec decode 활성화**. 1.5B/7B 는 MoE 전환 (E1 Mixtral + Fiddler) 또는 KV offload 로 batch ↑ 가 더 나음.

---

## 4. 상세 — 각 영역 Top 논문 + 매트릭스

### 4.1 [A] Speculative Decoding

| # | 논문 | 연도 | arXiv | 요약 |
|---|---|---|---|---|
| 1 | **DuoDecoding** | 2025 | 2503.00784 | **CPU draft + GPU verify**, hardware-aware γ=c 매칭. Llama-68m→Llama2-7B 2.61× lossless. 우리 구조 그대로 |
| 2 | **OmniDraft** | 2025 | 2507.02659 | Cross-vocab online adaptive drafter. **Llama-68m → Qwen2.5-32B 에서 2.05× (GSM8K)** — 우리 target 사이즈 일치 |
| 3 | **EAGLE-3** | 2025 | 2503.01840 | Feature fusion + training-time test, 최대 6.5×. SOTA draft accuracy. GPU draft 전제 |
| 4 | **Medusa** | 2024 | 2401.10774 | Extra decoding head + tree attention, 2.3-3.6×. Separate draft 불필요 → CPU 에 올릴 게 없음 |
| 5 | **SpecExec** | 2024 | 2406.02532 | CPU RAM 에 model params 저장, tree 10-20 accept. Offloading 시나리오 |
| 6 | **ML-SpecQD** | 2025 | 2503.13565 | **Intel 저자**. MXFP4 quantized drafts + multi-level, **2.72× on CPU**. AMX 맥락 결정적 |
| 7 | **Dovetail** | 2024 | 2412.18934 | **반대 방향** (GPU draft + CPU verify). Consumer hardware 대상. 우리 방향 반례로 유용 |
| 8 | **Leviathan et al.** (원조) | 2022 | 2211.17192 | α ∈ [0.5, 0.9] for "orders of magnitude smaller" drafters. 이론 baseline |

**매트릭스**:

| 기법 | Draft 구조 | Verify | Accept α | Multi-token | Lossless | CPU drafter 호환 |
|---|---|---|---|---|---|---|
| Leviathan | Small LM (68M~1B) | Linear | 0.5-0.9 | 고정 γ | ✓ | **Native** |
| Medusa | Target 내부 head | Tree | 0.6-0.8 | 2-5 | ✓ | ✗ (target 내부) |
| EAGLE-2 | 1 extra layer, feature | Dynamic tree | ~0.8 | 4-6 | ✓ | △ (target feature 필요) |
| EAGLE-3 | Multi-layer fusion | Tree | ~0.85 | 5-8 | ✓ | △ |
| SpecExec | Tree (Dijkstra) | Batched tree | 10-20 tokens | 깊이 D | ✓ | ○ |
| **DuoDecoding** | Small LM on CPU | Linear + multi-seq | ~0.7 | 동적 γ | ✓ | **✓✓✓** |
| **ML-SpecQD** | MXFP4 quantized self | Hierarchical | ~0.75 | 2-level | ✓ | ✓✓ (CPU 논문) |
| **OmniDraft** | Llama-68m cross-vocab | Linear | ~0.6 | γ=5 | ✓ | ○ |

**1순위 권고**: **0.5-1B drafter (BF16 또는 INT8) 을 CPU EngineCore 에서 기동, ZMQ 로 GPU verifier 에 token id 전달, vLLM rejection sampler 재사용.** `vllm/v1/spec_decode/` 내 `eagle.py`, `medusa.py`, `ngram_proposer.py` 가 이미 있음 → 신규 `cpu_drafter.py` 추가로 path 확장.

구현 차단 요소:
- **Block 1**: vLLM V1 이 현재 spec decode 에서 EAGLE/Medusa/ngram 만 지원, 순정 small LM drafter path 는 `NotImplementedError`
- **Block 2**: Verifier KV cache 를 accept length 에 맞춰 rollback — V1 scheduler `num_rejected_tokens` 경로 재사용
- **Block 3**: CPU→GPU token transfer 는 기존 ZMQ 에 얹으면 < 100 µs (무시 가능)

### 4.2 [B] KV Cache Hierarchy / Offload

| # | 논문 | 저자/연도 | arXiv | 요약 |
|---|---|---|---|---|
| 1 | **FlexGen** | Sheng 2023 | 2303.06865 | GPU/DRAM/SSD 3-tier, zig-zag block schedule, offline only |
| 2 | **vAttention** | Prabhu 2024 | 2405.04437 | CUDA VMM (cuMemMap) 로 physical-virtual 분리, block_table 없는 contiguous KV |
| 3 | **InfiniGen** | Lee OSDI'24 | 2406.19707 | **Layer i-1 의 partial attention 으로 layer i 의 critical KV 예측** → DRAM 에서 선별 prefetch |
| 4 | **LayerKV** | Xu 2024 | 2410.00428 | Layer-granular KV placement, TTFT 최적화 |
| 5 | **AttentionStore** | Gao/Chen 2024 | 2403.19708 | Multi-turn KV 재사용, hierarchical cache |
| 6 | **H2O** | Zhang NeurIPS'23 | 2306.14048 | Accumulated attention score 기반 Heavy-Hitter eviction (lossy) |
| 7 | **Quest** | Tang ICML'24 | 2406.10774 | Page-granular query-aware top-k KV selection, FA 수정 |
| 8 | **KIVI** | Liu 2024 | 2402.02750 | 2-bit KV 양자화 (key per-channel / value per-token), plug-in 가능 |

보조: **StreamingLLM** (2309.17453), **ALISA** (2403.17312), **FastGen** (2310.01801)

**매트릭스**:

| 기법 | Offload 대상 | Granularity | Trigger | Lossless | 적합 workload |
|---|---|---|---|---|---|
| FlexGen | DRAM+SSD | tensor/layer | static schedule | ✓ | offline, throughput-only |
| vAttention | (HBM, 가상화) | page (2MB) | VMM on-demand | ✓ | vLLM 대체재, online |
| **InfiniGen** | DRAM | block/head | **speculative (layer i-1 hint)** | ~✓ (top-k approx) | **online, long-context** |
| LayerKV | DRAM | layer | TTFT schedule | ✓ | prefill-heavy |
| AttentionStore | DRAM+SSD | request | cross-req reuse | ✓ | multi-turn |
| H2O/Scissorhands | evict | token | attention score | ✗ | long gen |
| Quest | (HBM 내) | page (16 tok) | query-aware top-k | ✗ (approx) | long-context |
| KIVI / INT4 | compress in-place | tensor | always | ✗ (2-4 bit) | **직교** (병행 가능) |

**1순위 권고 경로**:
- **Phase 1 (2주)**: KIVI INT4 KV + pinned DRAM overflow pool. PagedAttention block_table 에 `tier: {HBM, DRAM}` 필드만 추가, miss 시 blocking memcpy (correctness first). 목표: batch 1500 가동.
- **Phase 2 (4주)**: InfiniGen 식 layer-ahead speculative predictor + double-buffered cudaMemcpyAsync. 목표: stall 제거, throughput 2× 접근.
- **Phase 3 (선택)**: Quest 식 query-aware page top-k 를 FA fork 에 통합해 HBM 사용량 50% 추가 절감.

### 4.3 [C] Prefill/Decode Disaggregation

| # | 논문 | 연도/venue | arXiv | 요약 |
|---|---|---|---|---|
| 1 | **Splitwise** | ISCA'24 | 2311.18677 | Prefill/decode 를 다른 머신 pool 로 분리, InfiniBand/NVLink KV 전송. Heterogeneous cluster |
| 2 | **DistServe** | OSDI'24 | 2401.09670 | SLO-aware goodput 최적화. 독립 TP/PP, NCCL KV transfer |
| 3 | **Sarathi-Serve** | OSDI'24 | 2403.02310 | **Chunked prefill 을 decode step 에 stall-free interleave**, disagg 없이 단일 worker. 2.6× throughput, TBT 0.3× |
| 4 | **Mooncake** | FAST'25 | 2407.00079 | P/D disagg + RDMA 기반 KVCache store, cache reuse 중심 |
| 5 | **LoongServe** | SOSP'24 | 2404.09526 | Long-context 용 elastic SP, worker 동적 조정 |
| 6 | **FastServe** | 2024 | 2305.05920 | Skip-join MLFQ preemptive scheduling |
| 7 | NanoFlow/POD/Chunk-Attention | 2024-2025 | - | Intra-device prefill-decode co-execution |

**같은 머신 CPU/GPU disagg 의 현실성**:
- PCIe Gen5 ×16 ≈ **50 GB/s 실측**. Llama-32B × 16K tokens × 64 layers × 2 × 8192 × 2B ≈ **34 GB KV → 0.7 s 전송**. H100 16K prefill 이 ~1.5 s 라면 전송이 prefill 의 47% 소요
- **AMX BF16 prefill throughput**: SPR 8480+ 96c 이론 BF16 peak ~60 TFLOPS, 실효 20-30 TFLOPS. H100 990 TFLOPS BF16 의 **2-3%**
- **32B dense 16K prefill**: H100 0.06 s vs CPU **2-3 s** → **CPU prefill 이 GPU prefill 보다 30-50× 느림**

→ **full Splitwise 식 CPU prefill worker 는 latency 면에서 망한다**. 단 **GPU 가 decode 로 꽉 찬 순간 prefill queue 가 몇 초 쌓이면** CPU prefill (2-3 s) 이 GPU 대기 (>5 s) 보다 빠를 수 있음 → **load-adaptive overflow routing** 만 의미 있음.

**1순위 권고**:
1. **Sarathi chunked prefill 을 GPU 에 먼저 적용** (이미 vLLM V1 에 있음) — low-risk, 2.6× 이득 검증됨
2. **CPU 는 high-load overflow prefill worker 로만** (input ≥ 8K 한정). Queue depth 기반 overflow 는 현 `CapacityAwareRouter` (Property 2 gate) 와 구조 동일, 임계만 prefill time 으로 바꾸면 됨
3. Full Splitwise 는 layer-pipelined KV copy 없이는 의미가 없고, 그건 최소 4주짜리 작업이며 PCIe bound 상한도 낮음

### 4.4 [D] CPU Inference / AMX / Quantization

| # | 프로젝트/논문 | 연도 | 요약 |
|---|---|---|---|
| 1 | **llama.cpp (ggerganov)** | 2023- | Q4_0/Q4_K/Q8_0 weight-only quant + AMX tile GEMM. SPR 실측 7B Q4 **12-18 tok/s** |
| 2 | **llama.cpp AMX PR #6341** | 2024 | `ggml-amx.cpp` AMX-INT8/BF16 tile dispatch. 우리가 이식할 참조 구현 |
| 3 | **PowerInfer-2** | 2024 | arxiv 2406.06282. Mobile/CPU sparse activation — hot/cold neuron split. ReLU 모델만 |
| 4 | **QServe** | 2024 | arxiv 2405.04532. W4A8KV4 — weight INT4 + activation INT8 + KV INT4 |
| 5 | **SmoothQuant** | 2023 | arxiv 2211.10438. Activation outlier → weight migration, W8A8 품질 보존. IPEX 직접 지원 |
| 6 | **AWQ** | 2023 | arxiv 2306.00978. Activation-aware weight-only INT4 |
| 7 | **IPEX LLM (Intel)** | 2024 | `ipex.llm.optimize` + `WeightOnlyQuantConfig(INT4/INT8)` — AMX-INT8 직접 dispatch |
| 8 | **Neural Speed (Intel, archived)** | 2024 | Intel 자체 llama.cpp fork, IPEX LLM 에 흡수 |
| 9 | **DeepSparse (NeuralMagic)** | 2021- | 2:4 structured sparse + INT8, SPR 지원 |
| 10 | **FBGEMM (Meta)** | 2018- | INT8 GEMM reference, AMX tile 추가, PyTorch INT8 backend |

**매트릭스**:

| 기법 | Target | Quant | BW 절감 | 구현 난이도 (우리 stack) |
|---|---|---|---|---|
| llama.cpp Q4_0 | CPU-only | W4 | 3.5× | 높음 (kernel 이식) |
| llama.cpp Q8_0 | CPU-only | W8 | 1.9× | 높음 |
| **IPEX WOQ INT8** | CPU | W8A16 | 1.9× | **낮음 (한 줄)** |
| **IPEX SmoothQuant** | CPU | W8A8 | 1.9× + compute 2× | 중간 (calib 필요) |
| AWQ (CPU kernel) | CPU | W4A16 | 3.5× | 매우 높음 (vLLM CPU kernel 부재) |
| QServe W4A8KV4 | GPU 주 | W4A8 | 3.5× + KV 4× | 매우 높음 (GPU only) |
| PowerInfer-2 | CPU/Mobile | FP16 sparse | 1.5-2× | 매우 높음 (ReLU only) |

**BW bound 극복 예상 수치** (SPR 8480+, 7B):
- BF16 baseline (현재): 이론 ~14 tok/s, 실측 **2.3** → overhead 큰 신호
- INT8 W8A16 (IPEX WOQ): 이론 ~28, 실측 기대 **8-12 tok/s**
- INT8 W8A8 (SmoothQuant + AMX-INT8): compute 2× 추가 → 실측 기대 **10-15 tok/s**
- INT4 (Q4_0): 이론 ~55, 실측 기대 **15-22 tok/s** (llama.cpp 보고치)

**1순위 권고 — Phase 0 진단 (2-3일)**:
```python
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.quantization import WoqWeightDtype
qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
    weight_dtype=WoqWeightDtype.INT8)
model = ipex.llm.optimize(model, quant_config=qconfig, dtype=torch.bfloat16)
```
- `cpu_worker.py` 의 IPEX optimize 호출 지점에 `quant_config` 한 줄 추가
- AMX-INT8 dispatch 는 oneDNN 자동 → `ONEDNN_VERBOSE=1` 로 `s8s8` 확인
- **분기 결정**:
  - 6+ tok/s 나옴 → BW 가 진짜 천장이 아니라 overhead. Phase 1 에서 CPU 경로 본격 활용
  - 여전히 2-3 tok/s → BW 가 진짜 천장. CPU 경로 포기, spec decode / MoE / KV offload 로 방향 전환
- **둘 다 정보적 가치**: 실패해도 "BW bound 가 아닌 다른 병목" 이라는 진단이 됨

### 4.5 [E] MoE Expert Offload

| # | 논문 | 저자/연도 | arXiv | 핵심 아이디어 |
|---|---|---|---|---|
| 1 | **Fiddler** | Kamahori 2024 | 2402.07033 | **Hot expert=GPU, cold=CPU DRAM + CPU 에서 직접 compute**. PCIe transfer 대신 activation 을 CPU 로 보냄. Mixtral-8x7B, single GPU 8.2× |
| 2 | **Mixtral-offloading** | Eliseev 2023 | 2312.17238 | Expert LRU cache + 2/3-bit quant. CPU 는 **store only**, compute 는 GPU (swap in) |
| 3 | **PowerInfer-2** | Xue 2024 | 2406.06282 | "Hot neuron" 을 GPU/NPU, cold 를 CPU. Dense 모델의 activation sparsity (ReLU 기반) |
| 4 | **MoE-Infinity** | Xue 2024 | 2401.14361 | Expert activation trace 기반 prefetch + request-level expert reuse. SSD 확장 |
| 5 | **Pre-gated MoE** | Hwang ISCA'24 | - | Gate 를 **한 레이어 미리** 평가해 다음 레이어 expert prefetch |
| 6 | **EdgeMoE** | Yi 2023 | 2308.14352 | Expert-wise bitwidth adaptation + in-memory buffer management |
| 7 | ExpertFlow / Lina | MLSys'23-24 | - | Cross-request expert caching, expert-level scheduling |

**매트릭스**:

| 기법 | Expert 위치 | Compute 위치 | Routing | Prefetch | Lossless | 우리 적합성 |
|---|---|---|---|---|---|---|
| **Fiddler** | Hot=HBM, Cold=DRAM | **CPU 직접** | Top-k 후 동적 dispatch | ✗ | ✓ | ⭐⭐⭐⭐⭐ |
| Mixtral-offload | HBM cache + DRAM | GPU (swap in) | LRU | Speculative | 2-bit lossy | ⭐⭐⭐ |
| MoE-Infinity | DRAM + SSD | GPU | Trace 기반 | Learned | ✓ | ⭐⭐⭐⭐ |
| Pre-gated MoE | HBM | GPU | 1-layer 선행 gate | Deterministic | 약한 lossy | ⭐⭐⭐ |
| PowerInfer-2 | Mix | CPU+GPU | Neuron hotness | Predictor | ReLU only | ⭐⭐ |
| EdgeMoE | DRAM | GPU | LRU | ✗ | mixed-bit | ⭐⭐ |

**왜 Fiddler 가 best-fit 인가**:
1. Fiddler 의 전제 (dense GPU/CPU 분리 프로세스 + CPU 가 충분히 빠름 + Expert 크기 작음 + DRAM 여유) 가 **우리 stack 과 정확히 일치**
2. SPR 8480+ 는 Fiddler 원논문 Xeon 6430 보다 2-3× 강력 (AMX BF16 추가)
3. Mixtral-8x7B expert = ~200 MB BF16, AMX GEMM 15-25 ms vs PCIe 전송 5 ms — **CPU compute 가 win 되는 영역**
4. 944 GB DRAM 은 Mixtral-8x22B (141B) FP16 저장에도 여유

**`expert_offload.py` stub 분석** (실측 리서치로 확인):
- `num_gpu_experts=8`, LRU swap, `swap_threshold=100` → **Mixtral-offload 구조**
- `_compute_cpu_experts` 가 SiLU+mm 직접 수행, ThreadPoolExecutor per-expert → **Fiddler 구조**
- INT8 quant path, IPEX 감지 → Mixtral-offload 확장
- `_grouped_topk` DeepSeek-V2/V3 지원 → 미래 확장

**현재 stub 의 문제점**:
1. `_compute_cpu_experts` 가 `.cpu()` **blocking copy** — GPU worker 안에서 동기 CPU compute 수행, **dual-process 전제 위배**
2. Expert weight 가 GPU worker 프로세스 메모리에 상주, CPU EngineCore 로 넘어가지 않음
3. `ThreadPoolExecutor` 는 GIL 때문에 AMX 성능 안 남, `init_cpu_threads_env` 기반 OMP 1:1 pin 과 별개로 동작
4. `CapacityAwareRouter` 와 연동 없음

**1순위 권고 경로 (2주)**:
- **Phase 1 (3일)**: Mixtral-8x7B 를 현재 dual-process 인프라에서 **expert offload 없이** 돌려 baseline. GPU util, HBM 점유, batch 상한 측정
- **Phase 2 (5일)**: `expert_offload.py` 를 Fiddler 구조로 리팩터
  - Expert weight 를 CPU EngineCore 프로세스에 상주 (ZMQ 로 layer 별 expert shard 전달)
  - `_compute_cpu_experts` 를 thread pool 대신 CPU EngineCore 로 RPC (`cpu_in_flight` 카운터 재사용)
  - Hot expert → GPU 상주, cold → CPU AMX BF16 compute
  - Router 결정: `cpu_in_flight < cpu_max_num_seqs * num_numa` 면 CPU expert 허용
- **Phase 3 (4일)**: Pre-gated MoE 의 1-layer 선행 gate 도입 + expert activation trace 로깅
- **Phase 4 (2일)**: Qwen1.5-MoE-A2.7B 재측정 (tokenizer 호환성 확인)

**핵심 리스크**:
1. CPU↔GPU activation transfer: 크기 ~1-4 MB / step, PCIe Gen5 ~50 µs, AMX expert compute 10-20 ms 대비 무시 가능
2. **Top-2 routing load imbalance**: Mixtral expert 중 소수만 hot, burst 시 cold expert latency spike → `cpu_max_num_seqs` 대신 **per-expert in-flight counter** 필요

---

## 5. 직교성 매트릭스 — 곱셈 효과 찾기

| | A1 spec | B1/B2 KV offload | C1 chunked | D1 INT8 | E1 MoE |
|---|---|---|---|---|---|
| **A1 spec** | — | ✓ 독립 | ✓ 직교 | ✓ drafter 가 더 빠름 | △ Mixtral 에는 효과 작음 |
| **B1/B2 KV** | ✓ 독립 | — | ✓ 독립 | ✓ 독립 | ✓ batch ↑ 에 도움 |
| **C1 chunked** | ✓ | ✓ | — | ✓ | ✓ |
| **D1 INT8** | ✓ drafter 2× | ✓ | ✓ | — | ✓ expert 2× |
| **E1 MoE** | △ | ✓ | ✓ | ✓ expert 2× | — |

**최대 곱셈 경로**: **D1 → (A1 × B1) → Phase 2 에 B2**. 즉 IPEX INT8 진단/활성화 → 32B spec decode + KIVI INT4 KV → InfiniGen speculative prefetch.

---

## 6. 통합 로드맵

### Phase 0 — 진단 (2-3일, **가장 먼저**)
- **[D1] IPEX WoqWeightDtype.INT8 활성화 + 7B 재측정**
  - `cpu_worker.py` 의 IPEX optimize 호출 지점에 `quant_config` 한 줄
  - 성공 지표: `ONEDNN_VERBOSE=1` 에서 `s8s8` AMX dispatch 확인 + 7B decode 6+ tok/s
  - 결과에 따라 Phase 1 의 CPU 경로 우선순위 결정

### Phase 1 — Breadth-first 병렬 검증 (2주, 3 track)
- **[A1] Spec decode CPU drafter** (32B target 한정)
  - `vllm/v1/spec_decode/cpu_drafter.py` 신규
  - Qwen2.5-0.5B 또는 Llama-3.2-1B drafter
  - DuoDecoding 식 γ=c 동적 window
  - 검증: 32B TPOT 41 → 22-28 ms, accept rate ≥ 0.6
- **[E1] Mixtral-8x7B + Fiddler expert offload**
  - 기존 `expert_offload.py` stub 을 Fiddler 로 재설계
  - Expert 를 CPU EngineCore 에 상주, activation 을 CPU 로 dispatch
  - 검증: dense 32B 대비 throughput 2×
- **[B1] KIVI INT4 KV compression** (저위험 병행)
  - `vllm/v1/core/block_pool.py` 에 INT4 block 옵션
  - HBM KV footprint 4× 축소 → batch 3-4×
  - 독립 commit, 실패해도 revert 쉬움

### Phase 2 — Depth-first 최적화 (4주, Phase 1 결과 따라 1-2개 선택)
- **[B2] InfiniGen speculative top-k prefetch**
  - FA v3 fork 또는 Triton PagedAttention gather path
  - Pinned DRAM pool + double-buffered cudaMemcpyAsync
  - 검증: 70B 또는 batch 1500+ workload 에서 throughput 2×
- **[A4] llama.cpp AMX-INT4 kernel 이식** (Phase 0 성공 시)
  - `csrc/cpu/gemm_amx_int4.cpp` 신규, Q4_0 dequant+GEMM fused tile
  - llama.cpp `ggml/src/ggml-cpu/amx/mmq.cpp` 참조 (MIT)
  - 검증: 7B 15-22 tok/s
- **[C1] Sarathi chunked prefill 보강 + CPU overflow**
  - Long-context (16K+) workload 에서 high-load 시 P99 TTFT 1.2-1.5×

### Phase 3 — Ninja gap 전면 (1-2 개월)
- [A1 × A4] Drafter 를 INT4 로 → draft throughput 2× 추가
- [B2 × B1] InfiniGen + KIVI → batch 5000+ 가능
- [E1 × B1] Mixtral + KIVI → weight HBM 감소 + KV HBM 감소, batch ↑
- **70B baseline (Llama-3.3-70B)** — HBM 압력 + KV offload demo
- **32K context workload** — P/D disagg 발현 조건

---

## 7. 논문 전체 목록 (30+ 편)

**Spec decode**: DuoDecoding (2503.00784), OmniDraft (2507.02659), EAGLE-3 (2503.01840), Medusa (2401.10774), SpecExec (2406.02532), ML-SpecQD (2503.13565), Dovetail (2412.18934), Leviathan (2211.17192), Mirror Spec Decode (2510.13161)

**KV offload / long-context**: FlexGen (2303.06865), vAttention (2405.04437), InfiniGen (OSDI'24 2406.19707), LayerKV (2410.00428), AttentionStore (2403.19708), H2O (2306.14048), Quest (ICML'24 2406.10774), KIVI (2402.02750), StreamingLLM (2309.17453), ALISA (2403.17312), FastGen (2310.01801)

**P/D disagg**: Splitwise (ISCA'24 2311.18677), DistServe (OSDI'24 2401.09670), Sarathi-Serve (OSDI'24 2403.02310), Mooncake (FAST'25 2407.00079), LoongServe (SOSP'24 2404.09526), FastServe (2305.05920)

**CPU quant / AMX**: llama.cpp AMX PR #6341, PowerInfer-2 (2406.06282), QServe (2405.04532), SmoothQuant (2211.10438), AWQ (2306.00978), IPEX LLM, Neural Speed, DeepSparse, FBGEMM

**MoE offload**: Fiddler (2402.07033), Mixtral-offload (2312.17238), MoE-Infinity (2401.14361), Pre-gated MoE (ISCA'24), EdgeMoE (2308.14352)

---

## 8. 가장 먼저 해야 할 한 가지

**Phase 0 의 D1 (IPEX `WoqWeightDtype.INT8`, 2-3일)**.

이유:
1. **작업량 가장 작음** — `cpu_worker.py` 한 줄 수정
2. **결정적 진단** — "BW 가 정말 천장인가" 에 대한 명확한 답. 결과에 따라 Phase 1 우선순위가 크게 달라짐
3. **실패 비용 0** — 효과 없으면 1 커밋 revert, 있으면 즉시 CPU 경로 2× 가속
4. **A1 spec decode 의 drafter 속도 기준값** — 0.5B INT8 drafter throughput 예측 근거

두 번째로 빠른 검증: **Qwen2.5-32B 에서 DuoDecoding-style spec decode (A1) 의 accept rate 실측** (1주). 0.5B drafter + 32B target 으로 실험 가능. 논문 수치 (OmniDraft 2.05×) 와 재현성 대조.

---

## 9. 관련 파일 / 경로 인덱스

**영향 받는 코드**:
- `vllm/v1/spec_decode/eagle.py`, `medusa.py`, `ngram_proposer_dynamic.py` — [A1] 확장 지점
- `vllm/v1/engine/hybrid_core.py::_route_throughput_adaptive` — 라우팅 gate (이미 Property 2 구현)
- `vllm/v1/engine/core_client.py::HybridAsyncMPClient` — [A1] dispatch 확장
- `vllm/v1/worker/cpu_worker.py` — [D1] IPEX quant_config 추가 지점, [E1] Fiddler CPU expert compute 호스트
- `vllm/v1/attention/backends/cpu_attn.py` — decode path counter
- `vllm/v1/core/block_pool.py`, `kv_cache_manager.py` — [B1] KV tier 확장
- `vllm/model_executor/layers/fused_moe/expert_offload.py` — [E1] stub 재설계 대상
- `vllm/engine/disaggregated/` — [C1] stub 확장
- `csrc/cpu/gemm_vnni.cpp` — [A4] AMX-INT4 시작점
- `csrc/cpu/torch_bindings_utils.cpp` — `_C_cpu_ops` 등록

**참조 자료**:
- `docs/paper/main.tex` §3 Property 2 — 정량식 업데이트 필요 (TODO §4.1)
- `docs/HYBRID_OPTIONS_IMPLEMENTATION_PLAN.md:508` — 기존 `expert_offload.py` 70% 진단
- `experiment_result/20260411_143500_h100x4_isa_verification_and_ninja_gap_strategy/` — 이전 ninja gap 분석 (본 문서의 확장판)

**관련 이전 실험**:
- `20260411_082501_h100x4_qwen7b_smoke_cpu_bw_diagnosis` — BW bound 가설 최초 제기
- `20260411_141500_h100x4_qwen1.5b_routing_regression_root_cause_fix` — Property 2 gate 구현
- `20260411_142900_h100x4_qwen1.5b_7b_gpu_only_vs_hybrid_4runs` — 1.5B/7B 수정 검증
- `20260411_145900_h100x4_qwen32b_gpu_only_vs_hybrid_baseline` — 32B scaling

---

## 10. 다음 단계 (사용자 결정 대기)

본 ideation 노트의 통합 로드맵 중 다음 중 하나를 선택:

(a) **Phase 0 D1 바로 진행** — IPEX INT8 활성화 + 7B 재측정 (2-3일)
(b) **Phase 1 A1 선행** — 32B spec decode CPU drafter 구현 (1주)
(c) **Phase 1 E1 선행** — Mixtral-8x7B baseline + Fiddler 재설계 (2주)
(d) **리서치 확장** — 특정 논문 (예: InfiniGen, Fiddler) 의 원문 deep-dive
(e) **또 다른 주제** — 본 서베이에서 다루지 않은 영역 (예: continuous batching 개선, hardware-aware scheduling, 에너지 최적화)
