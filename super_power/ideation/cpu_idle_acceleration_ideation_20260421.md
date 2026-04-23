# CPU 유휴 자원 활용 LLM 추론 성능 향상 — Ideation

**작성일**: 2026-04-21
**작성 기준**: `vllm_hybrid` 프로젝트 단일 목표 — *"시스템 추론 성능을 CPU 유휴 자원을 활용하여 높인다"*
**문서 목적**: 추측 아닌 실측 / 논문 / OSS 근거가 있는 기법을 카테고리별로 정리하고, 그 위에 새로운 결합 아이디어를 추가.

---

## 0. 프로젝트 현 상태 (실측 기반)

- **환경**: Qwen2.5-32B × H100x8 (TP=8) × 500 req × 128/128
- **gpu_only 기준선**: outTP **11,523 tok/s**
- **hybrid 최고치** (§06-1 v1 seqs=1): outTP **1,196 tok/s** = gpu_only 의 **10.4%**
- **주원인** (Tech_done v8 §SSOT-3): CPU engine 의 batch 병렬화가 M>1 에서 super-linear 비용 증가 (seqs=1→8 에서 outTP 4.4× 감소)
- **기각 이력**: §03 Phase 2 (SPR TLB 역효과), §04 (IPEX Linear 비호환), §06-1 v2 (VNNI 반쪽 tile + compensation 오버헤드), §11 Phase 1 (remainder path = IPEX 와 동일 구조), §16 (unstructured pruning 은 GPU tensor core 미지원)
- **보류**: §28 xFT (AMX 성능은 Intel 독점 xDNN 바이너리 의존)

**남은 Tier 1 후보 (문서 기준)**: §22 NEO, §13 T-MAC — 각자 흠 있음.

이 문서는 **§22/§13 에 갇히지 않고** 처음부터 다시 탐색한다.

---

## 1. Survey — 카테고리별 정리 (실측 / 구현 근거 있는 것만)

### A. Asymmetric CPU-GPU Decode Split (decode step 단위 분할)

같은 step 의 sublayer 를 GPU (weight matmul 중심) + CPU (attention KV scan 중심) 로 나눠 파이프라인 병렬 실행.

| 시스템 | 논문 / 실측 수치 | 코드 | 모델 변경 |
|---|---|---|---|
| **NEO** | [arXiv 2411.01142](https://arxiv.org/abs/2411.01142) / [MLSys'25 poster](https://mlsys.org/virtual/2025/3346) / [PDF](http://minlanyu.seas.harvard.edu/writeup/mlsys25.pdf) — T4 환경 **최대 7.5× throughput**, H100+70B 14.3% | [github.com/NEO-MLSys25/NEO](https://github.com/NEO-MLSys25/NEO) | 없음 |
| **APEX** | [arXiv 2506.03296](https://arxiv.org/abs/2506.03296) (2025) — Asynchronous Overlap Execution, CPU attention 결과를 GPU unified batch 안에 숨김. 제한된 GPU 환경 전용 | 공개 여부 불명 | 없음 |
| **KTransformers** | [SOSP'25 PDF](https://dl.acm.org/doi/pdf/10.1145/3731569.3764843) / [DOI 10.1145/3731569.3764843](https://dl.acm.org/doi/10.1145/3731569.3764843) — MoE 대상. prefill 4.62~19.74×, decode 1.25~4.09× speedup. **AMX-specialized kernel** + **Expert Deferral** 로 CPU 활용도 75% → 100% | [github.com/kvcache-ai/ktransformers](https://github.com/kvcache-ai/ktransformers) | 없음 (weight 분배만) |
| **LIA** | [ISCA'52 (2024) DOI 10.1145/3695053.3731092](https://dl.acm.org/doi/full/10.1145/3695053.3731092) — 단일 GPU + AMX CPU + CXL memory. 온/오프라인 모두 | 공개 여부 불명 | 없음 |
| **HGCA** | [arXiv 2507.03153](https://arxiv.org/abs/2507.03153) (2025) — long-context 전용 GPU-CPU attention | 공개 여부 불명 | 없음 |

**핵심 관찰**:
- NEO 는 우리 환경에 가장 가까움 (H100+70B) — 단 70B 기준
- KTransformers 는 MoE 전용 — Qwen2.5 dense 엔 바로 적용 불가
- APEX 는 "constrained GPU" 전제 — H100 처럼 여유 있는 GPU 에선 이득 줄 수 있음
- LIA 는 CXL 까지 포함 — 우리 하드웨어 외

### B. Heterogeneous Speculative Decoding (drafter + verifier 분산)

작은 drafter 모델이 k 토큰 생성 → 큰 verifier 가 한 번에 검증 → accept 되면 여러 토큰 free.

| 시스템 | 논문 | 역할 분담 | 이득 |
|---|---|---|---|
| **Dovetail** | [arXiv 2412.18934](https://arxiv.org/abs/2412.18934) (2024) / [ACL Anthology 2025.emnlp-main.879](https://aclanthology.org/2025.emnlp-main.879.pdf) | **CPU: target / GPU: drafter** — 약한 GPU + 강한 CPU 환경용 | LLaMA2-7B 3GB VRAM 에서 5.86 tok/s (CPU-only 대비 2.77×). 7GB 시 8 tok/s |
| **DuoDecoding** | [arXiv 2503.00784](https://arxiv.org/abs/2503.00784) — Lv et al. "DuoDecoding: Hardware-aware Heterogeneous Speculative Decoding with Dynamic Multi-Sequence Drafting" | CPU drafter + GPU verifier | TPOT 2.1~2.61× (GPU drafter 대조) |
| **원전 spec decode** | [Leviathan et al. (arXiv 2211.17192)](https://arxiv.org/abs/2211.17192), [Chen et al. (arXiv 2302.01318)](https://arxiv.org/abs/2302.01318) | GPU-GPU | ~2× |
| **EAGLE / Medusa** | [EAGLE arXiv 2401.15077](https://arxiv.org/abs/2401.15077), [Medusa arXiv 2401.10774](https://arxiv.org/abs/2401.10774) | GPU-GPU, tree draft | ~2~3× |
| **vLLM upstream spec decode (CPU+GPU)** | [vllm PR action 10573538228](https://github.com/vllm-project/vllm/actions/runs/10573538228) — 실험 상태 | CPU drafter + GPU verifier | 프로덕션 미정착 |

**핵심 관찰**: 방향은 둘 다 가능. 우리 환경처럼 **강한 GPU + 유휴 CPU** 일 때는 Dovetail 의 반대 방향 (GPU=target, CPU=drafter) 이 자연스러움. 실측 수치는 CPU drafter + GPU verifier 조합이 GPU drafter + GPU verifier 보다 약간 떨어지는 것이 일반적.

### C. KV Cache Offload + Predictive Prefetch

CPU DRAM 에 KV block 저장, 필요 블록을 **예측 prefetch** 해 GPU 로 전송.

| 시스템 | 논문 / 코드 | 메커니즘 | 이득 |
|---|---|---|---|
| **InfiniGen** | [arXiv 2406.19707](https://arxiv.org/abs/2406.19707) — Lee et al. OSDI'24 | 현재 layer 의 일부 Q×K rehearsal 로 다음 layer 에 "필요한" KV 예측 → 그것만 prefetch | FlexGen 대비 inference speed 상승 |
| **FlexGen** | [arXiv 2303.06865](https://arxiv.org/abs/2303.06865) — Sheng et al. ICML'23 / [github.com/FMInference/FlexLLMGen](https://github.com/FMInference/FlexLLMGen) | weights / activations / KV 를 GPU/CPU/disk 에 분산. 4-bit 압축 결합 | single GPU + 대형 모델 가능 |
| **LMCache** | [arXiv 2510.09665](https://arxiv.org/abs/2510.09665) / [tech report](https://lmcache.ai/tech_report.pdf) / [lmcache.ai](https://lmcache.ai) | CPU / filesystem / Mooncake / ValKey 백엔드 저장 | enterprise-scale |
| **Mooncake** | [arXiv 2407.00079](https://arxiv.org/abs/2407.00079) — Moonshot AI / [github.com/kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake) | disaggregated + CPU KV pool | production 운영 |
| **KVSwap** | [arXiv 2511.11907](https://arxiv.org/abs/2511.11907) (2025) | disk-aware on-device | long-context |

**핵심 관찰**: CPU DRAM 은 H100 HBM 보다 훨씬 큼 (~1 TB vs 80 GB). Long-context 에서 critical. 현재 우리 workload (128/128) 에선 기여 0 — 하지만 long-context 로 가면 핵심.

### D. Prefill-Decode Disaggregation (단계 분리)

TTFT (prefill) 와 TPOT (decode) 의 SLO 다름 → 분리해서 다른 HW 에 배치.

| 시스템 | 논문 | 분리 방식 | 이득 |
|---|---|---|---|
| **DistServe** | [arXiv 2401.09670](https://arxiv.org/abs/2401.09670) — Zhong et al. OSDI'24 / [USENIX PDF](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf) | 다른 GPU 에 prefill / decode | 7.4× 더 많은 request, 12.6× tighter SLO |
| **Splitwise** | [arXiv 2311.18677](https://arxiv.org/abs/2311.18677) — Patel et al. MICRO'24 | 이종 GPU (H100 + A100). architectural | energy-efficient |
| **Mooncake** | (C 섹션 참조) | KV-centric disaggregation | production 대규모 |
| **DuetServe** | [arXiv 2511.04791](https://arxiv.org/abs/2511.04791) (2025) | prefill/decode harmonize | goodput 최적화 |
| **TetriInfer / DejaVu** | DistServe 후속 concurrent 작업. 세부 URL 확인 필요 | 유사 | — |

**핵심 관찰**: 현존 연구는 대부분 **GPU ↔ GPU** disaggregation. CPU 를 prefill 에 쓰는 논문은 드묾 (CPU prefill 느려서 TTFT 망가짐). 하지만 **prefill-bound 가 아닌 long-context 에서 CPU-prefill 로 GPU HBM 절약**은 고려 가능.

### E. CPU 보조 Attention Sparsification (top-k KV 선택)

전체 KV 에 attention 계산하지 말고 **중요한 top-k block** 만 선택.

| 시스템 | 논문 / 코드 | 메커니즘 | 이득 |
|---|---|---|---|
| **Quest** | [arXiv 2406.10774](https://arxiv.org/abs/2406.10774) — Tang et al. ICML'24 / [github.com/mit-han-lab/Quest](https://github.com/mit-han-lab/Quest) | KV page 단위 min/max key 추적, query 로 critical page 추정 | attention **7.03×**, FlashInfer 4-bit 대비 2.23× inference |
| **SparQ Attention** | [arXiv 2312.04985](https://arxiv.org/abs/2312.04985) / [OpenReview cp1hJ67l3M](https://openreview.net/pdf?id=cp1hJ67l3M) | query 상위 r 컴포넌트로 top-k 선택 | **8× compression**, 정확도 유지 |
| **H2O** | [arXiv 2306.14048](https://arxiv.org/abs/2306.14048) — Zhang et al. NeurIPS'23 | attention score 누적 → heavy hitter eviction | KV 크기 축소 |
| **Double Sparsity** | [arXiv 2408.07092](https://arxiv.org/abs/2408.07092) | post-training sparse attention | retrain 없이 |
| **Loki** | [NeurIPS'24 PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/1e027da6bec9ceb2ec37951ceeccae93-Paper-Conference.pdf) | low-rank keys 로 sparse attention | 효율 |
| **ScoutAttention** | [arXiv 2603.27138](https://arxiv.org/abs/2603.27138) (2025) | **layer-ahead CPU pre-computation** — CPU 가 한 layer 앞서 Q 예측 → top-k KV 선별 → GPU 는 hot block 만 | long-context decoding 5.1× |

**핵심 관찰**: ScoutAttention 이 우리 방향 정확히 맞음 — "CPU 가 layer 앞서 판단". 우리 프로젝트의 `NinjaGap_Todo/21_scout_attention.md` 에 이미 정리됨 (현재 "장거리 보류"). 단, 현재 workload (128 out) 에선 기여 작음.

### F. MoE Expert Offload (expert-sparsity 활용)

MoE 모델에서만 적용. Qwen2.5-32B 는 dense 라 **scope 밖**. 참고로 정리:

| 시스템 | 논문 / 코드 | 이득 |
|---|---|---|
| **Fiddler** | [arXiv 2402.07033](https://arxiv.org/abs/2402.07033) — ICLR'25 / [github.com/efeslab/fiddler](https://github.com/efeslab/fiddler) | Mixtral-8x7B (90GB+) 를 24GB GPU 에서 3+ tok/s |
| **KTransformers** | (A 섹션 참조) | MoE prefill 4.62~19.74× |
| **HybriMoE** | [arXiv 2504.05897](https://arxiv.org/abs/2504.05897) | hybrid schedule + cache |
| **PreScope** | [arXiv 2509.23638](https://arxiv.org/abs/2509.23638) | prefetch for MoE |
| **DALI** | [arXiv 2602.03495](https://arxiv.org/abs/2602.03495) | workload-aware offload for PC |

현재 Qwen2.5-32B 에는 적용 불가. 나중에 Qwen3 MoE / DeepSeek-V3 류 지원 시 재방문.

### G. Asynchronous CPU Overlap / Prefetch

CPU / CUDA stream 을 분리해 I/O 와 compute 를 겹치기.

| 시스템 | 논문 / 블로그 | 메커니즘 |
|---|---|---|
| **vLLM weight offload v2** | [vLLM blog 2026-02-03](https://blog.vllm.ai/2026/02/03/dsr1-gb200-part1.html) | 다음 layer weight 를 별도 CUDA stream 으로 async onload, kernel 실행과 겹침 |
| **APEX** | (A 섹션 참조) | async overlap |
| **Async Model Offload** | [SSDBM'25 poster PDF](https://ssdbm.org/2025/assets/poster/8884-Jie.pdf) — Ye & Nicolae | 별도 CUDA stream 으로 block prefetch |
| **Async KV Prefetch** | [arXiv 2504.06319](https://arxiv.org/abs/2504.06319) | I/O + compute overlap |
| **PRESERVE** | allReduce 중 prefetch | multi-GPU 시나리오 (single GPU 가치 작음) |

**핵심 관찰**: 이미 upstream vLLM 에 유사 기능 존재. 우리 hybrid 엔진에서 CUDA stream 분리 기반 CPU ↔ GPU pipelining 기회 있음.

### H. CPU 보조 Sampling / Logits Processing / Grammar

생성 단계에서 토큰 하나 뽑는 과정 (sampling + logits processing + grammar check) 을 CPU 로 오프로드.

| 시스템 | 논문 / 블로그 | 주장 |
|---|---|---|
| **XGrammar** | [MLC blog 2024-11](https://blog.mlc.ai/2024/11/22/achieving-efficient-flexible-portable-structured-generation-with-xgrammar) / [github.com/mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar) | grammar processing 을 GPU compute 와 overlap |
| **vLLM anatomy** | [vLLM blog 2025-09-05](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html) | CPU-bound detokenization → chunking 으로 완화 |
| **Guiding LLMs** | [arXiv 2403.06988](https://arxiv.org/abs/2403.06988) | non-invasive constrained generation |
| **SGLang** | [github.com/sgl-project/sglang](https://github.com/sgl-project/sglang) | RadixAttention + grammar 효율화 |

**핵심 관찰**: decode step 당 compute 중 매우 작은 비율이지만, 프로덕션에서 누적 latency 에 영향. 특히 **constrained decoding / JSON mode / tool calling** 워크로드에서 CPU 가 할 일 많음. 우리 프로젝트엔 아직 미도입.

---

## 2. 새 아이디어 — 위 카테고리 결합 / 우리 환경 특이성

### 2-1. **CPU-assisted 동적 배치 planner** [신규, 근거 C/D]

**관찰**: 현재 vLLM scheduler 는 GPU 주 프로세스에서 돌아감. batch planning 자체가 **요청 규모에 비례하는 Python 오버헤드**. CPU engine 이 놀고 있는 동안 **다음 step 의 batch 구성 / attention metadata / position ids / rope cache** 를 미리 계산해 GPU 에 전달하면 scheduler cost 을 hide 가능.

**근거 기반**:
- [vLLM anatomy blog (2025-09-05)](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html) — "CPU-bound output processing can create significant bottlenecks"
- [APEX (arXiv 2506.03296)](https://arxiv.org/abs/2506.03296) — asynchronous overlap 원리와 정합

**모델 변경**: 없음
**리스크**: scheduler 재설계 (§22 보다 작음, routing 축 건드리지 않음)
**기대 이득**: 불명 — 프로파일 먼저 해서 scheduler overhead 비중 확인 필요

### 2-2. **CPU drafter + GPU verifier (§18 재방문)** [신규, [Dovetail](https://arxiv.org/abs/2412.18934) 역전]

**관찰**: Dovetail 은 약한 GPU 환경에서 "CPU=target / GPU=drafter" 로 갔음. 우리는 **강한 GPU + 유휴 CPU** 반대 상황. GPU=full 32B target, CPU 가 **작은 모델 (Qwen2.5-0.5B) 의 drafter 역할**. 이건 [DuoDecoding (arXiv 2503.00784)](https://arxiv.org/abs/2503.00784) 의 기본 형태 — 우리 NinjaGap 문서 [§18 Spec Decode CPU drafter](../../NinjaGap_Todo/18_spec_decode_cpu_drafter.md) 에서 "CPU balance 조건 미충족" 으로 강등됐음.

**새 각도**: **CPU 가 drafter + GPU 가 verifier + 여러 request 를 큐로 관리** 해 CPU drafter 의 느림을 요청 다중화로 상쇄. single-request 가 아닌 batched spec decode. 이 스케일에서 ROI 재평가 필요.

**참고 구현**:
- [Leviathan et al. (arXiv 2211.17192)](https://arxiv.org/abs/2211.17192) — spec decode 원전 이론
- [EAGLE (arXiv 2401.15077)](https://arxiv.org/abs/2401.15077), [Medusa (arXiv 2401.10774)](https://arxiv.org/abs/2401.10774) — tree draft
- [vLLM upstream spec decode experiment](https://github.com/vllm-project/vllm/actions/runs/10573538228)

### 2-3. **CPU-side "Cold KV" staging (long-context 준비)** [신규, C 카테고리 확장]

**관찰**: 우리 workload 는 현재 128/128 단문 — [InfiniGen (arXiv 2406.19707)](https://arxiv.org/abs/2406.19707) / [Mooncake (arXiv 2407.00079)](https://arxiv.org/abs/2407.00079) / [LMCache (arXiv 2510.09665)](https://arxiv.org/abs/2510.09665) 등 KV offload 기법의 이득이 0. 하지만 **사용 시나리오가 long-context 로 이동하면 (8K/32K/100K input)** CPU DRAM 활용이 핵심이 됨.

**아이디어**: 지금부터 CPU 에 **"cold KV pool"** 을 구성해 두고, context 길이가 threshold 초과 시 자동으로 old KV block 을 CPU 로 eviction + InfiniGen 방식 prefetch 도입. 128 workload 에선 overhead=0 (코드 경로 안 탐), long-ctx workload 에선 이득 즉시 작동.

**근거**: InfiniGen OSDI'24, LMCache (tech report), Mooncake (Moonshot AI 프로덕션) 모두 production 검증.

**단점**: 현재 primary workload 에 기여 0. 하지만 **다음 workload 전환 대비 인프라** 로 가치.

### 2-4. **CPU Speculative Logits Rerank (GPU sampling 결과 재평가)** [신규]

**관찰**: GPU 가 top-k logits 출력 후 sampling 전에, CPU 가 별도로 **가벼운 보조 모델 (Q8_0 로 양자화된 동일 모델의 "last few layers" 만)** 을 돌려 logits 를 재평가. 두 결과가 합의하면 빠른 sampling, 불일치면 GPU 재확인.

**기존 연구 대응**: [EAGLE (arXiv 2401.15077)](https://arxiv.org/abs/2401.15077) / [Medusa (arXiv 2401.10774)](https://arxiv.org/abs/2401.10774) 계열 draft model 의 역방향.

**단점**: 정확도 영향 평가 필요, 추정 근거 약함 (D급). 실측 필수.

### 2-5. **Constrained Decoding 전담 CPU Worker** [신규, H 카테고리 실체화]

**관찰**: JSON mode / 함수 호출 / regex 제약 같은 **constrained decoding** 워크로드에서는 매 step grammar state update + token mask 계산이 필요. 이게 CPU-bound. [XGrammar 블로그](https://blog.mlc.ai/2024/11/22/achieving-efficient-flexible-portable-structured-generation-with-xgrammar) 가 "GPU compute 와 overlap" 주장.

**아이디어**: 현재 hybrid engine 의 CPU engine 1 개를 "grammar/constraint 전용 워커" 로 재정의. 일반 요청은 GPU, constrained 요청은 CPU 가 mask 계산 + GPU 가 masked sampling. 용도 특화.

**근거**:
- [XGrammar](https://blog.mlc.ai/2024/11/22/achieving-efficient-flexible-portable-structured-generation-with-xgrammar) / [github.com/mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar)
- [Guiding LLMs The Right Way (arXiv 2403.06988)](https://arxiv.org/abs/2403.06988) — Beurer-Kellner et al., non-invasive constrained generation
- [llama.cpp grammar](https://github.com/ggml-org/llama.cpp) — 실전 구현

**단점**: workload 특이 (일반 텍스트 생성엔 적용 안 됨)

### 2-6. **CPU prefill-assist for MEDIUM context (500~2K)** [신규]

**관찰**: 현재 workload 128 에선 GPU prefill 빠름. 하지만 **512~2K 중간 구간** 에서 GPU 는 prefill-compute-bound, 동시에 CPU 는 유휴. CPU 가 **초기 1~2 layer embedding / position encoding / rope table** 을 미리 준비해 두면 GPU prefill TTFT 단축 가능.

**근거**: [DistServe (arXiv 2401.09670)](https://arxiv.org/abs/2401.09670) 의 prefill 분리 원리를 CPU 측에서 **보조적으로** 적용. 기존 논문에 직접 대응은 없음 — "CPU 측 prefill 보조" 라는 gap 이 연구 지형에 있음.

**단점**: 효과 미검증 (D). 프로파일 필요.

### 2-7. **"CPU Background Compiler"** — layer weights preprocessing / repacking [신규]

**관찰**: 우리 §06 에서 load-time Q8_0 변환은 이미 존재. 하지만 **runtime 중** 에도 CPU 는 대부분 유휴. CPU 가 background 에서 다음 layer 의 weight layout 을 **AMX-friendly tile packing** 하거나, activation 범위 통계 (calibration) 를 계속 수집해 **다음 batch 의 동적 quantization scale** 을 업데이트.

**근거**: [APEX (arXiv 2506.03296)](https://arxiv.org/abs/2506.03296) / [Async KV Prefetch (arXiv 2504.06319)](https://arxiv.org/abs/2504.06319) 의 "PCIe prefetch" 원리의 "CPU 로 compute 도 옮기는" 확장. 직접 대응 논문 없음 — 확장.

### 2-8. **GPU-idle-phase CPU Burst** [신규]

**관찰**: Decode step 에서 GPU 가 attention 수행 중 (memory-bound) CPU 는 대부분 idle. 역으로 GPU 가 linear matmul 수행 중 (compute-bound) 도 CPU idle. 매 step **GPU 의 phase 에 따라 CPU 가 다른 보조 작업** 수행.

| GPU phase | CPU 작업 후보 |
|---|---|
| attention (memory-bound) | 다음 step scheduling, detokenize, grammar update |
| linear (compute-bound) | KV prefetch/evict, speculative draft, logits post-process |

**근거**: [NEO (arXiv 2411.01142)](https://arxiv.org/abs/2411.01142) 의 asymmetric pipeline 을 **phase-aware** 로 확장. 논문 수준은 없음 (D).

---

## 3. 현 프로젝트 기반 vs 새로 해볼 것 — Matrix

| # | 아이디어 | 근거 등급 | HW 일치도 | 모델 변경 | 구현 공수 | 현재 workload 기여 | 장기 가치 |
|---|---|:---:|---|---|---|---|---|
| §22 NEO (Tier 1 현재) | A | B | H100+70B 실측 | 없음 | 크다 (scheduler 재설계) | 추정 5~10% | 높음 |
| §13 T-MAC (Tier 1 현재) | C | 재검증 | **있음** (INT4) | 크다 (LUT kernel) | 불명 | 중 |
| §28 xFT (보류) | B (wrapper) | SPR | 없음 | 중 + xDNN 의존 | 모름 | 낮음 (closed binary 리스크) |
| 2-1 CPU-assisted batch planner | C | 동일 | 없음 | 중 | 가능성 있음 (프로파일 후) | 중 |
| 2-2 CPU drafter (§18 재방문) | B | 유사 | 없음 | 크다 | 가능성 있음 | 높음 (구조적) |
| 2-3 CPU cold KV pool (long-ctx 준비) | A | 동일 | 없음 | 중 | **0** (128/128) | **높음 (장기)** |
| 2-4 CPU spec logits rerank | D | 동일 | 없음 | 크다 | 불명 | 낮음 |
| 2-5 Constrained decoding CPU worker | B | 동일 | 없음 | 중 | workload 의존 | 중 (특수) |
| 2-6 CPU prefill-assist (medium ctx) | D | 동일 | 없음 | 중 | 불명 | 중 |
| 2-7 CPU background compiler | D | 동일 | 없음 | 크다 | 불명 | 중 |
| 2-8 Phase-aware CPU burst | D | 동일 | 없음 | 크다 | 불명 | 높음 |

**판정 규칙 (Tier 기준)**: 근거 A/B + 현재 workload 기여 있음 → Tier 1 후보.

**이 기준으로 Tier 1 승격 후보**:
- §22 NEO (유지)
- 2-1 CPU-assisted batch planner (측정 선행 시)
- 2-2 CPU drafter 재방문 (구조적 가치)

**장기 인프라 투자**:
- 2-3 CPU cold KV pool (long-context 시나리오 대비)

**실측 선행 필요**:
- 모든 D 급은 profile 데이터 본 후 재평가

---

## 4. 관찰 — 논문 지형의 "빈 공간"

Survey 정리하다 드러난 **직접 대응 논문이 없는** 영역:

1. **CPU-assisted scheduler** (2-1): 대부분 논문은 CPU 를 compute 로 씀. Scheduler 자체를 CPU 에 오프로드하는 연구 거의 없음
2. **Phase-aware CPU burst** (2-8): NEO 의 파이프라인을 sublayer phase 단위로 세분화하는 연구 없음
3. **CPU Background Compiler** (2-7): runtime 중 CPU 가 다음 layer weight/activation 준비 — quantization 파이프라인을 "offline pre-pack" 아닌 "online incremental" 로 돌리는 연구 없음
4. **CPU prefill-assist for medium context** (2-6): prefill disaggregation 은 GPU-GPU 전용. CPU 가 "GPU prefill 보조" 하는 연구 미발견

이 4 개가 **새로운 연구 기회** 일 수 있음. 단 우리 프로젝트는 논문 작성이 아닌 production 개선이 목적이므로, 가치 평가는 **우리 실측 기준** 으로.

---

## 5. 다음 단계 제안 (사용자 판단용)

**A. 측정 선행 (권장)**
- 현재 §06-1 v1 상태에서 `VLLM_HYBRID_PROFILE=1` + sublayer breakdown 측정
- 어느 구간 (scheduler / MLP / attention / sampling) 이 실제 bottleneck 인지 확정
- 그 결과에 따라 위 Matrix 의 Tier 1 후보 재정렬

**B. 구조적 재방문 — §18 CPU drafter**
- Dovetail 역방향 + KTransformers async scheduling 조합
- 구현 공수 크지만 장기 가치 큼

**C. 장기 인프라 투자 — 2-3 CPU cold KV pool**
- 현재 workload 기여 0 이지만 long-ctx 시나리오 대비
- InfiniGen 구현 이식 + 현 hybrid engine 에 통합

**D. 문서 공식 우선순위 유지 — §22 NEO**
- 기존 계획대로 착수

A 가 근거 확보 측면에서 가장 안전. D 가 문서 일관성 측면에서 가장 단순. B/C 는 더 큰 투자.

사용자 결정 대기.

---

## 6. 참고 문헌 (arXiv / DOI / GitHub / 블로그 직접 링크)

> 본 섹션은 본문 인라인 링크의 정본. 접근 불가 시 arXiv abstract URL 로 대체 가능.

### Asymmetric CPU-GPU
| 시스템 | 논문 | 출판처 | 코드 |
|---|---|---|---|
| NEO | [arXiv 2411.01142](https://arxiv.org/abs/2411.01142) | MLSys'25 ([poster](https://mlsys.org/virtual/2025/3346)) | [github.com/NEO-MLSys25/NEO](https://github.com/NEO-MLSys25/NEO) |
| APEX | [arXiv 2506.03296](https://arxiv.org/abs/2506.03296) | 2025 preprint | — |
| KTransformers | [DOI 10.1145/3731569.3764843](https://dl.acm.org/doi/10.1145/3731569.3764843) | SOSP'25 | [github.com/kvcache-ai/ktransformers](https://github.com/kvcache-ai/ktransformers) |
| HGCA | [arXiv 2507.03153](https://arxiv.org/abs/2507.03153) | 2025 preprint | — |
| LIA | [DOI 10.1145/3695053.3731092](https://dl.acm.org/doi/full/10.1145/3695053.3731092) | ISCA'52 | — |
| Challenging GPU Dominance | [arXiv 2505.06461](https://arxiv.org/abs/2505.06461) | 2025 (on-device) | — |

### Heterogeneous Spec Decoding
| 시스템 | 논문 | 출판처 |
|---|---|---|
| Dovetail | [arXiv 2412.18934](https://arxiv.org/abs/2412.18934) | EMNLP'25 ([2025.emnlp-main.879](https://aclanthology.org/2025.emnlp-main.879.pdf)) |
| DuoDecoding | [arXiv 2503.00784](https://arxiv.org/abs/2503.00784) | 2025 preprint |
| Spec decode 원전 | [Leviathan et al. (arXiv 2211.17192)](https://arxiv.org/abs/2211.17192) | 2022 |
| Chen et al. spec sampling | [arXiv 2302.01318](https://arxiv.org/abs/2302.01318) | 2023 |
| EAGLE | [arXiv 2401.15077](https://arxiv.org/abs/2401.15077) | 2024 |
| Medusa | [arXiv 2401.10774](https://arxiv.org/abs/2401.10774) | 2024 |
| vLLM CPU+GPU spec PR | [actions run 10573538228](https://github.com/vllm-project/vllm/actions/runs/10573538228) | 실험 |

### KV Offload
| 시스템 | 논문 | 출판처 | 코드 |
|---|---|---|---|
| InfiniGen | [arXiv 2406.19707](https://arxiv.org/abs/2406.19707) | OSDI'24 | — |
| FlexGen | [arXiv 2303.06865](https://arxiv.org/abs/2303.06865) | ICML'23 | [FMInference/FlexLLMGen](https://github.com/FMInference/FlexLLMGen) |
| LMCache | [arXiv 2510.09665](https://arxiv.org/abs/2510.09665) / [tech report](https://lmcache.ai/tech_report.pdf) | 2025 | [lmcache.ai](https://lmcache.ai) |
| Mooncake | [arXiv 2407.00079](https://arxiv.org/abs/2407.00079) | Moonshot AI production | [kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake) |
| KVSwap | [arXiv 2511.11907](https://arxiv.org/abs/2511.11907) | 2025 preprint | — |

### P/D Disaggregation
| 시스템 | 논문 | 출판처 |
|---|---|---|
| DistServe | [arXiv 2401.09670](https://arxiv.org/abs/2401.09670) / [USENIX PDF](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf) | OSDI'24 |
| DuetServe | [arXiv 2511.04791](https://arxiv.org/abs/2511.04791) | 2025 |
| Splitwise | [arXiv 2311.18677](https://arxiv.org/abs/2311.18677) | MICRO'24 |

### Sparse Attention
| 시스템 | 논문 | 출판처 | 코드 |
|---|---|---|---|
| Quest | [arXiv 2406.10774](https://arxiv.org/abs/2406.10774) | ICML'24 | [mit-han-lab/Quest](https://github.com/mit-han-lab/Quest) |
| SparQ | [arXiv 2312.04985](https://arxiv.org/abs/2312.04985) / [OpenReview](https://openreview.net/pdf?id=cp1hJ67l3M) | 2024 | — |
| H2O | [arXiv 2306.14048](https://arxiv.org/abs/2306.14048) | NeurIPS'23 | — |
| Double Sparsity | [arXiv 2408.07092](https://arxiv.org/abs/2408.07092) | 2024 | — |
| Loki | [NeurIPS'24 PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/1e027da6bec9ceb2ec37951ceeccae93-Paper-Conference.pdf) | NeurIPS'24 | — |
| ScoutAttention | [arXiv 2603.27138](https://arxiv.org/abs/2603.27138) | 2025 | — |

### MoE Offload
| 시스템 | 논문 | 출판처 | 코드 |
|---|---|---|---|
| Fiddler | [arXiv 2402.07033](https://arxiv.org/abs/2402.07033) | ICLR'25 | [efeslab/fiddler](https://github.com/efeslab/fiddler) |
| KTransformers | (위 참조) | SOSP'25 | (위 참조) |
| HybriMoE | [arXiv 2504.05897](https://arxiv.org/abs/2504.05897) | 2025 | — |
| PreScope | [arXiv 2509.23638](https://arxiv.org/abs/2509.23638) | 2025 | — |
| DALI | [arXiv 2602.03495](https://arxiv.org/abs/2602.03495) | 2026 preprint | — |

### Async Overlap
- [vLLM weight offload v2 blog (2026-02-03)](https://blog.vllm.ai/2026/02/03/dsr1-gb200-part1.html)
- [Async Model Offload poster (SSDBM'25)](https://ssdbm.org/2025/assets/poster/8884-Jie.pdf) — Ye & Nicolae
- [Async KV Prefetch (arXiv 2504.06319)](https://arxiv.org/abs/2504.06319)

### CPU-side Sampling / Grammar
- [XGrammar blog (MLC 2024-11)](https://blog.mlc.ai/2024/11/22/achieving-efficient-flexible-portable-structured-generation-with-xgrammar) / [github.com/mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar)
- [Guiding LLMs (arXiv 2403.06988)](https://arxiv.org/abs/2403.06988) — Beurer-Kellner et al.
- [vLLM anatomy blog (2025-09-05)](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html)
- [SGLang](https://github.com/sgl-project/sglang)
- [llama.cpp grammar](https://github.com/ggml-org/llama.cpp)

### 인접 기법 — 본 문서 서술에는 포함 안 됐으나 관련성 있음
- [Wanda pruning (arXiv 2306.11695)](https://arxiv.org/abs/2306.11695) — Sun et al.
- [SparseGPT (arXiv 2301.00774)](https://arxiv.org/abs/2301.00774) — Frantar et al.
- [SmoothQuant (arXiv 2211.10438)](https://arxiv.org/abs/2211.10438)
- [GPTQ (arXiv 2210.17323)](https://arxiv.org/abs/2210.17323)
- [AWQ (arXiv 2306.00978)](https://arxiv.org/abs/2306.00978)
- [T-MAC (arXiv 2407.00088)](https://arxiv.org/abs/2407.00088) — Microsoft EuroSys'25 / [github.com/microsoft/T-MAC](https://github.com/microsoft/T-MAC)
- [SparAMX (HF 2502.12444)](https://huggingface.co/papers/2502.12444) / [arXiv 2502.12444](https://arxiv.org/abs/2502.12444) / [IntelLabs SparAMX](https://github.com/IntelLabs/Hardware-Aware-Automated-Machine-Learning/tree/main/SparAMX)
- [xFasterTransformer](https://github.com/intel/xFasterTransformer) — Intel 공식 (AMX 는 xDNN closed binary 의존)
- [Intel Extension for PyTorch (IPEX)](https://github.com/intel/intel-extension-for-pytorch)
- [StreamingLLM (arXiv 2309.17453)](https://arxiv.org/abs/2309.17453)

### 일반 조사 / 목록
- [LLM Inference Optimization Papers 목록](https://github.com/chenhongyu2048/LLM-inference-optimization-paper)
- [NVIDIA Mastering LLM Techniques](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)
- [vLLM Driving Large-Scale Serving (Blackwell, 2026-02)](https://blog.vllm.ai/2026/02/03/dsr1-gb200-part1.html)

---

**문서 버전**: v1 (초안, 2026-04-21)  
**위치**: `/super_power/ideation/cpu_idle_acceleration_ideation_20260421.md`
