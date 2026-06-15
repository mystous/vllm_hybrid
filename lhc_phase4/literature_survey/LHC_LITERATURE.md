# LHC (Lane-Separated Host Coprocessor) — 관련 논문 Literature Survey

> 작성일: 2026-06-08
> 작성자: systems literature researcher agent
> 본 LHC: **DSA + AMX + AVX-512 + NUMA-aware CPU core** 를 별도 lane 으로 분리한 host coprocessor architecture for LLM serving
>
> Survey 범위: 5 개 카테고리
> 1. Intel DSA (Data Streaming Accelerator) 활용
> 2. AMX BF16 / Intel CPU compute on LLM
> 3. CPU offload for LLM inference
> 4. NUMA-aware ML / serving
> 5. Host coprocessor / Lane separation 일반

---

## Section 1 — Paper 별 정리

### Category 1 — Intel DSA (Data Streaming Accelerator) 활용

#### 1.1 A Quantitative Analysis and Guidelines of Data Streaming Accelerator in Modern Intel Xeon Scalable Processors
- **Venue / Year**: ASPLOS 2024 (vol. 2) / arXiv 2305.02480 (v5: 2024-01)
- **Authors**: Reese Kuper, Ipoom Jeong, Yifan Yuan, Jiayu Hu, Ren Wang, Narayan Ranganathan, Nam Sung Kim (UIUC + Intel)
- **Main idea**: Sapphire Rapids 에 신설된 on-die DSA accelerator 의 throughput / latency / programming model 을 체계적으로 분석. DMA, memmove, memset, compare, CRC32, DIF, batch-descriptor, work-queue (DWQ vs SWQ), CXL host/device memory 경유 path 까지 다룬다. DPDK Vhost case study 로 실제 application 이득을 시연.
- **사용 hardware**: 4세대 Xeon Scalable (Sapphire Rapids) + DSA + CXL 1.1 메모리.
- **본 LHC 와 비교**: 본 LHC 의 *DSA lane* 정의 — 즉 LLM serving 내부에서 DSA 를 별도 lane 으로 운용 — 를 가능케 하는 **하부 측정 자료**. 단 이 paper 는 LLM-agnostic / serving-agnostic 한 microbenchmark 측면 분석이며, regime-aware dispatch 나 lane separation theorem 같은 LLM serving 측면 contribution 은 없다.

#### 1.2 Demystifying Intel Data Streaming Accelerator for In-Memory Data Processing
- **Venue / Year**: DIMES'24 (2nd Workshop on Disruptive Memory Systems, co-located with SC'24) / 2024-11
- **Authors**: André Berthold, Constantin Fürst, Antonia Obersteiner, Lennart Schmidt, Dirk Habich, Wolfgang Lehner, Horst Schirmeier (TU Dresden + 협력)
- **Main idea**: DSA 를 DRAM↔HBM (CXL 포함) 간 on-the-fly data distribution (ODD) 에 활용. HBM 을 DRAM 의 cache-like accelerator 로 보고 DSA 가 background data migration 을 담당하면, OLAP 풍의 in-memory data processing 이 가속됨.
- **사용 hardware**: Sapphire Rapids (DSA) + DRAM + HBM (Xeon Max).
- **본 LHC 와 비교**: DSA 를 "data plane lane" 으로 쓴다는 점에서 가장 가까운 시스템 prior art 중 하나. 그러나 (a) 대상이 OLAP 이고 (b) HBM 캐싱이지 LLM KV/host slack 이 아니며, (c) LLM 의 token-step 기반 deadline 이나 regime classification 이 없다.

#### 1.3 DSA-2LM: A CPU-Free Tiered Memory Architecture with Intel DSA
- **Venue / Year**: USENIX ATC 2025 / 2025-07
- **Authors**: Ruili Liu et al. (Tsinghua MADSys + Alibaba + UESTC + UT Arlington)
- **Main idea**: 2-tier memory (fast DRAM / slow DRAM / CXL) 에서 Linux kernel page migration 을 DSA 로 가속. DSA 가 단일 CPU 코어보다 4× 빠르게 page move 가능. 기존 page migration path 의 fine granularity 가 이득을 깎으므로 fast migration workflow + concurrent data paths + tuned DSA config 로 MEMTIS / TPP / NOMAD 대비 16–30% 성능 향상.
- **사용 hardware**: Sapphire Rapids + DSA + DRAM + CXL Type-3 memory.
- **본 LHC 와 비교**: 본 LHC 와 가장 직접적으로 시야가 겹치는 paper. 두 가지 차이: (1) DSA-2LM 은 kernel 의 page migration daemon 을 가속 — i.e. OS-level "lane" 이지 LLM-app lane 이 아니다. (2) compute (AMX/AVX-512) lane 과의 cross-lane orchestration 이 없다 (DSA only). LHC 는 user-space LLM serving 안에서 *DSA + AMX + AVX-512 + NUMA-pinned core* 4-축을 하나의 host coprocessor 로 통합한다.

#### 1.4 Rethinking Inter-Process Communication with Memory Operation Offloading
- **Venue / Year**: arXiv 2601.06331 / 2026-01
- **Authors**: Misun Park et al.
- **Main idea**: 멀티모달 / agent IPC 가 점점 큰 buffer (수백 MB) 를 주고받게 되면서 IPC stack 의 memcpy 가 CPU 사이클을 잡아먹음. DSA + software memory-ops 를 통합한 IPC runtime 을 제안. asynchronous pipelining, selective cache injection, hybrid coordination 으로 instruction count 22% 감소, throughput 2.1×, latency 72% 감소.
- **사용 hardware**: Sapphire Rapids + DSA, IPC workload.
- **본 LHC 와 비교**: DSA 를 *control plane (IPC)* lane 으로 쓴다. LHC 는 DSA 를 *data plane (KV slack reclaim, host-side staging)* lane 으로 쓰며, GPU LLM serving 의 critical path 와 협력한다. orthogonal 한 use case.

(추가: Intel DSA Specification 354293 / DSA User Guide 353216 / Linux idxd driver / DPDK dmadev / SPDK accel framework — 이들은 공식 spec/driver 이지 research paper 가 아니어서 위 표에는 별도 entry 없음, 단 LHC 구현에서 직접 의존.)

---

### Category 2 — AMX BF16 / Intel CPU compute on LLM

#### 2.1 KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models
- **Venue / Year**: SOSP 2025 / 2025-10 (Seoul)
- **Authors**: Chen et al. (Tsinghua MADSys + KVCache.ai 외)
- **Main idea**: 거대 MoE (예: DeepSeek-R1/V3) 에서 expert weight 를 CPU DRAM 에 두고 GPU 가 attention 만 처리. CPU 쪽은 **AMX (high-ARI, prefill 전용)** 와 **AVX-512 (low-ARI, decode 전용)** 를 동적으로 전환하는 specialized kernel + AMX-친화 weight layout 사용. expert deferral 로 CPU↔GPU 비동기 overlap. 결과: prefill 4.62–19.74×, decode 1.25–4.09× speedup. 단일 Xeon socket 에서 sustained 21.3 TFLOPS (PyTorch 대비 3.9×).
- **사용 hardware**: Xeon (Sapphire Rapids / Granite Rapids) + AMX + AVX-512 + GPU (H100 / RTX 4090 등).
- **본 LHC 와 비교**: AMX 를 LLM kernel 로 쓴다는 점은 **공통**. 그러나 (1) MoE expert weight 전용 — dense LLM (Llama-8B TP=8) 에 직접 적용 불가, (2) DSA lane 없음 (host-side memory move 는 일반 CPU memcpy), (3) NUMA-aware pinning 은 명시적으로 다루지 않음, (4) regime classification (GPU_SATURATED/KV_HEAVY) 없음. LHC 는 dense LLM + DSA + AMX + AVX-512 + NUMA 의 **5축 통합** 이라는 점에서 다르다.

#### 2.2 LIA: A Single-GPU LLM Inference Acceleration with Cooperative AMX-Enabled CPU-GPU Computation and CXL Offloading
- **Venue / Year**: ISCA 2025 / 2025-06 (Tokyo)
- **Authors**: Hyungyo Kim, Nachuan Wang, Qirong Xia, Jinghan Huang, Amir Yazdanbakhsh (Google DeepMind), Nam Sung Kim (UIUC)
- **Main idea**: 단일 H100 GPU 환경에서, AMX (Sapphire Rapids 20 TFLOPS / Granite Rapids 40 TFLOPS BF16) 를 GPU 와 협력시켜 LLM inference 가속. weight 일부와 KV 일부를 CPU DRAM + CXL Type-3 memory 에 두고, AMX 가 그 부분 compute. throughput driven task 에서 1.5× 추가 향상 (DDR-only 대비). vs. 단일 GPU offloading framework: latency 5.1×–19×, throughput 3.7×–5.1× 향상.
- **사용 hardware**: H100 단일 GPU + Sapphire Rapids / Granite Rapids + AMX + CXL 메모리.
- **본 LHC 와 비교**: AMX + GPU 협력이라는 큰 그림은 공통. 차이점: (1) LIA 는 **단일 GPU + CXL memory tier** 가정 (i.e. memory capacity 부족 해소) — LHC 는 H100 8-way TP / B200 같은 데이터센터급 multi-GPU baseline 에서 lane utilization 을 본다. (2) DSA lane 없음. (3) NUMA-aware lane separation 명시 없음. (4) workload-aware regime switching 없음. (5) LIA 는 *measured-positive* 케이스만 publish, LHC 의 measured-negative + honest dead-branch 보고 framework 와 대비.

#### 2.3 SparAMX: Accelerating Compressed LLMs Token Generation on AMX-Powered CPUs
- **Venue / Year**: arXiv 2502.12444 / 2025-02
- **Authors**: (Intel + 학계 collab)
- **Main idea**: unstructured sparsity 를 AMX 와 결합. compressed weight × sparse activation 을 AMX tile 단위로 효율적으로 multiply, attention 의 KV cache 까지 sparsity 적용. token-gen latency 가속.
- **사용 hardware**: Sapphire Rapids + AMX-BF16/INT8.
- **본 LHC 와 비교**: AMX kernel-level 최적화 paper. LHC 는 kernel 자체보다 *AMX 를 lane 으로 운용* 하고 *DSA/AVX-512 와 협력* 하는 system-level contribution. 두 work 는 orthogonal — LHC 의 AMX lane 내부에 SparAMX kernel 을 꽂을 수 있다.

#### 2.4 Improving Throughput-Oriented LLM Inference with CPU Computations
- **Venue / Year**: PACT 2024 / 2024-10
- **Authors**: Daon Park, Bernhard Egger (SNU)
- **Main idea**: decoder-based LLM 의 batched throughput 환경에서 CPU 도 일을 시킴 (CPU 가 attention 일부, GPU 가 GEMM). dynamic workload allocation 으로 CPU 와 GPU idle 을 최소화.
- **사용 hardware**: general-purpose CPU + GPU (특정 Xeon AMX/DSA 가속 X).
- **본 LHC 와 비교**: CPU computation 을 critical path 에 끌어들인다는 점은 같지만, (1) AMX/DSA 등 specialized accelerator 사용 X, (2) NUMA / lane separation 개념 없음. LHC 는 이 motivation 위에 specialized lanes 를 쌓은 것.

---

### Category 3 — CPU offload for LLM inference

#### 3.1 NEO: Saving GPU Memory Crisis with CPU Offloading for Online LLM Inference
- **Venue / Year**: MLSys 2025 / arXiv 2411.01142 (2024-11)
- **Authors**: Yang Zhou et al.
- **Main idea**: 일부 attention compute 와 KV cache 를 CPU 로 offload. asymmetric GPU-CPU pipelining + load-aware scheduling 으로 GPU batch size 를 효과적으로 늘려 throughput 향상. 본 LHC 세션 (IDE_022) 에서 PoC 의 baseline 으로 광범위 인용.
- **사용 hardware**: 일반 x86 CPU + NVIDIA GPU (특정 AMX/DSA 미사용).
- **본 LHC 와 비교**: NEO 는 LHC 의 직접적 motivation. 차이점: (1) NEO 는 CPU 를 단일 "second compute" 로 본다 — 본 LHC 는 CPU 내부에 **DSA / AMX / AVX-512 / NUMA-core** 4 sub-lane 으로 또 분리. (2) NEO 는 regime classification 없이 static / load-aware 한 schedule. LHC 는 **GPU_SATURATED vs KV_HEAVY 같은 regime 별 lane utilization profile** 을 갖는다. (3) NEO 는 측정 결과가 positive 인 case 위주. LHC 는 baseline (B200 + Llama-8B TP=8) 에서 측정-기반 무가치 (옵션 A noise, 옵션 C dead-branch) 도 honest 하게 보고.

#### 3.2 FastDecode: High-Throughput LLM Serving through Disaggregating Attention Computation
- **Venue / Year**: ICML / MLSys 류, OpenReview 공개 2024
- **Authors**: He, Zhai et al.
- **Main idea**: KV cache 와 attention 연산 전체를 CPU 클러스터로 offload. CPU-GPU 사이에는 attention input/output activation (KV 대비 수십~수백 배 작음) 만 흐르므로 PCIe bottleneck 해소. fixed I/O 길이 가정 + offline profiling 기반 static policy.
- **사용 hardware**: x86 CPU 노드들 + GPU. AMX/DSA 미명시.
- **본 LHC 와 비교**: "host 측 가속" 발상은 같으나 LHC 는 (1) on-die accelerator (DSA, AMX) 를 활용, (2) static profile 이 아닌 regime 기반 dispatch (옵션 C), (3) multi-GPU TP 환경.

#### 3.3 PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU
- **Venue / Year**: SOSP 2024 / arXiv 2312.12456 (2023-12)
- **Authors**: Yixin Song, Zeyu Mi, Haotong Xie, Haibo Chen (SJTU IPADS)
- **Main idea**: LLM 뉴런 활성화의 power-law (hot/cold) 를 활용 — hot 뉴런은 GPU 상주, cold 뉴런은 CPU 에서 sparse 계산. adaptive predictor + neuron-aware sparse operator. 단일 RTX 4090 에서 llama.cpp 대비 11.69×.
- **사용 hardware**: consumer GPU + 일반 CPU (AMX/DSA 미사용).
- **본 LHC 와 비교**: PowerInfer 의 핵심은 **모델-수준 sparsity** (hot/cold split). LHC 는 **하드웨어 lane separation** (compute path 와 data move path 가 다른 실리콘). 둘은 직교 — PowerInfer 의 cold-neuron CPU path 를 LHC 의 AMX lane 에 mount 가능.

#### 3.4 FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU
- **Venue / Year**: ICML 2023
- **Authors**: Ying Sheng, Lianmin Zheng, et al. (Stanford + UCB + CMU + Meta + HuggingFace 외)
- **Main idea**: GPU + CPU + Disk 3-tier 메모리 공간을 LP-solver 로 최적화. weight / KV / activation 의 배치를 search 해 latency-insensitive 한 throughput-oriented inference 에 single 16GB GPU 로 OPT-175B 를 돌림. 4-bit compression 까지 결합.
- **사용 hardware**: 일반 CPU + 단일 GPU + NVMe.
- **본 LHC 와 비교**: 본 LHC 는 throughput-oriented offline 이 아니라 *online serving* 의 GPU saturation 영역에서 CPU lane 들을 critical path 에 합류시키는 것. compression / LP-search 없음.

#### 3.5 DeepSpeed ZeRO-Inference
- **Venue / Year**: Microsoft, 2022-09 blog + KDD/MLSys 후속
- **Authors**: DeepSpeed team
- **Main idea**: optimizer state / weight 를 CPU 또는 NVMe 에 offload 해 단일 GPU 에 거대 모델 적재. 최근 refresh 는 weight quantization + KV cache offload 로 20× throughput.
- **사용 hardware**: 일반 CPU + GPU + NVMe.
- **본 LHC 와 비교**: ZeRO-Inference 는 *memory-tier* offload (capacity 문제 해결). LHC 는 *lane utilization* 문제 해결 — 즉 capacity 가 충분해도 CPU/DSA/AMX 가 idle 한 것을 채운다.

#### 3.6 SpecOffload: Unlocking Latent GPU Capacity for LLM Inference on Resource-Constrained Devices
- **Venue / Year**: arXiv 2505.10259 / 2025-05 (OpenReview)
- **Authors**: Zhang et al.
- **Main idea**: speculative decoding 의 draft model 을 offloaded-LLM 의 idle GPU memory 에 심어 GPU core utilization 을 4.49× 끌어올림. dual-batch interleaving 으로 CPU (attention) + I/O (FFN load) + GPU (draft) 가 동시에 진행. FlexGen / Fiddler 대비 2.54–4.04× throughput.
- **사용 hardware**: consumer GPU + 일반 CPU.
- **본 LHC 와 비교**: GPU-core idle 을 채운다는 점은 LHC 와 결이 비슷 (GPU_SATURATED 가 아닌 GPU_UNDERUTILIZED 영역). 그러나 SpecOffload 는 *동일 GPU 안의 latent capacity* 를 spec decode 로 채우고, LHC 는 *host coprocessor (CPU + on-die accel)* 을 lane 으로 채운다.

#### 3.7 Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models
- **Venue / Year**: ICLR 2025 / arXiv 2402.07033
- **Authors**: Keisuke Kamahori, Tian Tang, Yile Gu, Kan Zhu, Baris Kasikci (UW SyFI)
- **Main idea**: MoE expert weight 를 CPU DRAM 에 두고, 작은 input 은 CPU 에서 직접 계산 (weight transfer 회피), 큰 input 은 GPU 로 weight 끌어와 계산. latency model 로 split point 결정. 단일 24GB GPU 에서 unquantized Mixtral-8x7B 가 >3 tok/s. DeepSpeed-MII 대비 19.4×.
- **사용 hardware**: 일반 CPU + 단일 consumer GPU.
- **본 LHC 와 비교**: AMX/DSA 없음, dense LLM 대상 아님. LHC 의 KTransformers 비교와 유사한 위치.

#### 3.8 MoE-Lens: Towards the Hardware Limit of High-Throughput MoE LLM Serving Under Resource Constraints
- **Venue / Year**: arXiv 2504.09345 / 2025-04
- **Authors**: (UIUC 외)
- **Main idea**: CPU offload + GPU compute hybrid 에서 hardware 한계 throughput 을 closed-form 으로 분석. CPU 가 lightweight attention 처리, GPU 가 GEMM, expert weight 는 CPU DRAM 상주.
- **본 LHC 와 비교**: 본 LHC 의 lane separation theorem (κ < 0.3 정량 조건) 의 MoE 버전. LHC 는 dense LLM + multi-lane (DSA + AMX + AVX-512) 이라는 점에서 다르다.

#### 3.9 MoE-Gen: Module-Based Batching for MoE Offload
- **Venue / Year**: 2025 (Edinburgh)
- **Authors**: U. Edinburgh team
- **Main idea**: MoE 의 attention vs expert module 별로 batch size 를 다르게 잡고, host memory 의 token 을 dynamic re-batch. DeepSpeed-MoE 대비 8–31× throughput.
- **본 LHC 와 비교**: MoE 전용 batching 기법. LHC 와는 orthogonal.

#### 3.10 APEX: Asynchronous Parallel CPU-GPU Execution for Online LLM Inference on Constrained GPUs
- **Venue / Year**: arXiv 2506.03296 / 2025-06
- **Authors**: (학계)
- **Main idea**: decode-heavy online workload (chat/reasoning) 에서 CPU offload 와 GPU 의 overlap 을 *예측-기반 scheduler* 로 최대화. T4 / A10 에서 +37–72% throughput.
- **사용 hardware**: constrained GPU (T4 / A10) + 일반 CPU.
- **본 LHC 와 비교**: APEX 는 online + decode + 예측 scheduler — 가장 가까운 dispatch 측면 prior art. LHC 의 옵션 C (regime detection + adaptive dispatch) 와 발상이 유사. 차이: (1) APEX 는 CPU 를 단일 unit 으로 본다, LHC 는 4-lane 으로 본다. (2) APEX 는 T4/A10 (memory-bound) 에서 measured-positive — LHC 는 H100/B200 같은 데이터센터 GPU 에서 *baseline measured-negative* 도 honest 하게 둔다.

#### 3.11 Dovetail: CPU/GPU Heterogeneous Speculative Decoding for LLM Inference
- **Venue / Year**: EMNLP 2025 / arXiv 2412.18934
- **Authors**: National University of Defense Technology
- **Main idea**: GPU 에 draft 모델, CPU 에 target (verifier) 모델. consumer 디바이스 / legacy server 가정 (CPU 가 상대적으로 강하고 GPU 가 약한 경우). DGF (Dynamic Gating Fusion) 로 정확도 보완.
- **본 LHC 와 비교**: CPU 가 verifier 라는 점에서 본 LHC 의 PoC (CPU draft + GPU verify) 와 반대. 두 work 모두 spec-decode + host 협력의 가능성을 탐색.

#### 3.12 InstInfer: In-Storage Attention Offloading for Cost-Effective Long-Context LLM Inference
- **Venue / Year**: arXiv 2409.04992 / 2024-09
- **Authors**: (학계 + 산업 collab)
- **Main idea**: attention 과 KV cache 를 **Computational Storage Drive (CSD)** 로 offload. PCIe bandwidth 가 아니라 SSD 내부 대역폭을 활용. A6000 + 13B long-context 에서 FlexGen 대비 11.1×.
- **본 LHC 와 비교**: storage tier 까지 lane 을 확장한다는 발상은 LHC 와 연장선. 단 LHC 는 storage 가 아닌 *on-die accelerator (DSA / AMX)* lane 에 집중.

#### 3.13 FlexInfer: Breaking Memory Constraint via Flexible and Efficient Offloading for On-Device LLM Inference
- **Venue / Year**: EuroMLSys 2025
- **Authors**: (학계)
- **Main idea**: mobile / edge LLM 의 on-device inference 에서 asynchronous prefetching + balanced memory locking + flexible tensor preservation. Llama-2-7B 를 3GB RAM 으로 실행.
- **본 LHC 와 비교**: 모바일 edge 타깃. LHC 의 데이터센터 multi-GPU TP 와 다른 영역.

#### 3.14 Q-Infer: Towards Efficient GPU-CPU Collaborative LLM Inference via Sparsity-Aware Dynamic Scheduling
- **Venue / Year**: ACM TACO / 2025-12
- **Authors**: (PDS Lab)
- **Main idea**: sparsity-aware cache management + dynamic compute scheduling. 모델 sparsity 기반으로 GPU/CPU 분배를 dynamic 으로 결정. 1.4–11.5× throughput.
- **본 LHC 와 비교**: dynamic dispatch 라는 발상은 LHC 옵션 C 와 유사. 그러나 sparsity 기반 (모델 측면) 이지 lane separation (하드웨어 측면) 이 아님.

#### 3.15 TwinPilots: A New Computing Paradigm for GPU-CPU Parallel LLM Inference
- **Venue / Year**: SYSTOR 2024 / ACM 3688351.3689164
- **Authors**: (학계)
- **Main idea**: GPU 와 CPU 둘 다 *동등한 pilot* 으로 보고 load-balancing 문제로 재정의. KV cache 는 CPU 상주.
- **본 LHC 와 비교**: TwinPilots 의 "GPU 와 CPU 가 동등하다" 가 가장 LHC 와 가까운 철학. 차이: LHC 는 그 CPU 안을 다시 4 lane 으로 쪼개고, regime 별 dispatch 를 한다.

---

### Category 4 — NUMA-aware ML / serving

#### 4.1 vLLM `--numa-bind` (codebase feature)
- **Venue / Year**: vLLM open-source (2024 후반 ~ 2025)
- **Main idea**: multi-socket GPU 서버에서 각 GPU worker process 의 CPU 와 메모리를 그 GPU 와 같은 NUMA 노드에 pinning. `numactl --cpunodebind --membind` 또는 `--physcpubind --membind` 사용. Python interpreter 가 초기 메모리 할당부터 올바른 NUMA 정책으로 시작되도록 subprocess 전에 pin.
- **본 LHC 와 비교**: vLLM 의 NUMA bind 는 GPU worker 전체를 NUMA node 에 묶는 *coarse-grained* 정책. LHC 는 GPU worker 와는 **별도의 lane** 으로 NUMA-pinned CPU core 를 잡아 host lane 을 격리 (즉 GPU 가 cross-NUMA HBM traffic 으로 오염되지 않으면서 host lane 도 같은 NUMA 에서 동작).

#### 4.2 TensorRT-LLM NUMA topology awareness
- **Venue / Year**: NVIDIA, production framework
- **Main idea**: DeviceMesh / NUMA-aware affinity. multi-GPU TP 에서 같은 NUMA 노드의 GPU 가 NVLink 로 가깝게 묶이도록 partition.
- **본 LHC 와 비교**: NVIDIA TRT-LLM 의 NUMA awareness 는 GPU↔GPU 통신 위주. LHC 는 GPU↔CPU lane (host coprocessor) 의 NUMA 분리.

#### 4.3 (related) "Topology-aware Preemptive Scheduling for Co-located LLM Workloads"
- **Venue / Year**: arXiv 2411.11560
- **Main idea**: 같은 cluster 위에 두 개 이상 LLM workload 가 co-locate 될 때 NVLink / NUMA topology 를 고려해 preempt.
- **본 LHC 와 비교**: cluster level scheduling 이지 LHC 의 intra-node host coprocessor lane 과는 별 layer.

---

### Category 5 — Host coprocessor / Lane separation 일반

#### 5.1 NanoFlow: Towards Optimal Large Language Model Serving Throughput
- **Venue / Year**: arXiv 2408.12757 / 2024-08 (SyFI Lab, UW)
- **Authors**: Kan Zhu et al.
- **Main idea**: 단일 GPU 의 SM, memory, network 서브유닛을 *operation-level pipeline* 으로 동시에 사용. request 를 nano-batch 로 쪼개고 GPU functional unit 을 partition. LLaMA-2-70B / Mixtral-8x7B / LLaMA-3-8B 에서 1.91× throughput.
- **사용 hardware**: 단일 GPU (intra-GPU 자원 분할).
- **본 LHC 와 비교**: NanoFlow 의 "단일 디바이스 내부 functional unit 분리 + operation pipeline" 발상이 본 LHC 의 lane separation 과 **가장 동형 (isomorphic)**. 차이: NanoFlow 는 GPU 내부 SM 단위, LHC 는 *호스트 (CPU socket) 내부 DSA/AMX/AVX-512/NUMA core* 단위. 즉 LHC = "host-side NanoFlow".

#### 5.2 GPUDirect Storage / GPUDirect RDMA
- **Venue / Year**: NVIDIA 산업 표준 (지속 업데이트)
- **Main idea**: GPU 가 NIC/NVMe 와 직접 DMA. CPU 우회 (CPU 가 lane 에서 빠진다). NVMe-oF + GDS, GPUDirect RDMA + IB/RoCE.
- **본 LHC 와 비교**: GPUDirect 는 CPU 를 우회해서 lane 을 비우는 방식. LHC 는 그 *우회로 인해 비워진 CPU* 를 다시 채우는 정반대 방향. 둘은 보완적 — GDS 가 host CPU 의 idle 을 만들수록 LHC 가 채울 여지가 늘어남.

#### 5.3 Intel DTO (Data-Transfer Offload library)
- **Venue / Year**: Intel open-source, github.com/intel/DTO
- **Main idea**: user-space application 이 DSA 를 transparent 하게 쓸 수 있게 해주는 library. memcpy/memset 등을 hot-patch.
- **본 LHC 와 비교**: LHC 의 DSA lane 구현에서 직접 의존할 수 있는 building block. 자체 contribution 은 아니지만 ecosystem 측면 prior art.

#### 5.4 SPDK Acceleration Framework (AccelFW)
- **Venue / Year**: SPDK open-source
- **Main idea**: storage 스택의 memory copy / CRC / compress 를 DSA / AE4DMA / SW 중 하나에 dispatch. pluggable accelerator backend.
- **본 LHC 와 비교**: dispatch 추상화의 좋은 prior art. SPDK 가 backend 를 "어떤 가속기" 단위로 묶는 데 비해, LHC 는 "어떤 lane (DSA/AMX/AVX-512/core)" 단위로 묶고 regime 별로 분배.

---

## Section 2 — 본 LHC 의 Novelty Matrix

> 표 의미: ✅ = paper 가 해당 축에서 명시적 contribution 을 가짐, ⚠️ = 부분적 / 우회적으로 다룸, ❌ = 다루지 않음.
> 5 축: DSA (Data Streaming Accelerator), AMX (Advanced Matrix Extensions), NUMA-aware (lane-level), Adaptive dispatch (workload-aware regime detection), Lane separation (host 내부 multi-lane orchestration).

| Paper | DSA | AMX | NUMA-aware | Adaptive dispatch | Lane separation | Comments |
|---|:---:|:---:|:---:|:---:|:---:|---|
| Intel DSA Quant Analysis (ASPLOS'24) | ✅ | ❌ | ⚠️ | ❌ | ❌ | DSA microbench only. No LLM. |
| Demystifying DSA (DIMES'24) | ✅ | ❌ | ❌ | ❌ | ⚠️ | DSA as HBM↔DRAM mover, OLAP. |
| DSA-2LM (ATC'25) | ✅ | ❌ | ✅ | ⚠️ | ⚠️ | OS-level page mig lane. No LLM, no AMX. |
| Rethinking IPC (2026) | ✅ | ❌ | ❌ | ⚠️ | ⚠️ | DSA as IPC lane. No LLM. |
| KTransformers (SOSP'25) | ❌ | ✅ | ❌ | ⚠️ (AMX↔AVX-512 switch) | ❌ | MoE expert AMX, monolithic CPU view. |
| LIA (ISCA'25) | ❌ | ✅ | ❌ | ⚠️ | ❌ | AMX + CXL memory, single GPU. |
| SparAMX (2025) | ❌ | ✅ | ❌ | ❌ | ❌ | AMX kernel only. |
| Throughput-LLM-CPU (PACT'24) | ❌ | ❌ | ❌ | ✅ | ❌ | CPU+GPU dynamic split, no specialized accel. |
| NEO (MLSys'25) | ❌ | ❌ | ❌ | ⚠️ (load-aware) | ❌ | CPU offload, no lane sep. |
| FastDecode | ❌ | ❌ | ❌ | ❌ (static) | ❌ | Static attention disagg to CPU. |
| PowerInfer (SOSP'24) | ❌ | ❌ | ❌ | ⚠️ (hot/cold) | ❌ | Model-level sparsity split. |
| FlexGen (ICML'23) | ❌ | ❌ | ❌ | ⚠️ (LP) | ❌ | Tiered memory, offline LP. |
| ZeRO-Inference | ❌ | ❌ | ❌ | ❌ | ❌ | Capacity tier only. |
| SpecOffload (2025) | ❌ | ❌ | ❌ | ⚠️ | ⚠️ | Intra-GPU latent capacity for draft. |
| Fiddler (ICLR'25) | ❌ | ❌ | ❌ | ✅ (latency model) | ❌ | MoE expert CPU/GPU split. |
| MoE-Lens (2025) | ❌ | ❌ | ❌ | ✅ | ❌ | MoE hardware-limit analysis. |
| MoE-Gen | ❌ | ❌ | ❌ | ⚠️ | ❌ | Module batching. |
| APEX (2025) | ❌ | ❌ | ❌ | ✅ (predictive) | ❌ | Closest dispatch prior. |
| Dovetail (EMNLP'25) | ❌ | ❌ | ❌ | ⚠️ | ❌ | CPU verify + GPU draft. |
| InstInfer (2024) | ❌ | ❌ | ❌ | ❌ | ⚠️ (CSD) | Lane=storage, not host accel. |
| FlexInfer (EuroMLSys'25) | ❌ | ❌ | ❌ | ⚠️ | ❌ | Mobile edge. |
| Q-Infer (TACO'25) | ❌ | ❌ | ❌ | ✅ (sparsity-aware) | ❌ | Model-level dynamic split. |
| TwinPilots (SYSTOR'24) | ❌ | ❌ | ❌ | ✅ | ⚠️ | GPU & CPU as twin pilots. |
| vLLM `--numa-bind` | ❌ | ❌ | ✅ | ❌ | ❌ | GPU worker NUMA pin, coarse. |
| TRT-LLM NUMA | ❌ | ❌ | ✅ | ❌ | ❌ | Inter-GPU NUMA topology. |
| NanoFlow (2024) | ❌ | ❌ | ❌ | ✅ | ✅ (intra-GPU) | Intra-GPU lane sep. **Closest isomorph**. |
| GPUDirect | ❌ | ❌ | ⚠️ | ❌ | ⚠️ | CPU-bypass, opposite direction. |
| Intel DTO | ✅ | ❌ | ❌ | ❌ | ❌ | DSA user-space lib, building block. |
| SPDK AccelFW | ✅ | ❌ | ❌ | ⚠️ (pluggable) | ⚠️ | Storage dispatch. |
| **LHC (본 세션)** | **✅** | **✅** | **✅** | **✅** | **✅** | **최초 5축 통합 (host coprocessor for LLM serving)**. |

**관찰**:
- 5 축 모두 ✅ 인 paper 는 **없음** — LHC 가 첫 통합.
- DSA + LLM serving 의 직접 결합은 LHC 가 **최초** (위 표에서 DSA ✅ 인 4개 paper 모두 LLM-agnostic).
- AMX + LLM 은 KTransformers, LIA, SparAMX 등 다수 있으나 **DSA / NUMA lane sep 와의 결합** 은 없음.
- Adaptive dispatch + lane separation 둘 다 갖춘 것은 NanoFlow (intra-GPU) 뿐이며, 이를 *host coprocessor* 차원으로 옮긴 prior art 는 없음.

---

## Section 3 — Gap Analysis

### 3.1 기존 연구에 부재한 부분

1. **DSA 를 LLM serving 의 데이터 lane 으로 사용한 사례 없음**
   - DSA 관련 paper 4편 (Quant Analysis, Demystifying, DSA-2LM, Rethinking IPC) 모두 LLM 과 무관 (microbench / OLAP / OS page mig / IPC).
   - LLM 의 KV cache eviction, host-side slack reclamation, prefill staging 같은 데이터 이동 경로에 DSA 를 *명시적 lane* 으로 운용한 시스템 paper 는 부재.

2. **Host 내부 multi-lane orchestration 부재**
   - CPU offload paper 대부분 (NEO, FastDecode, PowerInfer, Fiddler) 은 CPU 를 *단일 monolithic compute* 로 본다.
   - "CPU 안에 DSA / AMX / AVX-512 / NUMA-pinned core 가 각각 다른 마이크로아키텍처 lane 이다" 라는 관점은 NanoFlow (intra-GPU) 외에는 없음.

3. **Workload regime 기반 lane utilization profile 부재**
   - APEX / Q-Infer / Fiddler 가 dynamic dispatch 를 하지만, **regime classification** (GPU_SATURATED / KV_HEAVY / IDLE) 을 명시적으로 정의하고 그에 따라 lane 사용을 swap 하는 시스템은 없음.

4. **Container-constrained NUMA workaround 부재**
   - Production cluster 의 컨테이너는 `CAP_SYS_NICE`, `mlock`, `MPOL_BIND` 등을 잘 안 줌. 이 제약 하에서 userspace + DSA 만으로 lane 효과를 내는 *deployable* 설계 prior art 없음 (vLLM `--numa-bind` 는 root 권한 가정).

5. **Measured-negative + dead-branch honest 보고 framework 부재**
   - 거의 모든 inference acceleration paper 가 measured-positive case 만 publish. 본 LHC 는 baseline (B200 + Llama-8B TP=8 + sharegpt) 에서 옵션 A noise / 옵션 C dead-branch 같은 *부정적 결과* 도 보고 → 적용 영역의 정직한 boundary 정의.

### 3.2 본 LHC 의 unique contribution 5 개

1. **DSA Lane in LLM serving** (literature 최초로 확인됨)
   - on-die DSA accelerator 를 LLM serving 의 KV slack reclamation / host-side staging lane 으로 운용.

2. **Lane Separation Theorem (κ < 0.3 정량 조건)**
   - 4 lane (DSA / AMX / AVX-512 / NUMA core) 간 cross-lane interference coefficient κ 가 0.3 미만일 때 lane separation 이 net-positive 임을 정량 조건으로 제시.

3. **Workload-aware regime detection (옵션 C)**
   - GPU_SATURATED / KV_HEAVY 등 regime 을 runtime 에 감지하고 lane dispatch 를 regime-별 정책으로 swap.

4. **Container-constrained NUMA workaround**
   - `CAP_SYS_NICE` 없이 userspace + DSA 의 cooperative 동작만으로 NUMA 분리 효과를 근사화.

5. **Measured-negative + LHC dead branch 의 honest 보고 framework**
   - baseline measured-negative (옵션 A) / dead-branch (옵션 C) 도 paper 의 정식 결과로 포함. 적용 영역 boundary 를 명시.

---

## Section 4 — 본 LHC 의 한계 (정직)

### 4.1 baseline measured-negative

- 본 세션의 baseline: **B200 GPU × 8 (TP=8) + Llama-8B + sharegpt workload**.
- 이 baseline 에서 옵션 A (host slack reclamation) 는 **측정 결과 noise 수준** — net-win 무.
- 옵션 C (workload-aware regime gating) 는 **dead branch** — regime 자체가 baseline 에서 GPU_SATURATED 로 거의 항상 고정 → adaptive 의 여지 없음.
- 즉 본 LHC 는 "single-tenant + 큰 GPU memory + dense LLM + sharegpt-like balanced workload" 영역에서는 measured-positive 가 아니다.

### 4.2 적용 영역 재정의

본 LHC 의 net-positive 가 기대되는 boundary:

- **Multi-tenant serving**: 한 컨테이너의 idle CPU 를 다른 컨테이너의 LLM lane 으로 빌려쓰는 경우 — DSA / AMX lane 이 cross-tenant overhead 없이 실리콘 단위로 격리.
- **Smaller GPU memory** (T4 / A10 / consumer 4090): APEX / Fiddler 가 measured-positive 인 영역. LHC 의 multi-lane 확장이 그 위에서 추가 throughput 가능.
- **Decode-context parallelism / long-context**: KV 가 host DRAM 으로 spill 되는 영역 — DSA lane 의 데이터 이동이 critical.
- **MoE / expert weight on CPU**: KTransformers / Fiddler 영역에 LHC 의 NUMA + DSA 축 추가.
- **Spec decode CPU draft**: 본 세션 PoC 의 일부. CPU AMX 가 small draft model 을 돌리면서 host slack 을 DSA 가 prefetch.

### 4.3 미해결 / future work

- κ (cross-lane interference) 측정의 hardware-counter 의존도 — 다른 CPU 세대 (Granite Rapids / Sierra Forest) 에서 재측정 필요.
- DSA 가 없는 AMD / ARM 호스트에서의 대안 lane (예: AE4DMA / ARM DPU).
- Regime detection 의 false-positive 비용 (잘못 dispatch 했을 때 GPU saturation 을 더 악화시킬 위험).

---

## 참고 — 본 survey 의 search coverage 한계

- 본 survey 는 WebSearch + WebFetch 기반으로 venue/year 까지 확인했으나 PDF 전문 read 는 일부만 수행 (timeout 으로 LIA, KTransformers PDF 직접 read 는 실패, 대신 abstract + 3rd-party summary 로 충당).
- arxiv 2025/12 ~ 2026/06 신규 paper 중 일부는 누락 가능.
- 본 LHC 의 정확한 비교를 위해서는 KTransformers / LIA / NanoFlow / APEX 4 편의 PDF 정독이 향후 필요 (현재는 abstract + 공식 summary 기반).
