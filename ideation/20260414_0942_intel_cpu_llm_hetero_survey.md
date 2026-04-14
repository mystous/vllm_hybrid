# Intel CPU 기반 LLM 추론 최적화 및 CPU-GPU 헤테로지니어스 연구 조사

> 작성일: 2026-04-14  
> 목적: CPU-GPU 헤테로지니어스 추론 연구 참고 자료

---

## 1. 개요

GPU 메모리 한계 및 비용 문제로 인해 CPU를 추론 파이프라인에 통합하는 헤테로지니어스 연구가 활발히 진행되고 있다. 크게 세 방향으로 분류된다.

- **CPU 단독 최적화**: Intel AMX/AVX-512 활용, 양자화·희소화 커널 설계
- **CPU-GPU 분업(Heterogeneous Cooperation)**: 뉴런 분류, Speculative Decoding, KV Cache 오프로드
- **시스템 특성 분석**: Prefill/Decode 단계의 하드웨어 바운드 분석

---

## 2. Intel CPU 전용 최적화 연구

### 2.1 Efficient LLM Inference on CPUs (Intel, NeurIPS 2023)

| 항목 | 내용 |
|------|------|
| **핵심 기법** | INT4 weight-only quantization + CPU 전용 고성능 커널 |
| **대상 모델** | GPT-J, LLaMA, MPT |
| **arXiv** | [arXiv:2311.00502](https://arxiv.org/abs/2311.00502) |
| **GitHub** | [intel/neural-compressor](https://github.com/intel/neural-compressor) |

INT4 weight-only quantization 자동화 플로우와 LLM 전용 런타임을 설계. GPU 없이도 실용적 추론이 가능함을 처음으로 체계적으로 보인 Intel 연구.

---

### 2.2 SparAMX: Accelerating Compressed LLMs on AMX-powered CPUs (Intel + Cornell, 2025)

| 항목 | 내용 |
|------|------|
| **핵심 기법** | Intel AMX + Unstructured Sparsity (선형 레이어 + Attention KV Cache) |
| **하드웨어** | Intel Xeon Sapphire Rapids (AMX 탑재) |
| **성능** | PyTorch 대비 end-to-end latency **1.42×** 감소, KV Cache sparsity **1.14×** 추가 가속 |
| **arXiv** | [arXiv:2502.12444](https://arxiv.org/abs/2502.12444) |
| **GitHub** | [IntelLabs/Hardware-Aware-Automated-Machine-Learning](https://github.com/IntelLabs/Hardware-Aware-Automated-Machine-Learning) |

Decode 단계의 memory-bound 특성에 주목하여 AMX 타일 연산 + unstructured sparsity를 결합한 오픈소스 PyTorch C++ 확장 커널 제공. KV Cache에 희소화를 적용한 첫 논문.

---

### 2.3 Inference Performance Optimization for LLMs on CPUs (Intel, ICML 2024 Workshop)

| 항목 | 내용 |
|------|------|
| **핵심 기법** | INT8→INT32→FP32 변환 명령어셋 활용, KV Cache 축소, oneCCL 기반 분산 추론 |
| **대상 모델** | Qwen, LLaMA, ChatGLM, Baichuan, OPT |
| **arXiv** | [arXiv:2407.07304](https://arxiv.org/abs/2407.07304) |
| **GitHub** | [intel/intel-extension-for-pytorch](https://github.com/intel/intel-extension-for-pytorch) |

`mm512_cvtepi32_ps` 명령어 기반 양자화 파이프라인 설계. 멀티소켓 분산 추론 시 토큰 ID 브로드캐스트(임베딩 값 대신)로 통신 오버헤드 최소화.

---

### 2.4 Understanding Performance Implications of LLM Inference on CPUs (IISWC 2024)

| 항목 | 내용 |
|------|------|
| **핵심 기법** | AMX + HBM CPU에서의 NUMA 구성, 코어 수 영향 실측 분석 |
| **발견** | GPU 메모리를 초과하는 대형 모델(short seq)에서 CPU가 GPU를 능가 |
| **PDF** | [IISWC 2024 논문](https://seonjinna.github.io/assets/pdf/iiswc24_CPULLM.pdf) |
| **GitHub** | [intel/intel-extension-for-pytorch](https://github.com/intel/intel-extension-for-pytorch) |

LLaMA2-13B, OPT-66B에 대한 LLC miss, load/store 명령어, 코어 활용률 등 마이크로아키텍처 수준 분석. Prefill(compute-bound) vs Decode(memory-bound) 분기 특성 실증.

---

### 2.5 Lossless Speculative Decoding for Heterogeneous Vocabularies (Intel + Weizmann, ICML 2025)

| 항목 | 내용 |
|------|------|
| **핵심 기법** | 어휘 불일치 모델 간 lossless speculative decoding 3가지 알고리즘 |
| **성능** | 최대 **2.8×** 추론 가속, 정확도 손실 없음 |
| **통합** | HuggingFace Transformers 오픈소스 통합 완료 |
| **발표 링크** | [Intel Newsroom](https://newsroom.intel.com/artificial-intelligence/intel-weizmann-institute-speed-ai-with-speculative-decoding-advance) |
| **GitHub** | [huggingface/transformers](https://github.com/huggingface/transformers) |

Draft 모델과 Target 모델의 어휘(vocabulary)가 달라도 적용 가능한 vendor-agnostic speculative decoding. CPU-GPU 이기종 환경에서 소형 draft 모델을 GPU에, 대형 target 모델을 CPU에 배치하는 시나리오와 직접 연결됨.

---

### 2.6 IPEX-LLM: Intel LLM Library for PyTorch (Intel 오픈소스)

| 항목 | 내용 |
|------|------|
| **특징** | 온더플라이 모델 변환, INT4/FP4/INT8/FP8, AVX-512 VNNI + AMX 활용 |
| **통합** | HuggingFace, LangChain, LlamaIndex, vLLM, Ollama, llama.cpp |
| **지원 모델** | LLaMA, Mistral, Qwen, DeepSeek, ChatGLM, Mixtral, Gemma 등 |
| **GitHub** | [intel/ipex-llm](https://github.com/intel/ipex-llm) |
| **화이트페이퍼** | [Intel 공식 문서 834133](https://cdrdv2-public.intel.com/834133/Intel%20AI_LLM%20Model%20Inference%20Using%20IPEX-LLM_Whitepaper_rev1.0.pdf) |

Self-Speculative Decoding 지원으로 FP16/BF16 추론 레이턴시 약 30% 향상. CPU + iGPU + Arc GPU + NPU 전체 Intel XPU 스택을 단일 라이브러리로 커버.

---

## 3. CPU-GPU 헤테로지니어스 협력 연구

### 3.1 PowerInfer: Fast LLM Serving with a Consumer-grade GPU (상하이 교통대, SOSP 2024)

| 항목 | 내용 |
|------|------|
| **핵심 아이디어** | 뉴런 활성화 power-law 분포 활용: hot neuron → GPU, cold neuron → CPU |
| **성능** | llama.cpp 대비 최대 **11.69×**, RTX 4090에서 A100 대비 82% 처리량 달성 |
| **대상 모델** | OPT-175B, LLaMA 시리즈 등 |
| **DOI** | [10.1145/3694715.3695964](https://dl.acm.org/doi/10.1145/3694715.3695964) |
| **arXiv** | [arXiv:2312.12456](https://arxiv.org/abs/2312.12456) |
| **GitHub** | [SJTU-IPADS/PowerInfer](https://github.com/SJTU-IPADS/PowerInfer) |

**핵심 설계**: Offline phase에서 뉴런 배치 정책(ILP)을 사전 계산하고, Online phase에서 Adaptive Predictor로 활성 뉴런을 식별. GPU-CPU 독립 연산으로 PCIe 데이터 전송 최소화. llama.cpp C++/CUDA 4,200줄 확장으로 구현.

---

### 3.2 Dovetail: CPU/GPU Heterogeneous Speculative Decoding (2024)

| 항목 | 내용 |
|------|------|
| **핵심 아이디어** | Draft 모델 → GPU, Target 모델 → CPU 병렬 검증 |
| **대상** | 소비자 기기, AI 미적용 레거시 서버 (약한 GPU + 강한 CPU 환경) |
| **특징** | 이기종 HW 상보적 특성 + speculative decoding을 결합한 lossless 가속 |
| **arXiv** | [arXiv:2412.18934](https://arxiv.org/abs/2412.18934) |

GPU는 draft token 생성, CPU는 target 모델로 병렬 검증 수행. PCIe 통신 대역폭 절감과 함께 기존 offloading 방식의 직렬 병목 해소.

---

### 3.3 HGCA: Hybrid GPU-CPU Attention for Long Context LLM Inference (2025)

| 항목 | 내용 |
|------|------|
| **핵심 아이디어** | GPU VRAM 초과 KV Cache를 CPU로 오프로드하는 하이브리드 Attention |
| **특징** | Locality-aware KV Cache 관리 + Hybrid Attention 메커니즘 |
| **장점** | 모델 재학습 불필요, 기존 LLM 프레임워크에 seamless 통합 |
| **arXiv** | [arXiv:2507.03153](https://arxiv.org/abs/2507.03153) |

긴 시퀀스·대형 배치에서 GPU 메모리 한계를 CPU DRAM으로 확장. sparse attention 베이스라인 대비 정확도·성능 모두 우세.

---

### 3.4 Challenging GPU Dominance: When CPUs Outperform for On-Device LLM (2025)

| 항목 | 내용 |
|------|------|
| **핵심 발견** | 소형 모델(1B)에서 CPU 2스레드 F16이 GPU 가속보다 빠름 (17 vs 12.8 tokens/s) |
| **플랫폼** | iPhone 15 Pro, llama.cpp |
| **원인 분석** | GPU kernel launch overhead, CPU thread 최적화, 소규모 GEMM에서 GPU 불리 |
| **arXiv** | [arXiv:2505.06461](https://arxiv.org/abs/2505.06461) |

Thread oversubscription, 양자화 전략, 하드웨어 제약이 CPU vs GPU 성능 역전에 미치는 영향 분석. 헤테로지니어스 라우팅 기준 설계에 실증적 근거 제공.

---

### 3.5 A Systematic Characterization of LLM Inference on GPUs (2025)

| 항목 | 내용 |
|------|------|
| **분석 프레임워크** | ① Two-Phase Heterogeneity ② 마이크로아키텍처 원인 분석 ③ 스케일링 원리 ④ 신규 패러다임 경계 |
| **RAG 발견** | RAG 워크로드에서 bottleneck이 GPU compute → CPU-side retrieval/memory로 이동 |
| **arXiv** | [arXiv:2512.01644](https://arxiv.org/abs/2512.01644) |

Roofline 모델로 Prefill(compute-bound)과 Decode(memory-bound) 위상 분리 실증. MoE의 sparse activation이 Decode 단계 routing overhead를 심화시킴을 확인. RAG에서 CPU-GPU 헤테로지니어스 파이프라인의 bottleneck shift를 최초 정량화.

---

## 4. 핵심 소프트웨어 및 도구

| 도구 | 설명 | GitHub |
|------|------|--------|
| **IPEX-LLM** | Intel CPU/GPU 전용 PyTorch LLM 라이브러리 | [intel/ipex-llm](https://github.com/intel/ipex-llm) |
| **OpenVINO** | Intel 범용 딥러닝 추론 최적화 (CPU/GPU/NPU) | [openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino) |
| **llama.cpp** | CPU 우선 경량 LLM 추론 엔진 | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) |
| **PowerInfer** | CPU-GPU 뉴런 분리 추론 엔진 | [SJTU-IPADS/PowerInfer](https://github.com/SJTU-IPADS/PowerInfer) |
| **neural-compressor** | Intel 양자화·희소화 프레임워크 | [intel/neural-compressor](https://github.com/intel/neural-compressor) |

---

## 5. 연구 분류 요약

```
LLM 추론 최적화 분류
├── CPU 단독
│   ├── 양자화: INT4/INT8 weight-only (arXiv:2311.00502)
│   ├── AMX 커널: SparAMX (arXiv:2502.12444)
│   ├── 분산: oneCCL 멀티소켓 (arXiv:2407.07304)
│   └── 성능 분석: AMX+HBM NUMA 실측 (IISWC 2024)
│
└── CPU-GPU 헤테로지니어스
    ├── 뉴런 분리: PowerInfer hot/cold (DOI:10.1145/3694715.3695964)
    ├── Speculative Decoding: Dovetail (arXiv:2412.18934)
    ├── KV Cache 오프로드: HGCA (arXiv:2507.03153)
    ├── 어휘 이기종 SD: Intel+Weizmann (ICML 2025)
    └── 성능 경계 분석: CPU vs GPU 역전 조건 (arXiv:2505.06461)
```

---

## 6. Intel 하드웨어 핵심 기능

| 기능 | 세대 | 용도 |
|------|------|------|
| **AMX** (Advanced Matrix Extensions) | Xeon 4세대 (Sapphire Rapids) 이상 | BF16/INT8 행렬 연산 전용 가속 |
| **AVX-512 VNNI** | Xeon 3세대 이상 | INT8 벡터 연산 |
| **HBM** (High Bandwidth Memory) | Xeon Max (Sapphire Rapids HBM) | 메모리 대역폭 병목 완화 |
| **oneCCL** | 소프트웨어 라이브러리 | 멀티소켓 분산 추론 통신 |

---

*참고: arXiv DOI는 `https://doi.org/10.48550/arXiv.XXXX.XXXXX` 형식으로도 접근 가능.*
