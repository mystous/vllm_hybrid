# vLLM Hybrid 프로젝트 - 학술 논문 및 기술 참고문헌

> 프로젝트와 관련된 주요 학술 논문 및 기술 참고문헌 목록.
> 마지막 업데이트: 2026-04-11

---

## 1. vLLM 및 PagedAttention

### [P1] Efficient Memory Management for Large Language Model Serving with PagedAttention
- **저자**: Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, Ion Stoica
- **발표**: SOSP 2023 (29th ACM Symposium on Operating Systems Principles)
- **링크**: [arXiv:2309.06180](https://arxiv.org/abs/2309.06180) | [ACM DL](https://dl.acm.org/doi/abs/10.1145/3600006.3613165)
- **관련성**: vLLM의 핵심 논문. OS 가상 메모리의 페이징 기법에서 영감을 받은 PagedAttention을 제안하여 KV 캐시 메모리 단편화를 거의 제거하고, 기존 시스템 대비 2-4배 처리량 향상을 달성. 본 프로젝트의 기반 시스템.

---

## 2. CPU-GPU 이종(Heterogeneous) LLM 추론

### [P2] KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models
- **저자**: Chen et al. (Tsinghua University MADSys Lab)
- **발표**: SOSP 2025 (31st ACM SIGOPS Symposium on Operating Systems Principles)
- **링크**: [ACM DL](https://dl.acm.org/doi/10.1145/3731569.3764843) | [GitHub](https://github.com/kvcache-ai/ktransformers)
- **관련성**: MoE 모델의 희소성을 활용한 CPU/GPU 하이브리드 추론 프레임워크. AMX 특화 커널과 비동기 CPU-GPU 태스크 스케줄링으로 prefill 4.62-19.74배, decode 1.25-4.09배 속도 향상. DeepSeek-V3/R1 671B 지원. 본 프로젝트의 MoE 하이브리드 모드 설계에 직접적인 참고.

### [P3] PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU
- **저자**: Yixin Song, Zeyu Mi, Haotong Xie, Haibo Chen (SJTU IPADS)
- **발표**: SOSP 2024 (30th ACM SIGOPS Symposium on Operating Systems Principles)
- **링크**: [arXiv:2312.12456](https://arxiv.org/abs/2312.12456) | [GitHub](https://github.com/SJTU-IPADS/PowerInfer)
- **관련성**: 뉴런 활성화의 power-law 분포를 활용한 GPU-CPU 하이브리드 추론 엔진. hot neuron은 GPU에 사전 로드, cold neuron은 CPU에서 계산하여 GPU 메모리 요구량과 데이터 전송을 대폭 감소. llama.cpp 대비 최대 11.69배 속도 향상.

### [P4] PowerInfer-2: Fast Large Language Model Inference on a Smartphone
- **저자**: Zhenliang Xue, Yixin Song et al. (SJTU IPADS)
- **발표**: arXiv 2024
- **링크**: [arXiv:2406.06282](https://arxiv.org/abs/2406.06282)
- **관련성**: 스마트폰의 이종 컴퓨팅 자원(CPU/NPU/메모리/I/O)을 활용한 LLM 추론. fine-grained neuron cluster 단위로 연산을 분해하여 dense 클러스터는 NPU, sparse 클러스터는 CPU에서 처리. 47B 모델을 스마트폰에서 11.68 tok/s 달성.

### [P5] HeteGen: Heterogeneous Parallel Inference for Large Language Models on Resource-Constrained Devices
- **저자**: Zhao Xuanlei, Bin Jia, Haotian Zhou, Ziming Liu, Shenggan Cheng, Yang You
- **발표**: MLSys 2024
- **링크**: [MLSys Proceedings](https://proceedings.mlsys.org/paper_files/paper/2024/hash/5431dca75a8d2abc1fb51e89e8324f10-Abstract-Conference.html) | [arXiv:2403.01164](https://arxiv.org/abs/2403.01164)
- **관련성**: CPU와 GPU를 활용한 이종 병렬 추론 프레임워크. 비동기 오버랩과 하이브리드 병렬리즘으로 I/O 병목을 해결하여 기존 방법 대비 317% 추론 속도 향상. 본 프로젝트의 이종 병렬 설계에 참고.

---

## 3. Intel CPU에서의 LLM 추론 최적화

### [P6] SGLang: Efficient Execution of Structured Language Model Programs
- **저자**: Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Jeff Huang, Chuyue Sun, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, Ying Sheng
- **발표**: NeurIPS 2024
- **링크**: [arXiv:2312.07104](https://arxiv.org/abs/2312.07104) | [NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html)
- **관련성**: RadixAttention 기반 KV 캐시 자동 재사용과 구조화된 출력 디코딩 최적화를 제공하는 고성능 LLM 서빙 프레임워크. Intel CPU 백엔드를 지원하며, AMX 기반 C++ 네이티브 백엔드로 BF16/INT8/FP8 지원.

### [P6a] Cost Effective Deployment of DeepSeek R1 with Intel Xeon 6 CPU on SGLang
- **저자**: Intel PyTorch Team / LMSYS
- **발표**: LMSYS Blog, 2025
- **링크**: [LMSYS Blog](https://lmsys.org/blog/2025-07-14-intel-xeon-optimization/)
- **관련성**: SGLang에서 Intel Xeon 6 프로세서를 사용한 DeepSeek R1의 CPU 전용 추론 최적화. AMX 기반 RadixAttention 백엔드, SiLU 퓨전, 동적 양자화 퓨전 등의 최적화 기법 설명. 본 프로젝트의 CPU 커널 최적화에 직접 참고.

### [P7] Inference Performance Optimization for Large Language Models on CPUs
- **저자**: Pujiang He, Shan Zhou, Wenhuan Huang, Changqing Li, Duyi Wang, Bin Guo, Chen Meng et al. (Intel)
- **발표**: arXiv 2024
- **링크**: [arXiv:2407.07304](https://arxiv.org/abs/2407.07304) | [GitHub (xFasterTransformer)](https://github.com/intel/xFasterTransformer)
- **관련성**: CPU에서 LLM 추론 가속을 위한 INT8 KV 캐시, 분산 추론 최적화, SlimAttention(Q-K 스코어 1차원 분해) 제안. Intel CPU에서의 KV 캐시 크기 절감 및 어텐션 최적화 기법이 본 프로젝트에 참고.

### [P8] Understanding Performance Implications of LLM Inference on CPUs
- **저자**: Seonjin Na, Geonhwa Jeong, Byung Hoon Ahn, Jeffrey Young, Tushar Krishna, Hyesoon Kim
- **발표**: IISWC 2024 (IEEE International Symposium on Workload Characterization)
- **링크**: [IEEE Xplore](https://ieeexplore.ieee.org/document/10763564/)
- **관련성**: 최신 CPU(SPR)에서 LLM 추론 성능의 체계적 분석. DDR/HBM 메모리, NUMA 구성, 클러스터링 모드에 따른 성능 영향을 분석하여 단일 NUMA 노드 노출 + HBM 명시적 사용이 최적임을 확인. 본 프로젝트의 NUMA 구성 결정에 참고.

---

## 4. NUMA 인식 메모리 할당

### [P9] Optimization of NUMA Aware DNN Computing System
- **저자**: (Various)
- **발표**: Springer, 2024
- **링크**: [Springer Link](https://link.springer.com/chapter/10.1007/978-981-97-5591-2_11)
- **관련성**: NUMA 인식 DNN 컴퓨팅 시스템의 메모리 접근 패턴 표준화와 정적 NUMA 최적화를 통해 최대 1.63배 단일 레이어, 1.37배 전체 가속을 달성. 본 프로젝트의 NUMA 메모리 할당 최적화에 참고.

### [P10] ParaX: Boosting Deep Learning for Big Data Analytics on Many-Core CPUs
- **저자**: (Various)
- **발표**: PVLDB, Vol. 14
- **링크**: [PVLDB](http://vldb.org/pvldb/vol14/p864-zhang.pdf)
- **관련성**: 다중 코어 CPU에서 DL 플랫폼 비효율의 원인(레이어별 배리어)을 규명하고, one-instance-per-core 방식을 제안. 2-NUMA Intel 8280 CPU에서 학습 1.73-2.93배, 추론 2.08-2.11배 처리량 향상.

### [P11] NUMA-Caffe: NUMA-Aware Deep Learning Neural Networks
- **저자**: Intel
- **발표**: Intel Technical Document
- **링크**: [Intel PDF](https://www.intel.com/content/dam/www/public/us/en/ai/documents/NUMA-Caffe.pdf)
- **관련성**: 다중/매니코어 CPU 아키텍처에서 NUMA 인식 멀티솔버 기반 CNN 설계. DNN 토폴로지에 독립적이며 기존 Caffe 변형 대비 우수한 확장성 제공.

---

## 5. AVX-512 / AMX 최적화

### [P12] Deep Learning with Intel AVX-512 and Intel DL Boost
- **저자**: Intel
- **발표**: Intel Developer Guide
- **링크**: [Intel Guide](https://www.intel.com/content/www/us/en/developer/articles/guide/deep-learning-with-avx512-and-dl-boost.html)
- **관련성**: AVX-512 VNNI(Vector Neural Network Instructions)를 활용한 딥러닝 추론 최적화 가이드. INT8 연산에 대한 VNNI 명령어 활용 방법 설명. 본 프로젝트의 VNNI INT8 GEMM 커널(gemm_vnni.cpp) 구현에 참고.

### [P13] Accelerate PyTorch Training and Inference using Intel AMX
- **저자**: Intel
- **발표**: Intel Technical Article
- **링크**: [Intel Article](https://www.intel.com/content/www/us/en/developer/articles/technical/accelerate-pytorch-training-inference-on-amx.html)
- **관련성**: 4세대 Intel Xeon의 AMX(Advanced Matrix Extensions)를 활용한 PyTorch 추론 가속. TDPBF16PS(BF16 타일 내적), TDPBSSD(INT8 타일 내적) 명령어를 통해 AVX-512 VNNI 대비 INT8/BF16에서 더 높은 성능 달성. FP32 대비 1.9-9.6배 추론 속도 향상.

---

## 6. Speculative Decoding

### [P14] Fast Inference from Transformers via Speculative Decoding
- **저자**: Yaniv Leviathan, Matan Kalman, Yossi Matias (Google)
- **발표**: ICML 2023 (40th International Conference on Machine Learning)
- **링크**: [arXiv:2211.17192](https://arxiv.org/abs/2211.17192) | [ICML Proceedings](https://proceedings.mlr.press/v202/leviathan23a.html)
- **관련성**: 작은 근사 모델의 출력을 대상 모델에서 병렬로 검증하는 speculative decoding 알고리즘 제안. 모델 수정이나 재학습 없이 T5-XXL에서 2-3배 가속. 출력 분포를 정확히 보존.

### [P15] Accelerating Large Language Model Decoding with Speculative Sampling
- **저자**: Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, John Jumper (DeepMind)
- **발표**: arXiv 2023
- **링크**: [arXiv:2302.01318](https://arxiv.org/abs/2302.01318)
- **관련성**: [P14]와 독립적으로 제안된 speculative sampling 알고리즘. 빠른 드래프트 모델의 짧은 continuation을 대상 모델에서 병렬로 스코어링하고, 수정된 rejection sampling으로 대상 모델의 분포를 보존. Chinchilla 70B에서 2-2.5배 디코딩 속도 향상.

---

## 7. MoE Expert Offloading

### [P16] DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale
- **저자**: Samyam Rajbhandari, Conglong Li, Zhewei Yao, Minjia Zhang, Reza Yazdani Aminabadi, Ammar Ahmad Awan, Jeff Rasley, Yuxiong He (Microsoft)
- **발표**: ICML 2022
- **링크**: [arXiv:2201.05596](https://arxiv.org/abs/2201.05596) | [ICML Proceedings](https://proceedings.mlr.press/v162/rajbhandari22a.html)
- **관련성**: MoE 학습 및 추론을 위한 엔드투엔드 솔루션. 모델 압축으로 MoE 모델 크기 최대 3.7배 축소, 동등 품질 dense 모델 대비 4.5배 빠르고 9배 저렴한 추론 달성. 본 프로젝트의 MoE 하이브리드 모드에 참고.

### [P17] Fast Inference of Mixture-of-Experts Language Models with Offloading
- **저자**: Artyom Eliseev, Denis Mazur
- **발표**: arXiv 2023
- **링크**: [arXiv:2312.17238](https://arxiv.org/abs/2312.17238)
- **관련성**: MoE 모델의 expert 접근 패턴의 규칙성을 활용한 개선된 캐싱 전략으로 GPU-RAM 데이터 전송을 줄여 오프로딩 가속. Mixtral-8x7B를 혼합 양자화로 데스크톱 하드웨어에서 실행 가능하게 함.

### [P18] MoE-Infinity: Activation-Aware Expert Offloading for Efficient MoE Serving
- **저자**: (Various)
- **발표**: arXiv 2024
- **링크**: [arXiv:2401.14361](https://arxiv.org/abs/2401.14361) | [GitHub](https://github.com/TorchMoE/MoE-Infinity)
- **관련성**: 시퀀스 레벨 expert 활성화 추적과 sparsity-aware expert 캐시를 통한 효율적 MoE 오프로딩. vLLM, Ollama, DeepSpeed 대비 per-token latency 3.1-16.7배 개선.

### [P19] Mixtral of Experts
- **저자**: Albert Q. Jiang, Alexandre Sablayrolles et al. (Mistral AI)
- **발표**: arXiv 2024
- **링크**: [arXiv:2401.04088](https://arxiv.org/abs/2401.04088)
- **관련성**: Sparse Mixture of Experts 언어 모델 Mixtral 8x7B. 각 레이어에 8개 expert, 라우터가 2개 선택. 총 47B 파라미터 중 13B만 활성화. Llama 2 70B 및 GPT-3.5 능가. MoE 오프로딩 연구의 주요 벤치마크 모델.

---

## 8. Disaggregated LLM 서빙

### [P20] DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving
- **저자**: Yinmin Zhong et al.
- **발표**: OSDI 2024 (18th USENIX Symposium on Operating Systems Design and Implementation)
- **링크**: [arXiv:2401.09670](https://arxiv.org/abs/2401.09670) | [USENIX](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)
- **관련성**: Prefill과 Decode를 별도 GPU에 분리(disaggregate)하여 두 단계 간 간섭을 제거. 기존 시스템 대비 7.4배 더 많은 요청 처리 또는 12.6배 더 엄격한 SLO 달성. 현재 대부분의 프로덕션 LLM 서빙 프레임워크가 이 설계를 채택.

### [P21] Splitwise: Efficient Generative LLM Inference Using Phase Splitting
- **저자**: Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Inigo Goiri, Saeed Maleki, Ricardo Bianchini (Microsoft)
- **발표**: ISCA 2024 (51st Annual International Symposium on Computer Architecture) - **Best Paper Award**
- **링크**: [arXiv:2311.18677](https://arxiv.org/abs/2311.18677) | [ACM DL](https://dl.acm.org/doi/10.1109/ISCA59077.2024.00019)
- **관련성**: Prefill/Decode를 이종 하드웨어(H100 vs A100)로 분리하는 3-tier 풀 설계. 동일 비용에서 1.4배 처리량 + 20% 비용 절감, 또는 동일 예산에서 2.35배 처리량 달성. 15% 전력 절감.

---

## 9. LLM 서빙의 요청 스케줄링/라우팅

### [P22] Orca: A Distributed Serving System for Transformer-Based Generative Models
- **저자**: Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, Byung-Gon Chun (Seoul National University / FriendliAI)
- **발표**: OSDI 2022 (16th USENIX Symposium on Operating Systems Design and Implementation)
- **링크**: [USENIX](https://www.usenix.org/conference/osdi22/presentation/yu)
- **관련성**: **Continuous Batching(연속 배칭)**의 원조 논문. 요청 단위가 아닌 반복(iteration) 단위 스케줄링과 선택적 배칭을 제안. GPT-3 175B에서 동일 지연 시간 대비 36.9배 처리량 향상. 모든 현대 LLM 서빙 시스템의 기반.

### [P23] SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills
- **저자**: Amey Agrawal, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S. Gulavani, Ramachandran Ramjee
- **발표**: arXiv 2023
- **링크**: [arXiv:2308.16369](https://arxiv.org/abs/2308.16369)
- **관련성**: Prefill 요청을 균등 크기 청크로 분할(chunked-prefills)하고, 나머지 슬롯을 decode로 채우는 decode-maximal batching 제안. Prefill-decode 간 GPU 활용 불균형 문제 해결의 기초.

### [P24] Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve
- **저자**: Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav Gulavani, Alexey Tumanov, Ramachandran Ramjee
- **발표**: OSDI 2024
- **링크**: [arXiv:2403.02310](https://arxiv.org/abs/2403.02310) | [USENIX](https://www.usenix.org/conference/osdi24/presentation/agrawal)
- **관련성**: [P23]의 chunked-prefills를 활용한 stall-free 스케줄링으로 진행 중인 decode를 멈추지 않고 새 요청 삽입 가능. Mistral-7B에서 vLLM 대비 2.6배, Falcon-180B에서 최대 5.6배 서빙 용량 향상.

---

## 10. KV Cache 관리 및 최적화

### [P25] (= [P1]) PagedAttention (vLLM)
- KV 캐시를 비연속 메모리 블록으로 관리하는 핵심 기법. 위 [P1] 참조.

### [P26] Prompt Cache: Modular Attention Reuse for Low-Latency Inference
- **저자**: (Various)
- **발표**: MLSys 2024
- **링크**: [MLSys Proceedings](https://proceedings.mlsys.org/paper_files/paper/2024/file/a66caa1703fe34705a4368c3014c1966-Paper-Conference.pdf)
- **관련성**: 프롬프트의 모듈화된 어텐션 재사용을 통한 저지연 추론. 반복되는 프롬프트 프리픽스에 대해 KV 캐시를 캐싱하여 재계산을 방지.

### [P27] Exploring CXL-based KV Cache Storage for LLM Serving
- **저자**: Yupeng Tang et al.
- **발표**: NeurIPS 2024 (ML for Systems Workshop)
- **링크**: [NeurIPS](https://neurips.cc/virtual/2024/103619)
- **관련성**: CXL 메모리를 KV 캐시 저장소로 활용. CXL-CPU 인터커넥트가 CPU-GPU 인터커넥트와 유사한 지연/대역폭을 제공하여 배치 크기 30% 증가 가능. GPU 요구량 최대 87% 감소, prefill GPU 활용률 7.5배 향상.

### [P28] LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference
- **저자**: (Various)
- **발표**: Technical Report
- **링크**: [lmcache.ai](https://lmcache.ai/tech_report.pdf)
- **관련성**: GPU/CPU/스토리지/네트워크 계층에 걸친 모듈형 KV 캐시 커넥터. 배치 연산, 컴퓨트/I/O 파이프라이닝 등으로 최대 15배 처리량, 최소 2배 지연 시간 개선.

---

## 11. LLM 서빙의 Continuous Batching

### [P29] (= [P22]) Orca
- Continuous Batching의 원조 논문. 위 [P22] 참조.

---

## 12. Intel IPEX (Intel Extension for PyTorch)

### [P30] Intel Extension for PyTorch (IPEX)
- **저자**: Intel
- **발표**: Open-source project
- **링크**: [GitHub](https://github.com/intel/intel-extension-for-pytorch) | [Intel Developer](https://www.intel.com/content/www/us/en/developer/articles/technical/accelerate-with-intel-extension-for-pytorch.html)
- **관련성**: PyTorch에 대한 Intel 최적화 확장. TorchScript IR 기반 연산자 퓨전, oneDNN 활용, AVX-512 VNNI 및 AMX 가속 지원. 본 프로젝트의 cpu_worker.py에서 IPEX 감지 및 활용.

### [P31] Optimizing Large Language Model Inference on Intel CPUs with IPEX and IPEX-LLM
- **저자**: Intel
- **발표**: Intel Technical Paper, 2024
- **링크**: [Intel](https://www.intel.com/content/www/us/en/content-details/834133/optimizing-large-language-model-inference-on-intel-cpus-with-ipex-and-ipex-llm-technical-paper.html)
- **관련성**: IPEX와 IPEX-LLM을 사용한 Intel CPU에서의 LLM 추론 최적화. 배치 추론과 멀티인스턴스 추론 두 가지 접근법 설명.

### [P32] IPEX-LLM: Intel LLM Library for PyTorch
- **저자**: Intel
- **발표**: Open-source project
- **링크**: [GitHub](https://github.com/intel/ipex-llm)
- **관련성**: Intel XPU(iGPU, NPU, Arc/Flex/Max GPU)에서 LLM 추론 및 파인튜닝 가속. llama.cpp, vLLM, DeepSpeed 등과 통합 지원.

---

## 13. FlexGen - LLM 추론 오프로딩

### [P33] FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU
- **저자**: Ying Sheng, Lianmin Zheng, Binhang Yuan, Zhuohan Li, Max Ryabinin, Daniel Y. Fu, Zhiqiang Xie, Beidi Chen, Clark Barrett, Joseph E. Gonzalez, Percy Liang, Christopher Re, Ion Stoica, Ce Zhang
- **발표**: ICML 2023 (Oral)
- **링크**: [arXiv:2303.06865](https://arxiv.org/abs/2303.06865) | [ICML](https://icml.cc/virtual/2023/oral/25565) | [GitHub](https://github.com/FMInference/FlexLLMGen)
- **관련성**: GPU/CPU/디스크 메모리와 컴퓨팅을 유연하게 집계하여 제한된 GPU 메모리에서 대형 LLM 실행. 가중치와 어텐션 캐시를 4비트로 압축하여 OPT-175B를 단일 16GB GPU에서 실행, 기존 오프로딩 시스템 대비 100배 처리량.

---

## 14. llama.cpp CPU 추론 최적화

### [P34] llama.cpp / GGML
- **저자**: Georgi Gerganov et al.
- **발표**: Open-source project
- **링크**: [GitHub](https://github.com/ggml-org/llama.cpp)
- **관련성**: 최소 의존성으로 LLaMA 및 호환 모델을 CPU/GPU에서 실행하는 C/C++ 추론 엔진. AVX, AVX2, AVX-512, AMX 지원. 1.5-8비트 정수 양자화(Q4_0, Q4_1, Q5_K_M, Q8_0 등)로 메모리 절감 및 추론 가속. GGUF 포맷으로 메모리 매핑 zero-copy 로딩. 본 프로젝트의 Q8_0 양자화 커널(quant_q8_0.cpp) 설계에 참고.

### [P35] Which Quantization Should I Use? A Unified Evaluation of llama.cpp Quantization on Llama-3.1-8B-Instruct
- **저자**: (Various)
- **발표**: arXiv 2026
- **링크**: [arXiv:2601.14277](https://arxiv.org/abs/2601.14277)
- **관련성**: llama.cpp의 다양한 양자화 방식(Q2_K ~ Q8_0)에 대한 체계적 비교 평가. 각 양자화 수준별 정확도-속도-메모리 트레이드오프 분석.

---

## 15. DeepSeek V2/V3 MoE 아키텍처

### [P36] DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models
- **저자**: Damai Dai, Chengqi Deng, Chenggang Zhao, Runxin Xu et al. (DeepSeek AI)
- **발표**: ACL 2024
- **링크**: [arXiv:2401.06066](https://arxiv.org/abs/2401.06066) | [ACL Anthology](https://aclanthology.org/2024.acl-long.70.pdf)
- **관련성**: Fine-grained expert segmentation과 shared expert isolation 전략을 통한 극도의 expert 특화. 기존 MoE(GShard) 대비 더 유연한 expert 조합이 가능하여 40% 연산만으로 LLaMA2 7B 수준 성능 달성. 본 프로젝트의 MoE 하이브리드 모드 설계에 핵심 참고.

### [P37] DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model
- **저자**: DeepSeek AI
- **발표**: arXiv 2024
- **링크**: [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)
- **관련성**: Multi-head Latent Attention(MLA)과 DeepSeekMoE를 결합. MLA는 KV 캐시를 잠재 벡터로 압축하여 KV 캐시 93.3% 절감, 최대 생성 처리량 5.76배 향상. 236B 총 파라미터/21B 활성화, 128K 컨텍스트 지원.

### [P38] DeepSeek-V3 Technical Report
- **저자**: DeepSeek AI
- **발표**: arXiv 2024
- **링크**: [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
- **관련성**: 671B 총 파라미터/37B 활성화 MoE 모델. MLA + DeepSeekMoE + 보조 손실 없는 로드 밸런싱 + 멀티토큰 예측 학습. KTransformers 등 CPU/GPU 하이브리드 추론 연구의 주요 타겟 모델.

---

## 부록: 추가 관련 서베이

### [S1] A Survey on Large Language Model Acceleration based on KV Cache Management
- **링크**: [GitHub (Awesome-KV-Cache-Management)](https://github.com/TreeAI-Lab/Awesome-KV-Cache-Management)
- **관련성**: KV 캐시 관리 기법(압축, 공유, 오프로딩, 프리픽스 캐싱 등)에 대한 종합 서베이.

### [S2] A Comprehensive Survey of Speculative Decoding
- **링크**: [ACL Findings 2024](https://aclanthology.org/2024.findings-acl.456.pdf)
- **관련성**: Speculative decoding의 다양한 변형(draft model, self-speculation, Medusa, lookahead 등)에 대한 종합 서베이.

### [S3] A Survey on Inference Optimization Techniques for Mixture of Experts Models
- **링크**: [arXiv:2412.14219](https://arxiv.org/abs/2412.14219)
- **관련성**: MoE 모델 추론 최적화 기법(expert 캐싱, 오프로딩, 양자화, 라우팅 최적화 등)에 대한 종합 서베이.

### [S4] LLM Inference Serving: Survey of Recent Advances and Opportunities
- **링크**: [arXiv:2407.12391](https://arxiv.org/abs/2407.12391)
- **관련성**: LLM 추론 서빙의 최신 발전(배칭, 스케줄링, KV 캐시, 분산 추론, 하드웨어 최적화 등)에 대한 종합 서베이.
