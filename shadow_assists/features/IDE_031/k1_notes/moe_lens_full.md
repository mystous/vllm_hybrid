# MoE-Lens — K1 전문 정독 노트

- **서지**: Yichao Yuan, Lin Ma, Nishil Talati (U. Michigan). "MoE-Lens: Towards the Hardware Limit of High-Throughput MoE LLM Serving Under Resource Constraints". arXiv 2504.09345v1 (2025-04-12), 12쪽 본문 (ACM 템플릿, HPDC'26 채택 기록은 원문에 없음).
- **전문 확보 경로**: HF 데이터셋 `Chelsea707/arxiv-cs-2020-2025-pdfs` → `data/2025/2504_09xxx/2504.09345.pdf` (1.19 MB, 14쪽) → pypdf 텍스트 추출. **전문 확보**.

## 무대·토폴로지
- 무대 = **오프라인 배치 처리량** (온라인 서빙 아님). 원문: "Similar to prior works [9, 19, 38, 45], our focus is offline, batching processing inference tasks, such as model evaluation ... where maximizing inference throughput directly reduces total job completion time." / "This setting differs from traditional LLM serving systems ... optimized for latency-sensitive applications like chatbots ... resource-constrained inference systems ... prioritize overall throughput and can afford to trade off latency."
- 토폴로지 = **CPU attention + KV 상주, GPU 는 GEMM(모든 expert 포함) 전체를 PCIe 스트리밍**. 원문: "the GPU handles compute-intensive GEMM operations, while the CPU processes the relatively lightweight attention mechanism." / "All weights are stored in pinned CPU memory ... dynamically loaded into the GPU's weight buffer on demand. The size of the weight buffer is two times the model weight size divided by the number of layers." → GPU 에 hot expert 상주 개념 **없음** (weight buffer = 2개 층 분량). CPU 가 expert 연산을 하지 않음 (CPU 는 attention 만).
- 실험 머신: 2× Xeon Platinum 8380, DDR4-3200 8ch, 실측 소켓당 ≈150 GB/s, A40 48 GB (16–24 GB 로 제한), Mixtral 8x7B/8x22B, DBRX. KV 70–210 GB.

## 결정 변수
- 모델 입력: KV 용량 M (CPU 메모리 용량으로 주어짐), p/g (프롬프트·생성 길이), 요청 배치 K, 페이지 블록 b, δ = ModelSize / B_IO, T_GPU.
- 시스템이 결정하는 것: 이터레이션당 prefill 시퀀스 수 q (Eq. 8), prefill/decode 겹침, 선점(preemption). **expert 상주 수·KV 분할은 결정 변수가 아님** (GPU 메모리에 KV 도 expert 도 상주하지 않음).

## 비용 모델 파라미터 출처·검증·오차
- Stage 1: T_max = min(PME·M/δ, T_GPU) (Eq. 4). Stage 2: T = min(T1, T2) (Eq. 10–14), T1 은 IO-bound (매 이터레이션 = δ), T2 는 GPU-bound.
- 파라미터 출처: B_IO 는 **마이크로벤치 실측** — "We estimate CPU-GPU IO bandwidth B_IO at 19.5GB/s, based on 1GB tensor transfers." T_GPU / n_real 은 **Pipeline Profiler 로 실측** — "the profiler fits a line to capture the relationship between token count and GPU time. It also measures the time required to transfer a layer of weights to the GPU. It calculates the maximum number of parallel tokens using the line's slope and weight transfer time." (Fig. 7, n_real = 18236). 스펙-only 아님, end-to-end 회귀도 아님 — 연산자 단위 프로파일 + 해석식.
- 검증: 자기 시스템의 실측 처리량 대비 — "On average, MoE-Lens's performance model (§5.4) predicts throughput with 94% accuracy, as shown in Figures 11 and 12." 사전등록(pre-registration) 언급 **없음**. 오차 보고는 평균 정확도 한 숫자뿐 (셀별 분포·최악값 없음). 실패 사례 자인: "MoE-Lens falls short of the model's prediction when running Mixtral8x7B with a 210GB KV cache, due to contention between CPU attention computation and IO, which delays weight transfers."
- 예측 대상: 배치 전체 처리량 (tokens/s). 디코드 스텝 시간(TPOT) 격자 아님.

## expert·KV 경합 취급 (호스트 DRAM 대역폭 공유 항)
- **있음 (명시 식)**: Eq. (5) "the total CPU memory bandwidth requirement is the sum of the bandwidth needed to access the KV cache (B_KV) and the bandwidth to transfer weights from CPU to GPU (B_IO). B_Mem = B_KV + B_IO = M / (M_weight / B_IO)". 다만 **비-binding 으로 결론**: "it typically does not become a bottleneck on modern CPUs equipped with multiple memory channels ... B_Mem = 60 GB/s. This is well within the capabilities of modern CPUs."
- 실측 경합 관찰 (§8.2): "These two operations contend for shared CPU-side resources, particularly memory bandwidth. ... This contention at the CPU memory controller slows down weight transfers, increasing the time to transfer weights from CPU to GPU from approximately 5 seconds to 6 seconds."
- 즉 KV 읽기 + 가중치 스트리밍이 같은 DDR 을 쓴다는 **항은 존재**하나, (a) 분할 결정에 쓰이지 않고 (b) 모델은 이를 무시 (오차 원인으로만 사후 언급).

## D1·D2·D3 판정
| 조건 | 판정 | 근거 (원문) |
|---|---|---|
| D1 | **아니오** | expert 는 GPU 에 상주하지 않고 전량 스트리밍, KV 는 CPU 상주. 공통 예산으로 두 상주량을 결정하는 항 없음. 결정은 "maximize the number of concurrent tokens on the GPU by fully utilizing CPU memory for KV cache usage" — KV 용량 채우기 뿐 |
| D2 | **예 (형식) / 부분 (내용)** | 하이브리드 MoE 처리량을 해석 모델 + 마이크로벤치 파라미터(B_IO 1 GB 전송, GPU 프로파일 직선)로 예측, "average 94% accuracy". end-to-end 캘리브레이션 아님. 사전등록 없음. 대상은 배치 처리량이며 CPU expert 연산·DDR 경합 항은 모델에 없음 |
| D3 | **예 (다른 축)** | "As KV cache capacity increases, the system transitions from a CPU memory capacity-bound regime to a GPU-bound regime." Fig. 3(b)/4 "Turning point for Upper Bound / for Batch Sz. 200k". 전이 축 = **KV 용량** (CPU 메모리), binding = PCIe IO ↔ GPU 연산. hot expert 수 H 축 아님 |
| DDR 공유 항 | **예 (비-binding 취급)** | Eq. (5), §8.2 인용 위 |

## 우리와의 delta
- 같은 점: 하이브리드 MoE, 해석 모델 + 마이크로벤치 파라미터로 처리량 사전 예측 + 정확도 보고, binding 자원 전이점 도식, KV 읽기와 가중치 스트리밍의 DDR 합산 식. → "하이브리드 MoE 성능 모델 + 전이점" 자체는 신규성 아님.
- 다른 점: (1) 무대: 오프라인 배치 처리량 vs 우리 온라인 TPOT 스텝 시간 격자. (2) 토폴로지: CPU=attention·GPU=전 expert 스트리밍 vs 우리 CPU=cold expert(AMX)·GPU=attention+hot expert. (3) 전이 축: KV 용량 vs 우리 hot expert 수 H. (4) 결정 변수: MoE-Lens 는 GPU 상주 분할이 없어 D1 부재; 우리는 GPU 메모리를 hot expert↔KV 로 나누는 결정. (5) DDR 공유: MoE-Lens 는 식은 있으나 "not a bottleneck" 으로 버림, 우리는 이를 예산으로 씀 — 단 우리 쪽에서 실제 binding 임을 실측으로 보여야 함 (MoE-Lens 의 5→6 s 관찰이 오히려 우리 전제의 근거). (6) 오차 보고: 평균 한 숫자 vs 우리 셀별 ≤20% 사전등록.
- 위협도: **상**. 서베이·인트로에서 반드시 직접 비교해야 하며, "예측 모델 있음" 만으로는 우리 기여가 안 됨. 우리 기여는 (a)+(b)+(c)+(d) 결합과 사전등록 규율로 좁혀야 함.
