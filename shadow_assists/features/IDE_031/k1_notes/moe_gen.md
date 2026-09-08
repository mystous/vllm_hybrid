# MoE-Gen (arXiv'25) — K1 정독 노트

- **서지**: Tairan Xu, Leyang Xue, Zhan Lu, Adrian Jackson, Luo Mai (Univ. of Edinburgh/EPCC). "MoE-Gen: High-Throughput MoE Inference on a Single GPU with Module-Based Batching." arXiv 2503.09716 (v1, 2025-03-12). venue 미기재 (ICML 양식). 코드 github.com/EfficientMoE/MoE-Gen (현재 README 는 후속 "BatchGen" 으로 교체됨).
- **전문 접근 여부**: **전문 확보 (13쪽, 부록 A·B 포함)**.
- **전문 확보 경로**: HF 데이터셋 `Chelsea707/arxiv-cs-2020-2025-pdfs` `data/2025/2503_09xxx/2503.09716.pdf` (curl) → pypdf 추출. HF 초록 페이지로 교차 확인.

## 문제·결정 변수
- 대상: 단일 GPU (A5000 24GB/A6000 48GB) + 대용량 호스트 메모리에서 MoE (Mixtral, DeepSeek-V2/R1) 의 **offline 처리량**. 모듈 단위 배칭 (attention 배치를 누적해 expert 배치를 키움).
- 탐색 변수 (Eq.1, Table 2): `maximize B / T(B, S_Expert, S_Params, ba, be, ω)` — 누적 배치 B, attention 마이크로배치 ba, expert 마이크로배치 be, **attention 의 CPU 분할 비율 ω**, expert prefetch 버퍼 S_Expert, **GPU 에 상주 캐시할 파라미터 크기 S_Params**. KV 는 **전량 호스트** (설계 고정): "we demonstrate that fully offloading the KV-cache outperforms partial offloading".
- 인용 (§4.3): "It is possible that the entire process is memory-bound, so adding S_Expert or ba may not yield any benefit. In this case, using the spare GPU space to cache part of the model parameters reduces HtoD traffic for copying model weights, thereby alleviating memory-bound constraints."

## 비용 모델 파라미터 출처와 검증
- 모델: DAG 임계경로 DP (`dp[v] = max_{u∈pred} dp[u] + cost(v)`).
- **파라미터 출처 = 모듈별 오프라인 프로파일** (배치·길이 격자 실측): "Before runtime, each module is profiled offline across various batch sizes and sequence lengths, generating comprehensive profiling data." / "creating batching strategy based on hardware (e.g., connection speed, GPU memory capacity) and software (e.g., performance and memory usage of GPU and CPU kernels under various input batch sizes) profiling."
- 예측 검증·오차: **없음** (탐색 결과로 실측만). 오차 수치 인용 불가.

## expert·KV 경합 취급
- **PCIe (HtoD) 예산 아래 expert 스트리밍과 KV 복사가 경합한다는 인식은 명시**: "caching the KV-cache in GPU memory limits batch size, leading to increased fetching traffic for expert weights (e.g., up to 86GB for Mixtral-8x7B). By trading off KV-cache copying, MoE-Gen achieves up to 20× savings in fetching traffic" ; "CPU-based computation reduces the HtoD overhead and its bandwidth usage as the KV-cache stays in Host memory."
- 결정은 (GPU 상주 파라미터 크기 S_Params, KV 는 전량 호스트, CPU attention 비율 ω) — KV 배치 자체는 변수가 아니고 **KV 읽기를 PCIe 로 할지 CPU 에서 할지 (ω)** 가 변수. 호스트 DRAM 대역폭 항 없음 (병목은 HtoD).
- 전이점 실측 (§A.1): "this advantage holds only up to a certain breakeven point—approximately 60% CPU offloading in our experiments. Beyond this threshold, memory copying ceases to be the critical bottleneck, and further offloading results in GPU idling as it waits for CPU computations".

## D1/D2/D3 판정
| 조건 | 판정 | 근거 (원문) |
|---|---|---|
| D1 expert 상주 수 + KV 배치를 공통 대역폭 예산으로 함께 결정 (MoE) | **부분 예** | 같은 탐색에서 `S_Params` (GPU 상주 파라미터) 와 `ω` (KV 를 PCIe 로 읽을지 CPU 에서 읽을지) 를 공통 HtoD 예산 아래 결정: "using the spare GPU space to cache part of the model parameters reduces HtoD traffic". 단 KV 위치는 고정 (전량 호스트), 예산은 PCIe, expert 는 크기 단위이고 hot 집합/트레이스 항 없음. |
| D2 마이크로벤치-only 사전 예측 + 오차 | **아니오** | "each module is profiled offline across various batch sizes and sequence lengths" (엔드투엔드에 가까운 격자 프로파일); 오차 미보고. |
| D3 binding 자원 전이점 | **부분 (실측)** | ω 축의 breakeven ~60% 를 실측으로 보고; 배치 축 전이 (Fig.3: 2^10 토큰 이상이면 FLOPs 포화, 2^11 이상이면 GPU idle 0) — 예측 모델로 위치를 낸 것은 아님. |

## 우리와의 delta
- 그들이 이미 한 것: MoE 에서 "GPU 여유 메모리를 파라미터 캐시에 쓰면 HtoD 예산이 풀린다" 는 공통 예산 인식과 (S_Params, ω) 공동 탐색; CPU attention 분할 비율의 breakeven 실측.
- 그들이 안 한 것: (a) 예산이 PCIe 이며 **DDR** 아님 (그들 CPU 는 attention 만, expert 는 GPU 로 스트리밍). (b) hot expert 집합 H 와 라우팅 트레이스 항 없음 (균등 라우팅 가정: "the number of tokens routed to each expert is often uniformly distributed"). (c) 온라인 continuous batching 아님. (d) 사전 예측·오차 없음.
- 우리가 못 주장하게 되는 것: "GPU 메모리 분할 (파라미터 캐시 vs KV/버퍼) 을 공통 대역폭 예산으로 결정하는 최초 MoE 시스템". 위협도 **중~상** (D1 의 PCIe 판).
