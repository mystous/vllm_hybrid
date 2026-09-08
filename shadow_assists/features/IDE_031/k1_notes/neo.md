# NEO (MLSys'25) — K1 정독 노트

- **서지**: Xuanlin Jiang, Yang Zhou, Shiyi Cao, Ion Stoica, Minlan Yu (PKU/UC Berkeley/UC Davis/Harvard). "NEO: Saving GPU Memory Crisis with CPU Offloading for Online LLM Inference." MLSys 2025. arXiv 2411.01142 (v1, 2024-11-02, "Preprint, under review" 판).
- **전문 접근 여부**: **전문 확보 (13쪽)**.
- **전문 확보 경로**: HF 데이터셋 `Chelsea707/arxiv-cs-2020-2025-pdfs` `data/2024/2411_01xxx/2411.01142.pdf` (curl) → pypdf 추출. MLSys proceedings·저자 사이트·GitHub README 는 접속 불가/무내용.

## 문제·결정 변수
- 대상: **dense** LLM (LLaMa 7B/8B/70B) 의 **온라인** 서빙. 가중치는 전부 GPU. 결정 = 어느 요청의 **decode attention + KV 를 CPU 로 보낼지** (요청 단위, 전량 GPU 또는 전량 CPU) 와 두 비대칭 서브배치 구성.
- 인용 (§3.1): "For any request that has already been prefilled in the system, its KV cache will either reside entirely in the GPU-cache—designated as a 'GPU-request'—or entirely in the CPU-cache—designated as a 'CPU-request'. Requests are prioritized for storage in the GPU-cache to maximize GPU memory utilization."
- 인용 (§6): "NEO implicitly assumes putting all model weights on the GPU and only offloading attention computation to the CPU is the most efficient way to balance GPU and GPU loads."

## 비용 모델 파라미터 출처와 검증
- 스케줄러 모델 (§3.2): `T ≈ Ttr = L × (max{Tpo0 + Tpr0, Tca1} + max{Tpo1 + Tpr1 + Tga0, Tca0})`.
- **파라미터 출처 = 오프라인 프로파일 + 선형 보간** (커널별 실측): "To estimate Tl, Tga, and Tca, NEO does offline profiling for typical input/output lengths and uses linear interpolation to approximate the values for other lengths."
- 검증: 모델 오차 미보고. 오차 존재를 인정: "sometimes slightly worse due to suboptimal scheduling decisions caused by the inevitable inaccuracy of the offline performance profiling."
- 매 iteration GPU-only 스케줄과 비교해 큰 쪽 선택 (Greedy 원칙).

## expert·KV 경합 취급
- MoE/expert 없음. **CPU DRAM 대역폭이 CPU attention 의 binding 자원**이라는 실측은 있음 (§5.5): "The peak throughput gain is positively related to the CPU memory bandwidth. This supports the fact that the memory bandwidth, rather than computing power (i.e., number of cores), is the factor that determines the performance of attention operation on CPUs."
- 가중치 스트리밍과 KV 읽기의 경합은 다루지 않음 (가중치 전부 GPU). §6 에서 다른 구성요소 오프로드를 미래 과제로: "offloading some of the dense operations to the CPU could alleviate the GPU's pressure in these extreme workloads. Nevertheless, the actual gain needs to be validated by further exploration."
- CPU 과부하 회피는 규칙으로: "too many CPU requests might overload the CPU capacity either in memory bandwidth or compute."

## D1/D2/D3 판정
| 조건 | 판정 | 근거 (원문) |
|---|---|---|
| D1 expert 상주 수 + KV 배치 공동 결정 (MoE) | **아니오** | MoE 아님; 가중치 전부 GPU ("putting all model weights on the GPU and only offloading attention computation to the CPU"). 결정은 KV/attention 의 요청 단위 배치뿐. |
| D2 마이크로벤치-only 사전 예측 + 오차 보고 | **아니오** | 파라미터는 "offline profiling for typical input/output lengths and uses linear interpolation" (커널 프로파일; 스펙-only 아님). 오차 수치 인용 불가. |
| D3 binding 자원 전이점 명시 분석 | **부분 (정성)** | 출력 길이 축에서 균형점 서술: "As the output length increases, NEO's gains first grow to the maximum point, where GPU and CPU times are exactly balanced, then gradually drop as the system launches a larger proportion of GPU-only batches." 전이점을 모델로 예측·위치 특정하지는 않음. |

## 우리와의 delta
- 같은 점: 온라인 서빙에서 KV 일부를 호스트로 내리고 CPU attention 으로 읽는 구조 (우리 HiCache 경로와 데이터 흐름 유사); DDR 대역폭이 CPU attention 의 binding 자원이라는 실측.
- 다른 점: NEO 는 dense 이고 가중치 전부 GPU → **expert 스트리밍과 DRAM KV 읽기가 같은 DDR 을 나눈다는 항이 없음**. 우리는 그 경합 항 + H 축 전이점 + 마이크로벤치-only 사전 예측. 위협도 **중** (KV 오프로드 부분의 선행, 결합 예산은 없음).
