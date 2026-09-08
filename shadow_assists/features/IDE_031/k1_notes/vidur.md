# Vidur (MLSys'24) — K1 정독 노트

- **서지**: Amey Agrawal, Nitin Kedia, Jayashree Mohan, Ashish Panwar, Nipun Kwatra, Bhargav Gulavani, Ramachandran Ramjee, Alexey Tumanov (Georgia Tech / Microsoft Research India). "Vidur: A Large-Scale Simulation Framework For LLM Inference." MLSys 2024. arXiv 2405.05465. 코드 github.com/microsoft/vidur.
- **전문 접근 여부**: **불가** (arxiv.org, mlsys.org 타임아웃). WebSearch 스니펫 (arXiv abs/html 요약, MLSys 포스터, 2차 해설, 후속 시뮬레이터 논문의 비교 서술) 만. 인용은 스니펫 문장.

## 목적·결정 변수
- 이산 사건 시뮬레이터: 스케줄러 (vLLM, Orca, Sarathi 등), 배치 정책, 병렬화 (TP/PP/replica) 조합을 시뮬레이션해 latency / throughput / TTFT 등 추정. Vidur-Search 가 SLO 를 만족하는 최저 비용 배포 구성을 탐색.
- 스니펫 인용: "Vidur carefully models the performance of various operators involved in LLM inference using a combination of experimental profiling and predictive modeling".
- 결정 변수는 배포 구성 (GPU 종류·수, TP/PP, 스케줄러 파라미터). 메모리 계층 배치 결정 없음.

## 비용 모델 파라미터 출처와 검증
- 파라미터 출처: **실기 프로파일** — token-level 연산자 (matmul 등) 를 (total_tokens, TP_shards) 격자로 CUPTI 로 측정하고, 미측정 지점은 연산자별 **Random Forest** 회귀로 보간. attention 은 prefill / decode 로 나눠 프로파일. 스니펫 인용: "Token-level operators (e.g., matrix multiplies) are profiled over a grid of (total_tokens, TP_shards) using CUPTI instrumentation on a single large GPU."
- 즉 "서빙 무관 단일 GPU 연산자 프로파일 → 배포 구성 예측" 이라는 점에서 우리 마이크로벤치-only 규율과 방법론이 가장 가까움 (스펙 아님, 서빙 실측도 아님).
- 검증: 스니펫 인용 "Vidur estimates inference latency with less than 9% error across the range." 모델 LLaMA2-7B/70B, InternLM-20B, Qwen-72B (전부 dense). 정적 워크로드 중앙값 오차 3.33% 미만 (2차 서술).

## expert·KV 경합 취급
- MoE: **미지원**. 후속 논문 (Frontier 2605.21312) 의 서술: "Vidur's accuracy collapses on MoE because its operator library has no GroupedGEMM class and cannot express routing-dependent runtime." (2차 인용; Vidur 원문에서 MoE 를 future work 로 적었는지는 미확인.)
- CPU 연산 / 호스트 offload / KV 티어링 없음. KV 는 GPU 메모리 용량 제약으로만 (추측).
- **호스트 DRAM 대역폭 모델링**: 아니오.

## D1/D2/D3 판정
| 조건 | 판정 | 근거 |
|---|---|---|
| D1 | **아니오** | dense, GPU-only. 배치 결정 없음 |
| D2 | **부분 (아니오)** | 프로파일-only 파라미터로 서빙 지연을 사전 예측하고 오차 (<9%) 보고 → 방법론 선례. 하이브리드 MoE 아님 (MoE 자체 미지원) |
| D3 | **아니오** | 인용 불가. 전이점 분석 없음 |
| 호스트 DRAM BW | 아니오 | — |

## 우리와의 delta
- 우리 주장 2 의 "마이크로벤치 → 예측 → 오차 보고" 절차는 Vidur 가 dense/GPU 에서 이미 한 것. 우리가 더하는 것: (1) CPU expert 커널 (expert당·행당 비용), DDR 스트리밍, KV 호스트 읽기, 소켓 간섭이라는 **하이브리드 항** (2) 라우팅 트레이스에서 나온 distinct cold expert 기대값 (routing-dependent runtime — Vidur 가 못 하는 부분) (3) 전이점·분할 정책. 논문에서 Vidur 를 "방법론적 선례, 대상 다름" 으로 인용해야 함. 위협도 **중**.
