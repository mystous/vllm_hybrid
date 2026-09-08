# HybriMoE (DAC'25) — K1 정독 노트

- **서지**: Shuzhang Zhong, Yanfan Sun, Ling Liang, Runsheng Wang, Ru Huang, Meng Li (PKU/Beihang). "HybriMoE: Hybrid CPU-GPU Scheduling and Cache Management for Efficient MoE Inference." DAC 2025. **arXiv 2504.05897** (v1, 2025-04-08). 코드 github.com/PKU-SEC-Lab/HybriMoE.
- **ID 정정**: 과제 목록의 `2501.05087` 은 다른 논문 (cs.RO, "Enhanced Quantile Regression with Spiking Neural Networks ...") 임. HybriMoE 의 실제 ID 는 2504.05897 (GitHub README·HF 초록 페이지로 확인).
- **전문 접근 여부**: **전문 확보 (7쪽)**.
- **전문 확보 경로**: HF 데이터셋 `Chelsea707/arxiv-cs-2020-2025-pdfs` `data/2025/2504_05xxx/2504.05897.pdf` (curl) → pypdf 추출.

## 문제·결정 변수
- 대상: kTransformers 위에 구현, 엣지 (A6000 + Xeon 5220R 10코어) 에서 MoE (Mixtral, DeepSeek-V2-Lite, Qwen2-57B) 의 **단일 요청** prefill/decode 지연. GPU expert 캐시 비율 25/50/75% 를 외부 설정.
- 결정 (층 내, 동적): 활성 expert 를 GPU 큐 (캐시된 것, 부하 내림차순) / CPU 큐 (미캐시, 오름차순) 로 나누고, 시뮬레이션으로 `arg min max(CPU_TIME(cpu_expert), GPU_TIME(gpu_expert))` (Eq.2). 층 간 prefetch (다음 3층 gate 재사용), 캐시 교체 MRS (라우팅 점수 EMA).
- KV 배치: 변수 아님 (GPU).

## 비용 모델 파라미터 출처와 검증
- 시뮬레이터 파라미터 = **워밍업 실측**: "The system begins with a warmup phase to collect essential performance metrics, such as CPU and GPU processing speeds and data transfer latency."
- 모델 형태: "the GPU computation time is assumed to be constant, the CPU computation time is proportional to the expert's load, and the transmission time is fixed" (Fig.5 설명). 관찰: "GPU computation time scales linearly with the number of activated experts, while CPU computation benefits from overlapping memory access and computation due to its larger cache."
- 예측 검증·오차: **없음** (지연 실측 비교 1.33×/1.70× 만).

## expert·KV 경합 취급
- **없음**. GPU 캐시 비율은 실험 변수 ("we adjust the upper bound of the GPU expert cache ratio"), KV 와의 메모리 교환 논의 없음. 대역폭은 PCIe 전송만. 호스트 DRAM 대역폭 항 없음.
- 캐시 비율 ↑ 에 따른 병목 이동은 정성적으로만: "As cache capacity increases to 75%, the gap narrows ... as higher capacities reduce expert competition".

## D1/D2/D3 판정
| 조건 | 판정 | 근거 (원문) |
|---|---|---|
| D1 expert 상주 수 + KV 배치 공동 결정 | **아니오** | 캐시 비율은 외부 설정 ("adjust the upper bound of the GPU expert cache ratio"); KV 언급 없음. |
| D2 마이크로벤치-only 사전 예측 + 오차 | **아니오** | "warmup phase to collect essential performance metrics" 로 런타임 스케줄링용; 처리량 사전 예측·오차 보고 인용 불가. |
| D3 binding 자원 전이점 | **아니오** | 부하 균형 (`max(CPU_TIME, GPU_TIME)` 최소화) 이지 자원 전이점 분석 아님. 인용 불가. |

## 우리와의 delta
- 같은 점: kTransformers 기반, cold expert 를 CPU 에서 계산, 캐시 비율 스윕 (25/50/75%) — 우리의 H 스윕과 외형 유사.
- 다른 점: 단일 요청 지연·PCIe 중심·KV 무관·사전 예측 없음. 우리는 배치 서빙 처리량, DDR 예산, KV 결합, 사전등록 예측. 위협도 **하**.
