# MoE-Infinity — arXiv 2401.14361 (v3, 2025-03-12)

## 서지 / 전문 접근
- Leyang Xue, Yao Fu, Zhan Lu, Chuanhao Sun, Luo Mai, Mahesh Marina (Univ. of Edinburgh). v3 제목 "MoE-Infinity: Efficient MoE Inference on Personal Machines with Sparsity-Aware Expert Cache" (v1 제목은 "Activation-Aware Expert Offloading …"; 사용자 서베이는 v1 제목 기준). 학회 표기 없음(ICML 양식).
- 전문 접근: **예** — HF 미러 `Chelsea707/arxiv-cs-2020-2025-pdfs` 의 PDF(v3, 11쪽) 텍스트 추출.

## 결정 변수
- expert cache 교체 우선순위(요청 수준 EAM 매칭으로 재사용 확률 추정, 층 근접 가중) 와 prefetch 대상. EAMC(트레이스 저장소) 용량.
- 배치 1 고정("MoE inference on personal devices typically operates with a batch size of one"). 캐시 용량·KV 배치는 결정하지 않음.

## 비용 모델 파라미터 출처 / 예측 검증 / 보고 오차
- 성능 비용 모델 없음. 워킹셋 크기는 트레이스 연구로 추정: "For MoE models with around 100 experts ... fewer than 5% of experts are repeatedly activated when decoding tokens for a single request".
- 처리량 예측·오차 보고 없음. 전부 실측(A5000 24GB, PCIe 4.0).

## expert 와 KV 의 메모리·대역폭 경합 취급
- KV 는 전량 GPU: "MoE-Infinity keeps all KV-cache in GPU memory, resulting in less fetching under long context". 초기화: "we will reserve the amount of GPU memory in corresponding to the maximal output length we observed in the open LLM datasets".
- 경합을 관찰은 함: "Increasing the context length leads to a larger KV-cache size in GPU memory, which in turn reduces the available buffer size for caching experts." / vLLM 에 대해 "The KV-cache traffic causes contention with expert fetching, which delays and even blocks expert prefetching". 그러나 결정 변수로 다루지 않음.

## 캐시 크기 결정 방식
- **잔여(고정)**: GPU 메모리 − dense 파라미터 − KV 예약 = expert 버퍼. 예: 컨텍스트 2^16 에서 버퍼 2GB, 2^17 에서 1GB 로 줄어 on-demand 로 전락.

## 판정
- **D1: 아니오.** KV 전량 GPU, expert 캐시는 잔여. 인용: "we will reserve the amount of GPU memory in corresponding to the maximal output length".
- **D2: 아니오.** 인용 불가(비용 모델 없음).
- **D3: 아니오.** Table 1 의 GPU idle time 분해만 있음.

## 우리와의 delta
- 우리는 (a) 배치>1 서빙, (b) KV 위치를 결정 변수로, (c) expert 상주 수와 KV 를 공통 DDR 예산으로 결정, (d) 스텝 시간 사전 예측. MoE-Infinity 는 캐시 교체 정책 논문이며 위 어느 것도 다루지 않음. 단 "KV 가 커지면 expert 버퍼가 줄어 성능이 꺾인다" 는 관찰(Fig. 8)은 우리 주장 3 의 동기 인용으로 사용 가능.
