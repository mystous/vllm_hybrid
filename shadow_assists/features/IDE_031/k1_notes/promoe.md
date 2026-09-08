# ProMoE — arXiv 2410.22134 (v3, 2025-09-01)

## 서지 / 전문 접근
- Xiaoniu Song, Zihang Zhong, Rong Chen, Haibo Chen (SJTU IPADS). "ProMoE: Fast MoE-based LLM Serving using Proactive Caching". 학회 미표기.
- 전문 접근: **예** — HF 미러 PDF(14쪽) 텍스트 추출.

## 결정 변수
- 무엇을 prefetch 할지: 층별 학습 예측기(2-layer MLP, ~2M 파라미터, 정확도 84.7%) + stride prefetch. 어떻게: chunked prefetch, early preemption, reordered inference.
- 캐시는 per-layer LRU 고정. 캐시 비율(cache rate) 은 외부 파라미터로 10~70% 스윕. 배치 1 기본, 1~4 스윕.
- KV 는 결정하지 않음(non-expert 파라미터는 항상 GPU).

## 비용 모델 파라미터 출처 / 예측 검증 / 보고 오차
- 예측기 평가 지표 GoodPred = Accuracy × FetchRate. 시스템 latency 모델 없음.
- 측정 기반 분해: "The achieved bandwidth is 23 GB/s, which matches the achievable bandwidth (23.9 GB/s in our bandwidth test) from host to GPU using PCIe 4.0x8". 처리량 사전 예측·오차 없음.

## expert 와 KV 의 메모리·대역폭 경합 취급
- 없음. 캐시 비율 스윕은 "To simulate GPUs with varying memory capacities" 목적.

## 캐시 크기 결정 방식
- **고정(외부 파라미터)**: cache rate 를 실험 변수로 두고 스윕. 결정 절차 없음.

## 판정
- **D1: 아니오.** 인용 불가.
- **D2: 아니오.** 인용 불가.
- **D3: 부분(경험적 교차점).** 캐시 비율에 따라 CPU 연산(LO) 과 PCIe 스트리밍(캐시 계열) 의 우열이 바뀌는 지점을 실측으로 보고: "in the decode stage with a low cache rate, LO outperforms the other systems. Under low cache rates, the cache-based systems must fetch a significant number of experts through PCIe, while the limited computation makes offloading to the CPU more advantageous. As the cache rate increases, however, the cache-based systems quickly surpass LO." 또 Mixtral 에 대해 "the Mixt model activates a larger ratio of experts (25%) for each token, increasing the cost of fetching parameters to the GPU compared with directly computing them on the CPU." 모델·예측 없음.

## 우리와의 delta
- ProMoE 는 우리가 예측하려는 "전이점" 을 실험으로 한 지점 관찰(캐시 비율 축) 했으나 모델화·사전 예측·KV 항이 없다. 우리 논문에서 D3 관련 선행으로 인용하고, 차이를 "H 축 + 배치 축 격자에서 마이크로벤치 파라미터로 전이 위치를 사전 예측" 으로 서술.
- 예측기가 학습 기반(2M MLP/층) — 우리는 라우팅 트레이스 통계(해석식) 만 사용. 
