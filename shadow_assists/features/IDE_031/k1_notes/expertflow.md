# ExpertFlow — arXiv 2410.17954 (v1, 2024-10-23)

## 서지 / 전문 접근
- Xin He, Shunkang Zhang, Yuxin Wang, Haiyan Yin, Zihao Zeng, Shaohuai Shi, Zhenheng Tang, Xiaowen Chu, Ivor Tsang, Ong Yew Soon (A*STAR CFAR / HKUST 외). 학회 미표기.
- 전문 접근: **예** — HF 미러 PDF(14쪽) 텍스트 추출.

## 결정 변수
- 라우팅 경로 예측기(T5 encoder-decoder 기반, 7.21MB) 로 모든 층의 expert 활성 여부를 한 번에 분류 → PLEC(예측 기반 층간 캐시 슬롯 배분) + 실시간 교정. 토큰 스케줄러(K-means 로 비슷한 라우팅 경로끼리 재배치) 로 배치당 활성 expert 수 최소화. Dual-batch 파이프라인.
- 캐시 크기 CS 와 배치 BS 는 실험 설정 (CS,BS)=(16,32),(8,16),(4,8) 등 고정.
- 하드웨어 A40 48GB + Xeon 6338; 모델 Switch-32/64/128, Mixtral-8×7B.

## 비용 모델 파라미터 출처 / 예측 검증 / 보고 오차
- 비용 모델 없음. roofline 은 정성 인용: "According to the roofline model, the computational cost of processing a single token is nearly equivalent to that of multiple tokens." 
- 라우팅 예측 정확도(배치 수준 73~87%) 만 보고. 처리량 사전 예측·오차 없음.

## expert 와 KV 의 메모리·대역폭 경합 취급
- KV 는 정합성 문제로만: 재배치 후 "Merge consolidates the KV cache from different batches ... Reindex dynamically adjusts the indices of KV cache entries". 메모리 비용 집계에 KV 포함("includes model parameters, input/output data, and intermediate activations such as key-value (KV) caches") 되나 대역폭 예산 아님.
- 캐시·배치 상충을 관찰하고 향후 과제로 넘김: "an inverse relationship between cache size and batch size on hit ratio ... This suggests the potential for an adaptive system that adjusts these parameters in real-time".

## 캐시 크기 결정 방식
- **고정(CS 설정)**; 층간 분배는 예측 기반 동적(PLEC): "PLEC can use the predicted routing path to prefetch all 8 experts from the first two layers, fully utilizing the cache".

## 판정
- **D1: 아니오.** KV 는 정합성 처리만. 인용: 위 Merge/Reindex 문장.
- **D2: 아니오.** 인용 불가.
- **D3: 아니오.** 인용 불가.

## 우리와의 delta
- 우리와 겹치는 결정 변수 없음. "캐시 크기 ↔ 배치 크기" 상충을 적응적으로 풀자는 향후 과제 문장은 우리 M2 정책의 동기 인용으로 사용 가능. 토큰 재배치(라우팅 유사 토큰 묶기) 는 우리 범위 밖.
