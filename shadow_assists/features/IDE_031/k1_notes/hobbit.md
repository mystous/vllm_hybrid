# HOBBIT — arXiv 2411.01433 (v2, 2024-11-06)

## 서지 / 전문 접근
- Peng Tang, Jiacheng Liu, Xiaofeng Hou, Yifei Pu, Jing Wang, Pheng-Ann Heng, Chao Li, Minyi Guo (SJTU / CUHK). 학회 미표기(ACM 양식 placeholder). Llama.cpp 위 8,000 LoC.
- 전문 접근: **예** — HF 미러 PDF(15쪽) 텍스트 추출.

## 결정 변수
- 캐시 미스 expert 를 어느 정밀도로 적재할지(토큰별, 게이트 점수 기반 문턱 T1=0.6, T2=0.9 → 67% 고정밀 / 30% 저정밀 / 3% 스킵). 정확도를 바꾸는 기법.
- prefetch 깊이 p(권장 1~3), 캐시 교체 우선순위 = w_lru·LRU + w_lfu·LFU + w_lhu·LHU + w_fld·FLD (가중치는 사용자 하이퍼파라미터).
- 배치 1 고정. 캐시 용량·KV 는 결정하지 않음.

## 비용 모델 파라미터 출처 / 예측 검증 / 보고 오차
- 캐시 미스 페널티 모델만: 고정밀 미스 비용 C, 저정밀 미스 비용 (B_l/B_h)·C. 가중치는 "we determine suitable values by minimizing the mixed precision expert cache miss penalties on a calibration dataset" — 즉 캘리브레이션 데이터셋 튜닝.
- 동기 측정: "expert loading dominates the total inference time, consuming approximately 85.5% on the RTX 4090 and 94.5% on the Jetson Orin". 처리량 사전 예측·오차 보고 없음.

## expert 와 KV 의 메모리·대역폭 경합 취급
- 없음. GPU 메모리 구성만 서술: "GPU memory stores all non-expert weights, a subset of 'hot experts' (expert cache), and internal activations, while other experts are offloaded to CPU memory or SSD".

## 캐시 크기 결정 방식
- **고정**(장치 메모리로 정해짐). 고·저정밀 캐시를 분리: "the high-precision cache typically being larger than the low-precision cache" — 크기 결정 절차 없음.

## 판정
- **D1: 아니오.** 인용 불가.
- **D2: 아니오.** 인용 불가.
- **D3: 아니오.** "expert loading dominates" 는 한 지점 측정이지 전이 분석이 아님.

## 우리와의 delta
- 정밀도 혼합·캐시 정책 논문. 우리 주장과 직접 겹침 없음. CPU 협력 모드(§5.4) 에서 Fiddler 대비 0.99~1.46× 라는 수치는 "CPU expert 연산 vs 전송" 비교 참고치. 우리 모델의 정확도 제약(분포 유사성) 관점에서 HOBBIT 류 근사 기법은 비교군 제외 근거.
