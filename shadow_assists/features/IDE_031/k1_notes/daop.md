# DAOP — arXiv 2501.10375 (v2, DATE 2025)

## 서지 / 전문 접근
- Yujie Zhang, Shivam Aggarwal, Tulika Mitra (NUS). "DAOP: Data-Aware Offloading and Predictive Pre-Calculation for Efficient MoE Inference". DATE 2025 (Lyon). v1 2024-12-16, v2 2025-05-04. 코드 github.com/ecolab-nus/DAOP.
- 전문 접근: **예** — arXiv ID 는 WebSearch 로 확인, 본문은 HF 미러 `scholarweave/arxiv-latex` 의 LaTeX 소스(v2) 전문. 그림(tikz) 수치는 본문 서술만.

## 결정 변수
- 시퀀스별 GPU/CPU expert 배치: prefill 중 게이트 결과로 층마다 CPU 의 활성 상위 expert 와 GPU 의 활성 하위 expert 를 짝지어 교환(문턱 SwapInOut=1.05, 층당 최대 50%). decode 중에는 이동 없음.
- 한 층 앞 예측(다음 블록 게이트를 현재 hidden state 에 적용, i≥4, 정확도 84.11%) 으로 CPU expert 를 미리 계산; "Graceful Degradation" — 예측된 두 expert 가 모두 CPU 면 점수 낮은 쪽을 GPU 상주 next-best expert 로 대체(정확도 영향 있음, GSM8K 58.9→33.5 @ECR 25%).
- ECR(Expert Cache Ratio) 는 외부 파라미터(25~62.5%, 기본 46.9%). 배치 1 고정("setting the batch size to one").

## 비용 모델 파라미터 출처 / 예측 검증 / 보고 오차
- 비용 모델 없음. 동기 측정만: Table 1 (A100, Xeon 6326) CPU 블록 8.02ms / GPU 1.24ms / expert 이동 39.87ms / activation 전송 0.02ms — "migrating a single expert ... is approximately 32× slower than executing the entire block on the GPU".
- 적용 가정 3가지를 명시: "3) CPU-GPU transfer latency exceeds the time required for expert execution on the CPU." 처리량 사전 예측·오차 없음(A6000 48GB + i9-10980XE, HF transformers).

## expert 와 KV 의 메모리·대역폭 경합 취급
- 없음. KV 언급 없음(배치 1).

## 캐시 크기 결정 방식
- **고정(ECR) + 캘리브레이션 초기 내용**: "We standardize the expert cache size across all layers, ensuring it contains experts with the highest activation probabilities" — 초기 상주 집합은 ShareGPT 캘리브레이션의 층별 활성 확률, 크기는 ECR 로 외부 지정("All methods aim to maximize GPU memory utilization, with an ECR of 46.9%").

## 판정
- **D1: 아니오.** 인용 불가(KV 미취급).
- **D2: 아니오.** 인용 불가.
- **D3: 아니오.** Table 1 은 단일 지점 비교.

## 우리와의 delta
- compute-offloading 계열(Fiddler 개선) 이지만 배치 1, 비용 모델 없음, 정확도 손실 있는 근사. 우리 주장과 겹침 없음. "시퀀스별 상주 집합 교환" 은 우리 hotmap 온라인 갱신(선택) 의 참고. DAOP 의 ECR 스윕 결과(ECR 25% 에서도 Fiddler 대비 +35%) 는 "상주 비율이 낮을수록 CPU 연산 경로가 유리" 라는 정성 근거.
