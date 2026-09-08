# Pre-gated MoE — arXiv 2308.12066 (v3, ISCA 2024)

## 서지 / 전문 접근
- Ranggi Hwang, Jianyu Wei, Shijie Cao, Changho Hwang, Xiaohu Tang, Ting Cao, Mao Yang (KAIST / USTC / MSR). ISCA-51, 2024. v3 2024-04-27.
- 전문 접근: **예** — HF 미러 PDF(14쪽) 텍스트 추출.

## 결정 변수
- 모델 구조 변경(pre-gate: N번째 블록의 게이트가 N+1 블록의 expert 를 선택) 으로 prefetch 대상을 확정. 시스템은 "다음 블록의 활성 expert 만" CPU→GPU 이동.
- GPU 상주: dense 파라미터 + 현재·다음 블록 활성 expert(식 (1) Peak GPU mem). 캐시는 기본 없음; §VI-D 에서 1/10/20% 고정 비율 캐시 + LIFO/LFU/LRU 실험.
- 배치 1 고정("we primarily focus on single batch inference scenarios").

## 비용 모델 파라미터 출처 / 예측 검증 / 보고 오차
- 비용 모델 없음. Fig. 9 타임라인은 정성적 설명: "the compute-bound expert execution stage (green) to concurrently execute with the communication-bound expert selection stage (blue)".
- 예측·오차 보고 없음. A100 80GB + EPYC 7V12, PCIe gen4 32GB/s, FasterTransformer 구현, SwitchTransformer 로 실측.

## expert 와 KV 의 메모리·대역폭 경합 취급
- KV cache 언급 없음(encoder-decoder Switch, 배치 1). Peak GPU 메모리 식은 파라미터만: "Peak GPU mem = max(Non_MoE_M + Σ_{L=N}^{N+1} Act_Exp_L)".

## 캐시 크기 결정 방식
- **고정 비율 실험**: "we change the fraction of experts that are cached inside the GPU memory (from 1% to 20%)". 결정 절차 없음.

## 판정
- **D1: 아니오.** KV 미취급. 인용 불가.
- **D2: 아니오.** 인용 불가.
- **D3: 아니오(정성만).** 활성 expert 수 스윕(Fig. 14, 1→64 experts) 에서 "all CPU offloading based solutions ... experience a higher performance degradation vs. GPU-only as the number of activated experts is increased" — 경험적 추세 서술, 전이점 모델 없음.

## 우리와의 delta
- 정확도를 바꾸는 구조 변경 + 배치 1 latency 논문. 우리 주장(공통 예산 분할, 사전 예측, 전이점) 과 겹치는 부분 없음. Fig. 14 는 "활성 expert 가 늘면 offload 이득이 사라진다" 는 우리 주장 1 의 배경 인용으로만 유용.
