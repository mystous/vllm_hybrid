# LLMCompass — K1 전문 정독 노트

- **서지**: Hengrui Zhang, August Ning, Rohan Prabhakar, David Wentzlaff (Princeton). "A Hardware Evaluation Framework for Large Language Model Inference". arXiv **2312.03134** (ISCA'24 발표본 "LLMCompass"). 14쪽.
- **전문 확보 경로**: HF `Chelsea707/arxiv-cs-2020-2025-pdfs` → `data/2023/2312_03xxx/2312.03134.pdf` → pypdf. **전문 확보** (arXiv 판; ISCA 카메라레디와 수치 차이 가능).

## 무대·토폴로지
- 무대 = **하드웨어 설계 평가 프레임워크** (아키텍처 시뮬레이터 + 매퍼 + 면적·비용 모델). 서빙 시스템 아님. dense GPT-3 175B, TP/PP.
- 토폴로지: 단일 가속기 계층 (코어 로컬 버퍼 → 글로벌 버퍼 → 주메모리 HBM/DDR). CPU 오프로드·host 메모리 계층 **없음**. MoE 는 배경 문단에서 이름만 언급 ("Mixture-of-Experts [16]"), 모델링 대상 아님.

## 결정 변수
- 하드웨어 설계 파라미터 (코어 수, 시스톨릭 어레이, 버퍼, 메모리 종류·대역폭) 와 연산자 매핑·스케줄 (매퍼가 탐색). 서빙 정책·메모리 분할 결정 없음. 처리량 지향 설계 제안: "we use 512GB of DRAM powered by 256 PCIe 5.0 channels with an aggregated memory bandwidth of 1TB/s".

## 비용 모델 파라미터 출처·검증·오차
- 파라미터 출처 = **하드웨어 기술 템플릿 (스펙)** + 커널 런치 오버헤드 실측: "The kernel launch overhead including the framework overhead is measured by running the operator with an input of size 1." 연산자는 타일 단위 시뮬레이션.
- 검증·오차: "for Matmul, Softmax, LayerNorm, GELU, and all-reduce, LLMCompass achieves an average error rate of 9.0%, 12.0%, 11.3%, 5.0%, and 14.9% respectively. For LLM inference, LLMCompass achieves an average error rate of 0.69% and 7.5% for prefill and decoding respectively. On average, LLMCompass achieves a 10.4% error rate for different operators at various input sizes and a 4.1% error rate across the prefill and decoding stages." 플랫폼: 4×A100, 8×TPUv3, MI210. 사전등록 없음. 하이브리드·MoE 오차 없음.

## expert·KV 경합 취급
- expert 없음. KV 는 가속기 주메모리 읽기 항: "batching only reduces model parameter accesses but not KV cache reads. With a much larger batch, KV cache accesses become the new bottleneck" — 가속기 메모리 안의 파라미터↔KV 읽기 경합 (우리 host DDR 공유 항의 GPU 판). host DRAM 공유 항 **없음**.

## D1·D2·D3 판정
| 조건 | 판정 | 근거 |
|---|---|---|
| D1 | **아니오** | 인용 불가 |
| D2 | **부분** | 스펙 템플릿 기반 사전 예측 + 오차 (4.1% / 10.4%) 보고 — dense GPU/TPU. 하이브리드 MoE 아님 |
| D3 | **부분** | "prefill is compute-bound", "As latency is mostly IO-bound by reading parameters and KV cache" — 배치 크기에 따른 파라미터 읽기 → KV 읽기 병목 전이 서술. hot expert 수 축 아님 |
| DDR 공유 항 | **아니오** (가속기 메모리 내 파라미터+KV 합산은 있음) | 위 인용 |

## 우리와의 delta
- "스펙만으로 예측 + 오차 보고" 형식의 강한 선례 (D2 형식). 우리 delta: 하이브리드 (CPU AMX + host DDR) 항, 온라인 TPOT 격자, H 축 전이점, 사전등록. LLMCompass 의 "batching 이 파라미터 읽기는 줄이나 KV 읽기는 못 줄인다" 관찰은 우리 공통 예산 논리 (cold expert 스트리밍 vs KV 읽기) 의 GPU 판 선례로 인용 가치. 위협도 **중**.
