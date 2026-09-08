# LLMCompass (ISCA'24) — K1 정독 노트

- **서지**: Hengrui Zhang, August Ning, Rohan Prabhakar, David Wentzlaff (Princeton). "LLMCompass: Enabling Efficient Hardware Design for Large Language Model Inference." ISCA 2024 (doi 10.1109/ISCA59077.2024.00082). arXiv 2312.03134 (제목 "A Hardware Evaluation Framework for Large Language Model Inference", 2023-12). 코드 github.com/PrincetonUniversity/LLMCompass.
- **전문 접근 여부**: **불가** (arxiv.org, dl.acm.org, princeton.edu 미러 모두 타임아웃). WebSearch 스니펫 (arXiv abs, ACM DL, Semantic Scholar, 강의용 PDF 요약) 만. 인용은 스니펫 문장.

## 목적·결정 변수
- 하드웨어 설계 평가 프레임워크: 입력 = LLM 연산 그래프 + **하드웨어 기술 (description)** (코어 수, 벡터/행렬 유닛, 온칩 버퍼, 메모리 대역폭·용량, 링크). 내장 **mapper** 가 연산자별 최적 mapping/scheduling 을 탐색하고 (4-A100 노드 GPT-3 175B 시뮬레이션에 26,400 회 탐색, 16 분), 면적 기반 비용 모델 동반.
- 결정 변수는 설계자가 고르는 하드웨어 파라미터 (예: HBM → DRAM 교체, 연산 유닛 축소). 서빙 정책 결정 아님.
- 스니펫 인용: "two inputs are needed: the computational graph of the LLM and a hardware description."

## 비용 모델 파라미터 출처와 검증
- 파라미터: 스펙 수준 하드웨어 기술 (마이크로벤치 아님). 검증 대상 실기: NVIDIA A100, AMD MI210, Google TPUv3.
- 스니펫 인용: "LLMCompass' estimated latency achieves an average 10.4% error rate across various operators with various input sizes and an average 4.1% error rate for LLM inference." (ISCA 판은 10.9% 로 표기된 스니펫도 있음.) 연산자별: Matmul 9.0%, Softmax 12.0%, LayerNorm 13.8%, GELU 5.0%, all-reduce 14.9%; 추론 prefill 0.69%, decode 7.5%.
- 설계 결론: 스니펫 인용 "by reducing the compute capability or replacing High Bandwidth Memory (HBM) with traditional DRAM, new designs can achieve as much as 3.41x improvement in performance/cost compared to an NVIDIA A100." / "a larger batch size can compensate for the loss in memory bandwidth".

## expert·KV 경합 취급
- 평가 모델 GPT-3 175B (dense). MoE 언급 스니펫 없음 (추측: 미지원 또는 미평가). CPU offload / 호스트 메모리 tier 없음 — "HBM 대신 DRAM" 은 가속기 주메모리를 DRAM 으로 바꾸는 설계 탐색이지 호스트 offload 가 아님.
- **호스트 DRAM 대역폭 모델링**: 아니오 (가속기 부착 메모리로서의 DRAM 대역폭은 모델링).

## D1/D2/D3 판정
| 조건 | 판정 | 근거 |
|---|---|---|
| D1 | **아니오** | dense, 단일 가속기 메모리. expert·KV 배치 결정 없음 |
| D2 | **부분 (아니오)** | 스펙 파라미터 → 지연 예측 + 오차 4.1%/10.4% 보고는 선례. 하이브리드 MoE 아님 |
| D3 | **부분** | "LLM inference is mostly IO-bound ... HBM memory capacity limits the batch size" 등 compute/IO bound 논의와 배치에 따른 보상 분석 (스니펫). CPU↔GPU 하이브리드 전이점 아님 |
| 호스트 DRAM BW | 아니오 | — |

## 우리와의 delta
- GenZ 와 같은 부류: "스펙 → 예측 + 오차 보고" 형식의 선례. 차이는 (1) 하드웨어 설계 탐색 vs 우리의 서빙 배치 정책 (2) mapper 로 연산자 최적화를 모델 안에서 찾는 반면 우리는 실제 커널 (kt AMX/AVX-512, fused_moe) 을 마이크로벤치로 재는 점 (3) 하이브리드 DDR 예산·전이점. 위협도 **중** (D2 형식). 논문에서는 GenZ 와 묶어 "GPU-only 해석 모델" 로 한 문단 처리 가능.
