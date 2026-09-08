# GenZ — K1 정독 노트

- **서지**: Abhimanyu Bambhaniya, Ritik Raj, Geonhwa Jeong, Souvik Kundu, Sudarshan Srinivasan, Midhilesh Elavazhagan, Madhu Kumar, Tushar Krishna (Georgia Tech / Intel). arXiv 2406.01698. v1 제목 "Demystifying Platform Requirements for Diverse LLM Inference Use Cases" (2024-06), 개정판 제목 "Demystifying AI Platform Design for Distributed Inference of Next-Generation LLM models". 코드 github.com/abhibambhaniya/GenZ-LLM-Analyzer, Streamlit 데모.
- **전문 접근 여부**: **불가** (arxiv.org 타임아웃). WebSearch 스니펫 (arXiv abs, ADS, HF papers) 만. 인용은 스니펫 문장. 로컬 `eval/results/20260830_pln006_e1lit/DELTA.md` 에 이전 세션의 한 줄 판정 ("예측 모델이지 하한 아님, 사전등록 규율 없음") 있음.

## 목적·결정 변수
- 해석적 (analytical) 모델: 모델 구조 (Dense / GQA / MoE / Mamba) × 서빙 최적화 (chunking, speculative decoding, quantization) × 플랫폼 파라미터 (연산 성능, 메모리 용량, 메모리 대역폭, 네트워크 지연·대역폭) → prefill/decode 지연·처리량. 결정 변수는 사용자가 넣는 플랫폼 설계 값과 병렬화 (TP/PP/EP) — 시스템이 아니라 "설계 탐색 도구".
- 스니펫 인용: "GenZ ... an analytical tool to efficiently navigate the relationship between diverse LLM model architectures (Dense, GQA, MoE, Mamba), LLM serving optimizations (Chunking, Speculative decoding, quantization), and AI platform design parameters."
- MoE 취급: Mixtral-8x7B / 8x22B, GPT-4 (1.8T MoE 가정) 를 expert parallelism 으로 모델링. 스니펫 인용: "the TPOT of Mixtral 8x22B on 4 H100 with expert parallelism can vary between 3.23 ms and 11.33 ms for batch 32." — expert 는 전부 가속기 메모리에 상주하는 가정 (추측; CPU offload 경로 언급 없음).

## 비용 모델 파라미터 출처와 검증
- 파라미터: 하드웨어 **스펙** (FLOPS, HBM 용량·대역폭, 링크) + 효율 계수 (추측: 실측으로 보정한 efficiency factor 존재 가능성, 스니펫 미확인).
- 검증: 스니펫 인용 "validated against real hardware platforms running various different LLM models, achieving a max geomean error of 5.82%." — GPU 단독 (dense 포함) 실측 대비. 하이브리드 CPU-GPU 경로의 오차 보고는 없음.

## expert·KV 경합 취급
- KV 는 가속기 메모리 안에서 용량 제약으로만 취급 (배치 상한). 호스트 메모리 tier, CPU 연산, expert 스트리밍 **없음** (스니펫 범위, 추측 포함). "memory capacity / memory bandwidth requirements" 는 가속기 것.
- **호스트 DRAM 대역폭 모델링**: 아니오 (추측).

## D1/D2/D3 판정
| 조건 | 판정 | 근거 |
|---|---|---|
| D1 | **아니오** | 하이브리드 아님. expert 는 상주 가정, KV 배치 결정 없음 |
| D2 | **부분 (아니오)** | 스펙-only 파라미터로 처리량 예측 + 오차 5.82% 보고 → "사전 예측 + 오차 보고" 형식은 선례. 그러나 **GPU-only** 이며 CPU-GPU 하이브리드 MoE 처리량은 예측 대상 아님 |
| D3 | **부분** | roofline 식 compute/memory-bound 구분은 도구 성격상 내장 (추측). DDR↔GPU expert 연산 전이점 같은 하이브리드 전이 분석은 없음 |
| 호스트 DRAM BW | 아니오 (추측) | 가속기 메모리만 |

## 우리와의 delta
- GenZ 는 "스펙으로 GPU 서빙 예측 + 오차 보고" 의 선례 → 우리 주장 2 의 형식적 신규성은 약하다. 남는 delta: (1) CPU 측 expert 연산·DDR 스트리밍·KV 호스트 읽기가 있는 **하이브리드** 격자 (2) 파라미터를 스펙이 아니라 **서빙 무관 마이크로벤치** 에서 얻고 사전등록 (3) 전이점 위치 예측. `IDE_029/PLN_006` 에서 스펙-only 모델이 56% 오차로 기각된 사실이 GenZ 류의 한계 증거로 쓸 수 있음. 위협도 **중** (D2 형식).
