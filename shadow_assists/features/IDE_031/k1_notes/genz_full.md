# GenZ — K1 전문 정독 노트

- **서지**: Abhimanyu Bambhaniya, Ritik Raj, Geonhwa Jeong, Souvik Kundu, Sudarshan Srinivasan, Midhilesh Elavazhagan, Madhu Kumar, Tushar Krishna (Georgia Tech / Intel). "Demystifying AI Platform Design for Distributed Inference of Next-Generation LLM models" (개정판 제목). arXiv 2406.01698 (HF 데이터셋 사본, 19쪽, TABLE I–VIII 포함 개정판).
- **전문 확보 경로**: HF `Chelsea707/arxiv-cs-2020-2025-pdfs` → `data/2024/2406_01xxx/2406.01698.pdf` (2.4 MB) → pypdf. **전문 확보**.

## 무대·토폴로지
- 무대 = 플랫폼 **설계 탐색 도구** (해석 모델). TTFT/TPOT 를 use-case 별 (QA, Chat, RAG, 요약, 코드) 로 계산. 온라인 서빙 지표(TPOT) 를 쓰지만 실제 서빙 시스템은 아님.
- 토폴로지: NPU (가속기) + fast/slow 2계층 메모리. 원문: "Each NPU provides access to two external memories (fast and slow). Faster (smaller) memory represents an HBM/DDR Memory Bank providing high BW (BW_mem), while Slower (Larger) memory could be PCIe-accessible CPU or CXL-accessible SSD/Flash (BW_omem) for offload." → 오프로드 계층은 **대역폭 항으로 존재**. CPU 연산 오프로드는 Case Study IV 에서만: "Four 128×128 cores with CPU offloading for MHA (Logit + Softmax + Attend) and KV cache storage. CPU has 8 TOPS and GPU to CPU link of 128 GB/s using PCIe." (dense LLaMA-3-8B prefill, 가상 NPU).
- MoE: Mixtral-8x7B/8x22B, GPT-4 1.8T MoE 가정, **EP 로 가속기 상주** 전제. "TPOT of Mixtral 8x22B on 4 H100 with expert parallelism can vary between 3.23 ms (All tokens distributed equally) and 11.33 ms (All tokens going to a single expert) for batch 32." CPU 가 expert 를 계산하는 경로 **없음**.

## 결정 변수
- 사용자가 넣는 플랫폼 파라미터 (FLOPS, 메모리 용량·BW, 링크) 와 병렬화 (TP/PP/EP/SP). 시스템 정책 결정 아님. GPU 메모리 내 expert↔KV 분할 결정 없음.

## 비용 모델 파라미터 출처·검증·오차
- roofline + 효율 계수. 원문: "We follow a roofline-based approach combined with separate efficiency factors (extracted from real hardware or open source simulators) for computation FLOPs and memory BW to calculate each operator's runtime on the accelerator."
- 효율 계수 출처 = **실 하드웨어 커널 프로파일** (스펙-only 아님): "Our measured efficiency factors are derived from profiling real NPUs, following a methodology similar to Vidur [15]. We execute the same kernel multiple times and measure average utilization to obtain realistic efficiency estimates." 수치: "V100: 0.45, A100: 0.4, 1×H100: 0.55, 2×H100: 0.64, 4×H100: 0.66, and 8×H100: 0.75."
- 검증·오차: dense 모델, GPU-only. "The geomean error in prefill and decode predictions between real and GenZ-predicted values is 2.73% and 1.85%" / chunked "geomean error of 1.43%" / 타 아키텍처 "geomean error of 5.82%". MoE·오프로드 경로의 오차 보고 **없음**. 사전등록 없음 (효율 계수를 검증 대상 시스템에서 측정 → 사후 보정 성격).

## expert·KV 경합 취급 (호스트 DRAM 대역폭 공유 항)
- 슬로우 메모리 BW_omem 은 단일 링크 대역폭. KV 읽기와 expert 스트리밍이 같은 host DRAM 을 공유한다는 합산 항 **없음** (Case Study IV 도 KV 저장+MHA 만 CPU). 플랫폼 BW 요구는 `BW_Req ∝ O((ActiveModel + KVcache)/TPOT)` 의 가속기 메모리 항.

## D1·D2·D3 판정
| 조건 | 판정 | 근거 |
|---|---|---|
| D1 | **아니오** | expert 는 EP 로 가속기 상주 가정. 분할 결정 없음. 인용 불가 |
| D2 | **부분** | 해석 모델 + 실측 효율 계수로 TPOT 예측, 오차 1.4–5.8% 보고 — 그러나 **dense·GPU-only 검증**. 하이브리드 MoE 처리량 예측 오차 없음 |
| D3 | **부분** | 연산자별 compute/memory-bound 분류 ("more layers are compute-bound than memory-bound" 등). DDR↔GPU expert 연산 전이점 없음 |
| DDR 공유 항 | **아니오** | BW_omem 단일 링크 항만 |

## 우리와의 delta
- "해석 모델 + 프로파일 효율 계수 → TPOT 예측 + 오차 보고" 형식은 선례 (D2 형식 위협 **중**). 우리 delta: (1) 하이브리드 (CPU AMX expert + host KV) 격자, (2) 효율 계수를 서빙 무관 마이크로벤치에서 사전 고정 (GenZ 는 대상 플랫폼 커널 프로파일), (3) H 축 전이점, (4) DDR 공통 예산. GenZ 의 slow-memory 항 (BW_omem) 이 우리 모델의 특수 경우임을 인정하고 인용해야 함.
