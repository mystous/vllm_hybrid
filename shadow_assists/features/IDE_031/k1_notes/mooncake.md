# Mooncake — K1 정독 노트

- **서지**: Ruoyu Qin et al. (Moonshot AI / Tsinghua MADSys). "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving." arXiv 2407.00079 (v1 2024-06; v3). 학회판: FAST'25 "Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot"; 확장판 ACM TOS 2025 (doi 10.1145/3773772).
- **전문 접근 여부**: **불가** (arxiv.org / usenix.org / dl.acm.org 접속 타임아웃). WebSearch 스니펫 (arXiv html v1 요약, FAST'25 pdf 요약, Mooncake GitHub RFC #977, 해설 블로그) 만 확보. 인용은 스니펫 문장, 원문 대조 못 함.

## 문제·결정 변수
- 대상: Kimi 프로덕션. prefill / decode 클러스터 분리 + 클러스터 내 유휴 CPU·DRAM·SSD 를 모아 분산 KVCache 풀 (Mooncake Store) 구성, RDMA 기반 Transfer Engine 으로 이동.
- 결정 변수 (Conductor 글로벌 스케줄러): 요청별 prefill 인스턴스 선택 (prefix 매칭 길이·큐 대기·전송 시간으로 TTFT 추정), decode 인스턴스 선택, hot-spot 방지용 캐시 복제/스왑 (`kvcache_balancing_threshold`), 과부하 시 **prediction-based early rejection** (prefill 후 decode 부하 예측하여 사전 거절).
- 스니펫 인용: "estimates queue delay and prefill compute time, then aggregates these to estimate total TTFT as the sum of cache transfer time, queue wait time, and computation time."
- 스니펫 인용: "The scheduling of prefill nodes considers the KVCache distribution and the available DRAM size."

## 비용 모델 파라미터 출처와 검증
- 스니펫 인용 (RFC #977 / 해설): "Conductor estimates execution time based on request length and prefix_len (which varies by instance), using a polynomial regression model fitted with offline data." → 파라미터는 **오프라인 프로파일 회귀** (마이크로벤치·스펙 아님). 전송 시간은 캐시 위치 (HBM/DRAM/SSD) 별로 계산.
- decode 부하 예측: 출력 길이 미지 → "request-level" 과 "system-level" 두 방식; 스니펫 범위에서 예측 오차 수치 보고는 확인 못 함 (추측: 없음 또는 간접 — 거절률·SLO 만족률로 평가).
- 검증: 시뮬레이션 최대 +525% 처리량, 실배포 A800/H800 에서 +115%/+107% 요청 처리 — 예측 모델의 오차가 아니라 end-to-end 지표.

## expert·KV 경합 취급
- MoE / expert 언급: 스니펫 범위 없음 (TOS 확장판에 DeepSeek 계열 언급 가능성 있으나 미확인 — 추측).
- **호스트 DRAM 대역폭 모델링**: **아니오** (추측). 병목 서술은 RDMA/네트워크 대역폭과 GPU 유휴 자원 활용. "DRAM size" (용량) 는 스케줄 입력이지만 DRAM 읽기 대역폭 항은 스니펫에서 발견 못 함. Layer-wise prefill 로 KVCache 전송을 계산과 겹치는 것은 확인: "Layer-wise prefill enables stream transferring of KVCache to overlap with prefill computation."

## D1/D2/D3 판정
| 조건 | 판정 | 근거 |
|---|---|---|
| D1 | **아니오** | MoE 아님. expert 상주 결정 없음. KV 배치 결정은 노드·티어 선택 (용량 기준) |
| D2 | **아니오** | 오프라인 회귀 (측정 기반) 로 prefill 시간 추정. 마이크로벤치·스펙 파라미터 아님. 오차 보고 미확인 |
| D3 | **아니오** | 인용 불가. 전이점 분석 없음 (추측) |
| 호스트 DRAM BW | 아니오 (추측) | 용량·RDMA 만 |

## 우리와의 delta
- Mooncake 는 클러스터 수준 KV 재사용·스케줄. 단일 노드 내 DDR 대역폭을 expert 스트리밍과 KV 읽기가 나눠 쓰는 문제는 범위 밖. 우리 M2 의 HiCache L3 백엔드로 Mooncake Store 가 쓰일 수 있으나 연구 주장과 직접 경합 없음. 위협도 **하**.
