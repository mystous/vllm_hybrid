# Vidur — K1 전문 정독 노트

- **서지**: Amey Agrawal, Nitin Kedia, Jayashree Mohan, Ashish Panwar, Nipun Kwatra, Bhargav S. Gulavani, Ramachandran Ramjee, Alexey Tumanov (Georgia Tech / Microsoft Research India). "Vidur: A Large-Scale Simulation Framework for LLM Inference". arXiv 2405.05465, MLSys'24. 16쪽.
- **전문 확보 경로**: HF `Chelsea707/arxiv-cs-2020-2025-pdfs` → `data/2024/2405_05xxx/2405.05465.pdf` → pypdf. **전문 확보**.

## 무대·토폴로지
- 무대 = 온라인·오프라인 서빙 **시뮬레이터** (이벤트 구동, 클러스터/replica 스케줄러 포함). 지표: normalized latency, TTFT, TBT, MFU, KV 이용률.
- 토폴로지: **GPU-only, dense** (InternLM-20B, LLaMA2-70B, Qwen-72B, LLaMA2-7B; TP/PP). MoE·CPU 오프로드·host 메모리 항 **없음** (grep "MoE/offload/CPU memory/DRAM" 0건; "mixture" 는 일반어).

## 결정 변수
- 사용자 구성 (병렬화, 배치 정책, 스케줄러, replica 수) 을 Vidur-Search 가 탐색. GPU 메모리 내 분할 결정 없음.

## 비용 모델 파라미터 출처·검증·오차
- 연산자 단위 프로파일 + ML 보간. 원문: "Vidur models the performance of LLM operators using a combination of experimental profiling and predictive modeling". "we collect a limited set of data points and rely on small machine-learning models to interpolate the runtimes ... we find that random forest (RF) regression models achieve the right balance between data frugality and fidelity."
- 파라미터 출처 = **대상 GPU 에서의 마이크로 프로파일 (단일 GPU, 서빙 무관)**: "Vidur can simulate various parallelization schemes with minimal profiling performed on a single GPU." 디코드 attention 은 KV 총량으로 모델링: "it is sufficient to model the total amount of KV-Cache to be fetched in a batch of requests to determine the kernel runtime".
- 검증·오차: "it estimates inference latency with less than 9% error across the range" / 정적 "tail latency (P95) with upto 3.33% error" / 동적 "(< 5% error) in almost all scenarios with request rate set to 85% of the system capacity" / 부록 "up to 12.65% maximum error". 사전등록 명시 없음 (다만 프로파일→시뮬→실측 비교 구조라 사후 보정은 아님).

## expert·KV 경합 취급
- expert 없음, host DRAM 없음. KV 는 GPU 메모리 용량·attention 커널 입력으로만. DDR 공유 항 **없음**.

## D1·D2·D3 판정
| 조건 | 판정 | 근거 |
|---|---|---|
| D1 | **아니오** | 인용 불가 (MoE·분할 없음) |
| D2 | **부분** | "마이크로벤치(연산자 프로파일) → 스텝/요청 지연 사전 예측 + 오차 (<9%)" 형식 선례. dense GPU-only, 하이브리드 MoE 없음 |
| D3 | **아니오** | 전이점 분석 없음. capacity point (과부하 임계) 는 큐잉 현상이지 자원 전이 아님 |
| DDR 공유 항 | **아니오** | 인용 불가 |

## 우리와의 delta
- 방법론상 우리 주장 ② 와 가장 유사 (연산자 마이크로벤치 → RF → 격자 예측 → 오차 보고). 우리 delta 는 (1) CPU AMX expert·DDR 스트리밍·host KV 항을 포함한 하이브리드 격자, (2) 해석식 (RF 아님) + 사전등록 ≤20%, (3) H 축 전이점. Vidur 의 "<9%" 가 dense GPU-only 에서의 기준선이므로 우리 ≤20% 는 정당화 문장 (하이브리드 경합 항 때문에 느슨) 이 필요. 위협도 **중**.
