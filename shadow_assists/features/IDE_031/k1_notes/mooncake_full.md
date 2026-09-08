# Mooncake — K1 전문 정독 노트

- **서지**: Ruoyu Qin, Zheming Li, Weiran He, Mingxing Zhang, Yongwei Wu, Weimin Zheng, Xinran Xu (Moonshot AI / Tsinghua). "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving". arXiv 2407.00079v4 (2025-09-03), FAST'25. 23쪽.
- **전문 확보 경로**: HF `Chelsea707/arxiv-cs-2020-2025-pdfs` → `data/2024/2407_00xxx/2407.00079.pdf` → pypdf. **전문 확보** (v4).

## 무대·토폴로지
- 무대 = **온라인 서빙 (Kimi), TTFT/TBT SLO 아래 클러스터 처리량**. dense 모델 (LLaMA2-70B 구조의 더미 모델로 재현). 원문: "all the experimental results reported in this paper are based on replayed traces of real workloads, but using a dummy model that follows the same architecture as LLaMA2-70B." MoE·expert 항 **없음** (grep "expert/MoE" 0건).
- 토폴로지: prefill/decode 분리 + CPU·DRAM·SSD·RDMA 로 분산 KVCache 풀. "groups the CPU, DRAM, SSD, and RDMA resources of the GPU cluster to implement a disaggregated KVCache." KV 블록은 CPU 메모리에 페이지 단위, GPUDirect RDMA Messenger 로 이동.

## 결정 변수
- 요청을 어느 prefill 인스턴스에 보낼지 (prefix hit 길이 + 대기 시간으로 TTFT 추정), hot KV 블록 복제, 과부하 시 조기 거절. GPU 메모리 내 KV vs 가중치 분할 결정 **없음**.

## 비용 모델 파라미터 출처·검증·오차
- prefill 시간 예측: **오프라인 테스트 데이터로 만든 예측 모델** (end-to-end 회귀). 원문: "to predict the computation time of the prefill stage for a request, we employ a predictive model derived from offline test data. This model estimates the prefill duration based on the request's length and prefix cache hit length. Thanks to the regular computation pattern of Transformers, the error bound of this prediction is small as long as enough offline data is available." → 오차 수치 **미보고**. 사전등록 없음.
- 전송 시간 예측은 어렵다고 자인: "More difficulty lies in predicting the transfer time because it is determined not only by the size of the transferred data but also by the current network status".
- 디코드 부하 예측은 시스템 수준 근사: "we assume that each request's decoding stage takes a uniform time t_d."

## expert·KV 경합 취급
- expert 없음. DRAM 은 KV 풀 용량으로만 등장 ("the available DRAM size"), DRAM **대역폭** 이 다른 스트림과 공유된다는 항 없음. 전송 대역폭은 RDMA/네트워크.

## D1·D2·D3 판정
| 조건 | 판정 | 근거 |
|---|---|---|
| D1 | **아니오** | MoE 아님, expert 상주 결정 없음. 인용 불가 |
| D2 | **아니오** | 오프라인 회귀 예측 모델은 있으나 (위 인용) 오차 미보고, 마이크로벤치·스펙 파라미터 아님, 하이브리드 MoE 아님 |
| D3 | **아니오** | prefill compute-bound / decode memory-bound 의 일반 서술 (§2) 뿐, 전이점 분석 없음 |
| DDR 공유 항 | **아니오** | 인용 불가 |

## 우리와의 delta
- 온라인 서빙·TBT 무대는 같으나 클러스터 스케줄링 계층. 우리 노드 내부 (GPU 메모리 분할, DDR 예산) 결정과 직교. "오프라인 데이터 회귀 → 오차 미보고" 의 반례로 인용 가능 (우리 D2 의 사전등록·마이크로벤치 파라미터 대비). 위협도 **하**.
