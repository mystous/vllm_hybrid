# CoX-MoE — K1 전문 정독 노트

- **서지**: Muyoung Son, Yi Chen, Seungjae Yoo, Soongyu Choi, Joo-Young Kim (KAIST). "CoX-MoE: Coalesced Expert Execution for High-Throughput MoE Inference with AMX-Enabled CPU-GPU Co-Execution". arXiv 2605.17889 v1 2026-05-18 / v2 2026-05-19. comments: "7 pages, 8 figures, accepted to DAC '26".
- **전문 확보 경로**: HF `scholarweave/arxiv-latex` → `arxiv_part_0043.parquet` (2604.14703–2605.28601) 을 HTTP range 로 row-group 부분 읽기 → LaTeX 원문 (Script/0–8, Table/4_1·6_1·6_3, 193 KB). **전문 확보** (그림 PDF 는 없음, 캡션·본문만).

## 무대·토폴로지
- 무대 = **처리량 지향 오프라인 배치** (B=1024, L_in 97/800, L_out 32/256). 원문: "throughput-oriented workloads, such as offline benchmarking, large-scale data processing, or synthetic data generation ... motivates MoE inference systems that focus on single-GPU, throughput-intensive workloads."
- 토폴로지 = **AMX CPU + 단일 GPU**. expert 를 세 군으로 나눔: "We classify experts into three groups: EXP_R, EXP_M, and EXP_C" (R = VRAM 상주, M = PCIe 로 이동, C = CPU 계산). attention(OP_1)·out-proj(OP_2) 는 CPU/GPU 중 택일 ("assigns each of the three non-MoE operations exclusively to either the CPU or the GPU"). 즉 우리와 같은 "CPU 가 cold expert 를 AMX 로 계산 + GPU 에 hot expert 상주" 구조가 **이미 있음**. 단 attention 을 CPU 로 보내는 선택지도 포함 (우리는 GPU attention 고정).
- 머신: Xeon Platinum 8452Y 36코어, DDR5-4800 8ch 512 GB; RTX 6000 Ada 48 GB / A100 80 GB / H100 80 GB. 모델: Mixtral-8x7B, DeepSeek-V2-Lite, Qwen3-30B-A3B.

## 결정 변수
- 원문: "searches the feasible space of (x_0,x_1,x_2, EXP_R, EXP_M, EXP_C, m) that satisfies the VRAM/PCIe constraints and selects the configuration that yields the smallest T_tot". x_i = OP_i 의 장치, m = non-MoE 마이크로배치 크기, expert 는 배치 B 전체를 coalesce.
- VRAM 예산: "The budget explicitly accounts for EXP_R weights, computes allocation-induced non-MoE weights, intermediate data buffers (for the attention/expert computation), and temporary working space for kernels." → 예산은 **VRAM 용량**. KV 는 attention 을 CPU 로 보내면 host 에 두는 식 (OP_1 CPU 시 D_KV 를 PCIe 로 저장, Eq. t_store). KV 를 GPU 에 얼마나 둘지를 직접 변수로 두지는 않음 (attention 장치 선택에 종속).
- hot expert 선정: Expert-Aware Stratification — 데이터셋 임베딩 클러스터링 → 프로토타입 prefill probing → 활성화 맵으로 **정적** 사전 배치. "EAS is particularly advantageous in batch inference scenarios ... where the entire inference workload is available a priori."

## 비용 모델 파라미터 출처·검증·오차
- 모델: 층당 T_l = Σ_i [T_load + T_comp + T_store], T_comp = M·max((D_X+D_Y)/BW_DEV, C/TF_DEV) (roofline), expert 단계 T_comp(OP_3) = max(Latency_GPU, Latency_CPU), Latency_DEV = max((D_X3 + EXP_DEV·D_Y3)/BW_DEV, C_3/TF_DEV).
- 파라미터 출처: 장치 스펙 수치 (BW_PCIe ≈ 32 GB/s, DDR5 ≈ 300 GB/s, AMX ≈ 144 TFLOPs/socket, RTX 6000 Ada ≈ 364 TFLOPs — §2.2·§3.3 에 인용). 마이크로벤치 실측 언급 없음.
- 검증·오차: **모델 정확도·오차 보고 없음**. 성능 비교 (MoE-Lightning 대비 1.7–2.4×, FlexGen 대비 3.4–7.1×) 와 ablation (Table 6_3: 16.8 → 25.5 → 32.3 → 34.1 tokens/s) 만. 사전등록 없음.

## expert·KV 경합 취급 (호스트 DRAM 대역폭 공유 항)
- **없음**. CPU 측 BW_CPU 는 Latency_CPU 의 expert 가중치 읽기에만 쓰이고, EXP_M 의 PCIe 전송 (host DRAM 읽기) 이나 CPU attention 의 KV 읽기가 같은 DDR 을 쓴다는 합산 항이 없음. PCIe 와 DDR 은 "PCIe transfer rate (≈32 GB/s) ... is 10× slower than the CPU's DDR5 memory bandwidth (≈300 GB/s)" 로 별개 자원.
- 경합 논의는 **연산 불균형** 쪽: "simply offloading experts without considering their workload intensity can lead to a workload imbalance, shifting the bottleneck from the GPU to the CPU."

## D1·D2·D3 판정
| 조건 | 판정 | 근문 (원문) |
|---|---|---|
| D1 | **부분** | expert 상주 수 (EXP_R) 와 attention/KV 위치 (x_1) 를 **VRAM 용량 예산** 아래 함께 결정: "jointly optimizes two coupled decisions: (1) compute allocation for operations and (2) expert allocation to VRAM." 그러나 예산은 용량이지 DDR 대역폭이 아니고, KV 상주량이 연속 변수가 아님 (attention 장치 선택의 부산물) |
| D2 | **부분** | roofline 식으로 구성 선택은 하지만 처리량 사전 예측의 **오차를 보고하지 않음** ("인용 불가" — 정확도 문장 없음) |
| D3 | **부분** | 마이크로배치 축의 memory-bound 전이: "more experts become memory-bound as micro-batch size decreases." 및 CPU↔GPU 병목 이동 언급. hot expert 수 H 축의 전이점 분석 없음 |
| DDR 공유 항 | **아니오** | 인용 불가 (해당 항 없음) |

## 우리와의 delta
- **가장 가까운 토폴로지 선행** (AMX cold expert + GPU hot expert 상주 + 정적 hot 선정). "CPU AMX 가 cold expert" 자체는 우리 신규성이 아님.
- 남는 delta: (1) 무대 — 오프라인 배치(B=1024) vs 온라인 TPOT. (2) 예산 — VRAM 용량 vs 우리 DDR 대역폭 공통 예산 (cold expert 스트리밍 + host KV 읽기). (3) 예측 — 오차 미보고 vs 우리 사전등록 ≤20%. (4) 전이 — micro-batch 축 vs 우리 H 축. (5) hot 선정 — 데이터셋 사전 probing (워크로드 a priori) vs 우리 라우팅 트레이스 파라미터.
- 위협도: **상** (토폴로지·결정 변수 유사, 오차 미보고·온라인 부재가 유일한 빈틈). 7쪽 DAC 논문이라 모델 검증이 얕음 — 이를 우리 인트로에서 "결정 변수는 같으나 모델이 검증되지 않음" 으로 위치시킬 것.
