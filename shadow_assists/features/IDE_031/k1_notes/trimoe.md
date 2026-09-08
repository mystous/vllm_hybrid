# TriMoE — arXiv 2603.01058 (DAC 2026)

## 서지 / 전문 접근
- Yudong Pan, Yintao He, Tianhua Han, Lian Liu, Shixin Zhao, Zhirong Chen, Mengdi Wang, Cangyuan Li, Yinhe Han, Ying Wang (ICT CAS). "TriMoE: Augmenting GPU with AMX-Enabled CPU and DIMM-NDP for High-Throughput MoE Inference via Offloading". v1 2026-03-01, "Accepted by DAC 2026".
- 전문 접근: **예** — arxiv.org 는 본 네트워크에서 차단되어 HF 미러 `scholarweave/arxiv-latex` 의 LaTeX 소스(v1) 전문을 읽음. 그림 내용(roofline, hot/warm/cold 분포 수치)은 캡션·본문 서술로만 확인.

## 결정 변수
- decode 스텝마다 각 expert 의 실행 도메인: GPU(HBM hit / PCIe miss), AMX-CPU, DIMM-NDP. 온라인 greedy 배정 + bottleneck-aware 반복 이동(makespan 최소화).
- 예측(EMA, α=0.3) 기반 배경 작업: hot expert prefetch, 레이아웃 변환(striped↔localized), DIMM 간 cold expert 재배치.
- 배치는 256~768 (zigzag/offline batching) 고정 입력. KV 는 결정 변수 아님(호스트 DIMM 에 offload 고정).

## 비용 모델 파라미터 출처 / 예측 검증 / 보고 오차
- 파라미터: "we offline-profile both the GPU and CPU for various token counts, building lookup tables for f_calc_gpu(L_i) and f_calc_cpu(L_i)". 여기에 T_PCIe, T_DRAM(W_i, M_i)(striped=전체 대역폭 / localized=단일 DIMM 대역폭), T_Internal(W_i).
- 모델: T_GPU_Miss = max(f_calc_gpu, T_PCIe, T_DRAM); T_CPU = max(f_calc_cpu, T_DRAM); T_NDP = max(f_calc_ndp, T_Internal); T_Makespan = max(ΣGPU, ΣCPU, max_d[ΣNDP_d + T_contention^d]).
- 용도: 온라인 스케줄링 입력. **스텝 시간·처리량의 사전 예측값과 실측 대비 오차는 보고하지 않음.** 성능은 H100 + Xeon 8470(AMX) 프로토타입 + NDP 는 Ramulator 2.0 기반 사이클 시뮬레이터로 측정. 예측기 정확도만 보고("78% migration decision accuracy").

## expert 와 KV 의 메모리·대역폭 경합 취급
- "For storage, MLP and shared expert weights reside in GPU HBM, while the large KV Cache and all routed experts are offloaded to host DIMMs". KV 를 호스트에 두지만 KV 읽기 트래픽은 모델 항에 없음.
- DIMM 경합 항은 가중치 읽기만: "a DIMM is busy not only when performing local NDP computation but also when serving weight fetch requests from the GPU or CPU" (T_contention^d).
- GPU 메모리를 expert 와 KV 로 나누는 결정 없음.

## 캐시 크기 결정 방식
- **휴리스틱**: hot/warm/cold 는 활성화 트레이스 통계로 분류(cold >70% 의 expert 가 토큰 8%, warm 20~40% 가 토큰 최대 70%). GPU 상주 expert 는 shared expert + EMA 로 "hot" 예측된 것을 prefetch. GPU expert 슬롯 용량 자체를 결정하는 절차 없음.

## 판정
- **D1: 아니오.** KV 배치는 고정(호스트), expert 상주 수와 함께 대역폭 예산으로 결정하지 않음. 인용: "the large KV Cache and all routed experts are offloaded to host DIMMs".
- **D2: 아니오.** 마이크로벤치(lookup table) 기반 비용 모델은 있으나 처리량 사전 예측·오차 보고 없음. 인용: "we offline-profile both the GPU and CPU for various token counts, building lookup tables" (예측 오차 문장: 인용 불가).
- **D3: 부분(정성).** 도메인별 binding 자원을 명시: "GPU-CPU approaches are constrained by host memory bandwidth", "(1) GPU-side I/O bottleneck ... the H100 GPU requires at least 256 tokens per expert to reach 30% utilization ... (2) NDP-side compute bottleneck". 그러나 hot expert 수 H 에 따른 전이점 분석은 없음. makespan=max(·) 로 스텝별 병목 장치를 식별할 뿐.

## 우리와의 delta
- 공통: per-expert 비용 = max(연산, 전송, DRAM 읽기) 형태, "bottleneck-aware" 용어, AMX CPU 를 warm expert 에 배정.
- 차이: (a) 우리는 GPU-CPU 2도메인, NDP 없음. (b) 우리는 KV 위치(GPU/DRAM)를 결정 변수에 넣고 KV 읽기와 cold expert 스트리밍의 DDR 공유 항을 둔다 — TriMoE 는 KV 트래픽을 모델링하지 않음. (c) 우리는 H 를 스윕하며 binding 자원 전이점을 **사전 예측**하고 오차를 보고 — TriMoE 는 예측 오차 없음. (d) TriMoE 의 hot/warm/cold 3분류와 "tokens per expert ≥256" 문턱은 우리 모델의 g(D_h, B) 항 설계에 참고 가치.
