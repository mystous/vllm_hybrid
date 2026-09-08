# MoE-SpeQ — K1 전문 정독 노트

- **서지**: Wenfeng Wang, Jiacheng Liu, Xiaofeng Hou, Xinfeng Xia, Peng Tang, Mingxuan Zhang, Chao Li, Minyi Guo (SJTU / HKUST). "MoE-SpeQ: Speculative Quantized Decoding with Proactive Expert Prefetching and Offloading for Mixture-of-Experts". arXiv 2511.14102 (2025-11), ACM 템플릿 15쪽 (venue 미기재).
- **전문 확보 경로**: HF `Chelsea707/arxiv-cs-2020-2025-pdfs` → `data/2025/2511_14xxx/2511.14102.pdf` (1.8 MB) → pypdf. **전문 확보**.

## 무대·토폴로지
- 무대 = 단일 GPU 메모리 제약 환경의 MoE 추론 처리량 (accepted tokens/s) + TTFT SLO 제약 (k_max). 원문: "Real-world systems often face dual, competing objectives: high throughput for batch workloads and low Time-to-First-Token (TTFT) for interactive requests."
- 토폴로지: expert 는 host DRAM, GPU 가 모든 연산 (draft 4-bit 모델 + target). expert 를 PCIe 로 프리페치. CPU 연산 오프로드 **없음**. 머신: A100 40 GB + Xeon Silver 4310 256 GB, PCIe 4.0 x16. 모델: Phi-3.5-MoE, Qwen1.5-MoE-A2.7B, DeepSeek-V2-Lite.
- 핵심: 4-bit 양자화 draft 모델이 target 의 top-k expert 선택을 >90% 예측 → Expert Lookahead Buffer 로 프리페치·evict.

## 결정 변수
- draft 길이 k (Governor 가 온라인 최적화), 프리페치 대상·evict 정책 (Expert Scheduler). GPU 메모리 예산은 실험 변수 ("we select several representative GPU memory budgets"). KV 는 draft/target 간 **공유**로 footprint 만 줄임 — expert 상주 vs KV 분할 결정 없음.

## 비용 모델 파라미터 출처·검증·오차
- Amortization Roofline Model: Θ(k) = k_accept(k)/T_cycle(k), T_cycle(k) = max(T_draft(k), T_pcie,init) + T_pcie,new(k) + T_verify(k+1).
- 파라미터 출처 = **오프라인 프로파일 + 온라인 추정**: "T_draft(k): ... profiled offline and modeled as a linear function T_d,base + k·T_d,token." / "T_pcie,init: ... profiled as a system constant." / "T_verify(k+1): ... profiled for several values of k and interpolated." / "PCIe bandwidth B_PCIe is a profiled constant." 수락 확률: "measured empirically during a warm-up phase and are continuously updated using an exponential moving average".
- 검증·오차: **모델 예측 정확도·오차 보고 없음** (grep: governor 정확도 문장 부재). 보고된 정확도는 expert 선택 예측 (">90%") 뿐. 사전등록 없음.

## expert·KV 경합 취급
- 자원은 PCIe 대역폭과 GPU 연산 둘: "The system's throughput (Θ) is bound by either PCIe bandwidth (I/O Roof) or computation (Compute Roofs)." host DRAM 대역폭 항 **없음**, KV 읽기와 expert 스트리밍 경합 항 없음 (KV 는 GPU 상주).

## D1·D2·D3 판정
| 조건 | 판정 | 근거 |
|---|---|---|
| D1 | **아니오** | expert 상주는 캐시 정책, KV 는 GPU 상주·공유. 공통 예산 결정 없음. 인용 불가 |
| D2 | **부분** | 프로파일 상수 기반 roofline 으로 k 선택 (온라인 제어용). 처리량 사전 예측의 오차 보고 없음 |
| D3 | **부분** | "I/O Roof ↔ Compute Roof" 의 knee 를 k 축에서 탐색: "ideally near the 'knee' of the highest achievable compute roof." 전이 축 = draft 길이 k (PCIe ↔ GPU 연산). H 축 아님 |
| DDR 공유 항 | **아니오** | 인용 불가 |

## 우리와의 delta
- 전이점을 제어 변수 축에서 찾는 roofline 형식의 선례 (D3 형식, 축은 다름). 예측 정확도 미보고, CPU 연산 없음, 온라인 TPOT 격자 없음. 우리 delta 는 H 축 전이 + DDR 공통 예산 + 사전등록 오차. 위협도 **하~중**. 관련 연구에서 "expert 프리페치 계열 (투기 기반)" 으로 분류.
