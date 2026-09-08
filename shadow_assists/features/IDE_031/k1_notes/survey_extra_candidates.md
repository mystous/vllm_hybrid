# 사용자 서베이 인용 논문 전수 + 추가 후보 위협도 — K1 노트

- 원본: `~/.claude/uploads/dfe3454a-8b67-497c-afa5-34e0ac76691f/af3d1e40-moe_host_offloading_survey.md` (2026-09-07 작성).
- 접근: 서베이 본문은 로컬 읽음. "상" 후보의 arXiv 초록은 arxiv.org 접속 불가로 **WebSearch 스니펫** (abs 페이지 요약) 으로 대체 확인. 인용은 스니펫 문장.

## (a) 서베이 인용 논문 전수 (참고문헌 표 + 본문 언급)
참고문헌 표 (23): Fiddler 2402.07033 / KTransformers SOSP'25 / HybriMoE 2504.05897 / CoX-MoE 2605.17889 / Pre-gated MoE 2308.12066 / AdapMoE 2408.10284 / HOBBIT 2411.01433 / ProMoE 2410.22134 / MoE-Infinity 2401.14361 / fMoE(FineMoE) 2502.05370 / MoE-Lightning 2411.11217 / MoE-Gen 2503.09716 / MoE-Lens 2504.09345 / MoE-SpeQ 2511.14102 / SPICE 2608.21240 / CXL-NDP MoE 2512.04476 / TriMoE 2603.01058 / DynaNDE 2609.00407 / Local Routing Consistency 2505.16056 / LayerScope 2509.23638 / MoE-Beyond 2508.17137 / MoE Inference Survey (ACM CSUR 10.1145/3794845) / awesome-moe-inference (GitHub).
본문만 언급 (9): Mixtral-Offloading, ExpertFlow, SpecMoEOff (2508.21706), SP-MoE (2510.10302), DuoServe-MoE (2509.07379), FATE (2502.12224), DeepSpeed-Inference, llama.cpp, Marlin.

## (b) 정독 목록 21편 (대상 A 6 + 15) 에 **없는** 논문의 위협도
| 논문 | 서베이 설명 요지 | D1 | D2 | D3 | 위협도 |
|---|---|---|---|---|---|
| **MoE-Lens** (HPDC'26, 2504.09345) | HRM 이 CPU 메모리 용량·요청 길이를 빠뜨렸다고 지적; CPU 메모리 대역폭·연산도 필요 | 부분 | **예** | 부분 | **상** |
| **CoX-MoE** (DAC'26, 2605.17889) | 3그룹 expert 정적 층화, roofline 기반 per-device latency model 로 배치 | 부분 | 부분 | 부분 | **상** |
| **MoE-SpeQ** (2511.14102) | Amortization Roofline Model governor 가 speculation 전략 조정 | 아니오 | 부분 | 부분 | **상** (확인 후 중) |
| SpecMoEOff (2508.21706) | 이론+실측 roofline 으로 CPU/GPU 조율, 하이퍼파라미터 optimizer | 아니오 | 부분 | 부분 | 중 |
| CXL-NDP MoE (2512.04476) | prefill 통계로 decode expert 배치, hot expert HBM pin, 1–4bit | 아니오 | 아니오 | 아니오 | 중 |
| DynaNDE (2609.00407) | NPU-NDP 해석 성능 모델로 층별 expert 스케줄 | 아니오 | 부분 | 아니오 | 중 |
| Mixtral-Offloading, AdapMoE, fMoE, MoE-Beyond, SP-MoE, SPICE, DuoServe-MoE, FATE | 예측·prefetch 계열 (정확도·hit rate 목표) | 아니오 | 아니오 | 아니오 | 하 |
| Local Routing Consistency, LayerScope | 라우팅 분석 (SRP/SCH, 층군별 차이) | 아니오 | 아니오 | 아니오 | 하 (우리 트레이스 통계의 인용 근거로 유용) |
| MoE Inference Survey, awesome-moe-inference | 서베이/목록 | — | — | — | 하 |

## "상" 3편 초록 확인 결과 (WebSearch 스니펫)
### MoE-Lens (Yuan, Ma, Talati; HPDC'26)
- 스니펫 인용: "The performance model analyzes various fundamental system components, including CPU memory capacity, GPU compute power, and workload characteristics to understand the theoretical performance upper bound of MoE inference ... identifies CPU memory capacity as a primary limiting factor" / "the theoretical model predicting performance with 94% average accuracy" / "CPU attention kernels are 3-4× faster than KV cache transfer, which is close to the ratio of CPU memory bandwidth and the CPU to GPU memory bandwidth" / "the use of paged KV cache shifts the turning point to the right".
- 판정: **D2 예** — 하이브리드 MoE 처리량을 해석 모델로 예측하고 오차 (평균 94% 정확도 ≈ 6% 오차) 를 보고. 다만 파라미터가 마이크로벤치인지 스펙인지, 사전등록인지는 초록으로 불명 (전문 정독 필요). **D1 부분** — CPU 메모리 안의 KV·가중치 배분과 배치 크기의 관계를 모델링 (용량 축). "공통 **대역폭** 예산" 인지는 미확인. **D3 부분** — "turning point" (배치에 따른 GPU 활용 전환) 언급; binding 자원이 DDR→GPU 연산으로 바뀌는 우리 형태와 같은지 미확인. **정독 목록에 즉시 추가해야 하는 최대 위협.** 죽는 조건 "D2 선행 ≥2" 카운트 후보 1.
### CoX-MoE (KAIST; DAC'26)
- 스니펫 인용: "a static expert-aware stratification scheme that pre-assigns frequently activated experts to the GPU" / "T_comp follows a roofline model based on the hardware configuration of each device" / "1.7–2.4× higher throughput compared to MoE-Lightning and 3.4–7.1× higher throughput compared to FlexGen" / "In the baseline 'GPU Attn' ... the vast majority of VRAM (84.6%) is consumed by intermediate data ... which severely constrains the VRAM available for expert weights".
- 판정: **D1 부분** — hot expert 를 GPU 에 정적 배치하고 attention offload 로 VRAM 을 expert 에 돌리는 결정을 함께 다룸 (VRAM 용량 경합). DDR 대역폭 공통 예산은 스니펫에 없음. **D2 부분** — roofline 기반 per-device 지연 모델로 배치 결정; 예측 오차 보고 여부 미확인. **D3 부분** — micro-batching 으로 expert 가 memory-bound 가 된다는 진단 (operational intensity 전이). 우리와 같은 AMX CPU 하이브리드·throughput 영역. 전문 정독 필요. 죽는 조건 카운트 후보 2 (오차 보고가 있으면).
### MoE-SpeQ (SJTU)
- 스니펫 인용: "an adaptive governor, guided by an Amortization Roofline Model, dynamically tunes the speculation strategy to the underlying hardware" / "The Amortization Roofline Model quantitatively tunes the speculation window for throughput optimality" / "up to 2.34× speedup for Phi-MoE".
- 판정: D1 아니오 (KV 없음), D2 부분 (roofline 로 speculation 창 선택; 처리량 사전 예측 오차 보고 미확인), D3 부분 (speculation 창에 따른 I/O↔연산 전이). 위협도 **중** 으로 하향. 인용은 필요.

## 결론
- 정독 목록 21편에 **MoE-Lens 와 CoX-MoE 를 추가** 해야 K1 이 성립. 둘 다 "하이브리드 MoE 처리량 해석 모델 + 배치 결정" 이라 우리 주장 2·3 과 직접 겹침 가능. MoE-Lens 는 D2 에서 이미 "예" 로 보이므로, FlexGen·MoE-Lightning 과 함께 D2 카운트가 2 를 넘는지 전문으로 확정할 것.
