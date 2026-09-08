# KTransformers (SOSP'25) — K1 정독 노트

- **서지**: Hongtao Chen, Weiyu Xie, Boxin Zhang, Jingqi Tang, Jiahao Wang, Jianwei Dong, Shaoyuan Chen, Ziwei Yuan, Chen Lin, Chengyu Qiu, Yuening Zhu, Qingliang Ou, Jiaqi Liao, Xianglin Chen, Zhiyuan Ai, Yongwei Wu, Mingxing Zhang (Tsinghua MADSys 외). "KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models." SOSP '25 (2025-10, Seoul). DOI 10.1145/3731569.3764843. **arXiv 판 없음** (WebSearch 3회, HF PDF 데이터셋 2025 디렉토리에 해당 제목 없음).
- **전문 접근 여부**: **불가**. 전문 후보 3곳 모두 이 머신에서 차단: `dl.acm.org/doi/pdf|epdf/10.1145/3731569.3764843` (WebFetch 무응답), `madsys.cs.tsinghua.edu.cn/.../SOSP25-chen.pdf` (WebFetch 무응답 2회, curl HTTP 000), lmsys.org 블로그 (무응답). 확보한 1차 자료: (a) GitHub `kvcache-ai/ktransformers` 의 `kt-kernel/README.md` (raw.githubusercontent, 30KB, 전문), `doc/en/AMX.md`, `doc/en/kt-kernel/experts-sched-Tutorial.md`; (b) `doc/en/KTransformers Full Introduction for Motivation and Practice.pdf` (40쪽 슬라이드, **이미지 전용이라 텍스트 추출 0자**); (c) WebSearch 스니펫에 실린 초록 문장. 아래 논문 인용은 **스니펫에서 옮긴 초록 문장**이며 본문 대조 못 함. 본문에 기반한 판단은 "추측" 표시.

## 문제·결정 변수 (코드·문서 기준, 논문 본문은 추측)
- 배치: attention·shared expert·KV 는 GPU, routed expert 는 CPU 기본, **GPU 상주 expert 수는 사용자 파라미터**. README: "`--kt-num-gpu-experts` | Number of experts to keep on GPU | `32` (remaining experts go to CPU)" ; "Determine based on GPU memory and profiling: More GPU experts = lower latency but higher GPU memory usage (May cause OOM)".
- 상주 expert 선택: "`frequency`: Places the most frequently activated experts on GPU. Best performance when activation statistics are available; requires `--init-expert-location` pointing to a `.pt` statistics file." 동적 갱신: "During layerwise prefill, the system collects actual routing statistics and redistributes GPU experts accordingly." / "Particularly effective at lower GPU expert ratios (10%-70%)".
- Expert Deferral (초록 스니펫): "KTransformers further introduces an Expert Deferral mechanism that reorders expert execution across layers." README: "`--kt-max-deferred-experts-per-token` | Number of experts per token to defer for pipelined execution | `2` (0 to disable, 1-4 recommended)" ; "`5-7`: Highest latency reduction but may introduce noticeable accuracy loss".
- 커널 선택 (AMX.md): "AMX kernels are automatically selected during long prompt prefill phases (where each expert handles more than 4 tokens on average), while short prompt prefill and decode phases dynamically switch to AVX-512 kernels."
- KV 배치: GPU 고정 (문서에 KV 오프로드 결정 변수 없음; SGLang HiCache 와의 결합은 문서 밖).

## 비용 모델 파라미터 출처와 검증
- 문서·README·스니펫 어디에도 **처리량 예측 모델 없음**. GPU expert 수 결정은 "based on GPU memory and profiling" (사용자 수동). 초록 스니펫의 성능 서술은 실측: "achieving 4.62–19.74× prefilling speedups and 1.25–4.09× decoding speedups compared to existing systems" ; deferral 은 "increasing CPU utilization from typically below 75% to almost 100% ... up to 1.45× additional throughput ... average model accuracy drop of no more than 0.5%".
- 논문 본문에 roofline 류 분석이 있을 가능성은 배제 못 함 (**추측**; 전문 미확인). 초록·문서에서는 확인되지 않음.

## expert·KV 경합 취급
- 문서 기준 **없음**. GPU expert 수는 메모리 여유로 사용자가 정하고 KV 와의 교환은 서술 없음. DDR 대역폭은 NUMA 스레드풀 권고 문장뿐: "This enables better memory bandwidth utilization across NUMA domains."
- 초록 스니펫의 hardware 논리: "the sparsely activated experts can run efficiently on CPUs with large memory capacity, while the dense and compute-intensive components — attention and shared experts — execute on GPUs with higher bandwidth and throughput."

## D1/D2/D3 판정 (전문 미확인 — 초록·1차 문서 기준, 본문 추정 부분은 "추측")
| 조건 | 판정 | 근거 |
|---|---|---|
| D1 expert 상주 수 + KV 배치 공동 결정 | **아니오 (추측)** | 상주 수는 사용자 파라미터 "`--kt-num-gpu-experts` ... `32` (remaining experts go to CPU)"; KV 결정 변수 없음. 초록에 배치 결정 모델 언급 없음. |
| D2 마이크로벤치-only 사전 예측 + 오차 보고 | **아니오 (추측)** | 초록·문서에 예측 모델 문장 없음; 성능은 실측 speedup 만. 인용 불가. |
| D3 binding 자원 전이점 명시 분석 | **아니오 (추측)** | 초록에 없음. 관련은 커널 전환 임계 ("more than 4 tokens on average" 에서 AMX↔AVX-512) 와 deferral 의 CPU 활용률 서술뿐 — 자원 전이점 분석과는 다름. |

## 우리와의 delta
- 같은 점: 우리 시스템의 **기반 그 자체** (hot expert 수 H = `--kt-num-gpu-experts`, deferred N = `--kt-max-deferred-experts-per-token`, frequency 배치). IDE_030 의 hot-96 + deferred 4 는 이 파라미터의 설정값.
- 다른 점 (문서 기준): H 를 정하는 모델·KV 와의 결합·사전 예측·전이점이 없음 — 우리가 채우려는 자리. 단 **전문 미확인**이므로 본문에 "GPU expert 수에 따른 병목 분석" 이 있을 수 있음 (추측 위험). K1 판정 확정 전에 ACM DL 전문 확보 필요 (다른 네트워크에서 `SOSP25-chen.pdf` 다운로드).
- 우리가 못 주장하게 되는 것: "hot expert GPU 상주 + cold CPU 계산 + deferral" 메커니즘 자체는 KT 의 것. 우리 기여는 그 위의 **결정 모델**로 한정. 위협도 **중 (전문 확인 전까지 보류)**.
