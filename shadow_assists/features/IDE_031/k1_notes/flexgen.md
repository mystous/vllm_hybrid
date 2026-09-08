# FlexGen (ICML'23) — K1 정독 노트

- **서지**: Ying Sheng, Lianmin Zheng, Binhang Yuan, Zhuohan Li, Max Ryabinin, Beidi Chen, Percy Liang, Christopher Ré, Ion Stoica, Ce Zhang. "FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU." ICML 2023 (PMLR 202). arXiv 2303.06865 (v2, 2023-06-12).
- **전문 접근 여부**: **전문 확보 (23쪽, 부록 A.3 비용 모델 포함)**.
- **전문 확보 경로**: arxiv.org 직접 접속 불가 (HTTPS 타임아웃). HF 데이터셋 `Chelsea707/arxiv-cs-2020-2025-pdfs` 의 `data/2023/2303_06xxx/2303.06865.pdf` 를 curl 로 받아 pypdf (PyPI 휠, PYTHONPATH) 로 텍스트 추출. 교차 확인용으로 PMLR 판 (`proceedings.mlr.press/v202/sheng23a/sheng23a.pdf`) 도 확보 — 본문 동일.

## 문제·결정 변수
- 대상: **dense** LLM (OPT-6.7B~175B) 의 offline 배치 생성. GPU/CPU/디스크 3단 계층.
- 결정 변수 11개 (§4.3): "A policy includes 11 variables: block size bls, GPU batch size gbs, weight placement wg, wc, wd, activation placement hg, hc, hd, and KV cache placement cg, cc, cd."
- 연산 위치도 변수: KV 가 CPU 에 있으면 attention 을 CPU 에서 계산하는 위임 (§4.2 Computation delegation): "For long sequences (e.g., s ≥ 512), it is better to compute the attention scores on the CPU if the associated KV cache is not stored on the GPU."
- 스케줄: zig-zag block schedule (가중치 재사용 위해 열 방향 순회) + 6개 논리 스레드 overlap.

## 비용 모델 파라미터 출처와 검증
- 모델 형태: `T = Tpre·l + Tgen·(n−1)·l`, `Tgen = max(ctogg, gtocg, dtocg, ctodg, compg)` — 각 항은 바이트 수 / 대역폭, FLOPs / 처리율의 **선형 합** (부록 A.3). LP 로 배치 비율 9개 해결, (bls, gbs) 는 열거.
- **파라미터 출처 = 프로파일 피팅** (스펙·마이크로벤치가 아님): "To use the cost model, we run profiling on the hardware to sample some data points and fit the hardware parameters." 부록: "In real systems, these constants vary according to the total load. We handle such dynamics by using piece-wise functions and adding regularization terms."
- **예측 오차 보고 없음**. 모델 품질 언급은 정성적: "The cost model can usually return a good policy, but it is common that a better policy can be obtained by tuning manually." / "sometimes a strategy from the policy search can run out of memory. In this case, we manually adjust the policy slightly."
- MoE-Lightning 의 평가 (2411.11217 §1): FlexGen 은 "extensive data fitting (might take hours or days)" 이 필요하다고 지적.

## expert·KV 경합 취급
- MoE 아님 → expert 개념 없음. 대신 **가중치 비율 (wg,wc,wd) 과 KV 비율 (cg,cc,cd) 을 같은 LP 안에서 동시에 결정**하며, 둘 다 같은 I/O 항 (예: `dtocg = 1/dtoc_bdw · (weights·wd + KV·cd + act·hd)`, `ctogg` 도 동일 구조) 에 합산되므로 **가중치 스트리밍과 KV 읽기가 같은 링크 대역폭을 나눠 쓴다는 모델링은 이미 존재** (링크 = PCIe/디스크; 호스트 DRAM 대역폭은 별도 항 없음).
- 인용 (§4.3 Cost Model): "For I/O terms like dtocg, it is estimated by summing up the I/O events, which contain weights, activations, and cache reads."
- 인용 (§1): "our solution unifies the placement of weights, activations, and the KV cache, enabling a dramatically higher batch size upper bound".
- 메모리 제약도 공동: "gpu peak memory < gpu mem capacity, cpu peak memory < cpu mem capacity, disk peak memory < disk mem capacity".
- **호스트 DRAM 읽기 대역폭 항: 없음**. CPU attention 은 `cpu_compg = attg / cpu_flops` 로 FLOPS 항만 있음 (DRAM BW 항 부재).

## D1/D2/D3 판정
| 조건 | 판정 | 근거 (원문) |
|---|---|---|
| D1 expert 상주 수 + KV 배치를 공통 대역폭 예산으로 함께 결정 (MoE 맥락) | **아니오 (MoE 맥락)** / **예 (dense 가중치↔KV 공동 LP)** | "finding the best placement p = (wg, wc, wd, cg, cc, cd, hg, hc, hd) becomes a linear programming problem" — 가중치·KV 배치를 같은 I/O 예산 (`dtocg`, `ctogg` 합산) 아래 동시 결정. MoE/expert 는 다루지 않음. |
| D2 마이크로벤치/스펙-only 파라미터로 사전 예측 + 오차 보고 | **아니오** | 파라미터는 "run profiling on the hardware to sample some data points and fit the hardware parameters" (엔드투엔드 피팅). 예측 오차 수치 인용 불가 (본문에 없음). |
| D3 binding 자원 전이점 명시 분석 | **아니오 (부분)** | `Tgen = max(...)` 형태라 암묵적으로 binding 항이 있으나 전이점을 명시적으로 분석한 문장 인용 불가. 관련 정성 서술: "For 'No CPU compute', it degrades OPT-30B more than OPT-175B because the bottleneck for OPT-175B is on disk offloading." (Table 8: decoding 11315s 중 Cache(R) 7046s.) |

## 우리와의 delta
- 그들이 이미 한 것: 가중치 배치 비율과 KV 배치 비율의 **공동 결정**, 공통 링크 I/O 예산 (`max(I/O 항, compute)`), CPU attention 위임 — dense, offline, 링크 = PCIe/디스크.
- 그들이 안 한 것: (a) MoE expert 상주 집합 (hot H) 이라는 결정 변수 없음 → 라우팅 트레이스 기반 distinct-expert 기대값 항 없음. (b) **호스트 DRAM 대역폭** 항 없음 — 우리의 공통 예산은 PCIe 가 아니라 DDR (expert 스트리밍 + DRAM KV 읽기 동시 사용 간섭). (c) 파라미터가 엔드투엔드 프로파일 피팅이고 사전 예측·오차 보고가 없음. (d) 전이점 명시 없음.
- 우리가 못 주장하게 되는 것: "가중치와 KV 배치를 공동으로 결정하는 최초" 류의 주장. 주장 3 은 "MoE 서빙 + DDR 예산 + 사전등록" 으로 좁혀야 함. 위협도 **상 (주장 3 의 일반형)**.
