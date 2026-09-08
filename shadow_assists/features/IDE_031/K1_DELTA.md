# PLN_008 K1 — 선행 연구 delta 표 (작성 중, 2026-09-08)

> **접근 제약과 우회**: arxiv/usenix/acm 등 차단 (huggingface.co 만 열림). **전문 경로 확보**: HF 데이터셋 `Chelsea707/arxiv-cs-2020-2025-pdfs` (PDF) / `scholarweave/arxiv-latex` (LaTeX parquet range 읽기). 대상 B 8편은 전문 정독; 대상 C 6편은 스니펫 판정 (재확인 필요) → 분신 D 가 전문 재정독 중; 대상 A 7편은 전문 전환 지시됨.

죽는 조건: D1 = expert 상주 수와 KV 배치를 공통 대역폭 예산으로 함께 결정 (MoE 서빙) / D2 = 마이크로벤치·스펙-only 파라미터로 하이브리드 MoE 처리량 사전 예측 + 오차 보고 / D3 = binding 자원 전이점 분석.

## 대상 C+추가 (KV 계층·해석 모델·서베이 위협) — 분신 D, **전문 정독** (HF PDF / LaTeX; 노트 `k1_notes/*_full.md`)
| 논문 | 전문 | D1 | D2 | D3 | DDR 공유 항 | 한 줄 delta |
|---|---|---|---|---|---|---|
| **MoE-Lens** (2504.09345, HPDC'26) | PDF | 아니오 — expert 전량 PCIe 스트리밍 (버퍼 2층), KV CPU 상주, 분할 결정 없음 | **예(형식)/부분** — B_IO 실측 상수 + GPU 프로파일 직선 → "average 94% accuracy"; 사전등록 없음, 평균 한 숫자 | **예 (다른 축)** — "transitions from a CPU memory capacity-bound regime to a GPU-bound regime" (축 = KV 용량) | **예 (비-binding)** — Eq.5 "B_Mem = B_KV + B_IO"; §8.2 경합 실측 (5→6 s) 후 "does not become a bottleneck" | 오프라인 배치 / CPU=attention·GPU=전 expert / 전이 축 KV 용량. **위협 상** |
| **CoX-MoE** (2605.17889, DAC'26) | LaTeX | **부분** — (x0,x1,x2,EXP_R,EXP_M,EXP_C,m) 을 VRAM/PCIe **용량** 제약 아래 동시 선택; KV 상주량은 attention 장치 선택의 부산물 | 부분 — roofline 으로 구성 선택, **오차 미보고** | 부분 — micro-batch 축 ("shifting the bottleneck from the GPU to the CPU") | 아니오 | **AMX cold expert + GPU hot expert 상주 토폴로지 이미 존재**. 빈틈: 온라인 TPOT, DDR 대역폭 예산, 오차 보고, H 축. **위협 상** |
| InfiniGen (2406.19707) | PDF | 아니오 | 아니오 | 아니오 (PCIe 고정 병목) | 아니오 | 직교. 하 |
| Mooncake (2407.00079) | PDF | 아니오 | 아니오 (오프라인 회귀, 수치 없음) | 아니오 | 아니오 | 클러스터 계층. 하 |
| GenZ (2406.01698) | PDF | 아니오 | 부분 (실측 효율 계수 + roofline, dense 오차 1.4~5.8%) | 부분 | 아니오 (단일 링크 항) | 해석모델+오차 선례. 중 |
| Vidur (2405.05465) | PDF | 아니오 | 부분 (연산자 프로파일+RF, <9%, dense) | 아니오 | 아니오 | 방법론 최유사, 하이브리드·MoE 없음. 중 |
| LLMCompass (2312.03134) | PDF | 아니오 | 부분 (스펙 시뮬, 4.1%, dense) | 부분 (HBM 내) | 아니오 | 스펙-only 선례. 중 |
| MoE-SpeQ (2511.14102) | PDF | 아니오 | 부분 (오차 미보고) | 부분 (k 축 knee) | 아니오 | 투기 프리페치. 하~중 |

판정 (8편): D1 예 0건 (CoX-MoE 부분 — VRAM **용량** 예산; DDR **대역폭** 예산은 없음). D2 형식 선례 = MoE-Lens (하이브리드 MoE, 94%) + dense 3편; 사전등록·셀별 오차 분포 보고는 0건. D3: MoE-Lens (KV 용량 축), MoE-SpeQ (k 축), CoX-MoE (micro-batch 축) — **H 축 없음**. DDR 공유 항은 MoE-Lens Eq.5 에 이미 존재 (비-binding 결론) → 우리 주장 ③ 은 "이 항이 binding 인 워크로드 (멀티턴·HiCache 같은 소켓) 를 실측으로 보이고 그때 최적 분할이 바뀜을 사전 예측" 으로 좁혀야 함. 실증 실패 시 주장 ③ 은 MoE-Lens 각주 수준.

## 대상 A (최대 위협 7편) — 분신 A, **전문 정독** (KT SOSP'25 만 미확보; HybriMoE 실제 ID = 2504.05897)
| 논문 | 전문 | D1 | D2 | D3 | 한 줄 delta |
|---|---|---|---|---|---|
| FlexGen (ICML'23) | PDF | 아니오 (MoE) / 예 (dense: 가중치·KV 비율 9변수 LP 를 공통 I/O 항 아래 동시 결정) | 아니오 — "run profiling on the hardware to sample some data points and fit the hardware parameters", 오차 미보고 | 부분 — max(I/O, compute) 형태 | 가중치↔KV 공동 배치는 선행. 우리: MoE expert 집합 H + DDR 예산 + 사전등록 오차 |
| **MoE-Lightning** (ASPLOS'25) | PDF | **부분 예** — (N, μ, Ag, Fg, rw, rc) 를 같은 comm_cpu_to_gpu 예산으로 결정 (단위 = 비율, 예산 = PCIe) | 부분 — "theoretically calculated … with profiled peak performance and memory bandwidth" (스펙+피크벤치), 오차 미보고 | **예** — HRM turning point P1/P2/balance point (Eq.9–11) | **최대 위협.** 남는 delta: H 축 전이, DDR 예산, 라우팅 트레이스 항, 절대 오차 사전등록, 온라인 서빙 |
| NEO (MLSys'25) | PDF | 아니오 (dense, attention offload) | 아니오 (오프라인 프로파일 보간, 오차 미보고) | 부분 (출력 길이 축 balance point) | DDR 이 CPU attention 의 binding 자원이라는 실측은 선행 |
| KTransformers (SOSP'25) | **PDF (사용자 반입, 14:10)** | 아니오 — shared expert GPU / routed expert CPU, hot expert 수 변수 아님, KV GPU | 아니오 — 성능 모델 없음, 실측 speedup 만 | 아니오 — 배치 1, deferral 개수 = 단일 층 타임라인 실측 → "CPU 포화까지" 휴리스틱 | 기반 시스템. H 축·KV 결합·사전 예측·온라인 배치 서빙이 빈자리. deferral 창 기전은 우리 모델 항과 일치 (`ktransformers_sosp25_full.md`) |
| Fiddler (ICLR'25) | PDF | 아니오 (용량 순 popularity 배치, KV 없음) | 부분/아니오 (초기화 마이크로벤치 상수, 처리량 예측 없음) | 부분·정성 (입력 크기 임계) | batch-1 지연, PCIe 중심 |
| HybriMoE (DAC'25, 2504.05897) | PDF | 아니오 | 아니오 (warmup 프로파일) | 아니오 | kt 기반 단일 요청 지연. 하 |
| MoE-Gen (2503.09716) | PDF | **부분 예** — S_Params (GPU 상주 파라미터) 와 ω (KV PCIe/CPU) 를 공통 HtoD 예산으로 탐색 | 아니오 (오프라인 프로파일, 오차 없음) | 부분·실측 (ω breakeven ≈60%) | GPU 메모리를 파라미터 캐시에 쓰면 링크 예산이 풀린다는 인식은 선행 (PCIe) |


## 대상 B (expert 캐시·프리페치 8편) — 분신 B, **전문 정독** (HF 데이터셋 PDF/LaTeX 경로)
| 논문 | D1 | D2 | D3 | 캐시 크기 결정 | 한 줄 delta |
|---|---|---|---|---|---|
| TriMoE (2603.01058, DAC'26) | 아니오 (KV 호스트 DIMM 고정, 모델에 KV 트래픽 항 없음) | 아니오 (오프라인 lookup table, 오차 미보고) | 부분·정성 (도메인별 병목, "256 tokens/expert" 문턱) | 휴리스틱 (hot/warm/cold + EMA prefetch) | max(연산,전송,DRAM) 비용 형태·3분류 참고. H 축 전이·사전 예측 없음 |
| MoE-Infinity (2401.14361) | 아니오 (KV 전량 GPU, expert 캐시는 잔여) | 아니오 | 아니오 | 잔여 고정 | "KV 커지면 expert 버퍼 감소" 관찰 (Fig.8) → 주장 3 동기 인용 |
| Pre-gated MoE (2308.12066, ISCA'24) | 아니오 | 아니오 | 아니오 (경험적 추세) | 고정 비율 스윕 | 구조 변경·배치 1, 겹침 없음 |
| HOBBIT (2411.01433) | 아니오 | 아니오 (캘리브레이션 튜닝) | 아니오 | 고정, 정밀도 혼합 캐시 | 겹침 없음 |
| ProMoE (2410.22134) | 아니오 | 아니오 | **부분·경험적** (캐시 비율 축에서 CPU 연산 ↔ PCIe 스트리밍 교차점 실측) | 고정 (외부 스윕) | D3 선행으로 인용 필요 — 모델·사전 예측·KV 항 없음 |
| ExpertFlow (2410.17954) | 아니오 (KV 는 재배치 정합성만) | 아니오 | 아니오 | 고정 + 층간 분배 예측 | "캐시 크기↔배치 크기 상충의 적응 조정" 을 향후 과제로 남김 → M2 동기 인용 |
| **Klotski** (2502.06888) | 아니오·근접 (용량 제약 공통, 대역폭 항은 가중치 I/O 만, 상주 수 = top-k 고정) | **부분** (마이크로벤치로 부등식 풀어 n 계획; 처리량 예측·오차 없음) | 부분·경험적 (n 축 I/O 지배→버블 소멸) | 휴리스틱 | **K1 최근접 선행.** delta = 마이크로벤치→스텝시간 사전 예측 (격자·오차) → H/KV 분할; 항 차이 = CPU expert 연산, KV DRAM 읽기와 cold 스트리밍의 DDR 공유, H 결정 변수 |
| DAOP (2501.10375, DATE'25) | 아니오 | 아니오 (단일 지점) | 아니오 | 고정 (ECR) | 배치 1, 정확도 손실 근사, 겹침 없음 |

판정 (8편): D1 0편 / D2 완전 0편·부분 1편 (Klotski) / D3 완전 0편·경험적 부분 3편 (ProMoE·Klotski·TriMoE). H 축 전이점을 모델로 사전 예측한 선행 없음.

## K1 판정 (2026-09-08 14:10 확정, **22/22편 전문**)
- **D1** (expert 상주 + KV 배치를 공통 대역폭 예산으로 함께 결정, MoE 맥락): 완전 일치 0편, **부분 일치 3편** — MoE-Lightning (비율, PCIe 예산), MoE-Gen (S_Params+ω, PCIe 예산), CoX-MoE (VRAM 용량 예산, AMX cold + GPU hot 상주). 사전 등록 규칙 "≥1 → 주장 ③ 보조 강등" 을 **발동**으로 판정 (부분 일치라도 "함께 결정" 의 핵심이 선행됨).
- **D2** (마이크로벤치/스펙-only 파라미터 → 하이브리드 MoE 처리량 사전 예측 + 오차 보고): 완전 0편. 형식 선례 MoE-Lens (평균 94%, 파라미터 = 실측 B_IO 상수 + GPU 프로파일 직선, 사전등록 없음), MoE-Lightning (스펙+피크벤치, 오차 미보고). **연구 중단 조건 (≥2) 미발동.** 셀별 오차 분포·사전등록은 0편.
- **D3** (전이점 명시): MoE-Lightning (HRM turning points, 배치·비율 축), MoE-Lens (KV 용량 축), CoX-MoE (micro-batch 축), MoE-SpeQ (k 축). **hot expert 수 H 축 (DDR 대역폭 ↔ GPU expert 연산) 전이는 0편.**
- DDR 공유 항 (KV 읽기 + expert 스트리밍): MoE-Lens Eq.5 에 존재, 비-binding 결론. NEO 는 DDR 이 CPU attention 의 binding 자원임을 실측.

### 주장 재배치 (K1 결과)
1. **주력 ①**: H 축 전이점 — hot expert 수를 늘리면 binding 자원이 DDR 대역폭 (cold expert 스트리밍) 에서 GPU expert 연산으로 바뀌며, 마이크로벤치·라우팅 트레이스 파라미터만의 모델이 그 위치와 스텝 시간을 **사전 등록 후** 예측한다 (온라인 서빙 TPOT, 셀별 오차 분포 보고).
2. **주력 ②**: 위 모델의 예측력 자체 (격자 12셀, 중앙값 ≤20%, 순위 일치) — 선행은 평균 한 숫자·사후 비교뿐.
3. **보조 ③ (강등)**: KV 호스트 읽기와 cold 스트리밍의 DDR 경합 항 — HiCache 멀티턴 셀에서 binding 여부를 실측해 MoE-Lens 의 비-binding 결론이 어느 조건에서 뒤집히는지 보고. 정책 (H·KV 분할) 은 이 결과에 따라 본문 또는 부록.
- 무대 (온라인) 와 토폴로지 (CPU cold expert + GPU hot expert) 는 CoX-MoE·KTransformers 와 겹치므로 단독 delta 로 주장하지 않음.
- KTransformers SOSP'25 전문 확인 완료 (사용자 반입): GPU expert 수별 병목 분석 없음 (배치 1, shared/routed 고정 배치) → 주장 ① delta 유지. **K1 최종: 조건부 통과** (주장 ③ 보조 강등).
