# C-트랙: 원점 재탐색 — 자료 조사와 문제-발굴 방법론 (2026-08-29)

> 목적: "확인적 논문" 이 아니라 **큰 발견이 가능한 문제** 를 고르기 위한, 재사용 가능한 프로세스와 그 1차 적용 결과.
> 선행 처분: IDE_026 (SCED) E0~E4 기록보관, 안 A/B 미착수 보존.

---

## 1. 방법론 — 문제 발굴 5-필터 (6세대 실패에서 추출한 교훈의 규칙화)

지난 시도들이 죽은 이유는 기술이 아니라 **문제 선정** 단계에 있었다. 이를 필터로 규칙화한다. 후보 문제는 5개 필터를 전부 통과해야 착수한다.

| 필터 | 질문 | 탈락시킨 과거 사례 |
|---|---|---|
| **F1 물리 여유** | 우리가 이기려는 자원 비대칭이 ≥10× 인가? (대역폭·용량·가격 중 하나라도) | NinjaGap CPU 엔진 (연산 1:100 열세에서 정면승부) |
| **F2 GPU-자급 불가** | 그 작업은 GPU 혼자서는 **원리적으로** 못 하는 일인가? (용량 초과 / 요청-간 상태 / 품질 판단 / 제어) | IDE_006·NEO (GPU 가 스스로 할 수 있는 attention 을 분담 → query-dependency wall) |
| **F3 반직관 가설** | 결과를 한 줄로 줄였을 때 "당연한 얘기" 가 아닌, 틀릴 수 있는 예측이 있는가? | 8-27 캠페인 ("압박 있으면 캐시가 이긴다" = 상식), IDE_027 atlas (확인적) |
| **F4 red-ocean 회피** | 최근 6개월 arXiv 에서 동일 핵심 주장을 미는 그룹이 3개 미만인가? (주간 재확인) | SCED (CoX-MoE·2508.21706·ELMoE-3D 가 교차점 포위), CPU sparse-index (ShadowKV 외 다수) |
| **F5 자산 적합** | 우리 하드웨어(H100×8+SPR 2TB)·스택·데이터로 **6주 내** 결정적 증거를 만들 수 있는가? | GH200/CXL/PIM 계열 (하드웨어 부재) |

**운용 규칙**: ① 착수 전 2주 kill-test (가장 싸게 죽일 수 있는 실험) 를 반드시 설계 ② F3 의 가설은 사전 등록 (PLN_004 방식 유지 — 이번에 잘 작동했음) ③ F4 는 착수 시점과 4주 후 재검.

---

## 2. 지형 조사 (2026-08 기준) — 어디가 포화이고 어디가 열려 있나

| 영역 | 상태 | 근거 |
|---|---|---|
| 고전 hybrid (KV tier / CPU attention / MoE offload) | 🔴 포화 + 우리 실측으로 경계 확인 완료 | KTransformers SOSP'25, HybriMoE DAC'25, Expert Deferral OSDI'26, [CoX-MoE](https://arxiv.org/abs/2605.17889) |
| 비용-frontier (T/$ 최적화, 이기종) | 🟡 활발 — GPU-이기종은 선점, CPU-tier 포함 econ 은 부분 개척 | [Mélange](https://arxiv.org/pdf/2404.14527), [BOute](https://arxiv.org/pdf/2602.10729), [HBM Is Not All You Need](https://arxiv.org/pdf/2606.29986), [Tutti (SSD KV)](https://arxiv.org/pdf/2605.03375) |
| **Agentic / test-time-compute 서빙** | 🟢 **최활발 + 명시적 공백** — 주 단위로 논문이 나오는 형성기. 리뷰 문헌이 "**structured multi-path search 를 네이티브로 지원하는 서빙 시스템은 아직 없다**" 고 명시 | [KVFlow/Continuum 계열](https://arxiv.org/html/2604.26968v2), [KAIROS](https://arxiv.org/pdf/2604.16682), [TokenDance](https://arxiv.org/pdf/2604.03143), [SMetric](https://arxiv.org/pdf/2607.08565), [MORI](https://arxiv.org/pdf/2606.00866), [학습기반 agent KV](https://arxiv.org/html/2608.14624) |
| Coherent memory (GH200/GB200, 900GB/s C2C) | 🟡 벤더 자체 개척 중 + **우리 하드웨어 없음** (F5 탈락) | [NVIDIA 공식 KV offload 블로그](https://developer.nvidia.com/blog/accelerate-large-scale-llm-inference-and-kv-cache-offload-with-cpu-gpu-memory-sharing/) |
| CXL / PIM / 신메모리 | ⚪ 하드웨어 부재 (F5 탈락) | — |

핵심 판독: **발견의 시장은 "하드웨어 축" 에서 "워크로드 축" 으로 이동했다.** 하드웨어 축(무엇을 어디서 계산하나)은 3년간 채굴되어 경계가 그어졌고, 워크로드 축(추론이 *탐색* 이 된 시대에 서빙 시스템은 무엇을 최적화해야 하나)은 이제 막 형성 중이다.

---

## 3. 후보 문제 (필터 적용)

### C1 ★ 탐색-네이티브 서빙: "서빙 시스템의 목적함수를 토큰에서 정답으로" 

**문제**: reasoning·agent 시대의 추론은 단일 스트림이 아니라 **탐색** (best-of-N, 트리 탐색, 자기수정 루프) 인데, 모든 서빙 시스템은 여전히 **토큰 처리량** 을 최적화한다. 탐색 워크로드에서 진짜 목적함수는 **정답 품질 / (GPU-초 또는 달러)** 이고, 이 둘은 다른 스케줄을 낳는다.

**시스템 구상**: 가지(branch) 를 1급 스케줄링 객체로 — ① 가지별 가치 추정(PRM/휴리스틱) 기반 확장 순서 결정, ② 저가치 가지 KV 는 GPU→DRAM(2TB) 강등·부활 (recency 아닌 **value 기반 eviction**), ③ 가지 fork 시 prefix KV 공유, ④ CPU = 탐색 제어기 + 가치 채점 + cold-branch 보관층 (F2: GPU 혼자 못 하는 "요청-간 상태 + 품질 판단" 영역).

**반직관 가설 (F3)**: 
- (a) "동일 GPU 예산에서 **토큰 처리량을 의도적으로 낮추는** 스케줄이 정답률을 올린다" — 처리량 지표가 탐색 워크로드에서 반(反)최적임을 정량 입증
- (b) "가지 eviction 은 LRU 가 체계적으로 틀린다 — 같은 메모리에서 value-기반 eviction 이 정답률 +Δ" (LRU 는 오래 방치된 유망 가지를 죽임)

**필터**: F1 ✅ (DRAM 용량 3×, 가격 ~10×/GB — 다(多)가지 보관은 용량 게임) · F2 ✅ · F3 ✅ · F4 🟡 (영역은 활발하나 "**서빙 스케줄러의 목적함수 교체**" 를 정면으로 한 시스템은 미확인 — 착수 시 재검 필수) · F5 ✅ (Qwen3-30B + GSM8K/MATH 급 벤치로 quality 측정 가능, 기존 KV-tier·스택 재사용)

### C2 세션-수명주기 KV 경제학 (agent idle window × DRAM tier)

tool-call·human-turn 유휴창에 세션 KV 를 DRAM 으로 파킹/부활. **F4 ❌ 임박** — [MORI](https://arxiv.org/pdf/2606.00866) (idle-window offloading), Continuum (TTL pin), [2608.14624](https://arxiv.org/html/2608.14624) (학습기반) 이 이미 조밀. 기각.

### C3 비용-frontier 재구성 ("지는 구성이 이기는 구성")

SLO 제약하 $/token 최소화에 CPU-tier 를 넣으면 "느린 hybrid" 가 Pareto 최적이 되는 영역 도출. F3 는 좋으나 **F4 🟡** ([Mélange](https://arxiv.org/pdf/2404.14527)·[2606.29986](https://arxiv.org/pdf/2606.29986) 인접) + 단독으론 econ-분석 논문. **C1 의 평가축 ($ 당 정답) 으로 흡수** 하는 것이 최선.

---

## 4. 권고와 다음 단계

**권고: C1 을 단독 주 트랙으로.** C3 는 C1 의 평가 프레임으로 흡수, C2 기각.

### 2주 kill-test 설계 (시스템 구축 전에 가설만 최소비용 검증)

| 단계 | 내용 | 죽는 조건 |
|---|---|---|
| K1 (문헌 정밀) | "quality-aware serving scheduler" 정면 선행 유무 전수 조사 (F4 확정) | 동일 주장 시스템 2개 이상 발견 |
| K2 (가설 a 예비) | 기존 서빙 그대로 + 클라이언트측 탐색 (best-of-N, 트리) 에서 **스케줄 순서만 바꿔** 동일 GPU-초당 정답률이 유의미하게 (≥5%p) 변하는지 — 시스템 없이 순서 효과만 측정 | 순서 효과 < 2%p (스케줄이 품질에 무영향 → 시스템 지을 가치 없음) |
| K3 (가설 b 예비) | 가지 KV 를 인위적 메모리 제한에서 LRU vs value-proxy 로 evict — 정답률 차이 측정 | 차이 < 2%p |

K2/K3 가 살면 → 시스템 설계 (PLN 발급) / 죽으면 → 그 죽음 자체가 C1 기각 근거로 기록되고 다음 후보 탐색.


---

## 5. C1 kill-test 결과 (2026-08-29, 당일 실행)

**C1 기각.** GSM8K 100문제 × 12가지 트리 (GPU 98초) 위에서 사전등록 게이트로 판정:
K2 (스케줄 순서 효과) 최대 Δ +0.0%p < +5%p FAIL / K3 (eviction 정책) 전 구간 음수 FAIL.

사인 규명 2건 — 차기 후보의 사전값으로 편입:
1. **역신호**: mean-logprob 가치 프록시는 무용이 아니라 해롭다 (오답 가지가 더 자신만만: seg1 logprob 오답 −0.139 > 정답 −0.150). RANDOM 이 VALUE 를 전 예산에서 이김
2. **가지 = 표**: 다수결 체제에서 가지는 낭비가 아니라 표본이다 — 조기 절단은 정답률을 깎는다. C1 의 "GPU 시간 90% 낭비" 전제가 이 체제에서 거짓

상세: `eval/results/20260829_094237_pln005_killtest/K2K3_RESULTS.md` (+tree.json 원시 트리)

**방법론 성적**: 후보 1개 발굴→검증→기각에 반나절 미만·GPU 98초. 프로세스는 유지, 다음 후보 탐색 재개.

---

## 6. 후보 2 — Footprint-Elastic MoE Serving: kill-test 결과 (2026-08-29)

**후보 정의**: CPU-AMX expert 오프로드로 480B급 모델의 GPU footprint 를 4장까지 줄일 수 있으므로 (TSK_047), 부하에 따라 "통합 1-인스턴스(S)" ↔ "분할 2-인스턴스(D)" 를 동적으로 전환하는 elastic 서빙 시스템.

**사전등록 kill-test 와 판정**:

| 게이트 | 내용 | 결과 |
|---|---|---|
| K2 (간섭) | 480B×2 (4+4) 동시 실행 간섭 ≤20% | **PASS** (≈0%, `eval/results/20260829_124321_k2_dual480b/`) — 단 kt CPUInfer 절대-pin 결함 발견·해결 필요했음 |
| K3 (교차점) | 부하-정합 S vs D 곡선에서 승자 교차 존재 | **FAIL** — 총부하 1~64 전 구간 D ≥ S (+2~12%), 교차 없음 (`eval/results/20260829_175150_k3_crossover/RESULTS.md`) |
| K4 (diurnal 이득) | 실측 곡선 대입 시뮬에서 동적 전환 ≥15% | **moot** — K3 교차 부재로 전환 이득의 원천 자체가 없어 미실행 |
| F4 (선행) | red-ocean 재확인 | **FAIL** — AlpaServe (OSDI'23) 가 shard-vs-replicate 부하 의존 선택을 선점 (사용자 지적으로 재확인. 교훈: 신규성 조사 시 2023 이전 고전까지 소급 필수) |

**판정: 기각.** 남은 delta 는 "CPU-expert 가 GPU-메모리-초과 모델의 복제-서빙을 가능케 하며, 그 복제가 전 부하에서 통합보다 낫다 (+2~12%)" — 시스템 논문 분량이 아니라 강한 측정 결과 1건. → `features/IDE_023/dual480b_results_20260829.md` 및 K3 RESULTS 의 운영 지침으로 보존.

**부수 산출 (재사용 가치)**: ① kt pin 결함 + cgroup cpuset 해법 (upstream 제보 대상) ② "분할>통합 전구간" 곡선 ③ 전환 비용 83~86초 (page cache 상주 덕) ④ mmap 가중치 공유로 복제의 DRAM 한계비용 ≈0.

## 7. 신규성 탐색 4라운드 (2026-08-29, K3 진행 중 병행)

새 축 2개 조사, 모두 선점:
- **훈련-서빙 동거** (CPU-expert LoRA 훈련 + 서빙 co-location): MACE (2025, iteration-level 훈련·추론 공동 스케줄), mLoRA, Punica/S-LoRA, CARASERVE (CPU-assisted cold-start) — 일반형 선점. 남는 조각은 또 "희귀 하드웨어 위 조합"
- **zero-copy 가중치 복제 서빙**: IPC 공유 텐서로 multi-role 배포가 1벌 메모리로 동작하는 기법 기실용화, dedup/ref-count 도 LMCache 등에 존재

**4라운드 누적 패턴**: SCED·C1·Footprint·훈련동거·zero-copy — 생성한 모든 후보가 2023~2026 선행의 1~2보 이내에서 격추. 하이브리드 서빙 메커니즘 공간은 포화 상태라는 것이 데이터. → 권고 전환: novelty 논문 대신 **experience/measurement 논문** (capacity-초과 MoE 서빙 실측 corpus: 복제>샤딩 곡선, 양자화-병렬성 제약, kt pin 결함, DeepSeek 변환 결함, spec 상호작용 + upstream 제보 2건). 사용자 결정 대기.

---

## 8. 탐색 5라운드 — HPC 고전 기법의 이식 (2026-08-30, 사용자 지시)

**지시**: "고전 HPC/슈퍼컴퓨팅에는 메모리·캐시 로컬리티가 다른 계층을 다루는 기법이 많다. 가져와서 응용하는 방법을 구상하라."

### 8.1 고전 기법 → LLM hybrid 서빙 대응표

| HPC 고전 기법 | 우리 문제에서의 대응 | 선점 여부 |
|---|---|---|
| out-of-core (외부 메모리 알고리즘) | 가중치/KV offloading | 선점 (FlexGen, ZeRO-Inference 등) |
| latency hiding (double buffering, prefetch) | 전송-계산 overlap | 선점 (2508.21706, HybriMoE prefetch) |
| work stealing / 동적 부하 분산 | GPU↔CPU expert 작업 훔치기 | **선점 — HybriMoE, MoE-SpAc, TriMoE(AMX+NDP, hot/warm/cold 비용모델 배치)** |
| NUMA-aware 배치, read-only 복제 | K2/K3에서 우리가 운영적으로 수행 | 기법 자체는 상식 |
| temporal blocking (시간 타일링) | spec decode = 가중치 스트림의 시간 블로킹 | SCED에서 기각 완료 |
| huge pages / TLB, cache partitioning (CAT) | 232GB expert 가중치의 TLB, LLC 격리 | 엔지니어링 레버 (수 % 급) — 검증 노브로만 |
| **communication-avoiding / 데이터 이동 하한 (Hong-Kung, Ballard-Demmel)** | **"expert를 어디에 두어야 하는가"를 토큰당 최소 데이터 이동량의 하한 문제로 형식화** | **직접 선행 미발견 (검색 2종). 인접: FlashAttention 계열의 attention I/O 분석 (단일 디바이스 SRAM/HBM) — 계층 양쪽에 연산기가 있는 HBM/DRAM 배치 문제는 미형식화** |
| roofline model | 우리 E1 knee가 이미 암묵적 roofline | 단독으론 도구, 위 하한과 결합 시 골격 |

### 8.2 후보 3 — "MoE 배치의 데이터 이동 하한" (가칭 PlacementBound)

**초록 형식 (지시 6 준수)**:
> MoE 모델을 GPU(HBM)와 CPU(DRAM)에 걸쳐 서빙할 때 expert 가중치를 어디에 두고 어디서 계산할지는 지금까지 시스템마다의 경험 법칙이었다. 우리는 이 배치 문제를 고전 HPC의 데이터 이동 하한 문제로 형식화한다. 모델 구조(expert 크기, top-k, 층수)와 하드웨어 수치(HBM/DRAM 용량·대역폭, 연산 처리량)만 넣으면 토큰당 최소 데이터 이동량과 각 배치 정책의 실제 이동량이 계산되고, 어느 체제에서 어떤 배치가 이길 수밖에 없는지가 수식에서 바로 나온다. 이 모델은 우리가 H100×8+AMX 머신에서 실측한 세 현상 — ① GPU에 들어가는 모델은 CPU 이관 시 15배 손해 ② 배치 크기 knee에서 CPU expert 비용 43~53배 하락 ③ 용량 초과 모델은 NUMA 분할 복제가 통합보다 전 부하 +2~12% — 을 사후 설명이 아니라 **사전 예측**한다. 신규성은 "계층 양쪽에 연산기가 있는" 배치 문제의 하한 형식화다 (FlashAttention 계열의 I/O 분석은 단일 디바이스 내부 계층만 다룬다).

**이것이 기존 자산과 맞물리는 지점**: 보류된 IDE_027(regime atlas)이 골격이나, 당시 기각 사유가 "확인적(측정 서술)"이었음. 하한 형식화가 붙으면 서술→예측으로 바뀌어 그 사유가 해소됨. 실측 corpus(30B/70B/480B, knee, K3 곡선)는 전부 검증 데이터로 재사용.

### 8.3 사전등록 kill-test (착수 전 판정)

| 게이트 | 내용 | 죽는 조건 |
|---|---|---|
| K1 (선행 정밀) | "I/O complexity MoE placement", Hong-Kung→LLM 적용 전수 조사 (FlashAttention I/O 하한 계열과의 delta 명문화 포함) | 동일 주장 선행 ≥2 |
| K2 (예측력) | 모델이 **아직 측정 안 한 수치**를 먼저 예측하고 실측으로 검증 (예: 70B hybrid 처리량, 30B의 가상 crossover 배치 크기) | 예측 오차 > ±30% (그럼 이론이 장식) |
| K3 (비자명성) | 모델이 상식과 다른 검증 가능한 예측을 ≥1개 내놓아야 함 (예: HBM에 들어가는데도 hybrid가 이기는 특정 구조, 최적 expert 분할비가 0도 100도 아닌 지점) | 전 예측이 "당연한 얘기" (지시 3 위반) |

통과 시 → PLN 발급, characterization study(선택지 ①)의 이론 골격으로 착수. 실패 시 → 사인 기록.
