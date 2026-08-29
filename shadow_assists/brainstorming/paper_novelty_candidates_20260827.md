# 논문 신규성 후보 — CPU-GPU Hybrid 기능 구상 (2026-08-27)

> **목적**: PLN_003 캠페인 결과가 "vLLM 내장 기능 활용" (운영 성과) 에 그쳐 논문 기여가 없다는 사용자 판정에 따라,
> **논문으로서 값어치가 있는 CPU-GPU hybrid 기능**을 SOTA·문헌 조사 기반으로 재구상한다. **실험은 미실행** (타 실험 진행 중) — 설계·준비만.
>
> 신규 ID: `IDE_026` (P1) / `IDE_027` (P2) / `IDE_028` (P3)

---

## 1. 신규성 지형 지도 (2026-08 기준 — 무엇이 이미 점유되었나)

| 영역 | 점유 논문 (대표) | 상태 |
|---|---|---|
| KV cache CPU/디스크 tier | LMCache, Mooncake, InfiniGen, CachedAttention, llm-d | 🔴 포화 (우리 TSK_045 가 쓴 vLLM 기능이 이 계열) |
| Cold-KV CPU attention + merge | [NEO (MLSys'25)](https://arxiv.org/abs/2411.01142), FastDecode | 🔴 포화 + 우리 5세대 실측 기각 |
| MoE expert CPU offload (AMX) | [KTransformers (SOSP'25)](https://dl.acm.org/doi/10.1145/3731569.3764843), [HybriMoE (DAC'25)](https://arxiv.org/abs/2504.05897), [Expert Deferral (OSDI'26)](https://arxiv.org/html/2606.10493), [CoX-MoE (2026-05)](https://arxiv.org/abs/2605.17889) | 🔴 포화 — **단 전부 no-speculation** |
| Spec decode 오프로딩 결합 | [SpecOffload (2505.10259)](https://arxiv.org/abs/2505.10259) (draft 를 GPU 유휴에), [2508.21706](https://arxiv.org/pdf/2508.21706) (spec 으로 expert **전송** 은닉), [ELMoE-3D (2604.14626)](https://arxiv.org/html/2604.14626v1) (HW hybrid-bonding), [Less-Experts (2607.12696)](https://arxiv.org/html/2607.12696) | 🟡 활발 — **"spec × CPU-연산-expert" 교차점은 미점유** |
| Spec 제어 (serving, GPU-only) | TurboSpec, [FASER (2604.20503)](https://arxiv.org/pdf/2604.20503), [AdaSpec](https://arxiv.org/pdf/2503.05096), [BanditSpec](https://arxiv.org/pdf/2505.15141) | 🔴 포화 — 단 CPU-hybrid 비용모델 없음 |
| CPU-측 sparse-attention 인덱스 | ShadowKV, Squeezed Attention, [LServe](https://arxiv.org/pdf/2502.14866), [SparseServe](https://www.alphaxiv.org/overview/2509.24626), [CTkvr](https://arxiv.org/pdf/2512.15550), [2605.07719](https://arxiv.org/pdf/2605.07719), [ScoutAttention (2603.27138)](https://arxiv.org/html/2603.27138) | 🔴 포화 (구상 D 폐기 사유) |
| CPU/GPU coupled 특성화 | [2504.11750 (CMU/Samsung)](https://arxiv.org/html/2504.11750v1), [Oneiros (2507.11507)](https://arxiv.org/pdf/2507.11507) 의 단편 관찰 | 🟡 단편적 — **통합 형식화·regime atlas 는 미점유** |
| CPU drafter (이기종 spec) | [Dovetail](https://arxiv.org/abs/2412.18934), [DuoDecoding](https://arxiv.org/pdf/2503.00784), [AHASD (모바일)](https://arxiv.org/pdf/2604.25326) | 🟡 전부 **single-request latency** 지향 — batch-serving throughput 관점 미점유 |

**요약**: 단일 메커니즘의 빈 땅은 사실상 소진. 남은 신규성은 ① **교차점의 co-design** (특히 speculation × CPU-연산), ② **형식화·특성화** (우리만 가진 실측 corpus), ③ **serving-scale 재정의** (latency 논문들의 throughput 전환) 세 방향.

---

## 2. P1 (`IDE_026`) — SCED: Speculation-Coupled Expert Dispatch ★ 주력 후보

### 2.1 핵심 주장 (novelty claim, 한 문장)

**"Speculative decoding 의 검증 배치가 CPU-상주 expert 의 연산을 GEMV→GEMM 으로 전환시켜 CPU expert 의 경제성을 근본적으로 바꾼다 — 따라서 spec depth K 는 지연 은닉 수단이 아니라 CPU-GPU expert 배치의 1급 제어 변수이며, (K, 배치, deferral) 을 단일 최적화로 묶으면 기존 어느 시스템도 도달 못 하는 throughput regime 이 열린다."**

### 2.2 메커니즘

1. **관찰 (분석 모델의 핵심)**: CPU expert 연산은 memory-bound — 비용/token ≈ `W_expert / (BW_dram × n_tok/expert)`. 검증 스텝은 배치 M × (K+1) 토큰을 한 번에 처리 → expert 당 토큰 수가 (K+1)× 증폭 → **weight 1회 read 로 다수 토큰 처리 (AMX GEMM 화)** → CPU expert 의 실효 비용이 K 에 따라 급감.
2. **경제 규칙 (per-step dispatch)**: 매 스텝 router 결과로 expert 별 라우팅 토큰 수가 확정되면 — `토큰 多 expert → GPU (전송/상주 가치 있음)`, `토큰 少 expert → CPU AMX in-place (전송 미상각)`. 이 임계값이 K·α(수락률)·routing skew 의 함수로 닫힌형 도출됨.
3. **제어 루프**: 수락률 α 를 온라인 측정 → goodput-최적 K\* 를 CPU-hybrid 비용모델로 갱신 (기존 FASER/TurboSpec 의 GPU-only 모델과 **다른 최적점**을 가짐 — 논문의 검증 가능한 예측).
4. **예측→실증 서사**: "CPU-hybrid 에서 최적 K 는 GPU-only 보다 크다 (verify 배치 증폭이 CPU 비용을 낮추므로)" 라는 반직관 예측을 모델이 내고 실측으로 확인 — 강한 논문 구조.

### 2.3 선행연구 delta (novelty 방어)

| 선행 | 그들이 한 것 | 우리가 다른 것 |
|---|---|---|
| CoX-MoE (2605.17889) | micro-batch→ordinary-batch coalescing + **정적** stratification, **no spec** | coalescing 의 동력이 spec depth K (동적·제어 가능) + per-step 동적 dispatch |
| 2508.21706 | spec 으로 expert **PCIe 전송** 은닉 (GPU 연산) | expert 를 **CPU 가 연산** — 전송 자체를 제거, spec 은 CPU 연산의 경제성 변환기 |
| SpecOffload | draft 모델을 GPU 유휴 자원에 배치 | draft 는 종속 요소 아님 — 검증 배치 증폭 효과가 본체 |
| HybriMoE (DAC'25) | 동적 스케줄·캐시 (consumer, no spec) | K 결합 + 분석 모델 + 서버급 (TP8 + 2S SPR AMX) |
| Expert Deferral (OSDI'26) | deferral 로 CPU util ↑ | deferral 을 (K, dispatch) 와 **공동 최적화** 대상으로 흡수 |
| FASER/TurboSpec/AdaSpec | GPU-only serving 의 K 제어 | CPU-hybrid 비용항이 든 새 목적함수 — 최적 K 자체가 이동함을 보임 |

### 2.4 우리 자산 (왜 우리가 이걸 쓸 수 있나)

- PLN_003 로 **SGLang+kt-kernel 스택 가동 실적** (R1 642GB 서빙, 패치 4건, AMXINT4 변환 파이프라인) — 구현 출발점 확보
- 5월 spec decode 계측 자산 (수락률 워크로드별 실측 R/K: sonnet 0.388 / chat 0.812 / code 0.014 — α 모델의 초기값)
- regime 실증: GPU-only 불가 (OOM) 조건의 워크로드·설정 재현 절차 보유
- 하드웨어: SPR AMX ×2 + H100×8 — CoX-MoE 류 consumer 설정보다 큰 스케일에서의 최초 검증 가능

### 2.5 Kill risks (정직한 사전 검토)

1. **CoX-MoE v2** 또는 후속이 spec 결합을 선점할 위험 — v2 게재 확인 필요 (착수 시 재검)
2. 수락률 낮은 워크로드 (code α≈0) 에선 증폭 무효 → 논문은 α-조건부 이득으로 정직하게 서술 (오히려 regime 경계가 기여)
3. `SUB_167` (R1 품질 결함) 미해결 시 e2e 실증 모델이 제한 — Qwen3-계열 (품질 정상 확인됨) + 후속 대형 MoE 로 대체 가능
4. Q-dilemma 무관 검증: expert 연산은 라우팅 확정 후의 weight-resident 작업 — Q-dependency 없음 ✓

### 2.6 검증 계획 (실험 재개 허가 후)

microbench (AMX GEMM throughput vs tokens/expert 곡선 — 모델 §2.2-1 검증) → K-sweep × α-워크로드 grid (최적 K 이동 실증) → e2e (capacity-exceeded MoE, vs KT/CoX-MoE 재현 베이스라인). 목표 벤류: MLSys / ATC / EuroSys.

---

## 3. P2 (`IDE_027`) — Hybrid Legality: CPU 가 "합법적으로" 계산할 수 있는 것의 형식화 + Regime Atlas

### 3.1 핵심 주장

**"LLM 추론에서 CPU 로 옮길 수 있는 작업은 데이터-의존성 클래스 (Q-dependent / KV-resident / weight-resident / control-plane) 로 완전 분류되며, 클래스별 이득 상한이 자원비 (BW·FLOPS·용량) 의 닫힌형으로 결정된다 — 본 논문은 이 분류를 형식화하고, 동일 하드웨어에서 5세대에 걸친 시도·기각 실측 corpus 로 각 경계를 정량한 최초의 regime atlas 를 제공한다."**

- **Q-dependency dilemma** (layer-내 Q 확정 후 μs 창 안에 CPU 결과가 도달 불가 + Q 를 가진 GPU 는 자급 가능 → layer-수준 CPU attention 보조의 구조적 무효) 를 **명명·형식화한 논문은 부재** — NEO 의 H100 저이득, [Oneiros](https://arxiv.org/pdf/2507.11507) 의 GH200 관찰, [2504.11750](https://arxiv.org/html/2504.11750v1) 의 coupled-arch 특성화에 단편만 산재
- 우리 고유 자산: **타 그룹이 재현 불가능한 실패 corpus** — X/B1~B3, IDE_006 (merged 0%), NEO port (vanilla 2.6~4.1×), 그리고 성공측 경계 (KV-tier +51.8% 의 압박 임계, MoE 용량 임계, co-location 간섭 곡선)
- 산출물에 **regime-aware 런타임 라우터** (5월 AGSD 게이팅의 일반화 — 워크로드 신호로 hybrid 메커니즘 on/off) 를 얹으면 measurement+system 복합 논문
- 장점: **신규 GPU 실험 최소** (기존 eval/results corpus 대량 재사용) — 타 실험과 자원 충돌 없이 진행 가능한 유일 후보. 약점: "기능" 이라기보다 특성화 논문 — 사용자 지향 (기능 신규성) 과는 결이 다를 수 있음 → P1 의 동반 논문 포지션 권장

## 4. P3 (`IDE_028`) — 통합 Speculative Prefetch Oracle (예비)

draft 토큰이 공짜로 주는 두 신호 — ① router 출력 (다음 검증 스텝의 expert 집합), ② sparse-attention 접근 페이지 — 를 **단일 speculative prefetch 계약**으로 추상화해 expert·KV page·weight 를 tier 간 선반입. [2508.21706](https://arxiv.org/pdf/2508.21706) (expert 만)·InfiniGen (KV 만, prev-layer Q) 의 상위 일반화. **약점**: 구성적 (composition) 성격이 강해 novelty 방어력 3순위 — P1 의 확장 섹션으로 흡수하는 것이 안전.

---

## 5. 권고 우선순위와 다음 단계 (실험 없이 가능한 것)

| 순위 | 후보 | 이유 | 실험 전 준비 작업 |
|---|---|---|---|
| 1 | **P1 SCED** | 미점유 교차점 + 검증 가능한 반직관 예측 + 우리 스택 실적 위 | ① 비용모델 닫힌형 유도 (수식) ② CoX-MoE v2/신간 재검 ③ kt-kernel/SGLang 개입 지점 설계 ④ microbench 설계서 |
| 2 | **P2 Atlas** | 유일 corpus, 실험 최소, P1 과 상보 | ① 의존성 클래스 형식화 초안 ② 기존 eval/results 데이터 인벤토리 ③ 관련연구 비교표 |
| 3 | P3 Oracle | composition 위험 | P1 확장 섹션으로만 유지 |

**폐기 확정** (지형 조사 결과): CPU sparse-attention 인덱스 (포화), 단순 vLLM-native KT 이식 (엔지니어링, 신규성 없음 — 단 P1 의 구현 기반으로는 유효), KV-tier 자체 (내장 기능).
