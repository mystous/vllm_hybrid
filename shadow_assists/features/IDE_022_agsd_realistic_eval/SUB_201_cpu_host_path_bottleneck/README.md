# SUB_201 — spec-decode가 만든 host(CPU) 병목 재배치: CPU 병렬성 활용처 재정립

> **parent**: `TSK_042` (IDE_022). 선행 데이터: `TSK_042_realistic_workload_oracle/RESULTS.md` (**222셀, B200, 10모델 × {vanilla, suffix, llm-d, ngram} × 7조건**, XL 매트릭스 = 405B + 671B 추가 완료).
> **status**: 🔵 분석 완료 (XL 데이터 반영, 2026-06-04 갱신) / 결정적 실험(§5) 대기 — **AGSD 개념 전제 폐기**, CPU 병렬성 활용처를 데이터에서 재유도.
> **date**: 2026-06-04 (XL 추가 갱신)
> **HW**: DGX B200×8 (183GB HBM3e sm_100) + Intel Xeon Platinum 8570 (224스레드 2 NUMA, 2TB DRAM)
> **분류**: 연구/방향 재정립 (lever 탐색의 상위 thesis 결정)

---

## 0. 이 SUB의 목적

사용자 지시: **"AGSD 개념을 꼭 유지할 필요 없다. 지금 상황을 철저히 논리적으로 분석하고 CPU 병렬성을 활용할 곳을 찾자."**

따라서 본 문서는 AGSD(워크로드 레벨 게이팅) thesis 를 전제하지 않고,
TSK_042 의 173셀 실측 데이터에서만 출발해
**프로젝트 목표(CPU 활용률 극대화 → GPU 서버/클러스터 전체 성능↑)**를
달성할 수 있는 CPU 병렬성의 *실제* 적용 지점을 논리적으로 유도한다.

결론(선요약): **CPU의 올바른 역할은 "GPU 흉내(매트멀 오프로드)"가 아니라,
spec-decode가 만든 host-path 병목을 떠안아 GPU slack을 회수하는 것**이다.
이는 죽은 길(IDE_018 phase-burst, drop-in 커널, branchy busywork)과
산 길(IDE_019 CPU 드래프팅 + host-path 오프로드, IDE_017 KV tiering)을 가르는 단일 기준이 된다.

---

## 1. 데이터가 강제하는 사실 (TSK_042 XL 매트릭스 222셀, 반박 불가)

### 1.1 throughput·util 핵심 — 10모델 (mix 조건, 사이즈 오름차순)

| 모델 | size | vanilla tps | suffix tps | ΔSuf | GPU% v→s | ΔGPU | CPU% v→s | suf α |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| Qwen-2.5-7B | 7B(TP4) | 4,169 | 7,803 | +87% | 82.4→**36.9** | **+45.5** | 2.7→2.5 | 0.84 |
| Llama-3.1-8B | 8B | 8,850 | **27,851** ⭐ | **+215%** | 94.6→**66.8** | +27.8 | 4.6→4.4 | 0.85 |
| DS-Qwen-7B | 7B | 9,058 | 24,458 | +170% | 91.4→**66.6** | +24.7 | 2.8→2.6 | 0.64 |
| Qwen-2.5-32B | 32B | 3,056 | 6,597 | +116% | 94.1→**71.2** | +22.9 | 4.4→4.3 | 0.67 |
| DS-Qwen-32B | 32B | 4,938 | 9,056 | +83% | 97.4→81.5 | +15.9 | 4.8→4.4 | 0.56 |
| Qwen-2.5-72B | 72B | 2,735 | 5,268 | +93% | 97.1→83.7 | +13.4 | 4.6→4.3 | 0.48 |
| Llama-3.1-70B | 70B | 3,129 | **10,400** | **+232%** | 98.2→85.4 | +12.7 | 4.8→4.3 | 0.72 |
| DS-Llama-70B | 70B | 3,164 | 6,127 | +94% (mix only) | 98.2→85.9 | +12.3 | 4.9→4.4 | **0.40** |
| **Llama-405B FP8** | 405B | 1,252 | 2,829 | +126% | 98.8→93.1 | **+5.8** | 4.8→4.3 | 0.69 |
| **DeepSeek-R1 671B (MoE)** | 37B act | 1,538 | **781** | **−49%** ← worst | 97.6→93.7 | **+3.9** | 4.8→4.3 | **0.46** |

(전체 데이터: `TSK_042_realistic_workload_oracle/RESULTS.md`, raw `runs/tput_t1t3_20260602/metrics_table.parquet`)

**⚠️ XL 매트릭스에서 새로 드러난 thesis caveat**:
- **DS-Llama-70B 의 mix 외 6/7 셀, R1 671B 의 7/7 셀에서 suffix 가 vanilla 보다 느림** (총 vanilla-win 15셀)
- **R1 671B 는 매트릭스 worst suffix match** (mix −49%, α 0.46) — MoE expert routing 의 dynamism + reasoning 다양성이 suffix cache prefix-match 율 무력화
- **사이즈가 클수록 ΔGPU 작아짐**: 7B +45 → 8B +28 → 32B +23 → 70B +13 → 405B +6 → 671B +4. **spec-decode 의 GPU un-saturate 효과 = 작은 모델에서 강함**
- 즉 § thesis 의 "host-bound 전환" 은 **작은~중형 모델 + 높은 α 모델군에 강하게 적용**, **큰 모델 / 낮은 α 모델 (DS distill 70B+, MoE R1)** 은 별도 대책 필요 (→ §4.2 A1 CPU drafting 의 primary target)

### 1.2 세 가지 반박 불가 사실

1. **GPU가 throughput 병목**, CPU는 **전 구성 2.5–5.6%**로 거의 완전 유휴.
2. **지배 lever는 spec-decode(suffix)** — mix +83~232%. 다른 모든 lever(NUMA +3.13%, branchy +5.28%, AMX, NEO 등)는 한 자릿수 %이고 그나마 미검증/네거티브.
3. **결정적 단서: suffix를 켜면 GPU util이 *떨어진다*** (Qwen-7B 82.5→26.5%, 32B 94.3→64.8%, 8B 94.9→62.8%) — **처리량은 오르는데 GPU가 더 논다.**

### 1.3 지연 구조 (mix, p50)

| 모델 | TTFT v→s (ms) | TPOT v→s (ms) | suffix α |
|---|--:|--:|--:|
| DeepSeek-Qwen-7B | 16.8→22.3 | 3.3→**0.9** | 0.876 |
| Llama-3.1-70B | 28.4→56.3 | 9.2→**2.5** | 0.915 |
| Qwen-2.5-32B | 29.6→65.4 | 9.3→**3.1** | 0.857 |
| Llama-405B FP8 | 71.1→122.2 (+72%) | — | 0.766 |
| **DeepSeek-R1 671B** | **65.6→196.8 (+200%)** | — | **0.451** |

suffix는 TPOT(토큰당 지연)를 크게 줄이지만 TTFT는 늘린다 → step 수 급감 + step당 host 비용 증가. **R1 의 TTFT 가 3× 폭증** 은 MoE expert routing + draft+verify overhead 의 host-path 부담이 매우 큼을 시사 (§5 결정적 실험의 핵심 단서).

### 1.4 vanilla > suffix 셀 (15개) — α 임계치 분석

XL 매트릭스의 결정적 evidence: **suffix 가 vanilla 보다 느린 셀들의 공통 분모 = 낮은 α**.

| set | 셀 수 | α median | α range |
|---|--:|--:|--:|
| **suffix gain > 0** (suffix 이긴 셀) | 55 | **0.684** | 0.28 ~ 1.41 |
| **suffix gain < 0** (vanilla 이긴 셀) | **15** | **0.374** | 0.27 ~ 0.54 |

→ **α ~0.5 가 break-even 임계치** (K=32 num_speculative_tokens 셋업). 그 아래에선 spec 의 wasted compute (draft 생성 + reject) 가 gain 을 초과.

**15개 vanilla-win 셀 분포**:

| 모델군 | 셀 | α 범위 | s_wall/v_wall |
|---|---|---|---|
| **R1 671B** | 7/7 | 0.36~0.54 | **1.62~2.88×** (overhead 매우 큼) |
| **DS-Llama 70B** | 6/7 (mix 만 suffix) | 0.27~0.38 | 1.0~1.18× (overhead 작음, α 가 결정) |
| **Qwen 72B** | wildchat / mbpp | 0.27~0.47 | 1.05~1.08× (marginal) |

`s_ctok / v_ctok ≈ 0.88~1.05` — suffix 와 vanilla 가 거의 같은 토큰 수 생성 (모델 동일, lossless). **tps 차이 = wall time 차이**.

**근본 원인 (모델 패턴별)**:
- **DeepSeek-R1 MoE**: 256 expert dynamic routing + reasoning 출력 다양성 → suffix cache prefix-match 율 ↓ + verify 의 큰 forward pass cost
- **DS-Llama-70B**: distill reasoning chain 의 다양성 → α 0.27~0.38 (mean accept length ≈ 1.3, spec 효과 거의 없음)
- **Qwen-72B wildchat/mbpp**: 큰 모델 + multi-turn 다양성 → marginal break-even

---

## 2. 핵심 추론 — spec-decode가 병목을 GPU → host(CPU)로 이동시킨다

### 2.1 메커니즘

LLM decode 의 vanilla 경로는 **메모리 대역 bound 인 짧은 순차 step의 연속**이다(GPU util 82–98%로 포화).
spec-decode는 이 다수의 순차 step을 **소수의 큰 verify step**으로 대체한다(TPOT 3–9ms→1–3ms, α 0.79–0.93).
그 결과:

- step 수가 급감 → GPU가 step을 launch하는 빈도↓.
- **step 사이의 host 작업(draft 제안 + 스케줄링 + 샘플링 + detok + 동기)이 상대적으로 커져 GPU를 놀리는 gap이 된다.**
- 이 gap이 GPU util 26–65%로 관측된다.

> **명제(SUB_201): spec-decode는 GPU를 "verify 전용 가속기"로 만들면서
> 시스템을 host(CPU)-bound로 전환한다. GPU util 하락분(최대 ~70pp)은 회수 가능한 slack이며,
> 그 회수처는 정확히 host-path 위에 있는 유휴 CPU다.**

### 2.2 왜 이것이 프로젝트 목표와 정합하는가

- 프로젝트 목표 = CPU로 GPU 서버 전체 성능↑.
- spec-decode가 GPU slack(util↓)과 CPU 유휴(2.5–5.6%)를 **동시에** 만들었다.
- 따라서 CPU가 host-path를 떠안아 GPU를 다시 채우면 → **유휴 CPU가 직접 throughput으로 환산**된다.
- 이것이 "CPU idle 불허" 원칙을 *유용한 작업*으로 달성하는 유일한 데이터-정합 경로다.

### 2.3 필수 caveat (이 SUB의 go/no-go가 §5에 걸리는 이유)

`nvidia-smi` util은 **"커널이 하나라도 돌았나"의 coarse 지표**라
"GPU util 하락 = host gap"을 *증명하지 못한다*. 가능한 대안 해석:

- (a) inter-kernel gap (host-bound) — CPU로 회수 가능 ✅
- (b) verify 커널 내부 occupancy/대역 저하 — CPU로 못 고침 ❌
- (c) 배치를 못 채워서 (KV/concurrency 한계) — KV tiering으로 회수 (§6)

→ **§5의 step 분해 프로파일이 (a)/(b)/(c)를 판별하는 linchpin.** 이것 없이는 방향 확정 불가.

---

## 3. CPU 기여 가능 영역 — "throughput 매핑 여부"로 분류

### Tier A — GPU slack을 직접 회수 (throughput↑)

| ID | 내용 | 근거 | 기존 자산 |
|---|---|---|---|
| **A1** | **CPU 드래프팅** (suffix tree 탐색 / AMX 소형 draft model)을 유휴 CPU에서 GPU verify와 병렬 실행 → GPU는 verify만 | suffix가 지배 lever이고 GPU util 하락이 곧 회수 대상. 드래프팅은 매트멀-light(suffix tree) 또는 AMX 적합(소형 draft) | IDE_019 (amx_draft_head SUB_187: AMX 0.524ms/step feasible; cpu_draft_ranker) |
| **A2** | **KV tiering to DRAM/CXL** (유휴 2TB DRAM) → 더 큰 배치/긴 컨텍스트로 in-flight 시퀀스↑ → GPU gap을 다른 요청으로 채움 | B200 HBM 한계 regime에서 batch/concurrency 확대가 util 회복 경로 | IDE_017 (dma_zero_copy), SUB_058 (CPU radix prefix KV) |

### Tier B — host 직렬화 stall 제거 (GPU feeding↑)

| ID | 내용 | 근거 |
|---|---|---|
| **B1** | **Detokenization 오프로드** | Llama-8B suffix = 27,851 tok/s. detok이 critical path 경쟁 시 idle 코어로 분리 (IDE_016/SUB_171 AVX-512 detok 69× microbench 존재) |
| **B2** | **Constrained-decode(문법/JSON/tool) FSM·mask CPU 오프로드** | 정확히 code/SWE-bench 워크로드의 host-heavy 경로. 구조적 출력은 mask 계산이 step마다 발생 |
| **B3** | **스케줄러/샘플링 파이프라이닝** | Chung et al. 2026 — multi-GPU의 병목이 CPU가 GPU를 못 채우는 것 |

### Tier C — 논리적으로 죽은 길 (명시적 폐기)

| ID | 내용 | 폐기 근거 |
|---|---|---|
| **C1** | CPU가 GPU 매트멀 대신 수행 | IDE_018 phase-burst(tasks_executed=0, +1.35%), drop-in 커널 0/7, SUB_180 Jacobi(GPU 대비 40–50× 느림): **H2D/D2H 동기 비용이 이득 상쇄. 물리적 패배.** |
| **C2** | branchy busywork로 CPU "warm" 유지 | SUB_196 cellB +5.28%(1-run, 미검증), 메커니즘 가설("C-state 회피")도 미확증. **유용한 서빙 작업이 아님.** SUB_189(branchy −0.82%)로 work-pattern sensitivity 확인됨 — 신뢰 불가. |
| **C3** | CPU에서 대형모델 병렬 추론 | CPU FLOPS가 GPU 대비 1–2 orders 부족 (NEO TSK_019 net-negative 확정). |

> **분류 기준(단일):** CPU가 *GPU를 흉내내면*(C) 패배, *GPU가 만든 host 병목을 떠안으면*(A/B) 승리.

---

## 4. 가장 강한 단일 타깃

**"CPU = spec-decode의 draft + host-path 엔진"** (A1 + B 묶음, A2 부차).

- **데이터 정합**: 지배 lever(suffix)의 GPU util 하락분(회수 대상)의 원인이 host-path이고, CPU가 정확히 그 자리를 메운다.
- **죽은 길과 명확히 구분**: 매트멀 오프로드(C) 아님 — host-path는 본래 CPU 작업이다.
- **기존 자산 재활용**: IDE_019(multi_source_drafter)가 바로 이 방향. SUB_187 AMX draft microbench feasible(0.524ms). IDE_018/branchy harvest 트랙은 폐기.

### 4.1 ROI 우선순위 — XL 매트릭스 (222셀) 기반 정량 추정

| 우선 | Lever | 대상 모델·워크로드 | 정량 ROI 추정 | 근거 (XL 매트릭스 evidence) |
|---|---|---|---|---|
| 🥇 1 | **A1 CPU drafting** | R1 (α 0.46), DS-Llama-70B (α 0.40), Qwen-72B (α 0.48), DS-Qwen-32B (α 0.56) — **α<0.5 의 7+ 모델군** | suffix -49% → **+20~+50% 회복 가능** (R1 mix 781→2,000~3,000 tps) | 15 vanilla-win 셀 모두 α<0.55. CPU draft = GPU 가 verify only → draft cost 0, α 회복 가능 |
| 🥈 2 | **B1 detok AVX-512** | 7B/8B 클래스 모든 워크로드, throughput 큰 셀 (Llama-8B mix 27,851 tps) | +10~+20% (37% wall slack 의 일부 회수) | Llama-8B suffix mix wall 136s 중 51s = 37% slack. detok 이 critical path 경쟁 시 idle 코어 분리로 wall ↓ |
| 🥉 3 | **B3 scheduler / inter-kernel gap** | R1 (TTFT +200%), 큰 모델 ttft 폭증 셀 | Nsight 측정 후 결정 (gate) | R1 vanilla 65ms→suffix 196ms 의 폭증이 scheduler/batching/KV alloc 의 host 작업 늘어남 신호. §5 의 결정적 실험 대상 |
| 4 | **A2 KV tiering** | 405B (GQA KV head 8), long-context | 우리 데이터로 직접 증명 X — 별도 실험 (long context, large batch) 필요 | 405B vanilla GPU 98.8% saturated, KV space 늘리면 batch ↑ 가능. R1 MLA(kv_lora_rank=512) 작음 → ROI 작음 |

### 4.2 모델군별 lever 매핑 (top-down)

**작은 모델 (7B/8B/32B) — host-bound 강함**:
- suffix 가 GPU 를 26~71% 로 떨어뜨림 → **§5 (a) host gap 가설 가능성 가장 높음**
- B1 detok / B3 scheduler 가 직접 효과
- 이미 suffix 가 +87~232% 가속, 추가 회수 여지 30~40%

**중대형 모델 (70B/72B) — 혼합**:
- Llama-70B suffix +88% (α 0.72), DS-Llama-70B suffix -10% (α 0.40)
- 같은 사이즈인데 α 가 갈리는 게 핵심 — **distill family 차이**
- A1 CPU drafting 으로 α 회복 가능성 (suffix tree 가 cache miss 인 distill chain 에서 ML draft head 가 더 정확할 가능성)

**초대형 모델 (405B dense, 671B MoE) — GPU bound 유지**:
- ΔGPU 4~6% 만 — host-bound 전환 효과 약함
- 405B suffix 는 정상 작동 (+91% avg) — A1 의 ROI 일반적 (작음)
- **671B 가 최우선 target**: vanilla > suffix 의 -49% 페널티 회복 시 절대 throughput +1,500 tps 가능 (큰 모델이라 환산 가치 큼)

---

## 5. 결정적 실험 (go/no-go linchpin) — step 분해 프로파일

골격 재정립 전 **이 측정 하나가 전부를 결정**한다.

### 5.1 목적
suffix ON 상태에서 **decode step 1개의 wall-clock을 구성요소로 분해**하여
GPU util 하락(§1.3)의 원인이 host gap(a)인지 / verify 내부(b)인지 / 배치 미충전(c)인지 판별.

### 5.2 대상 셀
**Qwen-2.5-7B suffix, mix** — GPU util이 26.5%로 가장 낮아 gap이 최대(신호 최강).
대조: **Llama-3.1-70B suffix**(util 83%, gap 작음)로 대비.

### 5.3 측정 항목 (per step)
1. **GPU 커널 실제 실행 시간** vs **커널 사이 gap** — Nsight Systems (CUDA event timeline, inter-kernel idle).
2. gap 구간의 host 작업 분해 — py-spy / Python profiler:
   draft 제안(suffix tree lookup) / 스케줄 / 샘플링·logits / detokenize / Python overhead / H2D·D2H 동기.
3. verify 커널 내부 SM occupancy·DRAM 대역 — Nsight Compute(ncu) 샘플 (b) 판별용.
4. GPU util을 nvidia-smi(coarse)와 timeline-derived(정밀)로 교차.

### 5.4 결정 트리 (gate)
- **gap ≥ verify의 ~30% 이고 host-bound 확인 (a)** → A1/B 회수 가능 = **본론 확정** (CPU draft + host-path 엔진).
- **gap 작고 verify 내부 memory-bound (b)** → CPU로 불가 → **A2(KV tiering)로 전환**해 배치 확대로 util 회복.
- **배치 미충전 (c)** → concurrency↑ 실험 + A2 우선.

### 5.5 산출물
`SUB_201_cpu_host_path_bottleneck/profile/` 에 Nsight rep + 분해 표 + 결정 verdict.
이 verdict가 §8 논문 골격·§9 TSK_043·child task의 입력.

### 5.6 한계·위험 (정직)
- B200 컨테이너 내 Nsight/ncu 권한 필요 (CAP_SYS_ADMIN). 불가 시 CUDA event 수동 계측 + py-spy로 대체.
- step 분해가 워크로드 의존적일 수 있음 → 최소 2모델(저/고 util) 비교 필수.

---

## 6. 부차 타깃 — KV tiering (유휴 2TB DRAM)

- B200 HBM 1.5TB는 단일 모델엔 충분하나 **multi-tenant / ultra-long context(≥128k)**에서 한계.
- 유휴 DRAM 2TB + (가능 시 CXL)로 cold KV tiering → 더 많은 in-flight 시퀀스 → GPU gap 충전(=util 회복).
- §5에서 병목이 (c)로 판명되면 A2가 1순위로 승격.
- 기존: IDE_017(dma_zero_copy), SUB_058(CPU radix prefix KV).

---

## 7. 폐기 목록 (thesis 재정립에 따른)

| 폐기 대상 | 근거 |
|---|---|
| **AGSD per-request 워크로드 게이팅을 1번 기여로 보는 framing** | TSK_042: method 최적값의 지배 축은 워크로드가 아니라 **모델 계열**. 단일 모델 배포 시 per-request 워크로드 게이팅 이득은 Llama·DeepSeek 계열 0, Qwen 중형만 ~10–22%. |
| **IDE_018 phase-burst / task-pool** | C1 (매트멀 오프로드, H2D/D2H 패배, stub) |
| **branchy busywork harvest (SUB_196 cellB 등)** | C2 (유용 작업 아님, 미검증, work-pattern sensitive) |
| **IDE_023 5-axis "CPU slack harvesting"의 busywork 성격 부분** | harvest가 *유용한 host-path 작업*이 아니면 폐기. A1/A2/B로 재흡수. CPU 하베스팅은 논문에서 **future work로 강등**(사용자 결정). |

---

## 8. 논문 골격 함의 (새 thesis)

> **새 thesis 후보: "Speculative decoding shifts the LLM-serving bottleneck from GPU
> memory bandwidth to the host path; the idle CPU on GPU servers is precisely the
> resource to relieve it — reclaiming the GPU slack that spec-decode creates."**

- 동기: spec-decode가 지배 lever이나 GPU를 un-saturate (§1, §2).
- 문제: 회수 대상은 host-path 병목 + 유휴 CPU/DRAM.
- 방법: CPU draft + host-path 오프로드(A1/B), KV tiering(A2).
- 결과: §5 프로파일 → 회수 실측. CPU slack "busywork" harvesting은 future work.
- llm-d 카드는 유지(복잡 라우터의 한계 이득 18%) — 어느 thesis든 보조 근거.

§5 verdict 확정 후 paper §4~§6·§8 재작성. (지금 논문은 spec-decode 우월성까지는 정합, CPU 역할 서사만 재정립 필요.)

---

## 9. TSK_043 재정렬 함의

- 기존 TSK_043 = AGSD 분류기(C0~C3) decision-regret. 사용자 결정 = **모델계열+워크로드 hybrid gate**.
- SUB_201 관점에서 TSK_043의 가치: gate는 **모델별 oracle 프로파일 1차 + (Qwen 중형) 워크로드 2차**. regret = hybrid-oracle 대비.
- 단, gate 자체는 throughput 지배 lever가 아님(§7). TSK_043은 "회귀 회피 안전장치"로 위상 축소, **본론은 §5의 host-path 회수**.

---

## 10. 다음 단계 (child / follow-up 제안)

1. **(즉시·필수)** §5 step 분해 프로파일 — Qwen-7B suffix + Llama-70B suffix. → verdict.
   - **XL 추가**: **R1 671B suffix** 도 분해 대상 (suffix 음수 gain -49% 의 원인이 host overhead 인지 verify 내부인지)
2. (a 판명 시) A1 CPU 드래프팅 real spec_decode 통합 PoC (IDE_019 자산 재활용, vllm/v1/spec_decode/ cpu proposer).
   - **primary target = R1 671B + DS-Llama-70B** (α<0.5 모델, vanilla-win 셀들의 회복 ROI 큼)
3. (a 판명 시) B1 detok + B2 constrained-decode 오프로드 PoC.
   - **primary target = Llama-8B / DS-Qwen-7B** (mix throughput 24~27k tps, wall slack 37%)
4. (c 판명 시) A2 KV tiering + concurrency sweep.
   - **primary target = 405B** (KV cache 큰 dense), R1 은 MLA 라 후순위
5. §5 verdict 기반 논문 골격 재작성 (별도 작업).

---

## 부록 — 데이터 출처

- throughput/util/latency/α: `features/IDE_022_agsd_realistic_eval/TSK_042_realistic_workload_oracle/RESULTS.md`
- raw: `vllm_config_perf/gating/realistic_eval/runs/tput_t1t3_20260602/` (per_request_raw.jsonl, summ_*.json ×140 v/s + 5 ngram + cells/, metrics_table.parquet)
- llm-d: `vllm_config_perf/gating/realistic_eval/runs/routing_llmd_20260603/` (summ_*.json ×56 conc=32 + ×21 concurrency sweep)
- 통합: `runs/routing_combined/` (3-routing 통합 산출물)
- 커버리지 (XL 매트릭스 갱신, 2026-06-04): **222셀 총합** = vanilla 70 + suffix 70 + ngram 5 (Qwen-7B baseline) + llm-d 56 + llm-d sweep 21. **405B + 671B llm-d 는 미측정** (별도 다른 세션에서 K8s 구성 후 진행 예정).
- mix=별도 500샘플(전체 2,159 아님), eagle 미측정.
- 관련 ID: IDE_019(CPU draft), IDE_017(KV/DMA), IDE_016(detok/AVX-512), SUB_187(AMX draft feasible), IDE_018/SUB_196(폐기 대상).

---

## 부록 B — XL 매트릭스 (2026-06-04 추가) 의 핵심 변경

| 항목 | 기존 (8모델 173셀) | XL 갱신 (10모델 222셀) |
|---|---|---|
| 매트릭스 1위 (mix tps) | DS-Qwen-7B suffix 24,458 | **Llama-8B suffix 27,851** (XL 추가 405B/671B 가 직접 신기록은 아님) |
| α 임계치 evidence | 일반론 | **vanilla>suffix 15셀의 α median 0.37, positive 55셀의 median 0.68** → **α≈0.5 break-even 정량 확정** |
| 최대 모델 GPU 거동 | 70B 까지 | 405B (suffix GPU 93.1%, ΔGPU +5.8) + 671B MoE (suffix GPU 93.7%, ΔGPU +3.9) — **사이즈 단조 host-bound 약화** 확정 |
| worst suffix match | DS-Llama-70B (6/7 셀 음수) | **R1 671B (7/7 셀 음수, -49%, α 0.46)** — A1 CPU drafting 의 primary target 확정 |
| thesis caveat | "spec-decode → host-bound" 일반 적용 가정 | **"작은 모델 / 높은 α 모델군에 강하게 적용", 큰 모델·낮은 α 는 별도 lever (A1)** 필요 |
