# K — SUB_015 F1~F6 통합 improvement roadmap

> 2026-05-18 KST. branch `feat/neo-amx-apply` HEAD `7200f41f1`.
>
> **통합 분석 문서 (read-only). 코드 변경 0**. `analysis/J_sub015_root_cause_analysis.md` 의 6 개 개선 후보 (F1~F6) 의 interaction + cumulative win + sequential phasing + risk mitigation 통합 plan.

---

## 1. 6 후보 개요 (J 문서 참조)

| ID | 명칭 | 차원 | Effort (단독) | 단독 win | 변경 영역 |
|---|---|---|---:|---:|---|
| F1 | async pipeline depth > 0 | 분산 | 3-5 일 | +5-15% | `attention.py:1233-1238` + race-safe deque |
| F2 | b0/b1 size 균형 | 분산 | 2-3 일 | +3-7% | `neo_scheduler_adapter.py:1078-1098` |
| F3 | K cache BF16 host store | 코드+HPC | 1-2 일 | +1-3% | swap path + dtype |
| F4 | TP=8 → TP=4 (M=8→16) | 코드+HPC | 2-4 주 | +5-10% | model parallelism |
| F5 | BLOCK_SIZE 16 → 32 | 코드+HPC | 2-4 주 | +3-7% | paging granularity |
| F6 | OMP dynamic schedule + nowait | 분산 | 1-2 일 | +1-3% | `core.h:308-363` |

---

## 2. Interaction matrix — 누적 시 시너지 / 충돌

각 cell = (i, j) 의 cumulative effect 평가:
- **+** = 시너지 (개별 win 의 단순 합 초과)
- **=** = 독립 (단순 합)
- **−** = 충돌 (단순 합 미만, 한 lever 의 win 이 다른 lever 의 base 를 깎음)
- **×** = mutually exclusive (동시 적용 불가)

|   | F1 async | F2 b0/b1 | F3 K BF16 | F4 TP=4 | F5 BLOCK=32 | F6 OMP dyn |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| **F1** | — | **+** | = | **+** | **+** | = |
| **F2** | **+** | — | = | **+** | = | = |
| **F3** | = | = | — | = | = | = |
| **F4** | **+** | **+** | = | — | = | **+** |
| **F5** | **+** | = | = | = | — | **+** |
| **F6** | = | = | = | **+** | **+** | — |

### 2.1 시너지 (+) 8 쌍

#### (F1, F2) — async pipeline + b0/b1 균형 ★
- F1 의 layer-level overlap 이 F2 의 step-level GPU↔CPU overlap 와 결합 → **double pipelining**
- 단독 F1 (+5-15%) + 단독 F2 (+3-7%) = 8-22% 단순 합
- 시너지 후: **+12-25%** (layer × step 의 cross-dimension overlap)

#### (F1, F4) — async pipeline + TP=4
- F4 의 M=16 (AMX tile 100% occupancy) win 이 F1 의 pipeline depth 안에서 누적 가속
- TP=4 의 cdec 영역 share 가 TP=8 대비 더 큼 → F1 win 의 base 증폭

#### (F1, F5) — async pipeline + BLOCK=32
- F5 의 N=32 (1 dpbf16ps 의 work 2× ↑) 가 cdec wall 줄이고 F1 의 pipelining 효율 ↑

#### (F2, F4) — b0/b1 균형 + TP=4
- F4 의 b1 (CPU) wall 단축이 F2 의 b0 enlargement 와 결합 → GPU idle window 거의 0

#### (F4, F6), (F5, F6) — TP=4 / BLOCK=32 + OMP dynamic
- F4/F5 의 더 큰 matmul 에서 OMP task 의 work imbalance 가 커짐 → F6 의 dynamic schedule 효과 ↑

### 2.2 독립 (=) 7 쌍

- F3 (K BF16 host store) 는 다른 lever 와 영역 분리 — 변환 cost 제거만 (모든 lever 와 독립)
- F1 ↔ F3, F2 ↔ F3, F4 ↔ F3 등 모두 독립
- F2 ↔ F5, F2 ↔ F6 — scheduler / OMP 영역 분리

### 2.3 충돌 / mutually exclusive 0 쌍

★ **본 6 후보는 모두 compatible**. 동시 적용 불가능한 lever 없음.

---

## 3. Cumulative win 정량 estimate

### 3.1 단순 합 (additive) — worst-case 추정

```
F1 (+10%) + F2 (+5%) + F3 (+2%) + F6 (+2%) = +19% (cheap subset)
  + F4 (+7%) + F5 (+5%) = +31% (full)
```

### 3.2 시너지 반영 — best-case 추정

(F1, F2) +25% 시너지 + F3 (+2%) + F6 (+2%) = **+29%** (cheap subset)
  + (F4, F1, F5, F6) 시너지 → **+45-55%** (full architectural)

### 3.3 Amdahl 한계 정량

현재 cdec_wait = 8.75 ms / layer (Phase 1 측정), pacpu 비율 ≈ 70% wall:
- Theoretical maximum throughput speedup at cdec = 0 (∞ accel) = 1/(1-0.7) = **3.33×**
- 실제 적용 가능 범위 = max ~2.0× (cdec wall 의 80% 줄이기 = pipeline depth 4 + AMX-equiv)

### 3.4 보수적 추정 (실측 noise 반영)

| 단계 | 적용 lever | 단독 합 | 시너지 후 | 보수 추정 (50% 적용) |
|---|---|---:|---:|---:|
| **Step A** (1주일) | F3 + F6 | +4% | +4% | **+2%** |
| **Step B** (+1주일) | + F1 | +14% | +18% | **+9%** |
| **Step C** (+1주일) | + F2 | +20% | +28% | **+14%** |
| **Step D** (+5주일) | + F5 | +25% | +35% | **+18%** |
| **Step E** (+8주일) | + F4 (full) | +35% | +50% | **+25%** |

→ 보수 추정 net throughput: 현재 2,238.6 tps × 1.25 = **2,798 tps** (Step E full applied).

---

## 4. Sequential phasing plan

### Phase α (1 주일, +2%) — Quick wins

**적용 lever**: F3 + F6
- F3 (K cache BF16 host store): 1-2 일, +1-3%
- F6 (OMP dynamic schedule + nowait): 1-2 일, +1-3%

**핵심**: 정확도 risk 0~낮음. quick + safe. AMX path 의 base 강화 (F3) + libgomp barrier wait 줄임 (F6).

**측정**: 각 lever 의 500p × 8192 × 3-run avg. CV < 3% 확인.

**Gate**: F3+F6 cumulative +2-5% 시 Phase β 진입.

### Phase β (+1 주일, +9% cumulative) — Distributed pipelining ★

**적용 lever**: + F1 async pipeline depth > 0
- depth=1 부터 시작 (50% overlap), 실측 측정 후 depth=2/4 확장
- race safety: cdec_future result 의 lifecycle + GPU output tensor write 충돌 검증
- 정확도 검증: TST_003 verdict (분포·의도 유사성)

**핵심**: 가장 high-ROI 영역. depth=1 으로 시작 후 점진 확장.

**Gate**: F1 cumulative +5-12% 시 Phase γ 진입.

### Phase γ (+1 주일, +14% cumulative) — Scheduler tuning

**적용 lever**: + F2 b0/b1 size 균형
- b0 (GPU) batch size 의 enlargement (4% → 10-15%)
- chain dispatch logic 의 b1 scope 재계산
- 단 CPU 가속 effect 감소 trade-off 고려

**Gate**: F1+F2 cumulative +12-22% 시 Phase δ 진입 결정.

### Phase δ (+5 주일, +18% cumulative) — Block layout 변경

**적용 lever**: + F5 BLOCK_SIZE 16 → 32
- NEO scheduler 의 paging granularity 변경
- KV cache memory 분배 변경 (block 수 절반, 각 block 2× size)
- prefill 영역 영향 검증

**Risk**: NEO core design 변경. 중간 규모 redesign.

**Gate**: F1+F2+F5 cumulative +15-25% 시 Phase ε 진입 결정.

### Phase ε (+8 주일, +25% cumulative) — Model parallelism 변경 (long-haul)

**적용 lever**: + F4 TP=8 → TP=4
- 모델 설정 변경 (model parallelism)
- NUM_Q_HEADS = 64/4 = 16 (AMX tile full occupancy)
- KV cache memory 2× per worker
- GPU 측 parallelism 감소 — prefill throughput trade
- 별도 architecture-level 검증 필요

**Risk**: 큰 architectural change. 전체 system 영향.

---

## 5. Risk mitigation

### 5.1 정확도 risk (F1, F3)

| Lever | Risk | Mitigation |
|---|---|---|
| F1 async pipeline | result deferred 시 cdec output 이 GPU compute 와 race 가능 | TST_003 verdict (분포·의도) 매 phase 종료 시 검증. token-level bit-exact 아님, 분포 유사성 binding. |
| F3 K BF16 host store | BF16 mantissa 7 vs FP16 10 = 3 bit precision drop | per-token logprob max diff < 1e-3 검증. 누적 cumulative drift 측정. |

### 5.2 Race safety (F1, F6)

| Lever | Risk | Mitigation |
|---|---|---|
| F1 async pipeline | pending future deque 의 producer/consumer race | thread-safe deque (`std::mutex` 또는 lock-free queue). cdec_future 의 lifecycle 관리 명시 (submit → pending → consumed). |
| F6 OMP dynamic | Step 0/2 partition 변경 시 store_kv ↔ attn race | `#pragma omp for schedule(dynamic) nowait` 의 implicit barrier 안 보장 — explicit barrier 유지. |

### 5.3 Performance regression risk (F2, F4, F5)

| Lever | Risk | Mitigation |
|---|---|---|
| F2 b0/b1 균형 | CPU 가속 effect 감소 — b1 scope 너무 작으면 net loss | b0 batch size 의 sweep 측정 (4% → 8/10/15%). best 점 결정. |
| F4 TP=4 | GPU parallelism 감소로 prefill ↓ | 500p × 8192 의 prefill_tps 측정 + output_tps 비교. net throughput 확인. |
| F5 BLOCK=32 | NEO BlockManager 의 paging logic 영향 | 500p × 8192 의 swap-in/out rate 측정. CPU pool full 빈도 확인. |

### 5.4 Build / dependency risk (F4, F5)

| Lever | Risk | Mitigation |
|---|---|---|
| F4 TP=4 | model config 변경 — vllm 의 weight loading + KV cache layout | feat branch 별도 분리. main 영향 0. |
| F5 BLOCK=32 | dtype.h `BLOCK_SIZE` macro 변경 → pacpu rebuild. NEO core 의 모든 block_table indexing 검증 | unit test (block 처리 정확성) 추가. |

---

## 6. Decision framework — 어떤 phase 까지 갈 것인가

### 6.1 ROI 기반 단계 결정

| Phase | 누적 effort | 누적 win (보수) | Marginal ROI |
|---|---:|---:|---:|
| α (F3+F6) | 1 주 | +2% | 2%/주 |
| β (+F1) | 2 주 | +9% | **+7%/주 ★** |
| γ (+F2) | 3 주 | +14% | +5%/주 |
| δ (+F5) | 8 주 | +18% | +0.8%/주 |
| ε (+F4) | 16 주 | +25% | +0.9%/주 |

★ **Phase β (F1 추가) 가 가장 high marginal ROI** — 1 주일 effort 로 +7% (cumulative +9%).

### 6.2 권고 stopping point

**보수 권고**: Phase γ 종료 (3 주, +14% cumulative)
- F3 + F6 + F1 + F2 적용
- 모두 분산 + cheap HPC 영역 — 정확도 risk 낮음
- architectural change (F4/F5) 없음 — 본 task 안전

**ambitious 권고**: Phase δ 종료 (8 주, +18% cumulative)
- + F5 BLOCK=32 — NEO core 중간 변경
- 5 주 추가 effort, +4% marginal

**maximal 권고**: Phase ε 종료 (16 주, +25% cumulative)
- + F4 TP=4 — model 전체 변경
- 8 주 추가 effort, +7% marginal

### 6.3 ★ 권고: **Phase γ 까지 (3 주, +14%)**

이유:
1. Phase β (F1) 의 marginal ROI 가 압도적 (+7%/주)
2. Phase γ (F2) 도 medium ROI (+5%/주)
3. Phase δ/ε (F5/F4) 의 marginal ROI 가 급격 ↓ (+0.8-0.9%/주) — architecture redesign 의 long-haul cost
4. Phase γ 까지 = 분산 dim + cheap HPC 영역. SUB_015 의 진정한 root (분산 차원) 해결.

---

## 7. Sequential implementation roadmap (3 주 plan)

### Week 1: Phase α (F3 + F6)

#### Day 1-2: F3 K cache BF16 host store
- `vllm/v1/core/sched/neo_cpu_kv_buffer.py` — host buffer dtype 변경 (FP16 → BF16)
- swap-out path: GPU FP16 → host BF16 변환
- `csrc/cpu/pacpu/amx_kernel.cpp` — vec K conv 의 FP16→BF16 변환 제거 (이미 BF16)
- `csrc/cpu/pacpu/pacpu.ispc` — K_cache read 의 dtype 변경 (별도 path 또는 unified BF16)
- 빌드 + 500p × 8192 × 3-run avg

#### Day 3: F6 OMP dynamic schedule
- `csrc/cpu/pacpu/core.h:331` — Step 1 의 task loop 를 `#pragma omp for schedule(dynamic, 1)`
- Step 0/2 의 partition 도 task partition 으로 통일
- 빌드 + 500p × 8192 × 3-run avg

#### Day 4-5: 통합 측정 + 정확도 검증
- F3+F6 통합 500p × 8192 × 3-run avg
- TST_003 verdict (분포 유사성)
- Gate: cumulative +2% 시 Phase β 진입

### Week 2: Phase β (F1 async pipeline depth)

#### Day 1-2: F1 minimal viable (depth=1)
- `vllm/model_executor/layers/attention/attention.py:1233-1238` — `_neo_pending_cdec_queue` size 1 → 2
- `_neo_async_cdec_mode = True` env-toggle
- result wait 의 deferred mechanism — layer N submit → layer N+1 시작 시 layer N result use
- 빌드 + short test (100p × 1024)

#### Day 3: depth=1 정확도 검증
- TST_003 verdict + token-level logprob diff
- race safety 검증 (cdec_future lifecycle)
- 500p × 8192 × 3-run avg

#### Day 4-5: depth=2/4 sweep
- depth=2, depth=4 측정
- best depth 결정 (보통 2-4 가 sweet spot)
- Gate: F1 cumulative +5-12% 시 Phase γ 진입

### Week 3: Phase γ (F2 b0/b1 균형)

#### Day 1-2: F2 b0 batch size sweep
- `vllm/v1/core/sched/neo_scheduler_adapter.py:1078-1098` — b0 size 의 env-tunable
- VLLM_NEO_B0_BATCH_RATIO sweep (0.04 → 0.08, 0.10, 0.15)
- 500p × 8192 × 3-run avg

#### Day 3: best b0 ratio 결정 + Phase γ 통합 측정
- F3 + F6 + F1 + F2 통합 측정
- 500p × 8192 × 3-run avg

#### Day 4-5: 정확도 + 회귀 검증
- TST_003 verdict
- 22 strict items 측정
- 최종 best configuration 정리 + README + measurements 저장
- commit & push (사용자 명시 후)

---

## 8. Phase 별 측정 plan

### 측정 sequence (각 phase 종료 시)

| Phase | 측정 | Workload | Run 수 | Gate |
|---|---|---|---:|---|
| α 후 (F3+F6) | 500p × 8192 | 표준 | 3-run avg | cumulative +2% |
| β 후 (+F1, depth=1) | 100p × 1024 short | quick | 1-run | functional + 정확도 |
| β 후 (+F1, depth sweep) | 500p × 8192 | 표준 | 3-run avg | 최적 depth 확정 |
| γ 후 (+F2) | 500p × 8192 | 표준 | 3-run avg | cumulative +12-22% |

### 비교 baseline

- **S1-S9 baseline** (3-run avg 2,238.6) — 본 task 의 absolute baseline
- **Phase α / β / γ 의 각 cumulative** — incremental win 추적

### 정확도 verification

- **TST_003 verdict** (분포·의도 유사성) 매 phase 종료 시
- per-token logprob max abs diff < 1e-3
- 시퀀스 PPL relative diff < 0.5%
- 22 strict items 통과 ≥ 19/19

---

## 9. 핵심 차이점 (J 와 K 의 분리)

| 측면 | J (root cause) | K (roadmap, 본 문서) |
|---|---|---|
| 목적 | 왜 실패했나 진단 | 어떻게 고치나 plan |
| 단위 | 6 후보 ranking | 6 후보 통합 phasing |
| 정량 | 단독 win 추정 | cumulative win + 시너지 + Amdahl |
| 실행 | (분석만) | sequential roadmap (week 1/2/3) |
| 결정 | 어느 lever 진행? | 어느 phase 까지 진행? |

### Decision 권고

**Phase γ 종료 (3 주, +14% cumulative, 2,238 → ~2,553 tps 추정)** 가 best ROI sweet spot.

Architectural change (F4 TP=4, F5 BLOCK=32) 은 long-haul, marginal ROI 급격 ↓ — 본 task 범위 밖. 별도 task 분리 권고.

---

## 10. 관련 문서 cross-reference

| 문서 | 역할 |
|---|---|
| `analysis/H_static_analysis.md` | HPC dim 의 정량 fact (FLOPs/AI/Roofline) |
| `analysis/H_dynamic_analysis.md` | HPC dim 의 perf dso 분포 (libgomp 43.75%) |
| `analysis/H_phase1_final_levers.md` | Phase 1 lever 정정 |
| `analysis/H_phase2_results.md` | Phase 2/3 측정 sweep 결과 |
| `analysis/I_amx_proper_design.md` | HPC dim 의 7 strategy ranking |
| `analysis/J_sub015_root_cause_analysis.md` | **3 차원 root cause 분석** |
| `analysis/K_sub015_improvement_roadmap.md` | **본 문서 — F1~F6 통합 roadmap** |

---

## 11. 결론

1. **F1~F6 모두 compatible** (mutually exclusive 0 쌍, 시너지 8 쌍)
2. **F1 (async pipeline depth) 의 marginal ROI 가 압도적** (+7%/주)
3. **Phase γ (F3+F6+F1+F2, 3 주, +14% cumulative)** 가 best ROI sweet spot
4. **F4/F5 architectural change** 는 long-haul, marginal ROI 급격 ↓
5. 본 분석은 plan only — 실제 구현은 별도 task, 사용자 명시 후 진행
