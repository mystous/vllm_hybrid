# L — SUB_015 evidence-based 우선순위 재책정 + 진행 추천

> 2026-05-18 KST. branch `feat/neo-amx-apply` HEAD `7200f41f1`.
>
> **K 문서의 win % 가 추정/가설 임을 인정 후 작성**. 본 문서는 **실측 backing 있는 fact 만으로** 우선순위 재책정 + 정직한 진행 추천. 코드 변경 0.

---

## 0. K 문서의 한계 (정직한 인정)

K 문서의 모든 win % (+5-15%, +25% 등) + 시너지 multiplier + Phase 별 cumulative 는 **실측 backing 없는 가설**. 일반적 HPC engineering ballpark estimate 일 뿐.

본 SUB_015-Phase 3 의 6 step 측정 자체가 다음을 입증:
- 단일 lever 의 **이론 win 추정과 실측 결과의 갭이 매우 큼**
- 1-run noise CV 3.23% — 단일 측정도 신뢰성 낮음
- 실측 전에는 어떤 lever 가 진정한 win 인지 모름

→ 진정한 우선순위 = **실측 backing 강도** 기반.

---

## 1. 실측 backing fact 정리 (출처 명시)

### Tier A — 강한 backing (정량 + 측정 dir)

| Fact | 값 | 출처 |
|---|---:|---|
| libgomp share | **43.75%** | `H_dynamic_analysis.md` — perf 60s, 413K samples, dso 분포 |
| libpacpu share | 26.38% (qk 8.75 + softmax 9.73 + av 7.90) | 동일 |
| libtorch_cpu share | 10.24% (index_put 4.85 + AVX2 copy 3.27 + index 2.12) | 동일 |
| cdec_wait | 8.75 ms / layer | Phase 1 측정 (VLLM_NEO_PROFILE=1 로그) |
| max_workers=4 시 회귀 | **-8%** | SUB_030 측정 |
| Step 5 AMX (B+A+vec K) 3-run avg | 2,186.1 (**-2.35%** vs S1-S9) | `sub015_p3_step5_amx_bav_500p_3run_20260518/` |
| Phase 3 A dropin AMX 3-run avg | 2,142.5 (**-4.3%**) | `sub015_p3_amx_500p_3run_20260518/` |
| OMP_NUM_THREADS sweep | 5=-5%, 8=-2%, **10=best**, 14=timeout, 16=init fail | `sub015_p2_omp*` Phase 2 G-2 |
| K_TILE_WIDTH sweep | **2=best**, 4=-1.6%, 8=-0.5% | `sub015_p2_c_k*` Phase 2 C-1 |
| 1-run CV | **3.23%** (Step 5 3-run std/avg) | Step 5 measurement |
| qk AI | 7.11 FLOP/byte | `H_static_analysis.md` — 정적 계산 |

### Tier B — 약한 backing (정성/이론 only)

| Fact | 출처 |
|---|---|
| AMX setup ~650-900 cycle vs ISPC ~400-500 | instruction count × per-op latency ballpark — 실측 cycle 측정 아님 |
| AMX tile occupancy 19% (M=8/16 × 3/8 tile) | 정적 계산 |
| Roofline ceiling 355 GFLOP/sec (L2-bound) | AI × L2 BW 추정 |
| OMP fork-join 184K sync/sec | step rate × 80 layer × 4 sync × 8 worker 산술 |

### Tier C — 가설 (실측 backing 0)

| 가설 | 출처 |
|---|---|
| **F1 async pipeline depth +5-15%** | pipelining theory 의 ballpark — 실측 안 함 |
| F2 b0/b1 균형 +3-7% | GPU idle 1 ms / step 시간 추정 — b0/b1 wall 정확 측정 없음 |
| F3 K BF16 host store +1-3% | setup cycle 제거 추정 — 누적 wall 영향 모름 |
| F4 TP=4 +5-10% | tile occupancy 추정 — GPU 측 영향 미고려 |
| F5 BLOCK=32 +3-7% | tile N occupancy 추정 |
| F6 OMP dynamic +1-3% | barrier wait 감소 추정 |
| 시너지 multiplier (+25% 등) | 상상 |

---

## 2. Evidence-based priority 재책정

### 2.1 ranking 기준 (재정의)

K 문서: win % (추정) × ROI
**L 문서**: (a) **fact backing 강도** + (b) **mechanism 명확성** + (c) **실측 비용**

### 2.2 새 ranking

| 순위 | 후보 | Tier A backing | Mechanism 명확성 | 실측 cost | 종합 |
|---|---|:-:|:-:|:-:|---|
| **P1** | **분산 dim 의 실측 측정** (F1 short test, b0/b1 wall breakdown) | — | ★★★ | 매우 낮음 | **시작점** — 가설 검증 |
| P2 | F6 OMP dynamic schedule | **부분 ✓** (libgomp 43.75%, Step 4 회귀 fact) | ★★ | 낮음 | quick test 가능 |
| P3 | F3 K BF16 host store | **부분 ✓** (Step 5 vec K conv 의 +0.4% 1-run win) | ★★ | 중 (swap path 변경) |
| P4 | F1 async pipeline depth | ✗ (가설) | ★★ | 중 (race safety) | **실측 우선** |
| P5 | F2 b0/b1 균형 | ✗ (가설) | ★ | 중 (scheduler) |
| P6 | F5 BLOCK=32 | ✗ (가설) | ★ | 고 (NEO core) |
| P7 | F4 TP=4 | ✗ (가설) | ★ | 매우 고 (model) |

**P2 (F6)** 가 K 문서보다 우선순위 ↑ — **libgomp 43.75% 실측** + Step 4 회귀 사실 backing.
**P4 (F1)** 가 K 의 #1 위치에서 4 위로 강등 — 가설만 있음. 실측 우선 필요.
**P6/P7 (F5/F4)** 는 가설 + 큰 effort + 불확실 — 별도 long-haul.

---

## 3. Tier A backing 의 진정한 mechanism 분석

### 3.1 libgomp 43.75% — 진정한 root

본 fact 가 가장 강한 lever. 그러나 진정한 mechanism:

- 매 layer call 의 fork-join 빈도 = 80 layer × 72 step/sec × 8 worker = **46,080 call/sec/system**
- 각 call 의 4 sync (fork + barrier #1 + barrier #2 + join) = 184,320 sync/sec
- libgomp 의 spin wait (pause loop, Phase 1 의 disassembly 확인) = busy-wait

**축소 가능 영역**:
- (a) fork-join 빈도 자체 ↓ → layer batched call (semantic 불가, D 확인)
- (b) sync 횟수 ↓ → barrier #1 or #2 제거 (semantic 불가 — store_kv ↔ attn race, attn ↔ gather race)
- (c) **spin wait time ↓** → KMP_BLOCKTIME 또는 OMP_WAIT_POLICY 변경 ★ 시도 가치 있음
- (d) thread imbalance ↓ → F6 (OMP dynamic schedule)

→ **(c) KMP_BLOCKTIME sweep** = 가장 cheap evidence-based 시도. Phase 3.2 (KMP=0) 측정 fact 이미 있음 — 기존 KMP=200 best (Phase 3.1+KMP=200 의 측정).

→ **(d) F6** 가 다음 cheap. Step 4 의 회귀는 5-tile cfg overhead 가 dominant 였음 — F6 단독 (dynamic schedule 만, 3-tile cfg 유지) 의 실측 안 함.

### 3.2 cdec_wait 8.75 ms/layer — Amdahl 한계

- 80 layer × 8.75 ms = 700 ms / step (worst-case, 모든 layer cdec firing 시)
- step rate 72/sec → 본 영역 의 wall time fraction = 70% (8 worker × 700 ms / step / wall) ≈ 추정
- **cdec wall 의 정확한 fraction 측정 필요**

→ 측정 plan: 한 step 의 cdec wall breakdown 정밀 측정. P1 의 일부.

### 3.3 max_workers=2 best — 분산 contention 의 실측

- max_workers=4 시 -8% 회귀 — 단일 측정 fact (SUB_030)
- 본 결과의 진정한 mechanism = GIL contention 또는 Python overhead (가설)

→ 실측 backing 강함. cap 자체는 best, 단 cap 의 mechanism (왜 4 가 안 됨) 분석 필요.

### 3.4 Step 5 -2.35% — AMX 한계 확정

- 3-run avg 의 노이즈 control 후도 net loss
- AMX path 의 모든 cheap variant 시도 후도 baseline 초과 못함
- **AMX 우선순위 ↓** — 추가 변형 (F3) 의 marginal win 도 작음 예상

---

## 4. 진행 추천 (fact-based)

### 4.1 ★ Phase 0 — 측정 only (1-2 일, 코드 변경 0)

**목적**: F1~F6 의 가설 win 의 일부를 **실측 backing 으로 전환**.

**측정 1**: cdec wall fraction 정밀
- workload: 100p × 1024 short (이미 baseline 측정 있음)
- VLLM_NEO_PROFILE=1 에서 cdec_wait_avg, cdec_count, gpu_count 의 step 별 분포
- 산출: 한 step 의 wall 안에서 cdec wall fraction 정량 (현재 추정 70% → 실측 확정)

**측정 2**: b0/b1 wall breakdown
- attention.py:1118 의 `_st["b1_count_sum"] += (cdec_t1 - cdec_t0)` 로그 활성
- 산출: b0 (GPU) wall vs b1 (CPU) wall ratio — F2 의 가설 backing 확인

**측정 3**: KMP_BLOCKTIME sweep 재측정
- KMP_BLOCKTIME ∈ {0, 10, 50, 200, 1000} short 측정
- libgomp share 변화 측정 (perf 30s × 각 setting)
- Phase 3.2 의 단발 측정 (KMP=0) 결과 보완

**측정 4** (선택): GIL contention 정량
- py-spy native unwind 으로 GIL hold time 측정
- TP=8 worker 의 cdec dispatch 의 GIL hold

**Effort**: 1-2 일 (모두 측정만, 코드 변경 0)
**Output**: 진정한 fact backing 으로 F1~F6 의 priority 재확정.

### 4.2 Phase 1 (선택, 측정 결과 dependent) — F6 OMP dynamic schedule 단독 시도

**근거**: Phase 2 의 Step 4 회귀 (5-tile cfg) 는 OMP dynamic 단독이 아닌 통합 시도. F6 단독 (3-tile cfg + `#pragma omp for schedule(dynamic, 1)`) 의 실측 없음.

**변경**: `core.h:331` 의 task loop 1 줄 변경 + barrier #1/#2 유지.

**Effort**: 0.5 일 (변경 + 빌드) + 측정 (100p short 1-run + 500p × 3-run)
**Risk**: 매우 낮음 (race 없음, 단순 schedule 변경)
**Expected**: Phase 2 의 Step 4 의 회귀 가 dynamic schedule 의 atomic counter overhead 인지 OR 5-tile cfg overhead 인지 분리.

### 4.3 Phase 2 (선택) — F1 functional short test

**근거**: K 문서의 F1 +5-15% 가설을 실측으로 검증.

**변경**: `attention.py:1233-1238` 의 `_neo_pending_cdec_queue` size 1 → 2, `_neo_async_cdec_mode = True` (env-toggle).

**Effort**: 1-2 일 (구현 + race safety + 정확도 검증)
**측정**: 100p × 1024 short 1-run — functional + tps + 정확도 verification

**Stopping criteria**: 만약 functional fail (정확도 손상 또는 race) 시 즉시 revert.

### 4.4 Phase 3 (선택) — Phase 0~2 결과 기반 다음 단계 결정

**측정 결과 dependent**:
- Phase 0 의 b0/b1 wall, cdec fraction, KMP sweep 결과
- Phase 1 의 F6 win/loss
- Phase 2 의 F1 functional + tps fact

이후 fact-based 진정한 priority 재확정. F2/F3/F4/F5 는 본 시점에서 가설만 — Phase 0 결과 보고 결정.

---

## 5. 권고 stopping point

### 5.1 Conservative 권고 (1 주, 측정 + cheap variant only)

| Step | 작업 | Effort |
|---|---|---:|
| 1 | Phase 0 측정 sweep (cdec fraction + b0/b1 + KMP) | 1-2 일 |
| 2 | F6 OMP dynamic schedule 단독 + 500p × 3-run | 1 일 |
| 3 | F3 K BF16 host store (cheap variant, no swap path 변경) + 측정 | 1-2 일 |
| **합계** | **3-5 일** | |

**Win**: 측정 + cheap quick wins only. fact backing 강함.

### 5.2 Aggressive 권고 (2-3 주)

위 + F1 functional test + 500p × 3-run + F2 b0 sweep.

**Win**: 가설 검증 + 진정한 high-ROI lever 확정.

### 5.3 ★ 본 분석의 권고

**Phase 0 (측정 only)** 부터 시작. 가설을 fact 로 전환 후 다음 단계 결정.

이유:
1. K 문서의 모든 ranking 이 가설 기반 — 실측 없이 implementation 진행은 도박
2. Phase 0 는 코드 변경 0, risk 0, effort 1-2 일
3. 측정 결과로 F1~F6 의 진정한 priority 재확정 가능
4. SUB_015-Phase 3 의 6 step 시도가 입증 — 추정과 실측 갭이 크다

**Phase 0 후의 진행 = 측정 결과 dependent**.

---

## 6. 진행 추천 요약 (3 단계)

### Step 1 — Phase 0 측정 sweep (1-2 일, 코드 변경 0)

| 측정 | 방법 | 산출 |
|---|---|---|
| cdec wall fraction | VLLM_NEO_PROFILE=1 의 cdec_wait_avg + step breakdown | 70% 가설 정량 확정 |
| b0/b1 wall ratio | attention.py b1_count_sum 로그 활성 | F2 가설 확정 |
| KMP_BLOCKTIME sweep | {0, 10, 50, 200, 1000} × 100p short | libgomp share 변화 |
| GIL contention (선택) | py-spy native unwind | F1/F2 GIL bottleneck 확인 |

### Step 2 — Phase 0 결과 review + F6 단독 시도 (1-2 일)

- Phase 0 측정 결과로 F1~F6 priority 재확정
- F6 OMP dynamic schedule 단독 (3-tile cfg 유지) 의 500p × 3-run 측정
- Step 4 회귀의 root 분리 (dynamic schedule 자체 vs 5-tile cfg)

### Step 3 — Phase 0/F6 결과 dependent

- 만약 F6 win 또는 KMP sweep 의 fact 가 backing 강하면 진행
- 만약 cdec fraction 측정 시 async pipeline depth 의 ceiling 확정 가능
- 측정 결과로 진정한 next lever 명시

---

## 7. K vs L 차이점

| 측면 | K (가설 기반) | L (fact 기반, 본 문서) |
|---|---|---|
| Win % | +5-15% 등 추정 | **추정 안 함** (실측만 인정) |
| Ranking 기준 | win × ROI | fact backing 강도 + mechanism 명확성 |
| 권고 시작점 | Phase α (F3+F6) | **Phase 0 측정 sweep** |
| Effort 약속 | Phase γ 3 주 +14% | 측정 1-2 일 + 다음 단계 dependent |
| 신뢰성 | 낮음 (가설) | 측정 backing only |

K 문서는 implementation priority 의 ballpark guide. L 문서는 measurement-driven 진행 plan.

---

## 8. 결론

### 8.1 정직한 fact

1. **실측 backing 있는 fact** = libgomp 43.75%, cdec_wait 8.75 ms, Step 5 -2.35%, OMP/K_TILE sweep 결과
2. **K 문서의 모든 win %** = 가설 (실측 backing 0)
3. **F1~F6 의 진정한 priority** = Phase 0 측정 후에만 확정 가능

### 8.2 진행 권고

**Phase 0 측정 sweep (1-2 일, 코드 변경 0) 부터 시작**.

이후 Phase 1 (F6 단독) → Phase 2 (F1 short test) → Phase 3 (measurement dependent) 으로 점진 진행.

각 phase 의 결정은 **실측 fact backing 후**. 가설 기반 implementation 직진 회피.

### 8.3 사용자 명시 후 진행

본 문서 = 분석 only. 코드 변경 + 측정 launch 모두 사용자 명시 후 진행.

**다음 결정**:
- (a) Phase 0 측정 sweep 시작 명시?
- (b) Phase 0 skip + F6 단독 시도 직접 명시?
- (c) K 문서 의 가설 기반 plan 그대로 진행 명시? (위험)
- (d) 별도 task 분리?

---

## 9. 본 sequence 의 P1~P5 sweep + long workload 검증 결과 (2026-05-19 KST)

### 9.1 P1~P5 sweep (short 100p × 1024 × 1-run each, baseline 573.0 = KMP=0)

| Phase | 변경 | tps | vs P1 baseline | 결과 |
|---|---|---:|---:|---|
| **P1** KMP_BLOCKTIME sweep {0, 10, 50, 200, 1000} | env-only | **573.0 (KMP=0 best)** | (best) | sweep finding |
| **P2** F6 OMP `schedule(dynamic, 1) nowait` | `core.h` | 564.9 | **-1.4%** → revert | 회귀 |
| **P3** F3 K BF16 host store | swap path 변경 1-2일 | — | — | **skip** (effort 큼, 다음 turn 진행) |
| **P4** F1 async cdec | env-only `VLLM_NEO_ASYNC_CDEC=1` | 562.3 | **-1.9%** | env-only 로는 회귀. 코드의 infra 가 deferred dispatch 미구현 (KV cache sequential dep 의 architectural blocker) |
| **P5** F2 MIRROR_MAX sweep {40, 60, 100, 120} | env-only | 570.0 / 568.3 / OOM / OOM | -0.5~0.8% (40/60) | sweet spot = 80 (baseline) |

### 9.2 P1 KMP=0 long workload 검증 (500p × 8192 × 1-run)

| Config | gpu_util | KMP | tps | wall | vs S1-S9 (3-run avg 2,238.6) |
|---|---:|---:|---:|---:|---:|
| 0.92 KMP=0 첫 3-run | 0.92 | 0 | **OOM (3-run 모두)** | — | 비교 불가 |
| 0.88 KMP=0 1-run | 0.88 | 0 | **1,900.9** | 2,131.7 | -15.1% (KV cache budget 부족) |
| 0.90 KMP=0 1-run (zombie 영향) | 0.90 | 0 | runtime OOM (18 min) | — | zombie hold root |
| **0.90 KMP=0 retry (zombie 정리 후)** | 0.90 | 0 | **2,190.8** | 1,856.5 | **-2.1% (1-run, CV 3.23% 안)** |

→ **long workload 의 KMP=0 vs KMP=200 = within noise**. KMP=0 의 win 은 short workload 한정.

### 9.3 진정한 결론

- **본 sequence 의 모든 변경 시도가 SUB_015 S1-S9 baseline (2,238.6 tps) 초과 win 없음**
- P2, P4, P5 = 모두 short 회귀
- P1 = sweep finding (short best, long 에서는 within noise)
- P3 = skip (1-2 일 effort, 다음 turn 진행)
- **S1-S9 (gpu=0.92, KMP=200 default, 3-run avg = 2,238.6)** = 여전히 best
- 본 OOM root = 외부 bentoml service 의 누적 GPU 메모리 + zombie 잔존. **코드 변경 영향 아님** (final 확정)

### 9.4 다음 turn 의 P3 진행 영역

| 영역 | file 영역 |
|---|---|
| host K buffer dtype FP16 → BF16 | `vllm/v1/core/sched/neo_cpu_kv_buffer.py` |
| swap_out path K 변환 | `vllm/v1/worker/gpu_model_runner.py` |
| Step 0 (store_kv) host K BF16 처리 | `csrc/cpu/pacpu/core.h` |
| 정확도 검증 (numerical precision) | sanity test |
| swap_in path K BF16 → GPU FP8 변환 | gpu_model_runner.py |
