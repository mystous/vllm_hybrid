# SUB_015-Phase 3 — 2026-05-18 1-day measurement timeline (상세)

> 2026-05-18 KST 의 SUB_015 Phase 3 AMX 최적화 측정 전체 진행. 모든 step 의 시각/tps/wall/변경 lever/결과 정리.
>
> branch `feat/neo-amx-apply`. commit chain: `00a35caae → 1824cbdb3 → da32e79dd → 25b635722 → df33b9d22 → 6b5c24dea`.

---

## Timeline 개요 (Mermaid)

```mermaid
timeline
    title SUB_015-Phase 3 — 2026-05-18 measurement timeline (KST)
    section Phase 3 A (dropin AMX)
        07:58 ~ 08:31 : round 1 = 2,226.0 tps (-0.56% vs S1-S9)
        08:43 ~ 09:15 : round 2 = 2,129.4 tps (-4.9%)
        09:17 ~ 09:50 : round 3 = 2,072.0 tps (-7.4%)
                      : 3-run avg = 2,142.5 (-4.3%) ✗ 회귀
    section AMX optimization Step 1~6 (1-run sweep)
        10:18 ~ 10:51 : Step 1 (B thread-Q-cache) = 2,237.4 (-0.05%, 회복)
        10:54 ~ 11:26 : Step 2 (+A K^T outer pre-pack) = 2,275.6 (+1.7%)
        11:28 ~ 12:01 : Step 3 (+G SW prefetch) = 2,184.7 (-2.4%) ✗
        12:09 ~ 12:42 : Step 4 (+C' 2-block fused) = 2,184.0 (-2.4%) ✗
        12:46 ~ 13:17 : Step 5 (+vec K conv) = 2,284.0 (+2.0%, 1-run best)
        13:57 ~ 14:29 : Step 6 (+all G+C') = 2,199.7 (-1.7%) ✗
    section Step 5 3-run 정식 검증
        12:46 ~ 13:17 : round 1 = 2,284.0
        14:36 ~ 15:08 : round 2 = 2,154.4
        15:09 ~ 15:44 : round 3 = 2,120.0
                      : 3-run avg = 2,186.1 (-2.35%) ✗
    section Best Configuration 정정
        15:44 ~ 15:50 : S1-S9 (3-run avg 2,238.6) 여전히 best
                      : AMX 의 모든 variant 가 baseline 초과 못함
```

---

## 측정 결과 표 (시각 순)

### 1. Phase 3 A (dropin AMX) — 3-run 사전 검증

| KST 시각 | round | tps | wall (s) | vs S1-S9 | result.json |
|---|:-:|---:|---:|---:|---|
| 07:58:55 ~ 08:31:21 | 1 | 2,226.0 | 1,828.6 | **-0.56%** | `measurements/sub015_p3_amx_500p_3run_20260518/round1_20260518_075855_amx/` |
| 08:43:33 ~ 09:15:39 | 2 | 2,129.4 | 1,906.9 | **-4.9%** | `round2_20260518_084333_amx/` |
| 09:17:22 ~ 09:50:21 | 3 | 2,072.0 | 1,958.9 | **-7.4%** | `round3_20260518_091722_amx/` |
| **avg** | — | **2,142.5** | **1,898.1** | **-4.3%** ★ | (CV 3.6%) |

**관찰**: round 단조 감소 (2226 → 2129 → 2072, -7%). thermal/cache drift. 정량 결론 = -4.3% 회귀.

### 2. AMX optimization Step 1~6 (1-run sweep, 빌드 + 측정)

| KST 시각 | Step | 변경 (cumulative) | tps | wall (s) | vs S1-S9 | 결과 |
|---|:-:|---|---:|---:|---:|---|
| 10:18:11 ~ 10:51:18 | **1** | + B thread-Q-cache | **2,237.4** | 1,814.2 | -0.05% | 회복 |
| 10:54:44 ~ 11:26:10 | **2** | + A K^T outer pre-pack | **2,275.6** | 1,784.8 | **+1.7%** | 개선 |
| 11:28:40 ~ 12:01:25 | **3** | + G SW prefetch | 2,184.7 | 1,864.8 | -2.4% | **revert** |
| 12:09:49 ~ 12:42:35 | **4** | + C' 2-block fused (5-tile) | 2,184.0 | 1,862.2 | -2.4% | **revert** |
| 12:46:24 ~ 13:17:49 | **5** | + vec K conv (`_mm512_cvtneps_pbh`) | **2,284.0** | 1,774.6 | **+2.0%** | **1-run best** |
| 13:57:09 ~ 14:29:55 | **6** | + G + C' 전체 통합 | 2,199.7 | 1,849.6 | -1.7% | **revert** |

**관찰**:
- 1-run 결과는 cumulative lever 의 효과 + thermal noise 가 혼재.
- Step 3/4/6 의 회귀 = G (prefetch overhead) + C' (5-tile setup overhead) 가 NEO 의 작은 matmul 에서 net loss.
- Step 5 (vec K conv) 는 정확도 risk 0 + 변환 cost 줄임 → 1-run best.

### 3. Step 5 (B+A+vec K conv) 3-run 정식 검증

| KST 시각 | round | tps | wall (s) | vs S1-S9 | result.json |
|---|:-:|---:|---:|---:|---|
| 12:46:24 ~ 13:17:49 | 1 | **2,284.0** | 1,774.6 | +2.0% | `step5_amx_bav_500p_3run_20260518/round1_20260518_124624/` |
| 14:36:12 ~ 15:08:53 | 2 | 2,154.4 | 1,880.8 | -3.76% | `round2_20260518_143612/` |
| 15:09:21 ~ 15:44:30 | 3 | 2,120.0 | 1,901.9 | -5.30% | `round3_20260518_150921/` |
| **avg** | — | **2,186.1** | **1,852.4** | **-2.35%** ★ | (CV 3.23%) |

★ **Step 5 의 1-run round 1 (+2.0%) 은 cold-start cache benefit + thermal noise**. 3-run avg = **-2.35% 회귀**.

---

## tps trend visualization (단조 감소 — thermal/cache drift)

### Phase 3 A (dropin)
```
2300 ┤
2226 ●━━━━━┓
     │     ┃
2129       ●━━━━━┓
     │           ┃
2072             ●          <-- round 3
2000 ┤
     07:58   08:43   09:17  (시작 시각 KST)
```

### Step 5 (B+A+vec K conv)
```
2300 ┤
2284 ●━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
     │                            ┃
2154                              ●━━━━━━━━━━━━━━━━━━┓
     │                                               ┃
2120                                                 ●     <-- round 3
2100 ┤
     12:46                         14:36              15:09
```

★ 두 측정 모두 round 별 단조 감소. 1-day 동안 host CPU 의 thermal accumulation + cache state pollution.

---

## 측정 host 환경 (전 시점 공통)

| 항목 | 값 |
|---|---|
| Host CPU | Intel Xeon Platinum 8480+ (SPR, 112 phys core, NUMA 2) |
| GPU | NVIDIA H100 80GB × 8 |
| Memory | 2 TB |
| AMX | amx_bf16 + amx_int8 + amx_tile (native) |
| AVX-512 | bf16, fp16, vnni 모두 native |
| Kernel | Linux 5.14.0-427.13.1.el9_4 (RHEL 9) |
| Workload | Llama-3.3-70B TP=8, 500p × max_tokens 8192 × target_input 8192, fp8 KV cache |
| env (common) | KMP_BLOCKTIME=200, OMP_NUM_THREADS=10, VLLM_NEO_PROFILE=1, VLLM_NEO_CPU_PIN_PER_WORKER=1, VLLM_NEO_CPU_PIN_CORES=12, VLLM_NEO_NUMA_BIND=1 |
| AMX env | VLLM_NEO_USE_AMX=1 (Phase 3 A 부터 모든 step) |

---

## Best Configuration 변화

| Phase 시점 | Best | tps (3-run avg) | 비고 |
|---|---|---:|---|
| ~ 2026-05-17 (Phase 1 분석 전) | S1-S9 | 2,238.6 | 기존 best |
| Phase 3 A 직후 (08:31 KST) | S1-S9 | 2,238.6 | dropin AMX = -4.3%, S1-S9 유지 |
| Step 5 1-run 직후 (13:17 KST) | (Step 5 임시) | 2,284.0 (1-run) | 1-run noise — 잘못된 promotion |
| Step 5 3-run 완료 (15:44 KST) | **S1-S9** ★ | **2,238.6** | **Step 5 3-run avg -2.35% — S1-S9 best 유지** |

★ **best 변경 없음 — S1-S9 (NEO 원본 100% 정합) 가 여전히 best**.

---

## SUB_015-Phase 3 의 7 lever 종합 평가

`analysis/reference/I_amx_proper_design.md` 의 lever ranking 과 실측 매핑:

| Strategy | 이론 win | 실측 (3-run avg 기준) | 결과 |
|---|---:|---:|---|
| **B** thread-Q-cache | +0.5-3% | (Step 5 안 내포, 단독 효과 측정 안 함) | win 확인 (1-run +4.4% vs Phase 3 A) |
| **A** K^T outer pre-pack | +1-5% | (Step 5 안 내포) | small win |
| **C** Multi-seq batched | — | algorithmic 불가 | 시도 안 함 |
| **C'** Multi-task fused N=32 | +3-6% | Step 4: -4% vs Step 2 (1-run) | **회귀** (작은 matmul 의 5-tile overhead) |
| **G** SW prefetch | +1-3% | Step 3: -4% vs Step 2 (1-run) | **회귀** (block_table indirection 으로 정확성 낮음) |
| **vec K conv** (custom) | +0.5-2% | Step 5: +0.4% vs Step 2 (1-run) | small win, but 3-run avg 으로 net loss |
| **E** AMX av_product | +0-3% | 시도 안 함 | deferred |

★ **모든 cheap lever 가 NEO 의 작은 matmul (M=8 head, N=16 token, K=128 dim) 의 fundamental setup overhead 한계 안에서 net loss 또는 marginal**. 진정한 큰 win 위해서는 NEO core design 변경 필요.

---

## 핵심 lesson (3 가지)

### Lesson 1 — 1-run noise CV >3%

본 day 의 모든 측정에서 CV 3.23% ~ 3.6%. **1-run 결과의 신뢰성 매우 낮음**. round 별 tps 차이가 lever 의 진정한 effect 보다 클 수 있음.

→ **3-run avg 만 신뢰 가능**.

### Lesson 2 — thermal/cache drift

sequential 측정 시 tps 단조 감소 patten:
- Phase 3 A: 2226 → 2129 → 2072 (-7%)
- Step 5: 2284 → 2154 → 2120 (-7%)

원인:
- L2/L3 cache state pollution (cold start cache benefit decay)
- CPU 가열로 frequency 점진 ↓
- GPU memory leak (worker 종료 후 부분 잔여)

→ round 1 의 result 는 cold-start benefit 으로 outlier. avg 가 ground truth.

### Lesson 3 — NEO 의 작은 matmul 한계

M=8 (NUM_Q_HEADS / TP=8), N=16 (BLOCK_SIZE), K=128 (HEAD_DIM):
- AMX tile = 16×16 (M=8 만 사용, 절반 낭비)
- AMX sweet spot = M ≥ 32, large batched matmul
- NEO 의 per-block matmul = setup overhead dominant

→ AMX 의 진정한 win 위해서는 **TP=8 → TP=4** (M=8 → 16), **BLOCK_SIZE 16 → 32** (N=16 → 32) 같은 architectural 변경 필요. SUB_015 범위 밖.

---

## 코드 변경 history

| Commit | 변경 | 시각 |
|---|---|---|
| `00a35caae` | (Phase 3 시작 직전) Reference measurements 표에 cpu112_500p row 추가 | 2026-05-17 |
| `1824cbdb3` | SUB_015-Phase 1 정적/동적 분석 산출 (`H_static_analysis.md` 등) | 2026-05-17 |
| `da32e79dd` | SUB_015-Phase 1 debug symbol 빌드 + CPU 112-core 분석 script | 2026-05-17 |
| `25b635722` | **Phase 3 A AMX qk_product Strategy B+A 적용** (initial dropin) | 2026-05-18 |
| `df33b9d22` | Step 4/5 AMX optimization (vec K conv) | 2026-05-18 |
| `6b5c24dea` | **Step 5 3-run 정식 검증 + Step 6 통합 측정 + best 정정** | 2026-05-18 |

---

## 다음 단계 후보 (별도 task, 본 1-day 범위 밖)

| 후보 | Effort | 예상 win | 위험 |
|---|---:|---:|---|
| K cache BF16 host store (full swap path 변경) | 1-2 일 | +1-3% | swap path 정확도 검증 필요 |
| TP=8 → TP=4 (M=8 → 16) | 2-4 주 | +5-10% | GPU parallelism 변경, model 영향 |
| BLOCK_SIZE 16 → 32 | 2-4 주 | +3-7% | paging granularity 변경 |
| softmax / av_product AMX | 2-3 일 | +0-3% | NEO size 한계 동일 |

★ 모든 후보가 baseline (S1-S9 2,238.6 tps) **5-15% 초과 가능성**. 단 effort 와 risk 큰 architectural change.

---

## 측정 자료 파일 위치

| 측정 | dir |
|---|---|
| Phase 3 A 3-run | `measurements/sub015_p3_amx_500p_3run_20260518/` |
| Step 1~6 1-run sweep | `measurements/sub015_p3_amx_steps_500p_1run_20260518/` |
| Step 5 3-run 정식 검증 | `measurements/sub015_p3_step5_amx_bav_500p_3run_20260518/` |
| 본 timeline 문서 | `measurements/sub015_p3_measurement_timeline_20260518.md` |
