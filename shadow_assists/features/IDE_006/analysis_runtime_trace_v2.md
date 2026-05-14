# [ANAL.v2] Phase 1 — runtime trace (try44 quantitative)

**작성**: 2026-05-08 / Phase 2-A 완료 후 retroactive 정리
**소스**: `eval/results/20260508_211625_try44_anal_v2_phase2A/engine.log.stdout`
**workload**: 500p × 50:50 / Llama-70B / TP=8 / VLLM_NEO_PREDICTIVE_THRESHOLD=0.5 / try22 skip 유지 (시나리오 A)

---

## 1. 개요 — try44 (시나리오 A) 정량 결과

| 지표 | 값 |
|---|---|
| init s | 163.81 |
| generate wall s | 3548.05 |
| total output tokens | 4,096,000 |
| **output_tps** | **1154.44** |
| prompt_tps | 763.53 |
| total schedule steps | ~16,400 (P1 step 발화 기준) |
| AssertionError | 0 |
| Traceback | 0 |

**비교 reference** (PLN_001 §5.6 NEO v41 500p): output_tps **2259.26** — try44 가 약 **51%** 수준. probe overhead 가 약 절반의 throughput 손실 야기 (acceptable for analysis).

## 2. 8 probe runtime hit count

| probe | total hit | 의미 | throttle 후 비율 |
|---|---|---|---|
| P1 | 4,591 | every 50 step + SWAPPED_OUT 등장 step | ~28% of all steps |
| P2 | 4,207 | try22 skip 발화 step (SWAPPED_OUT 출현) | swap_evt 와 거의 1:1 |
| P3 | 845,036 | resumed/not-prev transition logs | high — invariant transition 추적 |
| P5 | 535 | cdec_ids 분포 sample | every 50 + non-empty |
| P7 | 1,600 | cdec_future submit 결정 (warmup cap) | 80 layers × 8 worker × 첫 200 |
| P8 | 4,704 | _handle_neo_swaps event | every 50 + swap activity |
| **swap_evt** | **4,210** | swap_out chain fire (`new_swap_out>0` 등) | — |

## 3. P1 — `self.running` status 분포 (RUNNING count 빈도)

| RUNNING count | step 수 | 점유율 |
|---|---|---|
| 32 | 4195 | 91.4% — batch 동적 cap |
| 256 | 75 | 1.6% — 초기 peak |
| 3 | 58 | 1.3% — 종료 phase |
| 33 | 45 | 1.0% |
| 244 | 26 | 0.6% |
| ... | ... | ... |

→ **decode batch 가 step 대부분 (91%) RUNNING=32** 로 수렴. KV 압력 시 batch shrink 후 안정. 256 → 32 transition 이 있고 step 후반부 0~3 으로 종료.

## 4. P1 — `SWAPPED_OUT` 동시 잔류 max (try22 skip 의 직접 증거)

self.running 에 *동시에 잔류한 SWAPPED_OUT 의 max count* = **224**.

→ 즉 try22 skip 이 작동하는 동안 SWAPPED_OUT 이 self.running 에 *최대 224 개* 계속 잔류 (outer-loop skip 으로 schedule path 진입 안 함). 이들이 매 step iterated 되어 skip 만 반복 — overhead.

## 5. P2 — try22 skip 빈도 분포 (skipped_swapped_out value)

| skip count per step | step 수 |
|---|---|
| 224 | 1,776 (가장 빈번) |
| 212 | 272 |
| 159 | 85 |
| 178 | 79 |
| 160 | 54 |
| ... | ... |

→ 약 **1700+ step 동안 매 step 224 개** SWAPPED_OUT 을 skip. 즉 그 1700+ step 동안 NEO 가 *cdec 도 swap_in 도 못 하고 SWAPPED_OUT 잔류*. 이 phase 가 throughput 손실의 직접 원인.

## 6. P5 — cdec_ids 분포 (chain firing 핵심 지표)

**non-empty 발화 = 0 회.**

→ adapter 의 `cdec_ids = [rid for rid in vllm_ids if requests[rid].status == SWAPPED_OUT]` 가 항상 빈 list. SWAPPED_OUT reqs 가 schedule output 의 num_scheduled_tokens.keys() 에 포함 안 됨 (try22 skip 으로 outer loop 에서 미진입). **adapter 의 cdec dispatch path 가 firing input (cdec_ids non-empty) 을 못 받음.**

## 7. P7 — cdec_future submit 결정 (cdec dispatch 직접 지표)

| layer / call | tok_slice | seq_slice | req_ids |
|---|---|---|---|
| model.layers.79.self_attn.attn / call=80 | None | None | None |
| ... (모든 1600 entries 동일) | None | None | None |

→ `forward_context.neo_cdec_token_slice` / `seq_slice` / `req_ids` 가 항상 None. 즉 cdec dispatch wiring (worker side 에서 forward_context 에 stash) 이 *전혀 발화 안 함* — invariant chain 의 Step 5 (sub_batch split) 미진입에서 비롯.

## 8. P8 — `_handle_neo_swaps` event timeline (swap activity 의 시계열)

발화 시점 sample (early phase, step 154~):

| step | deferred_count | new_swap_in | new_swap_out | preempt_to_waiting (앞 5개) |
|---|---|---|---|---|
| 154 | 195 | 0 | **195** | 226-a7db02f2 ~ 218-876143eb |
| 159 | 202 | 0 | **202** | 233-b17cd9e7 ~ 222-97fa0b28 |
| 164 | 209 | 0 | **209** | 240-9cd47744 ~ 226-a7db02f2 |
| 169 | 216 | 0 | **216** | 247-83e857a6 ~ 233-b17cd9e7 |
| 174 | 223 | 0 | **223** | 254-bc5f6883 ~ 240-9cd47744 |

→ swap_out 매 ~5 step 마다 ~5-7 reqs *추가* 발화 (deferred_count 195 → 202 → 209 → 216 → 223). **swap_in 은 0 회**. 즉 NEO 가 swap_out 만 하고 swap_in 으로 복귀시키지 않음 → 모두 `_handle_neo_swaps` 의 deferred_preempt path 로 흘러서 vanilla preempt 처리 (status=PREEMPTED + waiting.prepend + KV freed).

이게 v1.2 의 *NEO ≈ vanilla preempt* 등가 — TSK_019 try26 의 deadlock 회피 fix 의 부작용.

## 9. P3 — assert_will_fire 발화 (invariant 위반 지표)

**0 회 — try22 skip 이 모든 SWAPPED_OUT 을 outer-loop 에서 제거했으므로 `_make_cached_request_data` 에 SWAPPED_OUT 진입 없음 → resumed slice 에 SWAPPED_OUT 가 안 나타남 → invariant 2 fire 0.**

## 10. 핵심 timeline 요약

```
Step 0-100:    warmup + 첫 batch (RUNNING=256)
Step ~150:     KV pressure → swap_out 195 reqs (deferred)
Step ~150-180: deferred 점진 증가 (195→223), all preempted to waiting
Step ~200~:    self.running 안정 RUNNING=32, SWAPPED_OUT 잔류 ~224
              try22 skip 매 step ~224 회
              cdec_ids = [] (어차피 skip 으로 outer loop 미진입)
              swap_in = 0 (NEO 의 swap-back path 미발화)
Step 1700~:    SWAPPED_OUT reqs 점진 종료 (vanilla preempt path 의 결과)
Step ~16400:   모든 reqs 종료
```

## 11. chain break point — 정확한 위치 (Phase 6 input)

| stage | 발화 여부 | break 위치 |
|---|---|---|
| swap_out fire | ✅ 4,210 | 정상 |
| RUNNING → SWAPPED_OUT 전이 | ✅ (P2=4,207) | 정상 |
| **outer scheduler loop 의 SWAPPED_OUT 처리** | ❌ skip | `scheduler.py:407` (try22) |
| KV alloc 우회 | ❌ 미진입 | P4=0 |
| adapter cdec_ids 추출 | ❌ 항상 [] | `neo_scheduler_adapter.py:605~609` |
| sub_batch split | ❌ 미생성 | adapter |
| worker b0/b1 분기 | ❌ P6=0 | `gpu_model_runner.py:1064~1092` |
| forward_context 의 neo_cdec_* set | ❌ None | worker side stash 미발화 |
| cdec_future submit | ❌ P7 _interesting=False | `attention.py:768~893` |
| forward_double 진짜 병렬 | ❌ submit 없음 → sequential | — |

→ **break root: scheduler.py:407 의 try22 skip.** 그 이후 chain stage 는 모두 *input 결핍* 으로 자연 미발화.

## 12. Phase 6 으로 넘기는 정량 fact

1. swap_out 자연 발화 횟수: **4,210**
2. SWAPPED_OUT 동시 max 잔류: **224** (즉 chain 활성화 시 cdec 처리 대상 max 224 reqs)
3. swap_in 자연 발화 횟수: **0** (TSK_019 의 swap-back path 미발화)
4. AssertionError 0 (try22 skip 의 가드 본분 작동)
5. NEO ON 상태에서 cdec/forward_double 발화 0 → output_tps 1154 (vanilla baseline 추정 1100~1300 과 동등) → **NEO 의 throughput 이득 0** (그러나 negative regression 도 0)

이 fact 들이 다음 *수정 plan* 의 baseline.
