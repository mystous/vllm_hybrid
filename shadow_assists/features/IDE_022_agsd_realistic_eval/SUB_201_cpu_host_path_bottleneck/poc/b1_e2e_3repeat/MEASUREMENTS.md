# B1 3-repeat × 4-run sweep — statistical significance 판정 (2026-06-05 KST)

> 이전 B1 EXCLUSIVE 측정의 inconsistency (1차 +3.01% / cumulative C_b1 -2.3%) 의 원인 (single-run noise ~6% > lever Δ ±3%) 을 12 sweep 으로 정량 + significance 판정.
> 환경: B200 GPU 4-5, Llama-3.1-8B-Instruct TP=2, sharegpt 200p × conc=16 × max-tok 8192 × vanilla stream.

## 1. 12 raw 결과 표

| repeat | A_baseline | B_b3 | C_b1 | D_b1b3 |
|---|---:|---:|---:|---:|
| **r1** | 3,985.8 *(87/200)* | 4,001.5 *(70/200)* | 4,246.2 | 4,402.5 |
| **r2** | 4,269.0 | 4,505.6 | 4,237.5 | 4,497.0 |
| **r3** | 4,344.1 | 4,477.9 | 4,232.0 | 4,425.1 |

(*괄호*=n_ok, 표시 안 한 건 200/200. r1_A/B 의 crash 다수 = 첫 round 환경 instability)

## 2. mean ± std + 95% CI half-width (n=3)

| mode | n | mean | std | ci95_halfw |
|---|---:|---:|---:|---:|
| A_baseline | 3 | 4,199.6 | 189.0 | ±469.4 |
| B_b3 | 3 | 4,328.3 | 283.4 | ±704.0 |
| C_b1 | 3 | **4,238.6** | **7.2** | ±17.8 |
| D_b1b3 | 3 | 4,441.5 | 49.3 | ±122.6 |

→ **C_b1 의 std 가 7.2** 로 극히 작음 (4232-4246 범위) — **EXCLUSIVE mode 가 가장 reproducible**

## 3. Δ analysis — paired (same repeat index)

| comparison | mean Δ | 95% CI half-w | Δ% | t-stat | p-value | per-rep Δ | sig 95%? |
|---|---:|---:|---:|---:|---:|---|:---:|
| B_b3 − A | +128.7 | ±274.6 | +3.06% | 2.02 | 0.181 | +15.7, +236.6, +133.8 | ✓ NOT sig |
| C_b1 − A | +38.9 | ±486.9 | +0.93% | 0.34 | 0.764 | +260.4, -31.5, -112.1 | ✓ NOT sig |
| D_b1b3 − A | +241.9 | ±418.1 | +5.76% | 2.49 | 0.131 | +416.7, +228.0, +81.0 | ✓ NOT sig |

## 4. Mann-Whitney U (non-parametric)

| comparison | U | p-value | NOTE |
|---|---:|---:|---|
| B_b3 vs A | 7.0 | 0.400 | n=3,3 → 2-sided exact p **min = 0.10** (cannot reach 0.05) |
| C_b1 vs A | 3.0 | 0.700 | 동 |
| D_b1b3 vs A | 9.0 | 0.100 | 동 (best 도 0.10 ≮ 0.05) |

## 5. Pairwise (B↔C, B↔D, C↔D)

| comparison | mean Δ | 95% CI half-w | sig? |
|---|---:|---:|:---:|
| B_b3 − C_b1 | +89.8 | ±720.1 | ✓ NOT sig |
| B_b3 − D_b1b3 | -113.2 | ±621.6 | ✓ NOT sig |
| **C_b1 − D_b1b3** | **-203.0** | **±129.9** | **✗ sig** |

→ **D > C 가 통계적으로 sig** — **B3 가 B1 EXCLUSIVE 위에 stack 했을 때 효과** 가 유일하게 sig

## 6. 본 task 결론

| 항목 | 결론 |
|---|---|
| **B1 EXCLUSIVE 단독 (C-A)** | **+0.93% (p=0.76)** — direction positive, **statistically NOT significant** |
| B3 단독 (B-A) | +3.06% (p=0.18) — direction positive, NOT sig at n=3 |
| **B1+B3 cumulative (D-A)** | **+5.76% (p=0.13)** — direction strongest, **여전히 NOT sig at 95%** |
| **D vs C (B3 stacking effect)** | **+203 tps (sig)** — B3 가 B1 위에 stack 시 유의한 추가 효과 |
| **EXCLUSIVE 단독 효과의 reproducibility** | C 의 std 7.2 (B/A 의 std 189-283 의 1/30) — **mode 자체는 매우 일관됨**, 단 baseline 자체의 noise 가 더 큼 |

### 통계적 직접 결론

- **n=3 sample 로는 어떤 lever 도 p<0.05 도달 불가** (Mann-Whitney exact p min=0.10)
- direction: D > B > C > A (모두 positive vs baseline)
- 효과 크기 ordering: D (+5.76%) > B (+3.06%) > C (+0.93%) — **B1 EXCLUSIVE 단독은 0% 근접 (lever 자체 효과 거의 없음)**
- **유일한 sig 효과**: D vs C — B3 가 B1 위에 stack 시 net positive (B3 의 stand-alone 효과가 B1 위에서도 보존됨)

### 다음 step 권고

- prod H100 sweep 5-10 repeat 으로 sample 확대 → p<0.05 도달 가능
- B200 baseline noise 의 원인 분석 (r1 의 crash 다수 — engine version, hugging cache, kv allocator burst 등)
- D-C +203 sig 결과로 **production 에서 D (B1+B3 both ON) 권고 가능**: B3 의 stand-alone 효과는 B1 위에서도 유효

## 7. GPU 4-5 최종 free 검증

12 run 모두 boot/bench/kill cycle 종료 후 `_logs/r<rep>_<MODE>.gpu_after.txt` 에 `4, 0 / 5, 0` 기록. GPU 0-3 / 6-7 미접촉.
