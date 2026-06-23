# SUB_250 — Rate-distortion water-filling mixed-precision (고전 응용 새 이론) — 전제 검증 (2026-06-16)

> **새 이론**: water-filling/rate-distortion(고전)을 응용 → 레이어별 "게이트 왜곡 vs 비트수"
> 곡선으로 **분포-게이트 왜곡 최소화 비트배분** 유도. NVFP4 uniform 4bit를 frontier에서 이김 목표.
> 이식이 아니라 "분포동등 게이트를 distortion 척도로 한 water-filling 재유도"가 기여 시도.

## 전제 검증 (Llama-3.1-8B, 레이어별 FP4 민감도, 6프롬프트 greedy 48토큰)
한 레이어만 FP4(나머지 BF16) → 출력 분포 왜곡(max_logprob_diff):
| 레이어 | max_logprob_diff | ppl_rel |
|---|---:|---:|
| L16 (최민감) | 0.319 | 0.035 |
| L8 | 0.225 | 0.018 |
| L1 | 0.178 | 0.073 |
| L0/L30 (최둔감) | 0.097/0.089 | 0.003/0.048 |

**min 0.089 ~ max 0.319 = 3.6× 편차.** → **비균일 민감도 = water-filling 헤드룸 실재.**

## Step 1 — 레이어별 D_i(b) 곡선 (b∈{2,4,8}, group-16 symmetric uniform)
`exp/waterfill_123.py`. teacher-forcing mean-KL distortion. D(b)는 **강한 볼록**:
2bit는 4bit 대비 한 레이어만으로도 KL 폭증(катастрофическая), 8bit는 4bit 대비 미세 개선.

## Step 2 — 고정 평균비트(B_avg=4) water-filling 배분
그리디 한계왜곡(ΔD/Δbit) 정렬 배분 → 둔감 레이어 일부 **2bit 강등**으로 8bit 승격 자금 조달
(예: 2개 레이어→2bit, 1개→8bit, 평균 4.00 유지).

## Step 3 — 결합 검증 ❌ **water-filling 이론 REFUTED**
| 배분(평균 4.00bit) | joint KL | uniform-4 대비 |
|---|---:|---:|
| uniform-4bit | **0.00465** | (기준) |
| water-filling(2/8 혼합) | 0.00626 | **+34.5% 나쁨** |

**기각 메커니즘**: D(b) 곡선이 강볼록 → 2bit 강등의 왜곡증가(>>)가 8bit 승격의 왜곡감소를
압도. **레벨 편차(3.6×) ≠ water-filling이 요구하는 한계기울기(marginal-slope) 편차.**
고전 water-filling은 D(b)가 매끈·오목할 때 성립 — 4-bit 격자 양자화는 그 가정 위반.

## Step 3b — 목표정렬 변종(bump): 2bit 미사용, 민감 top-k만 8bit
`exp/bump_variant.py`. 평균비트를 **올리는** 방향(고정예산 아님):
| 배분 | 평균비트 | joint KL | uniform-4 대비 |
|---|---:|---:|---:|
| uniform-4 | 4.00 | 0.00696 | (기준) |
| +top4→8bit | 4.50 | 0.00372 | −46.5% |
| +top8→8bit | 5.00 | −64.5% |  |
| +top12→8bit | 5.50 | 0.00152 | −78.1% |

→ 왜곡 크게↓ 하나 **평균비트↑ = 속도손해**(순수 FP4보다 느림). 이것은 **HAWQ류 sensitivity
mixed-precision** 그대로 = **신규성 없음**.

## Step 4 — 70B 실측 (bump-mixed 체크포인트, 사용자 "강행" 지시)
민감도 probe(`step4/probe70b.py`): 70B FP4 왜곡의 **70%가 첫 블록 L0-9에 집중**(KL 0.0009 vs
전체 0.00129; 다음 블록의 6.4×). → bump 대상 = L0-9만(bf16 유지), 나머지 70층 NVFP4 W4A4.
llm-compressor oneshot으로 mixed 체크포인트 생성(`step4/make_mixed.py`, ultrachat 256 calib).
70B TP4 서빙 sweep(`step4/step4_sweep.sh`):

| 구성 | tps | vs FP8(1810) | 게이트 | max_diff / ppl_rel |
|---|---:|---:|---|---|
| W4A4 (순수, RedHat) | **2273** | +25.6% | ✅ PASS | 0.43 / 0.069 |
| mixed (L0-9 bf16) | 2063 | +14.0% | ❌ FAIL | 0.62 / 0.103 |
| W4A4+spec | 4195 | +131.8% | ❌ FAIL | 0.43 / 0.128 |
| mixed+spec | 3910 | +116.0% | ❌ FAIL | 0.55 / 0.135 |

**Step 4 결과 — bump-mixed는 순수 W4A4에 두 축 모두 열세**: 느리고(2063<2273, 평균비트↑)
**정확도도 나쁨**(rel 0.103>0.069, diff 0.62>0.43). 8B in-place 시뮬에선 bump가 KL↓였으나
실제 70B는 **자가 calibration(ultrachat256) NVFP4 70층이 RedHat 정밀 calibration보다 나빠
L0-9 bf16 이득을 압도** → bump의 "속도희생→정확도" 위안조차 실측 미성립.

## Step 5 — 최종 판정
- **water-filling 이론**: REFUTED (+34.5%, D(b) 강볼록으로 수학적 성립불가).
- **bump 변종(HAWQ류)**: 70B 실측에서 순수 W4A4에 **완전 열세**(느리고 부정확) → 비신규 + 무이득.
- **부수확**: W4A4+spec 게이트 FAIL(ppl_rel 0.128) 재확인 → 10라운드 루프 R1(GPTQ)의 표적.
산출물: `exp/{per_layer_sensitivity,waterfill_123,bump_variant}.py`, `step4/{probe70b,make_mixed,step4_sweep}`.
**verdict: 전면 기각 — water-filling REFUTED / bump 70B 실측 dominated.**
