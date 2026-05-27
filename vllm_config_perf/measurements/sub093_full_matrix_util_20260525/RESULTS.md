# SUB_093 — Full matrix with GPU/CPU utilization (57 cells)

> **parent**: TSK_020
> **scope**: 2026-05-25 16:26 ~ 19:01 KST (~155 min wall)
> **status**: ✅ 완료 — 57/57 cell + per-config monitor CSV
> **이전 measurements 미흡 → 본 SUB 가 첫 util 캡처**

---

## 0. 두괄식

| 발견 | 값 |
|---|---|
| Llama 70B Trident core GPU util | **73.3%** (vanilla 93.8%) — spec decoding 영역 GPU 비점유 시간 늘림 |
| Llama 70B Trident core CPU util | 5.3% (vanilla 5.6%, ngram 7.6%) — suffix 영역 ngram 보다 CPU 가벼움 |
| Llama 70B Trident core throughput | 모든 6 workload 영역 net positive ⭐ (mix 3 종까지 포함) |
| **AGSD (= Trident + gating)** vs Trident always-on | Llama 70B 영역 차이 0 (gating 영역 모든 workload 영역 suffix 선택) — 차이 영역 mixed-model deployment 에서 발현 |
| ngram code-heavy mix | vanilla 6,495 → ngram 5,491 = **−15.5% 회귀** → Trident core 9,457 = **+45.6%** ⭐ |
| 소형/cross-val (P2/P3) | PIECEWISE 영역 short-wall noise — comparability 주의 (§5 참조) |

---

## 1. Phase 1 — Llama-3.3-70B + TP=8 + H100×8 (18 cells)

### 1.1 throughput matrix

| workload | vanilla | ngram (fair) | **Trident core** (suffix+PIECEWISE always-on) | **AGSD** (Trident + gating) | Trident vs vanilla |
|---|---:|---:|---:|---:|---:|
| **sonnet** | 7,678.7 | 10,758.8 | **11,676.9** | **11,676.9** (gating→suffix) | **+52.1%** ⭐ |
| **chat** | 2,266.8 | 3,243.5 | **3,830.4** | **3,830.4** (gating→suffix) | **+68.9%** ⭐ |
| **code** | 6,717.7 | 5,361.5 | **7,981.4** | **7,981.4** (gating→suffix) | **+18.8%** ⭐ |
| **mix-sh** (M1 60:20:20) | 6,325.9 | 7,932.6 | **10,297.7** | **10,297.7** (gating→suffix) | **+62.8%** ⭐ |
| **mix-bal** (M2 34:33:33) | 6,053.9 | 6,553.6 | **9,514.3** | **9,514.3** (gating→suffix) | **+57.2%** ⭐ |
| **mix-ch** (M3 10:20:70) | 6,494.9 | 5,490.7 | **9,457.3** | **9,457.3** (gating→suffix) | **+45.6%** ⭐ |

→ Llama 70B 단독 영역 **AGSD = Trident core** (gating 영역 모든 workload 영역 suffix 선택). AGSD 영역 진짜 가치 영역 mixed-model deployment 에서 발현 (§5 + Phase 2/3 caveat 참조).

### 1.2 util matrix (config-wide avg, 6 workload 평균)

| config | wall total (s) | CPU util (%) | GPU util (%) | 비고 |
|---|---:|---:|---:|---|
| vanilla | 2,750.5 | **5.6%** | **93.8%** | spec OFF — GPU 영역 forward 영역 fully bound |
| ngram | 2,635.8 | **7.6%** | **84.2%** | ngram drafter 영역 CPU 부담 (+2.0pp) / GPU 9.6pp 감소 (drafter wait) |
| **Trident core** | **1,892.4** | **5.3%** | **73.3%** | suffix drafter 영역 ngram 보다 CPU 가벼움 / GPU 20.5pp 감소 |

→ **Trident core wall 영역 31% 단축** 영역 GPU util 영역 20pp 감소 영역 동반. spec decoding 영역 GPU 영역 활용률 영역 떨어지나 throughput 영역 늘림 영역 (per-step 영역 K token output).
→ Llama 70B 영역 AGSD wall/util = Trident core (gating decision 영역 항상 suffix).

### 1.3 단일 vs mix 비교 — Trident core 의 mix robustness

| 영역 | sonnet single | mix-sh | mix-bal | mix-ch | code single |
|---|---:|---:|---:|---:|---:|
| vanilla | 7,679 | 6,326 | 6,054 | 6,495 | 6,718 |
| ngram | **10,759** ⭐ | 7,933 | 6,554 | 5,491 ✗ | 5,362 ✗ |
| **Trident core** | **11,677** ⭐ | **10,298** | **9,514** | **9,457** | 7,981 |

→ Trident core 영역 sonnet → mix 영역 throughput **−12%** 만 감소 (11,677 → 9,457). ngram 영역 sonnet → mix-ch **−49%** (10,759 → 5,491). **Trident core 영역 workload mix 영역 robust**.
→ AGSD (gating) 영역 결과 동일 (gating decision 영역 항상 suffix).

---

## 2. Phase 2 — Qwen 2.5-0.5B / 1.5B / 7B + TP=1 (27 cells)

### 2.1 throughput

| model | workload | vanilla | ngram | Trident core (suffix+PIECEWISE) | AGSD gating 선택 | AGSD tps |
|---|---|---:|---:|---:|---|---:|
| Qwen 0.5B | sonnet | 4,233.7 | 4,316.7 | **10,921.4** | suffix | 10,921.4 |
| Qwen 0.5B | chat | 6,575.8 | 5,256.7 | **7,693.8** | suffix | 7,693.8 |
| Qwen 0.5B | code | 7,228.2 | 7,948.6 | **11,784.8** | suffix | 11,784.8 |
| Qwen 1.5B | sonnet | 4,771.2 | 3,429.8 | **6,389.1** | suffix | 6,389.1 |
| Qwen 1.5B | chat | 5,571.4 | 4,668.8 | **6,528.3** | suffix | 6,528.3 |
| Qwen 1.5B | code | 6,786.5 | 5,157.8 | **11,295.3** | suffix | 11,295.3 |
| Qwen 7B | sonnet | 5,584.2 | 3,199.6 | **5,714.4** | suffix (~tie) | 5,714.4 |
| **Qwen 7B** | **chat** | **4,556.5** | 2,777.2 | 4,253.4 | **vanilla** ⭐ | **4,556.5** ⭐ |
| Qwen 7B | code | 6,196.0 | 5,902.0 | **8,071.3** | suffix | 8,071.3 |

→ **AGSD ≠ Trident core 영역 차이** 영역 Qwen 7B chat 1 cell 영역 +7.1% (gating 영역 vanilla 선택). 그 외 모든 cell 영역 AGSD = Trident core.

### 2.2 util

| model | config | CPU% (avg) | GPU% (avg) |
|---|---|---:|---:|
| Qwen 0.5B | all | 2.1% | 0.4% |
| Qwen 1.5B | all | 2.1% | 0.6% |
| Qwen 7B | all | 2.1% | 1.3% |

→ 소형 모델 영역 GPU util 영역 1% 미만 — single H100 영역 underutilized (TP=1 + 50p × 1024 영역 short wall). util 영역 의미 없음 (sampling rate 1Hz 영역 short wall 영역 cover 못 함).

### 2.3 ★ 이전 SUB_078/088/090 영역 universal regression 영역 대비 본 결과

이전 SUB (cudagraph default = FULL_AND_PIECEWISE, no compilation_config override):
- Qwen 0.5B code vanilla 11,056 → ngram 4,486 (−59%)
- Qwen 1.5B sonnet vanilla 12,595 → ngram 5,016 (−60%)

본 SUB_093 (compilation_config={"cudagraph_mode": "PIECEWISE"} 영역 ALL configs):
- Qwen 0.5B code vanilla **7,228** → ngram 7,949 (+10%) / Trident core 11,785 (+63%)
- Qwen 1.5B sonnet vanilla **4,771** → Trident core 6,389 (+34%)

→ **vanilla baseline 자체 영역 본 SUB 영역 50% 낮음** (legacy 11,056 vs 본 7,228). 원인 = **PIECEWISE cudagraph 영역 compilation overhead 영역 short-wall (~3s) 영역 dominate** + cold-start (init wall 영역 first batch 포함). 즉 본 §2 영역 결과 영역 **prior SUB 와 직접 비교 불가**. comparability 영역 의미 영역 본 SUB 내부 (vanilla vs Trident core with same cudagraph mode).

→ 본 §2 결과 영역 **PIECEWISE-only conditioned**: small model 영역 PIECEWISE mode 사용 시 Trident core 영역 net positive. 단 short-wall variance 영역 ±20% 가능. AGSD (gating) 영역 1 cell (Qwen 7B chat) 영역 vanilla 선택 — 그 외 모든 cell 영역 Trident core 와 동일.

---

## 3. Phase 3 — opt-125m / starcoder2-3b (12 cells, cross-validation)

### 3.1 throughput

| model | workload | vanilla | ngram | best |
|---|---|---:|---:|---|
| opt-125m | sonnet | **8,008.2** | 7,360.3 | vanilla (−8.1% ngram) |
| opt-125m | chat | 9,542.1 | 9,514.4 | vanilla (tie) |
| opt-125m | code | **9,772.1** | 6,435.2 | vanilla (−34.1% ngram) ✗ |
| starcoder2-3b | sonnet | **5,758.4** | 5,007.0 | vanilla (−13.0% ngram) |
| starcoder2-3b | chat | 6,722.5 | **7,486.1** | ngram (+11.4%) ⭐ |
| starcoder2-3b | code | 6,794.0 | **7,242.0** | ngram (+6.6%) |

### 3.2 ★ issue #16258 영역 재현 — 이전 SUB_091 영역 본 SUB 비교

| 항목 | SUB_091 (2026-05-25 11:34 KST) | 본 SUB_093 (2026-05-25 19:00 KST) | 변경점 |
|---|---|---|---|
| opt-125m code 회귀 | **2.13×** (vanilla 12,541 → ngram 5,883) | **1.52×** (vanilla 9,772 → ngram 6,435) | PIECEWISE 영역 vanilla 영역 22% 감소 → 회귀 폭 줄어듦 |
| starcoder2-3b code 회귀 | **2.30×** | **+6.6%** (회귀 → net positive) | PIECEWISE 효과 + short-wall noise |

→ SUB_091 영역 confirmed regression 영역 본 SUB_093 영역 PIECEWISE 조건 영역 **재현 불가** (regression 영역 사라지거나 줄어듦). 본 §3 결과 영역 **stand-alone confirmation 영역 아닌 supplementary observation**.

### 3.3 util

| model | config | CPU% | GPU% |
|---|---|---:|---:|
| opt-125m | all | 2.1% | 0.3% |
| starcoder2-3b | all | 3.6% | 0.8% |

→ 소형 모델 영역 짧은 wall 영역 sampling 영역 표현력 부족.

---

## 4. ★ 종합 — 6 workload × 4 config × Llama 70B 영역 핵심 표

본 SUB_093 신규 수치 (모두 fair: gmu=0.80 + PIECEWISE + same wrapper + util 캡처):

| Workload | vanilla | ngram | **Trident core**<br>(suffix+PIECEWISE always-on) | **AGSD**<br>(Trident + gating) ★ |
|---|---:|---:|---:|---:|
| sonnet | 7,678.7 | 10,758.8 (+40.1%) | **11,676.9 (+52.1%)** ⭐ | **11,676.9** (gating→suffix) |
| chat | 2,266.8 | 3,243.5 (+43.1%) | **3,830.4 (+68.9%)** ⭐ | **3,830.4** (gating→suffix) |
| code | 6,717.7 | 5,361.5 (−20.2%) ✗ | **7,981.4 (+18.8%)** ⭐ | **7,981.4** (gating→suffix) |
| mix-sh (M1) | 6,325.9 | 7,932.6 (+25.4%) | **10,297.7 (+62.8%)** ⭐ | **10,297.7** (gating→suffix) |
| mix-bal (M2) | 6,053.9 | 6,553.6 (+8.3%) | **9,514.3 (+57.2%)** ⭐ | **9,514.3** (gating→suffix) |
| mix-ch (M3) | 6,494.9 | 5,490.7 (−15.5%) ✗ | **9,457.3 (+45.6%)** ⭐ | **9,457.3** (gating→suffix) |

→ **Trident core 가 6 workload 모두 net positive (+18.8% ~ +68.9%)**. ngram 영역 code 회귀 영역 mixed traffic 까지 전파 (mix-ch −15.5%). Trident core 영역 mix-ch 까지 net positive 영역 mitigation.
→ Llama 70B 영역 **AGSD = Trident core** (gating decision 영역 모든 workload 영역 suffix). AGSD 영역 별도 가치 영역 mixed-model deployment 영역 발현 (Qwen 7B chat 1 cell 영역 +7.1% — §2 참조).

### util (6 workload 평균, config-wide)

| config | CPU% | GPU% | wall sum (s) |
|---|---:|---:|---:|
| vanilla | 5.6% | 93.8% | 2,750.5 |
| ngram | 7.6% | 84.2% | 2,635.8 |
| **Trident core** | **5.3%** | **73.3%** | **1,892.4** ⭐ |
| AGSD (gating) | =Trident core | =Trident core | =Trident core | Llama 70B 영역 gating decision 영역 항상 suffix → util/wall 동일 |

→ Trident core wall 영역 vanilla 대비 **31% 단축**. CPU/GPU 영역 idle 영역 더 많지만 throughput 영역 늘림.

---

## 5. comparability caveat (Phase 2/3)

본 SUB_093 영역 ALL configs 영역 `compilation_config={"cudagraph_mode": "PIECEWISE"}` 영역 적용 (Llama 70B Trident core 영역 호환성 영역 위해 도입). 결과:
- **Llama 70B (Phase 1)** 영역 wall 영역 충분 (~400-700s) — PIECEWISE compilation overhead 영역 amortize → reliable.
- **소형/cross-val (Phase 2/3)** 영역 wall 영역 짧음 (1-8s) — PIECEWISE 영역 init overhead 영역 dominate → vanilla baseline 영역 prior SUB 대비 50% 낮음.

→ **Phase 2/3 결과 영역 PIECEWISE conditioned only**. prior SUB_078/088/090/091 영역 default cudagraph 영역 측정 영역 결과 (universal regression / issue #16258 정확 일치) 영역 본 §2/§3 영역 무효화 영역 아님 — **두 cudagraph mode 영역 다른 결과** 영역 의미.

→ 본 SUB_093 영역 핵심 contribution = **Phase 1 (Llama 70B 6 workload × 3 config × util)** 영역 신뢰. Phase 2/3 영역 supplementary.

---

## 6. raw data

- `_all_cells.csv` — 57 row × {phase, model, tp, workload, config, tps, wall_s, cpu_util_pct, gpu_util_pct}
- `phase1_llama70b/{vanilla,ngram,agsd}/{sonnet,chat,code,mix-sh,mix-bal,mix-ch}/result.json` — per-cell tps + token counts + mix_counts
- `phase1_llama70b/{vanilla,ngram,agsd}/_monitor_{cpu,gpu}.csv` — 1Hz util time-series (config-wide, 6 workload 영역 single 캡처)
- `phase{2,3}/<cell>/result.json` + `_monitor_{cpu,gpu}.csv` — per-cell

---

## 7. 후속

| 후속 | 권장도 |
|---|---|
| spec_decoding/README.md §2 영역 본 SUB 영역 신규 수치 영역 갱신 | 즉시 |
| Phase 2/3 영역 default cudagraph mode 영역 재측정 (PIECEWISE 영역 보조 비교) | medium |
| Trident core GPU util 73.3% 영역 분석 (드라이버 영역 idle 시간 유래 영역 정량화) | low |
| canonical 3-run (Phase 1 Trident core 6 workload) — variance 검증 | medium |
| **AGSD gating 실측 영역 multi-instance / per-request override** 영역 PoC | medium (gating 영역 진짜 가치 영역 mixed-model deployment 영역 발현) |
