# SUB_213 — P-셀 측정 (uniform pad × FULL graph) 확정판, 2026-06-13

> **판정 요약 (positive ⭐⭐⭐ — serving 직접 가속)**: suffix draft 를 K 로 균일
> 패딩 (`VLLM_SUFFIX_PAD_UNIFORM=1`, K=8) 하면 uniform-decode FULL cudagraph 가
> 적중하여 **V2/V0 = +28.0% (기하평균), 최대 +51% (lmsys)** — suffix+FaP canonical
> 위에 얹는 순증. 메커니즘 = 워커 프로파일 (2026-06-13) 이 보인 **레이어당 Python
> op 디스패치 체인 (워커 CPU 50~75%) 의 우회**. 전 셀 100% 성공·0 에러,
> tpot p50 도 개선 (sharegpt 16.6→12.1 ms).

## 1. 경위 (프로파일 주도 재조준)

- py-spy 워커 프로파일 (70B suffix canonical 부하): total-time 의 75.5% =
  pybind11→torch.ops 디스패치, 45-49% = 레이어당 `unified_attention_with_output`
  Python 호출 — accept 0.72 라 step 대부분이 비균일 → PIECEWISE 경로,
  FULL `replay` 는 4.8% 뿐. → "균일화로 FULL 적중" = SUB_213 lever 그 자체.
- capture 한도 512 → K=8 (32 req × 9 tok = 288 ≤ 512) 채택. K=32 pad 는 1056 > 512 로 불가.

## 2. 결과 (70B, 7 corpus × 3셀, 셀별 fresh boot)

| corpus | V0 K=32 | V1 K=8 | V2 K=8+pad | V1/V0 | **V2/V0** | V2/V1 | acc V0→V2 |
|---|---:|---:|---:|---:|---:|---:|---|
| sharegpt | 4,531 | 4,492 | 6,080 | 0.992 | **1.342** | 1.353 | 0.73→0.32 |
| swebench | 4,776 | 4,618 | 6,674 | 0.967 | **1.397** | 1.445 | 0.80→0.49 |
| humaneval | 4,432 | 4,572 | 6,342 | 1.032 | **1.431** | 1.387 | 0.68→0.44 |
| mbpp | 2,607 | 2,153 | 2,574 | 0.826 | 0.987 | 1.195 | 0.45→0.17 |
| wildchat | 5,119 | 4,787 | 6,637 | 0.935 | **1.297** | 1.386 | 0.76→0.38 |
| lmsys | 3,992 | 4,410 | 6,049 | 1.105 | **1.515** | 1.372 | 0.66→0.35 |
| mix | 7,043 | 5,643 | 7,627 | 0.801 | 1.083 | 1.352 | 0.81→0.58 |
| **기하평균** | | | | 0.946 | **1.280** | **1.354** | |

- accept 하락 (0.7→0.3대) 은 **형식상** — pad 토큰은 기각-보장 (분모 증가) 이며
  정확도 무손실 (rejection sampling 등가). tps·지연 동시 개선이 실효 증거.
- 분해: K 축소 단독 = −5.4% (V1/V0) ↔ pad/FULL 효과 = +35.4% (V2/V1) → 순증 +28.0%.

## 3. 한계·후속

1. **저-accept corpus (mbpp 0.45) 는 중립** (0.987) — pad 낭비가 이득 상쇄.
   regime-aware 게이트 (IDE_024 TSK_046 의 α-EMA) 가 정확히 이 지점을 메움.
2. K sweep 미실시 — K∈{4,6,8} × pad, capture 1024 확장 시 K=16 pad 도 후보.
3. 단일 모델 (70B) — 8B/32B/671B 일반화는 후속.

## 4. 산출물

`runs/summ_*.json` 21셀 + `run_sub213.sh`. 근거 프로파일:
`../../IDE_026_rdt_guarded_harvest/profiling/worker0_profile.speedscope.json`

---

## 5. K-sweep 확장 (2026-06-13 03:48 확정 — pad 고정, K ∈ {4,6,8,12})

| corpus | V0 K=32 | K4pad | K6pad | K8pad | K12pad | winner |
|---|---:|---:|---:|---:|---:|---|
| sharegpt | 4,531 | 5,706 | 6,078 | 6,092 | 5,611 | K8 (+34.5%) |
| swebench | 4,776 | 6,578 | 6,711 | 7,179 | 6,565 | K8 (+50.3%) |
| humaneval | 4,432 | 6,060 | 6,218 | 6,120 | 5,927 | K6 (+40.3%) |
| mbpp | 2,607 | **3,814** | 3,481 | 3,277 | 2,891 | **K4 (+46.3%)** |
| wildchat | 5,119 | 6,471 | 6,883 | 6,632 | 6,198 | K6 (+34.5%) |
| lmsys | 3,992 | 5,804 | 6,060 | 6,050 | 5,626 | K6 (+51.8%) |
| mix | 7,043 | 8,099 | 9,496 | 8,358 | **14,389** | **K12 (+104.3%)** |
| **기하평균** | | 1.329 | **1.384** | 1.350 | 1.364 | oracle ≈ **1.49** |

판정:
1. **고정 K 최적 = K6 (+38.4%)** — 역U자 (디스패치-우회 고정 이득 vs 패딩 낭비 ∝ K).
2. **최적 K 는 accept regime 의 함수** (실측 스펙트럼): 저-accept mbpp → K4,
   고-accept mix (acc 0.81, tpot p50 1.9ms, wall 71s — 500/500·0err 검증) → K12 (+104%).
3. **per-corpus oracle ≈ +49%** vs 고정-최적 +38.4% — **적응형 K 선택 (TSK_046
   α-EMA regime 게이트) 의 상방 ≈ +11%p 가 정량 입증됨.** capture 1024 + K16 은
   K12 의 비-mix corpus 열세로 우선순위 하향 (mix-류 regime 한정 후보).

---

## 6. 8B 일반화 (2026-06-13 04:51 확정 — Llama-3.1-8B, K32 base vs K6+pad)

| corpus | G0 K=32 | G1 K6+pad | G1/G0 | acc G0→G1 |
|---|---:|---:|---:|---|
| sharegpt | 18,453 | 21,640 | 1.173 | 0.85→0.64 |
| swebench | 20,683 | 23,165 | 1.120 | 0.89→0.75 |
| humaneval | 13,514 | 18,918 | **1.400** | 0.77→0.60 |
| mbpp | 16,138 | 19,156 | 1.187 | 0.79→0.64 |
| wildchat | 18,723 | 21,697 | 1.159 | 0.86→0.67 |
| lmsys | 19,389 | 22,160 | 1.143 | 0.85→0.68 |
| mix | 24,908 | 24,951 | 1.002 | 0.93→0.89 |
| **기하평균** | | | **1.164** | |

판정: **lever 는 모델 크기 축에서 일반화** — 이득 크기는 host-bound 정도에 비례
(70B +38.4% > 8B +16.4%). 8B mix (acc 0.93) 는 중립 — K=32 긴 draft 가 이미
최적인 초고-accept regime 에선 pad 의 여지가 없음 (regime 게이트의 off-스위치
조건). 전 셀 100% 성공·0 에러.

---

## 7. TSK_046 dyn-K 판정 (2026-06-13 — 상세는 SUB_247)

D3 (버그 수정판) / D0 (고정 K6+pad) = **+1.1% (게이트 +3% 미달)**. 다중-K FULL
capture 인프라는 성공 (183 그래프·무사고·적응 실작동 텔레메트리 입증), batch-전역
EMA 정책은 oracle (+11%p) 미회수. **실용 결론: 고정 K6+pad 가 배포 권장값 유지,
워크로드를 아는 경우 K-sweep LUT (mbpp→K4 / mix류→K12) 정적 라우팅.**
데이터: `features/IDE_024_workload_adaptive_composite/SUB_247_dynk_judgment/`

---

## 8. E1/E2 confounder 최종 확정 (2026-06-13 09:23 KST)

8B vanilla × {PIECEWISE, FaP} × {sharegpt, mix} (셀별 fresh boot):

| 셀 | sharegpt | mix |
|---|---:|---:|
| E1 PIECEWISE | 8,550 | 8,376 |
| E2 FaP | 11,067 | 11,185 |
| **E2/E1** | **+29.4%** | **+33.5%** |

**판정: SUB_212 의 "+36% (TSK_042 8,850 vs 본 sweep 12,089)" 의 원인 = FaP
(cudagraph mode) 확정.** 호스트 DSA WQ 는 무죄 (SUB_213 가설 입증, SUB_212 의
host-DSA confounder 해석은 기각). FULL_MATRIX_6point 의 ①↔② 차이 해석에 반영 필요.

## 9. 정확도 게이트 — pad lever (2026-06-13 09:29 KST, TST_003 방식)

70B suffix K6+FaP, {no-pad vs PAD_UNIFORM}, 32 prompts × greedy 128 tok, logprobs=1:

| 지표 | 값 | 임계 | 판정 |
|---|---|---|---|
| **D-ii** worst_max_abs_logprob | **0.2743** | ≤ 0.5 (atol) | ✅ |
| **D-ii** worst_ppl_rel | **0.0730** | ≤ 0.1 (rtol) | ✅ |
| D-ii per-prompt pass | 32/32 | — | ✅ |
| D-i 완전 일치 (informational) | 18/32, 발산 mean 31.4 tok | — | BF16 cascade 범위 |

**VERDICT: PASS (binding = D-ii)** — pad lever 는 Constraint 의 분포-유사성 게이트
충족. main 머지 품질 증거 확보. 도구: `accuracy_gate.py` (1차 실행은 parquet 컬럼
오독으로 0-prompt 무효 — `raw_text` 수정 후 재실행, 교훈: collect 결과 건수 assert).

---

## 10. Full matrix (pad on/off × K, 70B, 2026-06-13)

K{4,6,8,12} × {nopad,pad} + base(K=32) × 7 corpus = 63셀. **padding 인과 확정**:

| cell | sharegpt | swebench | humaneval | mbpp | wildchat | lmsys | mix | vs base |
|---|---|---|---|---|---|---|---|---|
| base_k32 | 4496 | 5195 | 4907 | 3090 | 5096 | 4180 | 7143 | 1.000 |
| k4_nopad | 3641 | 4129 | 3693 | 2318 | 4055 | 3796 | 4505 | 0.774 |
| k4_pad | 5864 | 6522 | 6547 | 3448 | 6425 | 5846 | 7751 | 1.246 |
| k6_nopad | 4252 | 4657 | 4395 | 3006 | 4553 | 3555 | 5892 | 0.896 |
| k6_pad | 6104 | 6826 | 6251 | 3629 | 6751 | 5968 | 8346 | 1.289 |
| k8_nopad | 4465 | 5030 | 4482 | 2634 | 4919 | 3813 | 6944 | 0.938 |
| k8_pad | 5932 | 6475 | 6279 | 2909 | 6497 | 6013 | 9707 | 1.256 |
| k12_nopad | 4506 | 5011 | 4427 | 3233 | 5322 | 4629 | 6565 | 0.996 |
| k12_pad | 5550 | 6518 | 5896 | 2926 | 6131 | 5562 | 14565 | 1.284 |

- K4: pad vs base +25% / pad-vs-nopad +61%
- K6: pad vs base +29% / pad-vs-nopad +44%
- K8: pad vs base +26% / pad-vs-nopad +34%
- K12: pad vs base +28% / pad-vs-nopad +29%

**결론**: ① 같은 K 에서 pad on 이 항상 빠름 (+17~61%, 7/7 corpus) → 향상 원인은
K 축소가 아니라 **uniform pad → FULL graph**. ② nopad 는 전부 base 이하 (K 축소
단독은 손해). ③ K별 pad-vs-base 는 역U자, **K6 최적 (+29%)** — 어제 K-sweep
독립 재현. ④ mix 는 K8_pad 가 최대 (corpus 별 최적 K 상이 → 모델별 최적 K 탐색
근거 = §11 multi-model).

---

## 11. Multi-model 일반화 — 전 모델 × K{4,6,8,12}pad × 7corpus (SUB_213, 2026-06-14)

**목적**: SUB_212 6-point 표에 ⑦ (uniform-pad best-K) 열을 추가하기 위해, 10개 모델
전부에서 per-(model,corpus) 최적 K 와 그 tps 를 실측. 70B 는 §10 fullmatrix 에서,
나머지 9모델은 runs_multimodel/ 에서 측정.

**측정 규모**: 9모델 × K{4,6,8,12} × 7corpus = **252 셀** (+ 70B fullmatrix 별도).
전 셀 100% 성공·0 에러. 셀별 fresh boot (suffix tree 오염 회피, SUB_214 규칙).
조건: suffix+FaP, gmu 0.85, MML 16384, conc 32, max_tok 8192, 실 trace, mix=500p
shuffle seed0 (SUB_212 canonical 동일). 차이: `VLLM_SUFFIX_PAD_UNIFORM=1` +
K∈{4,6,8,12} + taskset 0-47,56-103 (절대비교 caveat).

### 11.1 모델별 ⑦ best-K pad 향상 (7-corpus 기하평균)

| model | ⑦ vs ④ suf(OFF) | ⑦ vs ⑤ suf(ON) | ⑦ vs ② van(ON) | best-K (mix) |
|---|---:|---:|---:|:---:|
| `Qwen2.5-7B-Instruct` | +8.8% | +12.5% | +16.0% | K6 |
| `DeepSeek-R1-Distill-Qwen-7B` | +33.7% | +42.0% | +51.9% | K12 |
| `Llama-3.1-8B-Instruct` | +29.2% | +34.2% | +113.5% | K12 |
| `Qwen2.5-32B-Instruct` | +18.9% | +27.3% | +37.0% | K8 |
| `DeepSeek-R1-Distill-Qwen-32B` | +19.6% | +24.6% | +39.6% | K12 |
| `Qwen2.5-72B-Instruct` | +22.8% | +27.9% | +39.8% | K6 |
| `Llama-3.1-70B-Instruct` | +29.1% | +33.4% | +130.4% | K12 |
| `DeepSeek-R1-Distill-Llama-70B` | +39.2% | +41.9% | +37.1% | K12 |
| `Llama-3.1-405B-Instruct-FP8` | +24.0% | — | +135.5% | K6 |
| `DeepSeek-R1` | +175.6% | +156.9% | +35.1% | K12 |

> ⑦ vs ⑤ = pad 레버의 순수 효과 (둘 다 suffix+FaP, K 만 다름). ⑦ vs ② =
> non-suffix 최강 baseline 대비. **671B 는 ④⑤(suffix K=32) 가 vanilla 보다
> net-negative 였으나 ⑦(pad) 가 ② 대비 +35% 로 역전** — pad 가 step-bound MoE
> 에서도 suffix 를 살려냄. 405B 는 ⑤⑥ 가 §11.1(SUB_212) 부팅한계로 미측정이라
> ⑦ vs ⑤ 는 N/A, ⑦ vs ④ +24% / ⑦ vs ② +135%.

### 11.2 best-K 의 regime 의존성

- **Qwen dense (7B/32B/72B)**: 단일 corpus → **K4**, mix → K6~K8.
- **DS-Qwen distill (7B/32B)**: **K6~K12** (긴 reasoning trace = 높은 accept).
- **Llama-8B**: 전 corpus **K12 석권** (최고 accept).
- **대형 dense (70B/405B)**: **K6** 중심, mix → K12.
- **671B MoE (37B active)**: 단일 corpus **K4**, mix → K12.
- 공통: **mix corpus 는 큰 K, 단일 corpus 는 작은 K** 유리 (accept rate ∝ 최적 K).

### 11.3 산출물

- 결과: `runs_multimodel/summ_*.json` (252 셀), `runs_fullmatrix/` (70B 63셀).
- LIVE 매트릭스: `LIVE_multimodel_results.md` (`/tmp/mm_report.py` 자동생성).
- ⑦열 반영: `SUB_212_optimal_dsa_6point/FULL_MATRIX_6point.md` (6→7 point 갱신,
  ⑦ 가 70셀 중 **68셀** 행 최대).

## 동적 K 컨트롤러 — 데이터-주도 오프라인 검증 (2026-06-15, GPU 0)

> **판정: 동적 K-선택은 헤드룸 없음 (천장 +6.5%, 온라인 회수 ~+1%). dyn-K +1.1%는 버그
> 아닌 천장.** `ktail_offline.py` (63셀 runs_multimodel 리플레이).

- **oracle / 고정K6 = 1.065 (+6.5%)** — 동적이 노릴 전부. (+49%는 K-pad 레버 vs K32였고
  *어떤 고정 K-pad든* 잡음. 동적 부분 ≠ +49%.)
- **α는 K를 예측 못 함**: 같은 α=0.367이 K4(Qwen-32B mbpp)·K12(DS-Qwen-7B humaneval).
  최적 단조 α-threshold 규칙조차 oracle의 97.3%. → α-버킷 무의미.
- **UCB 동적 컨트롤러**: 레짐길이 5~500 전부 **dyn/K6 = 99.1~101.3%** (고정 K6과 동률).
  dyn/oracle 레짐500에서도 95% — K-tps 곡면이 최적 근처 평평(near-tie)이라 탐색이
  못 건짐. 큰 갭은 mix/고-tail(K12 +30%) 소수뿐.
- **실속**: "mix/고-accept-tail → K12, else K6" 2-state가 갭 대부분. 동적 장치 불요.

산출물: `ktail_offline.py`.
