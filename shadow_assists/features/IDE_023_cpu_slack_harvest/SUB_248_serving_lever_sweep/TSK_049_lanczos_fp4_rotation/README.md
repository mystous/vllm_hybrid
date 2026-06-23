# TSK_049 — Lanczos eigen-rotation 으로 FP4 양자화 정확도 보강

> **상태**: 활성 (2026-06-16 신설) | **parent**: `IDE_023` / 연계 `SUB_248`·성공 `SR-003`
> **목표**: NVFP4 W4A4+spec(bf16 대비 **+194.5%**, 세션 최고속)이 분포동등 게이트를
> 근소 FAIL(ppl_rel 0.128>0.1)하는 것을, **회전(rotation) 기반 양자화**로 정확도만
> 끌어올려(속도 불변) 통과시켜 **+194% win 을 확정**한다. 란초스는 그 회전 기저를
> 거대 공분산에서 싸게 계산하는 도구.

---

## 1. 배경 / 동기

SUB_248 FP4 10-method sweep(2026-06-16, GPU 실측) 결과 (`runs/fp4_results.csv`):

| 방법 | tps | vs FP8(1810) | vs bf16(1437) | 게이트 | (diff/rel) |
|---|---:|---:|---:|---|---|
| **W4A4 NVFP4** | 2225 | +23.0% | +54.8% | **PASS ✅** | 0.43 / 0.068 |
| **W4A4 + spec** | 4232 | +133.8% | **+194.5%** | **FAIL (근소)** | 0.43 / **0.128** |
| W4A4 + TP8 | 2450 | +35.3% | +70.4% | FAIL | 0.45 / 0.103 |
| W4A16 | 763 | −58% | −47% | PASS | (dequant 오버헤드로 느림) |
| FP8 (ref) | 1810 | — | +25.9% | PASS | 0.135 / 0.042 |
| bf16 (ref) | 1437 | — | — | PASS | — |

- **plain W4A4 는 이미 게이트 통과(SR-003)**. 문제는 **W4A4+spec** — 최고속(+194.5%)인데
  ppl_rel 0.128 로 0.1 기준을 **근소하게** 넘겨 "빠르지만 부정확해서 실격".
- 정확도 여유(diff 0.43)는 FP8(0.135)보다 작다 = FP4 가 그만큼 lossy. **이 여유를 회전으로
  넓히면** +spec 까지 PASS 로 전환 가능.

## 2. 역할 분담 (핵심 — 오해 금지)

| 무엇 | 담당 | 효과 |
|---|---|---|
| **FP4 (비트폭 0.5B)** | B200 4-bit 텐서코어 | **속도 ↑** (+194% w/spec) |
| **회전 / 란초스** | 양자화-친화 직교변환 | **정확도 ↑ (속도 불변)** |

> 란초스는 **빠르게 만드는 게 아니다.** 빠르지만 부정확해서 못 쓰던 FP4 를 **"쓸 수 있게"**
> 만들어(양자화에도 불구하고 출력동등 회복) 이미 있던 속도 이득을 살린다. 회전은 추론
> 시점엔 가중치에 흡수되므로 **tps 비용 0**.

## 3. 회전 양자화 원리 (왜 정확도가 살아나나)

4-bit 의 적은 레벨(16개)은 **outlier** 에 취약하다 — 소수의 큰 값이 양자화 스케일을 키워
나머지 정상 값을 뭉갠다. 직교 회전 R 을 곱해:

```
선형층:  y = Wx        →  y = (W R)(Rᵀ x) = W' x'   (W'=WR, x'=Rᵀx)
R 직교:  R·Rᵀ = I      →  수학적으로 완전 동치 (무손실 변환)
```

R 은 **outlier 에너지를 전 차원에 분산**(분포를 Gaussian 화 / incoherent 화)시킨다. 그러면
어느 한 값이 지배하지 않아 **스케일이 작아지고 4-bit 격자 친화** → 양자화 오차 급감.
출력은 그대로(동치)인데 **양자화 격자가 보는 분포만** 친화적으로 바뀌는 것이 핵심.

## 4. 란초스가 하는 일 (정확한 수학)

란초스 = **거대 대칭행렬 A 를 작은 삼중대각 행렬 T 로 투영**하는 Krylov 부분공간 반복법.

```
입력:  A (d×d, 대칭, d=hidden dim 예 8192)         # 명시적으로 만들 필요 없음 — A·v 만 필요
Krylov: span{ v, Av, A²v, …, A^(k-1)v }            # A 를 v 에 k 번 반복 적용
정규직교 기저 Q (d×k),  T = Qᵀ A Q (k×k, 삼중대각)
출력:  T 의 고유값(Ritz) ≈ A 의 극단 고유값,
       Q·(T의 고유벡터) ≈ A 의 지배 고유벡터(분산 큰 주축)
```

- **비용**: matvec(A·v) **k 회** 뿐 → 전체 고유분해 O(d³) 회피. d 가 백만이어도 상위 k 개
  주축을 싸게 추출.
- **양자화 적용**: A = 가중치 또는 활성의 **공분산 행렬**. 란초스로 **상위 k 고유벡터(=outlier
  /분산이 집중된 주축)** 를 뽑아 그 축으로 **회전기저 R** 을 구성 → §3 의 직교변환에 투입.

> 한 줄: **란초스 = "거대 공분산에서 가장 의미있는 소수의 주축(작은 행렬/기저)을 matvec 만으로
> 싸게 추출"**. 그 작은 기저가 곧 양자화-친화 회전.

## 5. 기존 방식과의 차이

| 방식 | R 구하는 법 | 데이터 | 비용 | 성격 |
|---|---|---|---|---|
| 회전 없음 (plain FP4) | — | — | — | outlier 에 뭉개짐 |
| **QuaRot (Hadamard)** | 고정 구조행렬(랜덤 Hadamard) | 불요 | ~0 | outlier **균일 분산** |
| **SpinQuant (학습)** | 경사하강 학습(양자화손실 직접 최소화) | 필요 | **비쌈** | 데이터 최적 |
| **Lanczos eigen-rotation** | 공분산 **상위 고유벡터**(matvec) | 필요 | **쌈** | 데이터 **주축 정렬** |

- **vs Hadamard**: Hadamard 는 데이터 무관 고정. 란초스는 이 모델의 **실제 분산축**에 맞춤
  (대신 공분산+란초스 계산 비용).
- **vs SpinQuant**: SpinQuant 는 회전을 **학습**(무겁고 시간). 란초스는 **학습 없이 스펙트럼
  계산** → 훨씬 쌈. 단 직접 "양자화손실 최소화" 가 아니라 "분산 큰 축 정렬" 이라는 **대리목표**.

> **정직한 한계**: Lanczos-eigen 이 꼭 우월하지 않다. 문헌에선 **Hadamard(균일 분산)가 outlier
> 억제엔 더 강한 경우가 많다**(모든 차원을 똑같이 쉽게 만듦). eigen-rotation 은 분산 큰 축에
> 비트 예산을 몰아주는데 항상 유리하진 않음. 그래서 SpinQuant·QuaRot 가 주류. 란초스의 강점은
> **"데이터를 보되 학습은 안 하는" 중간 지점**(직접 70B 를 학습비용 없이 양자화하고 싶을 때).

## 6. 적용 설계

- **공분산 수집**: 캘리브레이션 입력으로 각 선형층의 입력 활성 X 의 공분산 `A = XᵀX/N`
  (활성 outlier 억제용), 또는 가중치 `W` 의 `WᵀW` (가중치 분포 정렬용).
- **R 구성**: per-layer(또는 블록대각) 직교 회전. 란초스 top-k 고유벡터로 부분 회전,
  나머지는 항등 — 또는 전체 직교화. attention 의 head-dim / FFN 차원 경계 고려.
- **추론 흡수**: R 을 인접 가중치에 미리 곱해(`W'=WR`, 다음층 입력측 `Rᵀ`) **체크포인트에 흡수**
  → 추론 시 추가 연산 0 → **tps = plain W4A4 와 동일**.
- **포맷**: NVFP4(compressed-tensors). vLLM 의 transform 경로
  (`vllm/model_executor/layers/quantization/compressed_tensors/transform/`) 가 online transform
  을 지원하나 **TP 와는 비호환**(`NotImplementedError: Online transforms with TP`) → 회전을
  **오프라인으로 가중치에 흡수**하면 TP 무관(권장).

## 7. 단계별 계획

### Phase 0 — 회전 효과 상한 즉시 검증 (저비용, 구현 0)
기존 **SpinQuant 사전양자화 체크포인트**(`daslab-testing/Llama-3.1-70B-Instruct-spinquantR1R2R4-nvfp4a16`,
이미 다운로드됨)를 **TP=1** 로 서빙(70B NVFP4=40GB → 단일 B200 가능). bench tps +
collect_logprobs → 게이트. **"회전이 FP4 정확도를 게이트 통과로 살리나"** 를 직접 측정.
> TP4 BOOT_FAIL 원인은 online-transform+TP 미지원이었음 → TP=1 이면 부팅 가능.
- 재사용: `bench_unique.py`, `collect_logprobs.py`, `runs/lp_bf16.json`(레퍼). TP=1 설정만 변경.

### Phase 1 — Lanczos eigen-rotation 양자화 구현
공분산 수집 → 란초스 top-k → R 구성 → 가중치 흡수 → NVFP4 양자화 체크포인트 생성.
도구: `llm-compressor`(미설치 → `uv pip install`) 또는 직접 구현. 캘리브 데이터 소량.

### Phase 2 — 3자 비교
Hadamard(QuaRot) vs SpinQuant(학습) vs Lanczos(eigen) — 동일 base(W4A4) + spec 에서
정확도(게이트 지표)·tps 비교. **어느 회전이 +spec 을 PASS 시키나** 판정.

## 8. 게이트 / 판정

- **정확도**: `ppl_rel ≤ 0.1` AND `max_abs_logprob_diff ≤ 0.5` (CLAUDE.md D-ii, vs bf16 70B).
- **속도**: `tps ≥ plain W4A4`(회전은 추론 비용 0 — 흡수). 즉 속도 무회귀.
- **성공 조건**: **W4A4+spec 이 FAIL→PASS 로 전환** → **bf16 대비 +194.5% 가 유효 win 확정**
  = SR-003(+54.8%) 상회 신규 최고. SR 등록 후보.

## 9. 리스크 (정직)

- **(a) 회전 불필요 가능**: plain W4A4 는 이미 PASS. +spec 만 0.128 로 근소 FAIL → 회전 없이도
  다른 작은 보강(예: KV 정밀도, calib)으로 넘을 수도. 회전이 과한 해법일 수 있음.
- **(b) Hadamard 우세 가능**: §5 한계 — 란초스 eigen 이 Hadamard 보다 못할 수 있음. Phase 2 가
  이를 가린다.
- **(c) heavy**: 직접 70B 양자화는 멀티-시간 + 도구 설치(llm-compressor). Phase 0 가 먼저
  "회전이 통하나" 를 싸게 답해야 Phase 1 착수 정당화.
- **(d) 대리목표**: Lanczos 는 양자화손실을 직접 최소화하지 않고 분산축 정렬을 함 — 둘은
  상관되나 동일하지 않음.

## 10. 참조

- 성공 등록: `shadow_assists/success.md` **SR-003**(FP4 W4A4) / **SR-002**(FP8).
- 메모리: `fp4-nvfp4-beats-fp8`, `fp8-weight-quant-win`.
- 측정: `../runs/fp4_results.csv`, `../runs/lp_*.json`, `../fp4_sweep.sh`, `../bench_unique.py`,
  `../collect_logprobs.py`.
- vLLM 코드: `vllm/model_executor/layers/quantization/compressed_tensors/transform/` (online
  transform, TP 미지원 확인 지점).
- 논문 계열: QuaRot(Hadamard rotation), SpinQuant(learned rotation), QuIP#(incoherence),
  Lanczos(Golub-Kahan, Krylov).

---

## Phase 0 실측 결과 (2026-06-16) — 회전 가설 반증

SpinQuant(daslab nvfp4a16) TP=1 서빙, 정확도 게이트(vs bf16):

| 방법 | match | max_logprob_diff | ppl_rel | 게이트 |
|---|---:|---:|---:|---|
| **SpinQuant(회전)** | 8.8% | **1.098** | **0.206** | **FAIL (최악)** |
| plain-W4A4 | 25.2% | 0.433 | 0.068 | PASS |
| W4A4+spec | 34.8% | 0.433 | 0.128 | FAIL(근소) |
| FP8 | 62.5% | 0.135 | 0.042 | PASS |

**판정: 회전이 정확도를 살리기는커녕 더 망침**(diff 1.1 ≫ plain 0.43). 리스크 (a)(b) 실현 —
plain W4A4 이미 PASS인데 회전 불필요·역효과. (daslab-testing=실험 repo, 품질 의심.) → **TSK_049
의 "rotation/Lanczos 로 FP4 정확도 보강" 가설은 본 체크포인트로는 반증.** Lanczos 직접 구현
(Phase 1)도 같은 위험 — **착수 정당화 약함.**

**대신 발견**: W4A4+spec 의 FAIL(0.128)은 spec-decode 가 분포보존(rejection sampling)이므로
**plain-W4A4(0.068 PASS)와 동등해야** 함 → 0.128 은 8-프롬프트 소표본+greedy 비결정 노이즈
가능성. **확정 경로 = 회전이 아니라 큰-표본 게이트 재측정.**

---

## Phase 1 (란초스 직접 구현·측정) 결과 (2026-06-16) — granularity 규명

란초스 회전(eigen/whiten)을 **직접 구현**(`exp/lanczos_fp4_error.py`, `exp/group_sweep.py`)해
실제 Llama-3.1-8B 가중치에서 회전별 FP4 양자화 상대오차를 granularity 별로 측정:

| group | none | Hadamard | whiten(란초스) |
|---|---:|---:|---:|
| **g=16 (NVFP4)** | 0.094 | +0% | **+0%** |
| g=128 | 0.110 | −1% | −1% |
| g=512 | 0.118 | −4% | −3% |
| **per-channel** | 0.139 | **−13%** | **−10%** |

**판정 (메커니즘 규명):**
1. **NVFP4(group=16)에선 회전 이득 ~0** — per-group(16) micro-scaling 이 outlier 를 국소적으로
   이미 처리 → 회전의 전역 outlier-분산 이득이 잠식. (Phase 0 SpinQuant 실패와 정합.)
2. **회전의 가치는 coarse 양자화(per-channel)에 실재** (−13% Hadamard). 회전법(QuaRot/SpinQuant)
   이 원래 coarse INT4 용으로 설계된 이유.
3. **란초스(whiten)는 Hadamard 보다 약간 못함** (−10% vs −13%) — eigen 기반이 균일분산을 못 이김.

**결론**: **란초스/회전으로 NVFP4(W4A4) 성능을 더 높이는 건 안 된다 — 포맷의 micro-scaling 이
회전을 불필요하게 만들기 때문.** 적용 경계 = coarse 양자화. byte-level 최고 win = **plain W4A4
(SR-003, +54.8%)** 유지. **TSK_049 회전 경로 = 판정 완료(NVFP4 부적용).**

## Phase 1b — 란초스 저랭크 양자화-오차 보정 (2026-06-16) — 부적용

`Ŵ=Q_fp4(W)`, `E=W-Ŵ`, top-k SVD(E)=Lₖ, `W̃=Ŵ+Lₖ`. 실 8B 가중치:

| 보정 | relerr | vs FP4 |
|---|---:|---:|
| FP4(보정無) | 0.0938 | — |
| +rank16 | 0.0930 | −0.9% |
| +rank64 | 0.0913 | −2.7% (저장 +7.6%) |

→ **FP4 양자화 오차 = random 반올림 노이즈(high-rank)** 라 저랭크(란초스)가 못 잡음. 나쁜 trade.

## TSK_049 종합 판정 (2026-06-16) — 란초스 NVFP4 부적용 (두 경로 모두 측정 확정)

| 란초스 적용 | 결과 | 메커니즘 |
|---|---|---|
| 회전(eigen/whiten) | NVFP4 +0% (per-channel만 −10%, 거기서도 Hadamard −13% 우세) | micro-scaling(g16)이 outlier 국소 처리 |
| 저랭크 오차보정 | −2.7%/저장+7.6% (나쁜 trade) | FP4 오차가 high-rank 노이즈 |

**결론**: 란초스는 NVFP4 성능 향상에 무용 — **회전(redundant)·저랭크(noise)** 둘 다 막힘.
신규성 후보였으나 측정으로 부적용 확정. byte-level win = plain W4A4(SR-003, 단 신규성 X, upstream).
란초스 가치는 *coarse 양자화 회전*(Hadamard에 약간 뒤짐)·*저랭크 구조 데이터*(KV 등, 별 regime)뿐.
