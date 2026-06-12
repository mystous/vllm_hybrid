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
