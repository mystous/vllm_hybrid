# SUB_247 — [TSK_046] dyn-K 판정 측정 (D-셀)

> **상태**: 활성 (2026-06-13) | **parent**: `TSK_046` (`IDE_024`) | GPU: 필요 (70B 판정)

## 정의

TSK_046 구현 (다중-K uniform FULL cudagraph + suffix α-EMA 동적 K_eff) 의 70B 판정
측정. 기준 = D0 (고정 K6+pad, SUB_213 의 고정-최적). 게이트: **dyn-K ≥ D0 +3%**
(SUB_213 K-sweep 의 per-corpus oracle ≈ 고정 K6 +11%p — 그 회수율 판정).

## 셀

| 셀 | 구성 | 판정 |
|---|---|---|
| D0_k6pad | K=6 + PAD (고정 anchor) | 기준 |
| D1_dynk | dyn-K 기본 임계 | ❌ **무효** — pop-버그로 적응 미작동 (= K6 고정 ± noise) |
| D2_dynk_t | dyn-K SAT0.6/BETA1.2 | ❌ 무효 (동일 버그) |
| D3_dynk_fix | **버그 수정판**, 기본 임계 | 진행 중 — 첫 진짜 판정 |

## 버그 기록 (방법론 교훈)

`suffix_decoding.py` 의 L3 스코어링이 `_l3_prev_proposal.pop()` 으로 항목을 먼저
소비 → dyn-K 수락길이 수집이 항상 None → **적응이 한 번도 안 돌았음**. 텔레메트리
부재로 D1/D2 (70분) 가 침묵 속에 무효화 — **"새 정책 코드는 텔레메트리를 처음부터"**
규칙 채택. 수정: 지역변수 `prev` 사용 + `[dyn-K]` 주기 로그 (steps/ema/k_eff/hist).
0.5B 검증: `k_eff=4 hist={4:199}` (낮은 수락길이 프롬프트에 정확 수렴).

## 산출물

- 측정: `runs_dynk/` (D3 종료 후 `features/IDE_023.../SUB_213.../runs_dynk` 에서 이관)
- 코드: TSK_046 구현 — `vllm/v1/cudagraph_dispatcher.py`, `gpu_model_runner.py`,
  `config/compilation.py`, `spec_decode/suffix_decoding.py` (커밋 ef4dbafd9 + 수정)

## ✅ 판정 (2026-06-13 08:29 — D3 확정)

| corpus | D0 K6 고정 | D3 dyn-K (수정판) | D3/D0 |
|---|---:|---:|---:|
| swebench | 6,762 | 7,399 | **1.094** |
| lmsys | 5,851 | 6,044 | 1.033 |
| mix | 9,356 | 9,656 | 1.032 |
| humaneval | 6,200 | 6,248 | 1.008 |
| sharegpt | 6,201 | 6,166 | 0.994 |
| wildchat | 6,684 | 6,466 | 0.967 |
| mbpp | 3,406 | 3,256 | 0.956 |
| **기하평균** | | | **1.011** |

텔레메트리 (61,200 steps): K 분포 {K4: 73%, K12: 16%, K6: 11%} — **적응은 실작동**.

**판정: 게이트 (+3%) 미달 — 부분 성공.**
- ✅ **인프라 성공**: 다중-K FULL capture·dispatch 정상 (그래프 183개 캡처,
  K 전환 무사고, 7corpus 100% 성공) — 향후 정책 개선의 enabler 로 유효.
- ❌ **정책 v1 미흡**: batch-전역 수락길이 EMA 로는 oracle (+11%p) 회수 실패
  (+1.1%). corpus 단위 비교에서 K-sweep 의 corpus별 winner 와 불일치
  (mbpp: 정책은 K4 다수 선택했으나 −4.4% — 일중 noise ± K 진동 비용 혼재).
- **결론**: regime 적응은 step-수준 EMA 보다 **워크로드-수준 정적 라우팅**
  (TSK_046 (a) — K-sweep LUT 기반) 이 실용 답. 동적 정책 고도화 (per-request
  regime, 진동 비용 모델) 는 후속 과제로 기록.

**교훈**: ① 새 정책 코드는 텔레메트리 동시 구현 (D1/D2 70분 침묵 무효) ②
정책 효과 판정 전에 "적응이 실제 일어났는가" 를 독립 검증할 것.
