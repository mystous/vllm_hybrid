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
