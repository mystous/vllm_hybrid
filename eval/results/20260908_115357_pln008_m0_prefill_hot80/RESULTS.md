# PLN_008 M0 — prefill 전용 벤치 (hot-80, sonnet 512+100 in / 1 out, 64 요청, 2026-09-08 11:54)

| 동시성 (admission 배치 b_p) | 요청 처리율 | 평균 TTFT | 요청당 prefill 시간 = TTFT/b_p (근사) | 토큰당 |
|---|---|---|---|---|
| 1 | 3.10 req/s | 322 ms | 322 ms | 0.53 ms |
| 8 | 14.03 req/s | 569 ms | 71 ms | 0.116 ms |
| 32 | 51.31 req/s | 617 ms | 19 ms | 0.031 ms |

해석 (모델 항):
- prefill 은 배치가 크면 토큰당 0.03ms 까지 떨어지지만, decode 벤치에서는 요청 완료 시점에 소수씩 admission 되므로 b_p 가 작아 요청당 70~300ms 가 든다. 이 시간 동안 decode 스텝이 정지 (mixed chunk 미사용) → TPOT 에 스텝당 B × T_pf(b_p)/b_p / out_tokens 만큼 가산.
- hot-96 C=32 실측: TPOT 51.5 − decode 스텝 39.0 = 12.5ms/스텝 → 요청당 ≈ 50ms → b_p ≈ 10 상당.
- 모델 규칙 (가정, M1 검증): b_p ≈ max(1, C/4). C=32 → 8 → 71ms/요청 → 스텝당 32×71/128 ≈ 17.8ms (실측 12.5, +40% — 보수적).
- C=64 (hot-96, KV 24576) 의 큰 잔차는 prefill 이 아니라 KV 포화 상태의 retraction 재-prefill + 스케줄 순환 (thrash) — 모델은 이 구간을 "B_eff < C 이면 회피" 로 다루고 ±20% 게이트 대상에서 thrash 셀은 별도 표기.
- 원본: prefill_c{1,8,32}.log
