# in-situ CPU 층 작업 phase 분해 — AMX MoE forward_prefill (FORWARD_TIME_PROFILE, 64회당 1회 표본) (2026-09-09 00:57~)
구성: callback-free + 빈 immediate 생략 + 코어 재배치, N=8, C32/C64 sonnet 512/128. 재현: hot-96 C32 579.9 / C64 **786.0** (앞선 600 / 760 재현).

## hot-96, decode 호출 (qlen 16..40, 표본 987건, 중앙값 qlen=32)
| 항목 | numa0 | numa1 |
|---|---|---|
| 활성 cold expert 수 (act_exp) 중앙값 | **6** | 6 |
| prepare / cpy_input / q_input | 3 / 9 / 17 µs | 3 / 8 / 14 |
| **up_gate** | **253** | 251 |
| act / q_down | 12 / 4 | 12 / 4 |
| **down** | **196** | 185 |
| weight(merge) | 11 | 13 |
| total (한 numa 절반) | **507** | 493 |
| max_local_num (expert 당 최대 행) | 2 | 2 |

활성 expert 수별 total 중앙값 (numa0): 0:24 1:162 2:222 3:310 4:378 5:454 6:508 7:597 8:676 9:734 10:810 11:873 µs → **≈ 90 + 71·D_c µs** (expert 당 71µs ≈ 마이크로벤치 55µs 의 1.3배; 고정비 90).

## 판정
1. 모델의 hot-96 CPU 항 (0.3ms) 이 in-situ (deferred 작업 평균 0.76ms) 대비 2.5배 과소인 이유 = 세 요인의 곱: (a) **층당 distinct cold expert 가 트레이스 기대 3.2 (B=32) 가 아니라 실측 중앙값 6, 평균 ≈6.5** (트레이스 18,776 토큰은 prompt 구간 라우팅; decode 생성 토큰은 hot-96 커버리지가 낮음) (b) expert 당 71µs (1.3×, GPU 트래픽·2소켓 동시 스트리밍 간섭) (c) 커널 밖 **wrapper 오버헤드 ≈0.2~0.25ms/작업** (do_numa_job 2-풀 디스패치/합류 + merge_results) — TaskQueue 작업 평균 0.76 vs 커널 total 중앙값 0.5.
2. 다음 손잡이 (효과 크기 순): ① wrapper 0.2~0.25ms × 62층 = 12~15ms/스텝 (스텝 43ms 의 30%) — numa 디스패치 cv 대기·합류 스핀 구조 판독 후 제거 ② decode 라우팅 트레이스로 hotmap 재구성 (D_c 6.5 → 트레이스 기대치에 근접시키면 CPU 항 −30~40%) ③ expert 당 71→55µs (스트리밍 간섭 원인 규명).
