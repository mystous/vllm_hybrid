# 최종 구성 GSM8K 100문항 + C32 재확인 (2026-09-09 03:12~03:20)
구성: hot-96 N=8, callback-free + 빈 immediate 생략 + 코어 재배치, α=0.25 hot set, KV 49152.
- **GSM8K 100: 94.0% (94/100)** — 기존 최종 (host 콜백, prompt map, N=4): 97.0% (97/100); 같은 구성의 GSM40: 97.5% (39/40).
- C32 sonnet 512/128: **658.7 tok/s, TPOT 36.5** (앞선 661.4 / 35.9 재현).
- 3문항 차이의 귀속 (메커니즘 N=8 vs hot set α=0.25 로 층당 ~10 expert 가 GPU FP8 → CPU INT4) 을 위해 같은 메커니즘 + prompt map GSM100 대조 (`…_ide034_final_gsm100_promptmap/`) 실행.
