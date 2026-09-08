# 대조: 새 메커니즘 (CF+skip+pin, N=8) + prompt map GSM100 (2026-09-09 03:15~03:25)
- **GSM100 93.0% (93/100)**, C32 579.3 tok/s / TPOT 43.9 (앞선 580~600 재현).
- α=0.25 hot set (94.0) 과 차이 없음 → GSM100 97 → 93~94 의 원인은 hot set 이 아니라 **N=8 전량 deferral (빈 immediate 생략의 전제)** 또는 잡음. 귀속: host 콜백 경로 N=4 / N=8 GSM100 (`…_ide034_gsm100_attribution/`).
