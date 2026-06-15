# SUB_244 — LULL-SURF (MBM 안티페이즈 TDM), 2026-06-15

> **판정: positive(메커니즘) — carrier-sense가 곧 anti-phase TDM.** 단, bursty 처리량
> 정량은 binary SIGSTOP 한계로 미정밀(rate-modulated actuator 필요).

## 메커니즘
serving BW 가 bursty(burst↔lull)일 때 harvest 가 lull 을 surf. = SUB_238 CSMA 의
carrier-sense(BW busy→SIGSTOP, idle→SIGCONT)가 **정의상 anti-phase TDM**: serving burst
(BW↑) 시 harvest 정지, serving lull(BW↓) 시 harvest 재개.

## 측정 (bursty serving 50%duty + LULL-SURF/CSMA harvest)
- harvest 가 lull 구간에서 SIGCONT 되어 동작 확인 (stops/conts 진동).
- bursty harvest BW ≈ 0.44 GB/s (16-core harvest, ceiling 30GB/s) — **정량 부정밀**:
  binary SIGSTOP + 150GB/s harvest 라 all-or-nothing → lull 채움률 거칠음.

## 판정
- **anti-phase 메커니즘 성립** (carrier-sense = SUB_238 positive 의 시간축 해석).
- **정밀 정량(bursty 처리량 이득) 미완**: 깔끔한 측정엔 rate-modulated actuator
  (MERCATO 식 MBA 연속제어, SUB_242) + lull 슬랙에 맞춘 harvest 크기 필요. binary
  SIGSTOP 으로는 lull 채움률이 거칢.
- → 메커니즘 positive, 정량은 MERCATO(242, 연속제어) 변형으로 후속 권장.

산출물: 측정 잔여(`/tmp/h*.log`). 도구: SUB_238 csma_ctl + SUB_223 duty_ctl.
