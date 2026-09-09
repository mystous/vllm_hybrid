# 최종 구성 정상 상태 도착률 sweep (2026-09-09 20:12~21:00) — 운영점 B (graph 224·청크 4096·KV 127,309) + RB/PF/FUSE

Poisson 도착, sonnet 512/128, 480 프롬프트, 동시성 상한 224. delayer = `--enable-prefill-delayer --prefill-delayer-queue-min-ratio 0.10 --prefill-delayer-max-delay-passes 8 --prefill-delayer-max-delay-ms 500` + 실행 배치 게이트 (`SGL_PD_MAX_RUNNING_FOR_DELAY`).

| 구성 | rate 5: tok/s / TPOT / TTFT p50·p99 | rate 7 | rate 9 | 버스트 C224 |
|---|---|---|---|---|
| delayer 끔 | 532.6 / **299.0** / 0.62·1.44s | 716.0 / 227.2 / 0.76·3.18s | 871.7 / 179.7 / 1.17·**7.22s** | **1110.3** / 149.0 |
| delayer + 게이트 128 | 536.1 / **199.8** (−33%) / 0.67·1.37s | 715.3 / 231.3 / 0.81·2.10s | 846.0 (−3%) / 196.7 / 0.90·**4.10s** (−43%) | 1095.5 / 151.0 |
| delayer + 게이트 225 | (진행 중) | | | |
| delayer + 게이트 225, 16패스/1000ms | (진행 중) | | | |

관찰:
- 도착이 흩어지면 요청마다 prefill forward 가 따로 돌고 (하이브리드 prefill forward 고정비 ≈ cold expert 64개 전량 스트리밍 ≈ 250 ms), 5 req/s 만으로 prefill 예산 초당 ~1.25 s → decode 굶음 (TPOT 299 vs 버스트 149). 도착률이 높아지면 대기열이 prefill 을 자연히 묶어 TPOT 가 내려감 (rate 9: 180).
- delayer 는 경부하 TPOT −33%, 과부하 TTFT p99 −43% (처리량 −3%). 게이트 128 은 rate 7 에서 실행 배치가 128 을 넘어 지연이 꺼져 이득 없음 → 게이트 225 변형 측정 중.
