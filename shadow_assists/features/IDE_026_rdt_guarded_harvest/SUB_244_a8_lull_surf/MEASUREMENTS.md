
## 70B 실서빙 스모크 (호스트, 2026-06-15) — CPU harvest −0.7%, guard 불요

실 Llama-3.1-70B serving(TP4) victim 에 naive CPU STREAM harvest(victim_aggressor
aggressor, node0 16코어 16-31, AVX-512 triad) 간섭. 부하 동일(conc24/reqs96 warm).

| 셀 | 70B gen tps |
|---|---|
| A 서빙 단독 | 2287.4 / 2295.4 |
| B +CPU STREAM harvest 16코어 | 2275.9 / 2275.0 |

→ CPU harvest 영향 = **−0.7%** (거의 노이즈). DSA(0%, SUB_236)보다 약간 크나 미미.
**LULL-SURF anti-phase guard 셀은 생략** — 회복 대상이 −0.7%뿐이라 무의미. harvest 가
실 70B serving 에 near-free 이므로 guard 의 실효 가치 낮음(SUB_236/228 과 동일 결론).

산출물: `host_smoke/`.
