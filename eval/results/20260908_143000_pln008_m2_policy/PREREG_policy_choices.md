# PLN_008 M2 — 정책 선택 사전 등록 (2026-09-08 14:30, 모델 = M1 과 동일 sha 705329d0…, 측정 전)

| 워크로드 | 정책 선택 (H, N, KV) | 예측 tok/s | binding | 비고 |
|---|---|---|---|---|
| C=8, ctx 640 | H=96, N=4, gpu | 198.5 | gpu_expert | |
| C=16, ctx 640 | H=96, N=4, gpu | 324.5 | gpu_expert | |
| C=32, ctx 640 | H=96, N=2, gpu | 524.6 | gpu_expert | N=2 vs 4 차이 미미 (실측 490.9 @N=4) |
| C=64, ctx 640 | H=96, N=2, gpu | 783.4 | cpu_ddr | KV 49152 ≥ 64×576 → thrash 없음 (전일 24576 설정과 다름) |
| C=32, ctx 2000 공유 prefix | H=96, N=4, gpu | 597.3 | gpu_expert | THRASH 표시 (B_eff 24) |
| **C=64, ctx 2000 공유 prefix** | **H=80, N=2, gpu** | 657.1 | cpu_ddr | **워크로드에 따라 H 가 바뀌는 첫 사례** — H=96 은 KV 49152 로 B_eff 24 → 정책이 hot 을 줄이고 KV 를 택함 |

- 정책은 hicache 를 한 번도 고르지 않음: 현 모델에 **HiCache 의 이득 항 (공유 prefix 의 prefill 재계산 절약)** 이 없고 간섭 페널티만 있기 때문. M2 에서 추가할 항: hicache + 공유 prefix 시 prefill 토큰 = 고유 토큰 (prefix 는 host hit) + host 읽기 비용 (토큰당, HiCache 단독 벤치로 측정). 이 항은 M1 hicache 셀 측정 전에 정의해 두었음 (예측은 그대로 — 사후 수정 금지).
- 검증 계획: C=64/ctx2000/공유 prefix 에서 H=80 vs H=96 실측 비교 (정책이 맞으면 H=80 이 더 높음). M1 셀 1·5 (H64/H96 C64 ctx2000) 와 함께 판정.
