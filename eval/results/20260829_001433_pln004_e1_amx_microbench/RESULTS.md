# PLN_004 E1 — AMX Expert GEMM Microbench 결과 (H1 판정)

- 노드: violet-h100-016 (SPR×2, **turbo OFF 2.0GHz**), kt-kernel 0.7.0.post2, cpuinfer 96 / threadpool 2 (NUMA×2)
- 대상: Qwen3-30B-A3B layer 0 expert (h=2048, m=768, E=128, k=8), KTMoEWrapper 단독 구동
- 방법: conc (T 토큰 전부 동일 8 expert → n_e=T 직접 제어) / unif (균등 라우팅) × T∈{1..1024}, warmup 3 + 12~30 iter 평균
- 원시: `int4.json`, `int8.json` / 하네스: `shadow_assists/features/IDE_026/e1_bench/e1_amx_microbench.py`

## 핵심 곡선 (conc, 토큰당 비용)

| n_e (tokens/expert) | INT4 (μs/tok) | INT8 (μs/tok) |
|---:|---:|---:|
| 1 | 157.1 | 201.6 |
| 8 | 39.4 | 44.4 |
| 32 | 21.7 | 19.8 |
| 64 | 18.5 | 16.4 |
| 128 | 4.9 | 5.3 |
| 256 | 3.9 | 4.1 |
| 512 | **3.6** | **3.8** |
| 1024 | 3.8 | 4.0 |

## 판정 — H1 ✅ PASS

1. **증폭 효과 실재**: n_e 1→512 에서 토큰당 비용 **43× (INT4) / 53× (INT8) 하락** — "expert 당 토큰을 늘리면 CPU expert 는 거의 공짜" 가 실측됨. SCED 의 물리적 기반 확인
2. **knee 위치**: 평탄화가 n_e ≈ 128~256 에서 발생. E1 실측 유효상수 (plateau 에서 C_eff ≈ 21 TFLOPS, weight-bound 구간에서 BW_eff ≈ 70~136 GB/s) 로 재계산한 닫힌형 `n* = b·C/(2·BW)` = **75~129** — 게이트 (±50%) 내 일치
3. 예상외 관찰: **T=64→128 에서 절대 시간 역전** (1.18→0.63ms, INT8 도 동일) — kt-kernel 이 T≥128 에서 다른 (prefill 용 추정) 커널 경로로 전환. 소배치 expert 호출의 비효율이 커널 수준에서도 존재한다는 부수 증거 (논문 §microbench 에 기재 가치)
4. unif 모드도 동일 경향 (T=1024 에서 n_e≈64, 4.2μs/tok) — serving 형 라우팅에서도 성립

## 이론 피드백

E0 캘리브레이션 (BW_eff 136GB/s, C 70TF → n*=258) 대비 layer-단독 실측은 C_eff 21TF 로 낮음 (turbo OFF + dequant 오버헤드) → **knee 가 오히려 왼쪽 (75~129) 으로 이동, memory-bound 탈출이 더 쉬움**. E0 시뮬레이터의 C 파라미터를 21TF 로 갱신할 것 (E3 해석 시).
