# TSK_019 — Best Configuration index

> Best configuration 영역 history + 개략 정보. 상세 fact = 각 best file 영역.

## Best history

| 시각 (KST) | best file | commit | tps | wall | workload | 22 strict | 비고 |
|---|---|---|---:|---:|---|:-:|---|
| 2026-05-15 00:14 | [`Best_v1.6_2157tps.md`](Best_v1.6_2157tps.md) | `64f9e0c48` | 2,156.9 | 1,882s | 500p × 8192 | 19/19 | 최초 strict 19/19 + shape mismatch=0 |
| 2026-05-15 20:31 | [`Best_Phase3_1_kmp50.md`](Best_Phase3_1_kmp50.md) | `099d23e54` | 2,038.7 | 1,591s | **400p** × 8192 | 19/19 | Phase 3.1 (OMP persist) + KMP_BLOCKTIME=50 |

## 개략 정보

### v1.6 (500p best)
- workload: 500p × 8192 in/out (4M total tokens)
- 핵심: shape mismatch fix (`NeoCpuKvBuffer._in_flight_swap_out`)
- vs vanilla 4,886: 44.1%

### Phase 3.1+KMP=50 (400p best)
- workload: 400p × 8192 (3.2M total tokens — v1.6 보다 80% 영역)
- 핵심: OMP persistent (omp_set_dynamic(0)) + KMP_BLOCKTIME=50ms
- cdec_wait 영역 2.68→2.38ms (−11.2%)
- Phase 3.4 baseline 영역 (400p, 1,930.5 tps) 대비 +5.61%

### 직접 비교 한계

v1.6 best 2,157 tps (500p) vs Phase 3.1+KMP=50 2,038 tps (400p) 영역 의 workload 영역 다름 — 직접 throughput 영역 비교 X. 동일 workload 영역 (400p) 영역 의 비교:
- Phase 3.4 baseline (400p, env Phase 3.1 적용 X): 1,930.5 tps
- Phase 3.1+KMP=50 (400p): 2,038.7 tps (+5.61%)

## 관련 영역 file

| 영역 | 위치 |
|---|---|
| 5-phase 분석 산출 (Phase A-F) | `analysis/` (13 .md) |
| 본 plan 구현 plan | `After_NEO_implementation_plan.md` |
| v1.6 성능 분석 | `Performance_analaysis_v1.6.md` |
