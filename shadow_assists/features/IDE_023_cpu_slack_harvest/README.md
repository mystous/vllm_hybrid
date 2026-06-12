# IDE_023 — HPC Theory-Grounded Multi-Axis CPU Slack Harvesting on DGX B200

> **status**: 활성 (계획)
> **parent**: TSK_020/SUB_072
> **registered**: 2026-05-29 (id_registry)
> **자식 SUB**: SUB_212 (Optimal+DSA 6-point coverage, 2026-06-11), SUB_213 (FaP 재검증 + uniform draft padding, 2026-06-11)

## 이론적 배경

HPC theory-grounded 5축 CPU slack harvesting on DGX B200:
- **A1 Temporal** — BSP + Amdahl + LogGP
- **A2 Compute SIMD** — Roofline + AVX freq license + AMX TMUL
- **A3 Data plane** — McCalpin STREAM + Shannon + LogGP
- **A4 Memory hierarchy** — Mattson + SLIT + CXL 1.1 + Smith 2-level + Denning + FP8
- **A5 SMT** — Tullsen + Snavely

HW lever 전면: AVX-512 / **AMX** / **DSA** / QAT / CMT / SMT / NUMA / **CXL** / NVLink5 / PCIe Gen5.

## 측정 진입 게이트

- **P0 이론 게이트**: ρ ≥ 1.5 (이론 speedup ≥ 1.5×)
- **P1 fit ≤ 20%** (모델 예측 vs 실측 오차)

## 자식 SUB 인덱스

| SUB | 상태 | 제목 | 위치 |
|---|---|---|---|
| `SUB_212` | ✅ 완료 (2026-06-11) | Optimal+DSA 6-point coverage on 10 models × 7 corpus + host DSA WQ confounder finding | [SUB_212_optimal_dsa_6point/](SUB_212_optimal_dsa_6point/) |
| `SUB_213` | 활성 (측정 대기) | SUB_212 confounder 재검증 (H-FaP: +36% = cudagraph FaP, host DSA 아님) + FaP×suffix 양립 lever (uniform draft padding) | [SUB_213_fap_suffix_uniform/](SUB_213_fap_suffix_uniform/) |

## 자세한 idea doc

[`../IDE_006/TSK_020/idea/IDE_023_cpu_parallelism_beyond_avx512.md`](../IDE_006/TSK_020/idea/IDE_023_cpu_parallelism_beyond_avx512.md)
