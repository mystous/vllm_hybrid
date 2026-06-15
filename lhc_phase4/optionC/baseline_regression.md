# Option C — baseline regression results

**Hypothesis**: Option C classifier detects GPU_SATURATED in
baseline regime → routes LHC OFF → throughput identical to
vanilla (within noise).

Source: `lhc_phase4/optionC/runs/bl_*` (24 cells)

## Output throughput (tok/s) mean ± std, 2 sweeps

| workload | vanilla | lhc_adaptive | Δ% |
|---|---|---|---|
| balanced | 18603.53 ± 204.74 | 18669.02 ± 211.51 | +0.35% |
| chat | 13414.21 ± 250.15 | 13886.75 ± 395.11 | +3.52% |
| code | 18140.31 ± 146.34 | 18187.15 ± 56.02 | +0.26% |
| code-heavy | 12405.78 ± 100.57 | 12460.17 ± 11.27 | +0.44% |
| sonnet | 15364.45 ± 2281.25 | 17280.11 ± 298.15 | +12.47% |
| sonnet-heavy | 14535.27 ± 1553.39 | 15165.65 ± 22.87 | +4.34% |

## Request throughput (req/s) mean

| workload | vanilla | lhc_adaptive | Δ% |
|---|---|---|---|
| balanced | 24.4866 | 24.5800 | +0.38% |
| chat | 55.8389 | 58.0197 | +3.91% |
| code | 35.4416 | 35.5235 | +0.23% |
| code-heavy | 12.2472 | 12.2462 | -0.01% |
| sonnet | 30.8003 | 34.6588 | +12.53% |
| sonnet-heavy | 7.1821 | 7.5037 | +4.48% |

## TTFT mean (ms)

| workload | vanilla | lhc_adaptive | Δ ms |
|---|---|---|---|
| balanced | 88.93 | 88.57 | -0.36 |
| chat | 60.76 | 62.29 | +1.53 |
| code | 170.74 | 167.91 | -2.83 |
| code-heavy | 418.29 | 388.94 | -29.35 |
| sonnet | 219.20 | 67.70 | -151.51 |
| sonnet-heavy | 638.36 | 300.54 | -337.81 |

## Per-cell raw

| run | output tok/s | req/s | TTFT mean (ms) | duration (s) |
|---|---|---|---|---|
| bl_balanced_vanilla_s1 | 18458.75 | 24.3439 | 86.91 | 20.54 |
| bl_balanced_vanilla_s2 | 18748.30 | 24.6293 | 90.95 | 20.30 |
| bl_balanced_lhc_adaptive_s1 | 18519.46 | 24.3992 | 87.92 | 20.49 |
| bl_balanced_lhc_adaptive_s2 | 18818.58 | 24.7608 | 89.22 | 20.19 |
| bl_chat_vanilla_s1 | 13237.33 | 55.0015 | 62.11 | 9.09 |
| bl_chat_vanilla_s2 | 13591.09 | 56.6763 | 59.41 | 8.82 |
| bl_chat_lhc_adaptive_s1 | 13607.36 | 56.8485 | 61.96 | 8.80 |
| bl_chat_lhc_adaptive_s2 | 14166.14 | 59.1908 | 62.62 | 8.45 |
| bl_code_vanilla_s1 | 18036.83 | 35.2508 | 169.37 | 14.18 |
| bl_code_vanilla_s2 | 18243.79 | 35.6324 | 172.12 | 14.03 |
| bl_code_lhc_adaptive_s1 | 18147.54 | 35.4479 | 161.76 | 14.11 |
| bl_code_lhc_adaptive_s2 | 18226.76 | 35.5991 | 174.06 | 14.05 |
| bl_code-heavy_vanilla_s1 | 12476.89 | 12.2616 | 368.64 | 16.31 |
| bl_code-heavy_vanilla_s2 | 12334.67 | 12.2327 | 467.94 | 16.35 |
| bl_code-heavy_lhc_adaptive_s1 | 12452.20 | 12.2707 | 349.68 | 16.30 |
| bl_code-heavy_lhc_adaptive_s2 | 12468.14 | 12.2217 | 428.20 | 16.36 |
| bl_sonnet_vanilla_s1 | 13751.36 | 27.4940 | 367.76 | 18.19 |
| bl_sonnet_vanilla_s2 | 16977.54 | 34.1065 | 70.65 | 14.66 |
| bl_sonnet_lhc_adaptive_s1 | 17490.94 | 35.0336 | 66.91 | 14.27 |
| bl_sonnet_lhc_adaptive_s2 | 17069.29 | 34.2841 | 68.48 | 14.58 |
| bl_sonnet-heavy_vanilla_s1 | 13436.86 | 6.5878 | 976.30 | 30.36 |
| bl_sonnet-heavy_vanilla_s2 | 15633.69 | 7.7764 | 300.41 | 25.72 |
| bl_sonnet-heavy_lhc_adaptive_s1 | 15181.83 | 7.4960 | 303.95 | 26.68 |
| bl_sonnet-heavy_lhc_adaptive_s2 | 15149.48 | 7.5113 | 297.14 | 26.63 |

## Summary

Mean Δ% across 6 workloads: **+3.56% ± 4.71%**

**Verdict**: |Δ%| = 3.56% > 2% noise band.