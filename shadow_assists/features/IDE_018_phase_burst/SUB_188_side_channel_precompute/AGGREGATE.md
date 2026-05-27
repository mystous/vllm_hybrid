# SUB_188 — Aggregate (side-channel batch precompute)

## tps table (9 cells × 2 modes)

| mix | scenario | OFF tps | ON tps | Δ% |
|---|---|---:|---:|---:|
| balanced | vanilla-only | 1607.9 | 1544.4 | -3.95% |
| balanced | trident-only | 1565.2 | 1621.0 | +3.57% |
| balanced | agsd-gated | 3942.8 | 3991.0 | +1.22% |
| sonnet-heavy | vanilla-only | 2013.3 | 2031.6 | +0.91% |
| sonnet-heavy | trident-only | 3283.3 | 3272.8 | -0.32% |
| sonnet-heavy | agsd-gated | 4292.6 | 4329.6 | +0.86% |
| code-heavy | vanilla-only | 1855.3 | 1873.3 | +0.97% |
| code-heavy | trident-only | 2968.1 | 2937.7 | -1.02% |
| code-heavy | agsd-gated | 4462.7 | 4610.6 | +3.31% |

## AGSD 3-mix avg

- OFF: 4232.7 tps
- ON : 4310.4 tps
- Δ  : +1.84%

## utilization (monitor.py 0.5s)

| mode | CPU avg % | GPU avg (8 GPU) % | per-GPU avg % |
|---|---:|---:|---|
| OFF | 4.73 | 17.50 | 18.0, 22.3, 21.4, 21.4, 13.8, 13.7, 14.6, 14.8 |
| ON | 4.41 | 17.71 | 18.4, 22.2, 23.1, 23.0, 13.3, 13.6, 14.1, 13.9 |