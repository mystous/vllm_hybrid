# SUB_189 — Aggregate (cpu draft ranker side-channel)

## tps table (9 cells × 2 modes)

| mix | scenario | OFF tps | ON tps | Δ% |
|---|---|---:|---:|---:|
| balanced | vanilla-only | 1599.8 | 1601.3 | +0.09% |
| balanced | trident-only | 1624.1 | 1644.1 | +1.23% |
| balanced | agsd-gated | 3973.9 | 4009.2 | +0.89% |
| sonnet-heavy | vanilla-only | 1975.7 | 1920.7 | -2.78% |
| sonnet-heavy | trident-only | 3336.6 | 3247.2 | -2.68% |
| sonnet-heavy | agsd-gated | 4337.8 | 4268.4 | -1.60% |
| code-heavy | vanilla-only | 1820.3 | 1782.9 | -2.05% |
| code-heavy | trident-only | 3035.9 | 2766.8 | -8.86% |
| code-heavy | agsd-gated | 4574.7 | 4503.8 | -1.55% |

## AGSD 3-mix avg

- OFF: 4295.5 tps
- ON : 4260.5 tps
- Δ  : -0.82%

## utilization (monitor.py 0.5s)

| mode | CPU avg % | GPU avg (8 GPU) % | per-GPU avg % |
|---|---:|---:|---|
| OFF | 4.29 | 17.67 | 21.9, 19.1, 21.3, 22.6, 13.8, 14.8, 13.4, 14.4 |
| ON | 6.31 | 17.48 | 19.2, 22.7, 21.7, 21.7, 13.1, 14.3, 13.4, 13.7 |