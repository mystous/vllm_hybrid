# SUB_184 — Aggregate (phase-burst dummy fill)

## tps table (9 cells × 2 modes)

| mix | scenario | OFF tps | ON tps | Δ% |
|---|---|---:|---:|---:|
| balanced | vanilla-only | 1564.8 | 1586.1 | +1.36% |
| balanced | trident-only | 1645.7 | 1309.9 | -20.41% |
| balanced | agsd-gated | 3992.6 | 3874.0 | -2.97% |
| sonnet-heavy | vanilla-only | 1988.1 | 2006.5 | +0.92% |
| sonnet-heavy | trident-only | 3468.9 | 2970.8 | -14.36% |
| sonnet-heavy | agsd-gated | 4382.7 | 4338.0 | -1.02% |
| code-heavy | vanilla-only | 1863.2 | 1877.2 | +0.75% |
| code-heavy | trident-only | 3013.3 | 2572.8 | -14.62% |
| code-heavy | agsd-gated | 4542.1 | 4478.9 | -1.39% |

## AGSD 3-mix avg

- OFF: 4305.8 tps
- ON : 4230.3 tps
- Δ  : -1.75%

## utilization (monitor.py 0.5s)

| mode | CPU avg % | GPU avg (8 GPU) % | per-GPU avg % |
|---|---:|---:|---|
| OFF | 4.08 | 17.72 | 19.9, 22.5, 21.7, 23.0, 14.1, 13.7, 13.3, 13.6 |
| ON | 5.61 | 18.30 | 18.1, 21.0, 21.7, 21.3, 16.9, 17.4, 12.7, 17.4 |