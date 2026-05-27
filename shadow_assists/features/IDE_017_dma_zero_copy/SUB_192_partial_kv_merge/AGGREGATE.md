# SUB_192 — Aggregate (side-channel partial KV merge)

## tps table (9 cells × 2 modes)

| mix | scenario | OFF tps | ON tps | Δ% |
|---|---|---:|---:|---:|
| balanced | vanilla-only | 1623.4 | 1611.0 | -0.76% |
| balanced | trident-only | 1624.0 | 1616.9 | -0.44% |
| balanced | agsd-gated | 4035.0 | 4025.0 | -0.25% |
| sonnet-heavy | vanilla-only | 2041.9 | 2045.0 | +0.15% |
| sonnet-heavy | trident-only | 3155.0 | 3266.1 | +3.52% |
| sonnet-heavy | agsd-gated | 4309.2 | 4428.9 | +2.78% |
| code-heavy | vanilla-only | 1896.3 | 1884.9 | -0.60% |
| code-heavy | trident-only | 2964.6 | 2963.8 | -0.03% |
| code-heavy | agsd-gated | 4579.4 | 4453.3 | -2.75% |

## AGSD 3-mix avg

- OFF: 4307.9 tps
- ON : 4302.4 tps
- Δ  : -0.13%

## utilization (monitor.py 0.5s)

| mode | CPU avg % | GPU avg (8 GPU) % | per-GPU avg % |
|---|---:|---:|---|
| OFF | 4.34 | 17.41 | 19.2, 21.9, 21.7, 21.7, 12.4, 14.3, 14.6, 13.4 |
| ON | 4.68 | 17.54 | 18.8, 22.0, 21.2, 21.4, 14.8, 14.7, 14.1, 13.4 |