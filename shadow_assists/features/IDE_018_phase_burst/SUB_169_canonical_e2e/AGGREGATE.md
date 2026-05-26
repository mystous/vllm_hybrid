# SUB_169 — Aggregate

## tps table (9 cells × 2 modes)

| mix | scenario | OFF tps | ON tps | Δ% |
|---|---|---:|---:|---:|
| balanced | vanilla-only | 2474.0 | 2519.2 | +1.83% |
| balanced | trident-only | 3914.7 | 3900.0 | -0.38% |
| balanced | agsd-gated | 5289.5 | 5476.8 | +3.54% |
| sonnet-heavy | vanilla-only | 2668.3 | 2654.9 | -0.50% |
| sonnet-heavy | trident-only | 5839.2 | 5911.0 | +1.23% |
| sonnet-heavy | agsd-gated | 6066.3 | 6184.1 | +1.94% |
| code-heavy | vanilla-only | 2546.4 | 2569.7 | +0.92% |
| code-heavy | trident-only | 6169.5 | 6077.4 | -1.49% |
| code-heavy | agsd-gated | 7023.8 | 6966.5 | -0.82% |

## AGSD 3-mix avg

- OFF: 6126.5 tps
- ON : 6209.1 tps
- Δ  : +1.35%

## utilization (monitor.py 0.5s)

| mode | CPU avg % | GPU avg (8 GPU) % | per-GPU avg % |
|---|---:|---:|---|
| OFF | 4.08 | 36.17 | 49.7, 44.1, 49.7, 50.1, 24.0, 23.9, 23.8, 23.9 |
| ON | 5.33 | 37.14 | 50.8, 51.2, 51.1, 45.7, 24.5, 24.8, 24.5, 24.6 |