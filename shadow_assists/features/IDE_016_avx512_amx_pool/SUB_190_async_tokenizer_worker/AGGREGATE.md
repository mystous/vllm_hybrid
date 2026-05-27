# SUB_190 — Aggregate (async tokenizer worker side-channel)

## tps table (9 cells × 2 modes)

| mix | scenario | OFF tps | ON tps | Δ% |
|---|---|---:|---:|---:|
| balanced | vanilla-only | 1603.7 | 1615.7 | +0.75% |
| balanced | trident-only | 1337.3 | 1562.7 | +16.85% |
| balanced | agsd-gated | 3909.2 | 4016.0 | +2.73% |
| sonnet-heavy | vanilla-only | 2058.8 | 1949.4 | -5.31% |
| sonnet-heavy | trident-only | 2960.0 | 3238.9 | +9.42% |
| sonnet-heavy | agsd-gated | 4277.4 | 4386.1 | +2.54% |
| code-heavy | vanilla-only | 1892.3 | 1852.2 | -2.12% |
| code-heavy | trident-only | 2969.7 | 2838.9 | -4.40% |
| code-heavy | agsd-gated | 4537.2 | 4533.4 | -0.08% |

## AGSD 3-mix avg

- OFF: 4241.3 tps
- ON : 4311.9 tps
- Δ  : +1.66%

## utilization (monitor.py 0.5s)

| mode | CPU avg % | GPU avg (8 GPU) % | per-GPU avg % |
|---|---:|---:|---|
| OFF | 4.23 | 17.94 | 19.8, 21.6, 18.7, 21.6, 13.0, 16.2, 16.6, 15.9 |
| ON | 5.28 | 17.69 | 19.2, 21.9, 23.0, 21.9, 13.2, 14.6, 14.5, 13.2 |