# Path 1 — AMX C3 prefix hash chain results

## Workload: chat_prefix

| config | sweeps | req_tps mean±std | out_tps mean±std | tot_tps mean±std |
|---|---|---|---|---|
| vanilla | 3 | 36.04±0.25 | 18453.21±127.15 | 95932.34±661.03 |
| lhc_amx_c3_prefix | 3 | 35.13±0.39 | 17986.94±198.12 | 93508.34±1029.96 |

**Δ (LHC vs vanilla)**: req -2.53%, out -2.53%, tot -2.53%

## Workload: sonnet

| config | sweeps | req_tps mean±std | out_tps mean±std | tot_tps mean±std |
|---|---|---|---|---|
| vanilla | 3 | 34.56±0.31 | 17234.12±157.07 | 33358.04±293.36 |
| lhc_amx_c3_prefix | 3 | 34.35±0.39 | 17149.37±191.24 | 33178.41±366.09 |

**Δ (LHC vs vanilla)**: req -0.59%, out -0.49%, tot -0.54%
