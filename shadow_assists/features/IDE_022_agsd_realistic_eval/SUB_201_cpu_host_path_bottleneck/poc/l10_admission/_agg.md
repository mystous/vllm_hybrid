# L10 aggregate (auto-generated)

### light load (1-seed, idle_mean=0.6s)  (mean±std over 2 seed runs)


**overall** (n=400 per run)

| metric | BASELINE | BURSTAWARE | Δ% |
|---|---:|---:|---:|
| ttft_ms_p50 | 40.2±2.5 | 38.2±4.3 | -5.2% |
| ttft_ms_p90 | 204.1±30.0 | 90.0±14.8 | -55.9% |
| ttft_ms_p99 | 397.9±33.5 | 378.0±19.8 | -5.0% |
| tpot_ms_p50 | 4.5±0.1 | 4.6±0.3 | +2.2% |
| tpot_ms_p99 | 6.2±0.7 | 7.0±0.1 | +12.1% |

**short** (n=272 per run)

| metric | BASELINE | BURSTAWARE | Δ% |
|---|---:|---:|---:|
| ttft_ms_p50 | 40.8±4.4 | 37.9±3.2 | -7.2% |
| ttft_ms_p90 | 251.2±21.0 | 157.1±104.7 | -37.5% |
| ttft_ms_p99 | 397.9±33.4 | 377.7±21.6 | -5.1% |
| tpot_ms_p50 | 4.6±0.1 | 4.7±0.1 | +2.2% |
| tpot_ms_p99 | 6.3±0.8 | 7.0±0.1 | +11.1% |

**long** (n=128 per run)

| metric | BASELINE | BURSTAWARE | Δ% |
|---|---:|---:|---:|
| ttft_ms_p50 | 39.1±0.9 | 38.4±5.5 | -1.9% |
| ttft_ms_p90 | 148.8±47.2 | 77.0±17.3 | -48.2% |
| ttft_ms_p99 | 400.1±38.3 | 327.9±52.4 | -18.0% |
| tpot_ms_p50 | 4.4±0.1 | 4.5±0.4 | +2.3% |
| tpot_ms_p99 | 5.7±0.7 | 6.4±0.6 | +12.3% |

**run-level**

| metric | BASELINE | BURSTAWARE |
|---|---:|---:|
| wall_total_s | 22.5±1.8 | 23.2±0.7 |
| n_ok (avg) | 400 | 400 |
