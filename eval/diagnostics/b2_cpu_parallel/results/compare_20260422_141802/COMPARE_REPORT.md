# X Phase 4/5 — Sync vs Async 비교 Report

결과 디렉토리: `compare_20260422_141802`

## 핵심 metric

| 항목 | sync (HYBRID_CPU_ASYNC_EXECUTOR=0) | async (=1) | Δ |
|---|---:|---:|---:|
| completed | — | 8 | — |
| total_output_tokens | — | 4096 | — |
| duration (bench, s) | — | 773.29 | — |
| bench wall (s) | — | 802 | — |
| request_throughput | — | 0.01 | — |
| output_throughput | — | 5.30 | — |
| mean_ttft_ms | — | 1375.98 | — |
| p99_ttft_ms | — | 3828.93 | — |
| mean_tpot_ms | — | 750.36 | — |
| p99_tpot_ms | — | 1508.89 | — |

## 판정

- ⚠ completed 수가 다름 (sync=None, async=8) — correctness 확인 필요

## 데이터 아티팩트

- `sync/` — sync baseline 전체 결과
- `async/` — async 실행 결과
- 각 디렉토리의 `hybrid.json` / `bench.log` / `server_boot.log` / `env_used.env`