# X Phase 4/5 — Sync vs Async 비교 Report

결과 디렉토리: `compare_20260422_134927`

## 핵심 metric

| 항목 | sync (HYBRID_CPU_ASYNC_EXECUTOR=0) | async (=1) | Δ |
|---|---:|---:|---:|
| completed | — | 4 | — |
| total_output_tokens | — | 625 | — |
| duration (bench, s) | — | 2.05 | — |
| bench wall (s) | — | 22 | — |
| request_throughput | — | 1.95 | — |
| output_throughput | — | 304.65 | — |
| mean_ttft_ms | — | 40.60 | — |
| p99_ttft_ms | — | 44.40 | — |
| mean_tpot_ms | — | 12.89 | — |
| p99_tpot_ms | — | 12.91 | — |

## 판정

- ⚠ completed 수가 다름 (sync=None, async=4) — correctness 확인 필요

## 데이터 아티팩트

- `sync/` — sync baseline 전체 결과
- `async/` — async 실행 결과
- 각 디렉토리의 `hybrid.json` / `bench.log` / `server_boot.log` / `env_used.env`