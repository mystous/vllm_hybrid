# Llama-3.1-405B-Instruct-FP8 × vanilla × mix

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 1252.1 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 923.3 |
| total_completion_tokens | 1156033 |
| TTFT p50/p99 ms | 71.1/436.3 |
| TPOT p50/p99 ms | 23.4/24.4 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 98.9 / 1272432 |
| CPU util | 4.8 |
| reqtps_avg | 42.2 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 70.59 | 65.88 | 20.09 |
| p50 | 540 | 12768.88 | 71.14 | 23.41 |
| p99 | 8192 | 193795.45 | 436.35 | 24.35 |
| max | 8192 | 193943.13 | 437.49 | 25.87 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-405B-Instruct-FP8_vanilla_mix.json`](../summ_Llama-3.1-405B-Instruct-FP8_vanilla_mix.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-405B-Instruct-FP8" and .method=="vanilla" and .condition=="mix")' ../per_request_raw.jsonl
  ```
