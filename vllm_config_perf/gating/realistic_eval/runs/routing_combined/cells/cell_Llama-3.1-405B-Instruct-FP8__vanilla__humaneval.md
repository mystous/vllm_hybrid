# Llama-3.1-405B-Instruct-FP8 × vanilla × humaneval

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 1253.4 |
| n_ok/n | 164/164 (err 0) |
| wall_total_s | 751.4 |
| total_completion_tokens | 941858 |
| TTFT p50/p99 ms | 93.1/329.7 |
| TPOT p50/p99 ms | 23.4/25.4 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 98.9 / 1272432 |
| CPU util | 4.8 |
| reqtps_avg | 42.7 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=164)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 14 | 570.05 | 68.18 | 20.64 |
| p50 | 8192 | 189769.71 | 93.14 | 23.41 |
| p99 | 8192 | 192535.74 | 329.74 | 25.41 |
| max | 8192 | 192670.78 | 330.22 | 36.46 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-405B-Instruct-FP8_vanilla_humaneval.json`](../summ_Llama-3.1-405B-Instruct-FP8_vanilla_humaneval.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-405B-Instruct-FP8" and .method=="vanilla" and (.condition=="humaneval" or (.condition==null and .corpus=="humaneval")))' ../per_request_raw.jsonl
  ```
