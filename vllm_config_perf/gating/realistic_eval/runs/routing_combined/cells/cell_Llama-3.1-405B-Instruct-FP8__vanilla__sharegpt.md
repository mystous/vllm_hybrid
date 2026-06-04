# Llama-3.1-405B-Instruct-FP8 × vanilla × sharegpt

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 1217.1 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 821.8 |
| total_completion_tokens | 1000263 |
| TTFT p50/p99 ms | 70.9/633.3 |
| TPOT p50/p99 ms | 23.3/24.6 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 98.9 / 1272421 |
| CPU util | 4.8 |
| reqtps_avg | 42.0 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 68.52 | 65.71 | 19.35 |
| p50 | 564 | 13326.81 | 70.88 | 23.34 |
| p99 | 8192 | 193044.79 | 633.32 | 24.58 |
| max | 8192 | 193588.96 | 637.09 | 28.23 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-405B-Instruct-FP8_vanilla_sharegpt.json`](../summ_Llama-3.1-405B-Instruct-FP8_vanilla_sharegpt.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-405B-Instruct-FP8" and .method=="vanilla" and (.condition=="sharegpt" or (.condition==null and .corpus=="sharegpt")))' ../per_request_raw.jsonl
  ```
