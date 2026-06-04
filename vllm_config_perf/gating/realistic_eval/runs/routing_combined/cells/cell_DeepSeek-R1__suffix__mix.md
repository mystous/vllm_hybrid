# DeepSeek-R1 × suffix × mix

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 781.4 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 1585.9 |
| total_completion_tokens | 1239192 |
| TTFT p50/p99 ms | 196.8/303.3 |
| TPOT p50/p99 ms | 51.4/83.0 |
| accept α (acc/draft) | 0.4512 (740923.0/1642116.0) |
| GPU util / mem MiB | 94.1 / 1273376 |
| CPU util | 4.3 |
| reqtps_avg | 36.7 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 83.93 | 78.1 | 4.24 |
| p50 | 1225 | 63684.38 | 196.81 | 51.4 |
| p99 | 8192 | 437874.36 | 303.28 | 83.05 |
| max | 8192 | 578915.43 | 308.1 | 89.56 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1_suffix_mix.json`](../summ_DeepSeek-R1_suffix_mix.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1" and .method=="suffix" and .condition=="mix")' ../per_request_raw.jsonl
  ```
