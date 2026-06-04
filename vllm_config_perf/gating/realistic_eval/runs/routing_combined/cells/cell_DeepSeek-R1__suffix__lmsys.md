# DeepSeek-R1 × suffix × lmsys

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 811.1 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 1241.9 |
| total_completion_tokens | 1007279 |
| TTFT p50/p99 ms | 174.8/251.3 |
| TPOT p50/p99 ms | 57.1/88.3 |
| accept α (acc/draft) | 0.453 (560594.0/1237433.0) |
| GPU util / mem MiB | 94.0 / 1273376 |
| CPU util | 4.3 |
| reqtps_avg | 27.2 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 79.4 | 75.63 | 4.73 |
| p50 | 921 | 53413.95 | 174.84 | 57.06 |
| p99 | 8192 | 417183.13 | 251.31 | 88.35 |
| max | 8192 | 651957.07 | 262.41 | 100.89 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1_suffix_lmsys.json`](../summ_DeepSeek-R1_suffix_lmsys.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1" and .method=="suffix" and (.condition=="lmsys" or (.condition==null and .corpus=="lmsys")))' ../per_request_raw.jsonl
  ```
