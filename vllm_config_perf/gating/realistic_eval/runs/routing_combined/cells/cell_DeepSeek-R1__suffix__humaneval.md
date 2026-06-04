# DeepSeek-R1 × suffix × humaneval

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 606.4 |
| n_ok/n | 164/164 (err 0) |
| wall_total_s | 206.5 |
| total_completion_tokens | 125251 |
| TTFT p50/p99 ms | 96.4/278.1 |
| TPOT p50/p99 ms | 29.5/43.6 |
| accept α (acc/draft) | 0.5381 (85037.0/158036.0) |
| GPU util / mem MiB | 91.5 / 1273376 |
| CPU util | 4.3 |
| reqtps_avg | 38.9 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=164)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 8 | 328.89 | 75.63 | 4.25 |
| p50 | 93 | 2793.42 | 96.45 | 29.49 |
| p99 | 8192 | 144594.01 | 278.14 | 43.6 |
| max | 8192 | 205070.35 | 278.6 | 49.92 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1_suffix_humaneval.json`](../summ_DeepSeek-R1_suffix_humaneval.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1" and .method=="suffix" and (.condition=="humaneval" or (.condition==null and .corpus=="humaneval")))' ../per_request_raw.jsonl
  ```
