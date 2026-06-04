# DeepSeek-R1 × suffix × wildchat

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 858.4 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 1606.5 |
| total_completion_tokens | 1379121 |
| TTFT p50/p99 ms | 179.4/288.0 |
| TPOT p50/p99 ms | 61.2/95.6 |
| accept α (acc/draft) | 0.528 (828616.0/1569367.0) |
| GPU util / mem MiB | 94.1 / 1273376 |
| CPU util | 4.3 |
| reqtps_avg | 33.6 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 84.58 | 80.91 | 4.57 |
| p50 | 1549 | 74400.34 | 179.37 | 61.23 |
| p99 | 8192 | 469424.48 | 288.05 | 95.6 |
| max | 8192 | 608332.4 | 289.28 | 105.79 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1_suffix_wildchat.json`](../summ_DeepSeek-R1_suffix_wildchat.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1" and .method=="suffix" and (.condition=="wildchat" or (.condition==null and .corpus=="wildchat")))' ../per_request_raw.jsonl
  ```
