# DeepSeek-R1-Distill-Qwen-7B × llm-d × humaneval

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 9612.9 |
| n_ok/n | 164/164 (err 0) |
| wall_total_s | 84.0 |
| total_completion_tokens | 807349 |
| TTFT p50/p99 ms | 23.4/69.6 |
| TPOT p50/p99 ms | 3.2/5.2 |
| accept α (acc/draft) | 0.5477 (342883.0/626026.0) |
| GPU util / mem MiB | 76.6 / 614764 |
| CPU util | 3.7 |
| reqtps_avg | 413.9 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=164)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 504 | 2074.85 | 14.82 | 0.57 |
| p50 | 3706 | 9557.54 | 23.41 | 3.23 |
| p99 | 8192 | 26473.89 | 69.57 | 5.24 |
| max | 8192 | 31823.97 | 70.2 | 5.46 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Qwen-7B_llm-d_humaneval.json`](../summ_DeepSeek-R1-Distill-Qwen-7B_llm-d_humaneval.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Qwen-7B" and .method=="llm-d" and (.condition=="humaneval" or (.condition==null and .corpus=="humaneval")))' ../per_request_raw.jsonl
  ```
