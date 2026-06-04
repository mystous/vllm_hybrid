# DeepSeek-R1-Distill-Qwen-32B × llm-d × humaneval

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 3288.2 |
| n_ok/n | 164/164 (err 0) |
| wall_total_s | 74.8 |
| total_completion_tokens | 246123 |
| TTFT p50/p99 ms | 28.2/74.9 |
| TPOT p50/p99 ms | 6.3/8.7 |
| accept α (acc/draft) | 0.4218 (63976.0/151664.0) |
| GPU util / mem MiB | 76.8 / 1230904 |
| CPU util | 5.1 |
| reqtps_avg | 166.5 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=164)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 7 | 108.15 | 19.09 | 0.94 |
| p50 | 529 | 3545.44 | 28.23 | 6.26 |
| p99 | 8192 | 51613.12 | 74.87 | 8.73 |
| max | 8192 | 51654.0 | 75.1 | 8.82 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Qwen-32B_llm-d_humaneval.json`](../summ_DeepSeek-R1-Distill-Qwen-32B_llm-d_humaneval.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Qwen-32B" and .method=="llm-d" and (.condition=="humaneval" or (.condition==null and .corpus=="humaneval")))' ../per_request_raw.jsonl
  ```
