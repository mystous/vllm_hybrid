# DeepSeek-R1-Distill-Qwen-32B × suffix × humaneval

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 3771.4 |
| n_ok/n | 164/164 (err 0) |
| wall_total_s | 63.0 |
| total_completion_tokens | 237779 |
| TTFT p50/p99 ms | 29.8/91.5 |
| TPOT p50/p99 ms | 8.4/10.8 |
| accept α (acc/draft) | 0.3808 (133101.0/349505.0) |
| GPU util / mem MiB | 81.1 / 1267776 |
| CPU util | 4.3 |
| reqtps_avg | 142.5 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=164)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 7 | 138.54 | 19.83 | 1.22 |
| p50 | 553 | 4564.67 | 29.76 | 8.37 |
| p99 | 8192 | 51777.07 | 91.45 | 10.83 |
| max | 8192 | 55713.15 | 94.37 | 11.21 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Qwen-32B_suffix_humaneval.json`](../summ_DeepSeek-R1-Distill-Qwen-32B_suffix_humaneval.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Qwen-32B" and .method=="suffix" and (.condition=="humaneval" or (.condition==null and .corpus=="humaneval")))' ../per_request_raw.jsonl
  ```
