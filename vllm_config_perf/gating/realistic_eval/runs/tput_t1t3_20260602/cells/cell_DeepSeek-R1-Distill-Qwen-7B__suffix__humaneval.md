# DeepSeek-R1-Distill-Qwen-7B × suffix × humaneval

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 11459.0 |
| n_ok/n | 164/164 (err 0) |
| wall_total_s | 73.3 |
| total_completion_tokens | 840013 |
| TTFT p50/p99 ms | 19.5/64.1 |
| TPOT p50/p99 ms | 4.1/5.7 |
| accept α (acc/draft) | 0.5476 (614282.0/1121720.0) |
| GPU util / mem MiB | 67.1 / 632552 |
| CPU util | 2.6 |
| reqtps_avg | 460.6 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=164)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 504 | 2504.24 | 15.64 | 0.66 |
| p50 | 5499 | 10889.16 | 19.51 | 4.07 |
| p99 | 8192 | 32350.37 | 64.08 | 5.68 |
| max | 8192 | 41550.36 | 64.65 | 5.84 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Qwen-7B_suffix_humaneval.json`](../summ_DeepSeek-R1-Distill-Qwen-7B_suffix_humaneval.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Qwen-7B" and .method=="suffix" and (.condition=="humaneval" or (.condition==null and .corpus=="humaneval")))' ../per_request_raw.jsonl
  ```
