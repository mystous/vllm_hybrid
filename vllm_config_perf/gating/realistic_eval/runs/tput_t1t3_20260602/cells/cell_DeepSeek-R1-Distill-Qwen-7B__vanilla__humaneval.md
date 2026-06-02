# DeepSeek-R1-Distill-Qwen-7B × vanilla × humaneval

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 8159.3 |
| n_ok/n | 164/164 (err 0) |
| wall_total_s | 104.2 |
| total_completion_tokens | 849912 |
| TTFT p50/p99 ms | 19.0/59.9 |
| TPOT p50/p99 ms | 3.3/3.3 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 91.0 / 632552 |
| CPU util | 2.8 |
| reqtps_avg | 305.0 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=164)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 483 | 1588.96 | 15.33 | 3.23 |
| p50 | 6291 | 20684.34 | 19.02 | 3.27 |
| p99 | 8192 | 26949.77 | 59.9 | 3.29 |
| max | 8192 | 26950.9 | 61.93 | 3.29 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Qwen-7B_vanilla_humaneval.json`](../summ_DeepSeek-R1-Distill-Qwen-7B_vanilla_humaneval.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Qwen-7B" and .method=="vanilla" and (.condition=="humaneval" or (.condition==null and .corpus=="humaneval")))' ../per_request_raw.jsonl
  ```
