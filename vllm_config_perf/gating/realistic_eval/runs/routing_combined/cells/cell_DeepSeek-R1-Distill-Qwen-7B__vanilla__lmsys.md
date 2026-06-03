# DeepSeek-R1-Distill-Qwen-7B × vanilla × lmsys

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 8810.8 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 157.4 |
| total_completion_tokens | 1386576 |
| TTFT p50/p99 ms | 16.8/65.8 |
| TPOT p50/p99 ms | 3.3/3.3 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 91.4 / 632552 |
| CPU util | 2.8 |
| reqtps_avg | 301.4 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 4 | 30.18 | 14.77 | 3.06 |
| p50 | 1146 | 3777.8 | 16.79 | 3.28 |
| p99 | 8192 | 27024.13 | 65.83 | 3.33 |
| max | 8192 | 27027.67 | 69.8 | 3.45 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Qwen-7B_vanilla_lmsys.json`](../summ_DeepSeek-R1-Distill-Qwen-7B_vanilla_lmsys.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Qwen-7B" and .method=="vanilla" and (.condition=="lmsys" or (.condition==null and .corpus=="lmsys")))' ../per_request_raw.jsonl
  ```
