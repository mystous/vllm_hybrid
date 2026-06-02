# DeepSeek-R1-Distill-Qwen-32B × suffix × wildchat

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 5728.6 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 247.4 |
| total_completion_tokens | 1417136 |
| TTFT p50/p99 ms | 34.2/99.8 |
| TPOT p50/p99 ms | 10.7/14.4 |
| accept α (acc/draft) | 0.5729 (958917.0/1673670.0) |
| GPU util / mem MiB | 82.0 / 1267776 |
| CPU util | 4.4 |
| reqtps_avg | 198.2 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 8 | 93.26 | 21.74 | 1.12 |
| p50 | 1386 | 12699.46 | 34.24 | 10.66 |
| p99 | 8192 | 57029.09 | 99.8 | 14.43 |
| max | 8192 | 82574.98 | 103.84 | 17.04 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Qwen-32B_suffix_wildchat.json`](../summ_DeepSeek-R1-Distill-Qwen-32B_suffix_wildchat.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Qwen-32B" and .method=="suffix" and (.condition=="wildchat" or (.condition==null and .corpus=="wildchat")))' ../per_request_raw.jsonl
  ```
