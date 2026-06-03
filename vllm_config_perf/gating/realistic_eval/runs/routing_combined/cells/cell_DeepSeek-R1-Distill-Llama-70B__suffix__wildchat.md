# DeepSeek-R1-Distill-Llama-70B × suffix × wildchat

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 2658.1 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 378.0 |
| total_completion_tokens | 1004834 |
| TTFT p50/p99 ms | 43.1/132.3 |
| TPOT p50/p99 ms | 14.7/18.3 |
| accept α (acc/draft) | 0.3542 (470175.0/1327312.0) |
| GPU util / mem MiB | 86.3 / 1268784 |
| CPU util | 4.3 |
| reqtps_avg | 91.2 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 41.21 | 31.89 | 1.51 |
| p50 | 1293 | 17955.93 | 43.09 | 14.7 |
| p99 | 8192 | 85716.27 | 132.35 | 18.31 |
| max | 8192 | 101171.91 | 134.56 | 26.69 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Llama-70B_suffix_wildchat.json`](../summ_DeepSeek-R1-Distill-Llama-70B_suffix_wildchat.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Llama-70B" and .method=="suffix" and (.condition=="wildchat" or (.condition==null and .corpus=="wildchat")))' ../per_request_raw.jsonl
  ```
