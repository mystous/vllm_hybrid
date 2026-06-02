# DeepSeek-R1-Distill-Qwen-7B × suffix × wildchat

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 11717.2 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 153.5 |
| total_completion_tokens | 1798103 |
| TTFT p50/p99 ms | 19.3/69.1 |
| TPOT p50/p99 ms | 5.3/7.8 |
| accept α (acc/draft) | 0.6094 (1291182.0/2118866.0) |
| GPU util / mem MiB | 67.9 / 632552 |
| CPU util | 2.6 |
| reqtps_avg | 428.5 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 8 | 104.66 | 14.93 | 0.6 |
| p50 | 1643 | 7702.91 | 19.3 | 5.34 |
| p99 | 8192 | 32039.05 | 69.1 | 7.75 |
| max | 8192 | 56696.74 | 71.93 | 8.36 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Qwen-7B_suffix_wildchat.json`](../summ_DeepSeek-R1-Distill-Qwen-7B_suffix_wildchat.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Qwen-7B" and .method=="suffix" and (.condition=="wildchat" or (.condition==null and .corpus=="wildchat")))' ../per_request_raw.jsonl
  ```
