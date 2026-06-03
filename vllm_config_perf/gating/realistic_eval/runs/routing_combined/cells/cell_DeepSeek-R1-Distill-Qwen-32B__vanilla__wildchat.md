# DeepSeek-R1-Distill-Qwen-32B × vanilla × wildchat

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 4890.9 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 300.1 |
| total_completion_tokens | 1467518 |
| TTFT p50/p99 ms | 24.2/100.6 |
| TPOT p50/p99 ms | 5.9/6.1 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 97.8 / 1267776 |
| CPU util | 4.8 |
| reqtps_avg | 168.4 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 8 | 66.37 | 19.72 | 5.67 |
| p50 | 1471 | 8706.68 | 24.23 | 5.9 |
| p99 | 8192 | 49052.28 | 100.59 | 6.09 |
| max | 8192 | 49121.74 | 103.77 | 6.92 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Qwen-32B_vanilla_wildchat.json`](../summ_DeepSeek-R1-Distill-Qwen-32B_vanilla_wildchat.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Qwen-32B" and .method=="vanilla" and (.condition=="wildchat" or (.condition==null and .corpus=="wildchat")))' ../per_request_raw.jsonl
  ```
