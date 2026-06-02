# DeepSeek-R1-Distill-Qwen-7B × vanilla × wildchat

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 8925.3 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 201.2 |
| total_completion_tokens | 1795468 |
| TTFT p50/p99 ms | 16.9/64.4 |
| TPOT p50/p99 ms | 3.3/3.3 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 91.6 / 632552 |
| CPU util | 2.8 |
| reqtps_avg | 301.8 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 8 | 85.71 | 15.01 | 3.22 |
| p50 | 1654 | 5459.9 | 16.9 | 3.29 |
| p99 | 8192 | 27116.41 | 64.35 | 3.34 |
| max | 8192 | 27128.05 | 68.08 | 3.39 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Qwen-7B_vanilla_wildchat.json`](../summ_DeepSeek-R1-Distill-Qwen-7B_vanilla_wildchat.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Qwen-7B" and .method=="vanilla" and (.condition=="wildchat" or (.condition==null and .corpus=="wildchat")))' ../per_request_raw.jsonl
  ```
