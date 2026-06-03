# DeepSeek-R1-Distill-Qwen-32B × vanilla × mix

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 4938.2 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 254.3 |
| total_completion_tokens | 1255581 |
| TTFT p50/p99 ms | 24.1/119.4 |
| TPOT p50/p99 ms | 5.9/6.1 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 97.9 / 1267760 |
| CPU util | 4.9 |
| reqtps_avg | 168.8 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 7 | 60.39 | 21.2 | 5.27 |
| p50 | 1103 | 6359.96 | 24.08 | 5.91 |
| p99 | 8192 | 48843.41 | 119.36 | 6.11 |
| max | 8192 | 48947.42 | 122.35 | 6.32 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Qwen-32B_vanilla_mix.json`](../summ_DeepSeek-R1-Distill-Qwen-32B_vanilla_mix.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Qwen-32B" and .method=="vanilla" and .condition=="mix")' ../per_request_raw.jsonl
  ```
