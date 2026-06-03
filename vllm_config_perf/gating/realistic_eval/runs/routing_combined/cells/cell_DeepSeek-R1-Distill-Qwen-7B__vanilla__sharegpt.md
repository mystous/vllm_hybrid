# DeepSeek-R1-Distill-Qwen-7B × vanilla × sharegpt

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 8723.7 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 182.5 |
| total_completion_tokens | 1591660 |
| TTFT p50/p99 ms | 17.0/299.0 |
| TPOT p50/p99 ms | 3.3/3.5 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 91.1 / 632543 |
| CPU util | 2.9 |
| reqtps_avg | 298.2 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 33.46 | 15.1 | 3.25 |
| p50 | 1467 | 4864.88 | 17.01 | 3.3 |
| p99 | 8192 | 27670.56 | 298.95 | 3.53 |
| max | 8192 | 27671.79 | 299.7 | 5.47 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Qwen-7B_vanilla_sharegpt.json`](../summ_DeepSeek-R1-Distill-Qwen-7B_vanilla_sharegpt.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Qwen-7B" and .method=="vanilla" and (.condition=="sharegpt" or (.condition==null and .corpus=="sharegpt")))' ../per_request_raw.jsonl
  ```
