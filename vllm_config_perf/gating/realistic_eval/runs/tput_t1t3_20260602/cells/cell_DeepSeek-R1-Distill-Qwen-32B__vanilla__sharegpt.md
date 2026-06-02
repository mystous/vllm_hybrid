# DeepSeek-R1-Distill-Qwen-32B × vanilla × sharegpt

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 4803.3 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 236.7 |
| total_completion_tokens | 1136980 |
| TTFT p50/p99 ms | 24.1/355.0 |
| TPOT p50/p99 ms | 5.9/6.1 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 97.6 / 1267751 |
| CPU util | 4.7 |
| reqtps_avg | 168.6 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 9 | 72.35 | 19.81 | 5.47 |
| p50 | 1259 | 7394.52 | 24.08 | 5.86 |
| p99 | 8192 | 48548.8 | 354.96 | 6.07 |
| max | 8192 | 48679.37 | 355.55 | 7.39 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Qwen-32B_vanilla_sharegpt.json`](../summ_DeepSeek-R1-Distill-Qwen-32B_vanilla_sharegpt.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Qwen-32B" and .method=="vanilla" and (.condition=="sharegpt" or (.condition==null and .corpus=="sharegpt")))' ../per_request_raw.jsonl
  ```
