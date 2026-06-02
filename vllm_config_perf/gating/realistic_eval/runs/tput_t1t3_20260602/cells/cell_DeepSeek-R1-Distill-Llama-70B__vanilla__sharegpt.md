# DeepSeek-R1-Distill-Llama-70B × vanilla × sharegpt

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 3033.1 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 277.0 |
| total_completion_tokens | 840125 |
| TTFT p50/p99 ms | 28.2/371.4 |
| TPOT p50/p99 ms | 8.9/9.3 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 98.2 / 1268896 |
| CPU util | 4.8 |
| reqtps_avg | 111.4 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 29.14 | 25.56 | 7.63 |
| p50 | 1135 | 10177.0 | 28.2 | 8.95 |
| p99 | 8192 | 74183.15 | 371.43 | 9.33 |
| max | 8192 | 74506.52 | 371.87 | 11.36 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Llama-70B_vanilla_sharegpt.json`](../summ_DeepSeek-R1-Distill-Llama-70B_vanilla_sharegpt.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Llama-70B" and .method=="vanilla" and (.condition=="sharegpt" or (.condition==null and .corpus=="sharegpt")))' ../per_request_raw.jsonl
  ```
