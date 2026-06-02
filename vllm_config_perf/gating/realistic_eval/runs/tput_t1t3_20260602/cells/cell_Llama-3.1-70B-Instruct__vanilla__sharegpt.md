# Llama-3.1-70B-Instruct × vanilla × sharegpt

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 3090.8 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 296.8 |
| total_completion_tokens | 917186 |
| TTFT p50/p99 ms | 28.9/376.3 |
| TPOT p50/p99 ms | 9.1/9.6 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 98.4 / 1268896 |
| CPU util | 4.9 |
| reqtps_avg | 107.9 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 29.06 | 25.63 | 8.12 |
| p50 | 552 | 5045.46 | 28.94 | 9.14 |
| p99 | 8192 | 75937.83 | 376.27 | 9.59 |
| max | 8192 | 76006.74 | 377.84 | 9.65 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-70B-Instruct_vanilla_sharegpt.json`](../summ_Llama-3.1-70B-Instruct_vanilla_sharegpt.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-70B-Instruct" and .method=="vanilla" and (.condition=="sharegpt" or (.condition==null and .corpus=="sharegpt")))' ../per_request_raw.jsonl
  ```
