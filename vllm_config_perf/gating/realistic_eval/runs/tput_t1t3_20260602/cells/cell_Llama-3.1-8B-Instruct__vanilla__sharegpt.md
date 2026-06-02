# Llama-3.1-8B-Instruct × vanilla × sharegpt

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 8868.5 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 426.9 |
| total_completion_tokens | 3785679 |
| TTFT p50/p99 ms | 25.2/300.7 |
| TPOT p50/p99 ms | 3.5/3.5 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 94.7 / 1265344 |
| CPU util | 4.7 |
| reqtps_avg | 288.0 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 129 | 489.52 | 16.16 | 3.43 |
| p50 | 8192 | 28335.11 | 25.18 | 3.46 |
| p99 | 8192 | 28829.25 | 300.71 | 3.51 |
| max | 8192 | 28831.13 | 302.47 | 3.75 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-8B-Instruct_vanilla_sharegpt.json`](../summ_Llama-3.1-8B-Instruct_vanilla_sharegpt.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-8B-Instruct" and .method=="vanilla" and (.condition=="sharegpt" or (.condition==null and .corpus=="sharegpt")))' ../per_request_raw.jsonl
  ```
