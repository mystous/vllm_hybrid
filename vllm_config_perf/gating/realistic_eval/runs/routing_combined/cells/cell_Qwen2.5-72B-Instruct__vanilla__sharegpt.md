# Qwen2.5-72B-Instruct × vanilla × sharegpt

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 2687.9 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 263.7 |
| total_completion_tokens | 708911 |
| TTFT p50/p99 ms | 32.3/473.4 |
| TPOT p50/p99 ms | 9.6/11.1 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 97.7 / 1269094 |
| CPU util | 4.5 |
| reqtps_avg | 103.2 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 27 | 270.95 | 25.35 | 8.16 |
| p50 | 704 | 6777.74 | 32.34 | 9.56 |
| p99 | 8192 | 81883.43 | 473.44 | 11.08 |
| max | 8192 | 82711.74 | 478.84 | 11.44 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-72B-Instruct_vanilla_sharegpt.json`](../summ_Qwen2.5-72B-Instruct_vanilla_sharegpt.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-72B-Instruct" and .method=="vanilla" and (.condition=="sharegpt" or (.condition==null and .corpus=="sharegpt")))' ../per_request_raw.jsonl
  ```
