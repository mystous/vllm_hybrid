# Qwen2.5-32B-Instruct × vanilla × sharegpt

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 3079.4 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 334.2 |
| total_completion_tokens | 1029247 |
| TTFT p50/p99 ms | 31.4/437.6 |
| TPOT p50/p99 ms | 9.0/11.2 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 94.1 / 1267760 |
| CPU util | 4.4 |
| reqtps_avg | 112.5 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 13 | 163.45 | 23.78 | 6.33 |
| p50 | 613 | 5399.29 | 31.44 | 9.04 |
| p99 | 8192 | 79533.77 | 437.65 | 11.24 |
| max | 8192 | 79677.7 | 517.7 | 11.56 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-32B-Instruct_vanilla_sharegpt.json`](../summ_Qwen2.5-32B-Instruct_vanilla_sharegpt.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-32B-Instruct" and .method=="vanilla" and (.condition=="sharegpt" or (.condition==null and .corpus=="sharegpt")))' ../per_request_raw.jsonl
  ```
