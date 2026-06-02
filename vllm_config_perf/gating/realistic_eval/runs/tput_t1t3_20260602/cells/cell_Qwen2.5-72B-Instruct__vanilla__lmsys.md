# Qwen2.5-72B-Instruct × vanilla × lmsys

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 2806.9 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 293.5 |
| total_completion_tokens | 823796 |
| TTFT p50/p99 ms | 32.9/164.3 |
| TPOT p50/p99 ms | 9.8/11.9 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 97.5 / 1269104 |
| CPU util | 4.5 |
| reqtps_avg | 100.5 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 2 | 39.38 | 25.63 | 8.49 |
| p50 | 505 | 4905.92 | 32.87 | 9.84 |
| p99 | 8192 | 85870.95 | 164.32 | 11.86 |
| max | 8192 | 86110.09 | 169.66 | 12.25 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-72B-Instruct_vanilla_lmsys.json`](../summ_Qwen2.5-72B-Instruct_vanilla_lmsys.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-72B-Instruct" and .method=="vanilla" and (.condition=="lmsys" or (.condition==null and .corpus=="lmsys")))' ../per_request_raw.jsonl
  ```
