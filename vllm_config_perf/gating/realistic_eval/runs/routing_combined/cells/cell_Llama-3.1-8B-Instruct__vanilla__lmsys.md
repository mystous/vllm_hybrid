# Llama-3.1-8B-Instruct × vanilla × lmsys

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 9073.8 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 430.2 |
| total_completion_tokens | 3903610 |
| TTFT p50/p99 ms | 35.3/72.8 |
| TPOT p50/p99 ms | 3.5/3.5 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 94.9 / 1265360 |
| CPU util | 4.6 |
| reqtps_avg | 286.5 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 47 | 195.81 | 16.07 | 3.41 |
| p50 | 8192 | 28521.96 | 35.27 | 3.48 |
| p99 | 8192 | 28890.81 | 72.8 | 3.52 |
| max | 8192 | 28893.03 | 78.51 | 3.66 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-8B-Instruct_vanilla_lmsys.json`](../summ_Llama-3.1-8B-Instruct_vanilla_lmsys.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-8B-Instruct" and .method=="vanilla" and (.condition=="lmsys" or (.condition==null and .corpus=="lmsys")))' ../per_request_raw.jsonl
  ```
