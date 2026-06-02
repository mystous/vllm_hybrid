# Llama-3.1-8B-Instruct × suffix × lmsys

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 19862.3 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 195.8 |
| total_completion_tokens | 3888816 |
| TTFT p50/p99 ms | 21.6/71.7 |
| TPOT p50/p99 ms | 1.3/8.4 |
| accept α (acc/draft) | 0.8486 (3345916.0/3943000.0) |
| GPU util / mem MiB | 67.0 / 1265376 |
| CPU util | 4.4 |
| reqtps_avg | 756.5 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 47 | 253.19 | 17.91 | 0.7 |
| p50 | 8192 | 10173.09 | 21.64 | 1.3 |
| p99 | 8192 | 35977.56 | 71.72 | 8.36 |
| max | 8192 | 45437.11 | 75.24 | 9.3 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-8B-Instruct_suffix_lmsys.json`](../summ_Llama-3.1-8B-Instruct_suffix_lmsys.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-8B-Instruct" and .method=="suffix" and (.condition=="lmsys" or (.condition==null and .corpus=="lmsys")))' ../per_request_raw.jsonl
  ```
