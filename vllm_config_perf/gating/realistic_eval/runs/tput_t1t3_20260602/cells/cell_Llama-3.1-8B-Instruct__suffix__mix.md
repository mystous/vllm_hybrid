# Llama-3.1-8B-Instruct × suffix × mix

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 27851.4 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 136.3 |
| total_completion_tokens | 3796101 |
| TTFT p50/p99 ms | 24.7/65.4 |
| TPOT p50/p99 ms | 1.0/4.4 |
| accept α (acc/draft) | 0.9332 (3447279.0/3694153.0) |
| GPU util / mem MiB | 62.8 / 1265376 |
| CPU util | 4.4 |
| reqtps_avg | 928.7 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 21 | 106.81 | 17.74 | 0.68 |
| p50 | 8192 | 7896.82 | 24.66 | 0.98 |
| p99 | 8192 | 22861.48 | 65.37 | 4.39 |
| max | 8192 | 43317.88 | 68.71 | 5.96 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-8B-Instruct_suffix_mix.json`](../summ_Llama-3.1-8B-Instruct_suffix_mix.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-8B-Instruct" and .method=="suffix" and .condition=="mix")' ../per_request_raw.jsonl
  ```
