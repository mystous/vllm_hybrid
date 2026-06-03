# Qwen2.5-32B-Instruct × suffix × mix

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 6596.9 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 168.7 |
| total_completion_tokens | 1113169 |
| TTFT p50/p99 ms | 65.4/96.8 |
| TPOT p50/p99 ms | 3.1/22.6 |
| accept α (acc/draft) | 0.8571 (972556.0/1134741.0) |
| GPU util / mem MiB | 64.8 / 1267776 |
| CPU util | 4.3 |
| reqtps_avg | 303.3 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 25 | 157.6 | 28.34 | 1.24 |
| p50 | 605 | 2033.88 | 65.42 | 3.1 |
| p99 | 8192 | 56253.52 | 96.82 | 22.57 |
| max | 8192 | 147352.99 | 114.61 | 27.88 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-32B-Instruct_suffix_mix.json`](../summ_Qwen2.5-32B-Instruct_suffix_mix.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-32B-Instruct" and .method=="suffix" and .condition=="mix")' ../per_request_raw.jsonl
  ```
