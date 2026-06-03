# Qwen2.5-72B-Instruct × suffix × mix

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 5268.4 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 125.6 |
| total_completion_tokens | 661862 |
| TTFT p50/p99 ms | 57.4/83.6 |
| TPOT p50/p99 ms | 2.6/18.2 |
| accept α (acc/draft) | 0.8516 (571592.0/671166.0) |
| GPU util / mem MiB | 82.5 / 1269136 |
| CPU util | 4.3 |
| reqtps_avg | 328.5 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 61.61 | 36.72 | 1.6 |
| p50 | 607 | 1656.2 | 57.42 | 2.61 |
| p99 | 8192 | 37576.94 | 83.56 | 18.16 |
| max | 8192 | 75574.13 | 100.39 | 23.19 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-72B-Instruct_suffix_mix.json`](../summ_Qwen2.5-72B-Instruct_suffix_mix.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-72B-Instruct" and .method=="suffix" and .condition=="mix")' ../per_request_raw.jsonl
  ```
