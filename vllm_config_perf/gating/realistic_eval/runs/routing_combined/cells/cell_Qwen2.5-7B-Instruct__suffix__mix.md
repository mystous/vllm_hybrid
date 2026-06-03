# Qwen2.5-7B-Instruct × suffix × mix

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 7803.4 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 178.0 |
| total_completion_tokens | 1388891 |
| TTFT p50/p99 ms | 69.3/85.6 |
| TPOT p50/p99 ms | 3.1/23.2 |
| accept α (acc/draft) | 0.8814 (1236294.0/1402603.0) |
| GPU util / mem MiB | 26.5 / 632567 |
| CPU util | 2.5 |
| reqtps_avg | 313.5 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 2 | 121.14 | 23.01 | 1.0 |
| p50 | 706 | 2322.46 | 69.26 | 3.14 |
| p99 | 8192 | 69436.72 | 85.56 | 23.23 |
| max | 8192 | 136125.03 | 114.45 | 39.78 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-7B-Instruct_suffix_mix.json`](../summ_Qwen2.5-7B-Instruct_suffix_mix.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-7B-Instruct" and .method=="suffix" and .condition=="mix")' ../per_request_raw.jsonl
  ```
