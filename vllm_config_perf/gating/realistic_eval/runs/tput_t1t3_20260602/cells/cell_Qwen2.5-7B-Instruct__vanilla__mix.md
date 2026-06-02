# Qwen2.5-7B-Instruct × vanilla × mix

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 4168.9 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 326.9 |
| total_completion_tokens | 1362873 |
| TTFT p50/p99 ms | 26.2/46.2 |
| TPOT p50/p99 ms | 6.9/8.7 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 82.5 / 632552 |
| CPU util | 2.7 |
| reqtps_avg | 157.9 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 2 | 24.37 | 16.56 | 3.53 |
| p50 | 693 | 4410.81 | 26.17 | 6.92 |
| p99 | 8192 | 62022.79 | 46.24 | 8.71 |
| max | 8192 | 62097.09 | 58.19 | 9.07 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-7B-Instruct_vanilla_mix.json`](../summ_Qwen2.5-7B-Instruct_vanilla_mix.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-7B-Instruct" and .method=="vanilla" and .condition=="mix")' ../per_request_raw.jsonl
  ```
