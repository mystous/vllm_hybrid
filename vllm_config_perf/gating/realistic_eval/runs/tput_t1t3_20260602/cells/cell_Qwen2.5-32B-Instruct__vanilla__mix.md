# Qwen2.5-32B-Instruct × vanilla × mix

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 3055.5 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 337.9 |
| total_completion_tokens | 1032438 |
| TTFT p50/p99 ms | 29.6/76.2 |
| TPOT p50/p99 ms | 9.3/10.4 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 94.3 / 1267776 |
| CPU util | 4.4 |
| reqtps_avg | 115.4 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 15 | 168.01 | 21.98 | 6.04 |
| p50 | 579 | 4929.16 | 29.65 | 9.26 |
| p99 | 8192 | 81483.66 | 76.24 | 10.39 |
| max | 8192 | 82142.76 | 79.02 | 11.39 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-32B-Instruct_vanilla_mix.json`](../summ_Qwen2.5-32B-Instruct_vanilla_mix.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-32B-Instruct" and .method=="vanilla" and .condition=="mix")' ../per_request_raw.jsonl
  ```
