# Qwen2.5-72B-Instruct × vanilla × mix

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 2734.8 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 241.7 |
| total_completion_tokens | 660869 |
| TTFT p50/p99 ms | 31.2/70.2 |
| TPOT p50/p99 ms | 9.3/10.6 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 97.6 / 1269104 |
| CPU util | 4.5 |
| reqtps_avg | 105.0 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 28.98 | 24.29 | 8.41 |
| p50 | 600 | 5621.69 | 31.15 | 9.35 |
| p99 | 8192 | 80848.46 | 70.19 | 10.57 |
| max | 8192 | 81323.0 | 73.37 | 10.77 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-72B-Instruct_vanilla_mix.json`](../summ_Qwen2.5-72B-Instruct_vanilla_mix.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-72B-Instruct" and .method=="vanilla" and .condition=="mix")' ../per_request_raw.jsonl
  ```
