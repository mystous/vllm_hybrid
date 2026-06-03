# Llama-3.1-8B-Instruct × vanilla × mix

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 8849.9 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 429.0 |
| total_completion_tokens | 3796241 |
| TTFT p50/p99 ms | 22.8/59.9 |
| TPOT p50/p99 ms | 3.5/3.5 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 94.9 / 1265360 |
| CPU util | 4.6 |
| reqtps_avg | 286.3 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 21 | 105.7 | 15.04 | 3.44 |
| p50 | 8192 | 28533.36 | 22.78 | 3.48 |
| p99 | 8192 | 28707.95 | 59.93 | 3.52 |
| max | 8192 | 28719.18 | 63.83 | 4.07 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-8B-Instruct_vanilla_mix.json`](../summ_Llama-3.1-8B-Instruct_vanilla_mix.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-8B-Instruct" and .method=="vanilla" and .condition=="mix")' ../per_request_raw.jsonl
  ```
