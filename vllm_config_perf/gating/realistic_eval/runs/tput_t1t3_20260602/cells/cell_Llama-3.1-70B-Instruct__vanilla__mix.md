# Llama-3.1-70B-Instruct × vanilla × mix

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 3129.2 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 326.3 |
| total_completion_tokens | 1020968 |
| TTFT p50/p99 ms | 28.4/114.0 |
| TPOT p50/p99 ms | 9.2/9.6 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 98.5 / 1268912 |
| CPU util | 4.8 |
| reqtps_avg | 108.4 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 26.61 | 24.7 | 8.16 |
| p50 | 503 | 4560.9 | 28.42 | 9.16 |
| p99 | 8192 | 76326.44 | 114.0 | 9.64 |
| max | 8192 | 76435.07 | 117.56 | 10.21 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-70B-Instruct_vanilla_mix.json`](../summ_Llama-3.1-70B-Instruct_vanilla_mix.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-70B-Instruct" and .method=="vanilla" and .condition=="mix")' ../per_request_raw.jsonl
  ```
