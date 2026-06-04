# DeepSeek-R1 × vanilla × mix

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 1537.6 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 775.2 |
| total_completion_tokens | 1191971 |
| TTFT p50/p99 ms | 65.6/259.2 |
| TPOT p50/p99 ms | 19.7/20.4 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 98.6 / 1277680 |
| CPU util | 4.8 |
| reqtps_avg | 48.9 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 64.06 | 51.56 | 15.11 |
| p50 | 1171 | 22800.73 | 65.58 | 19.66 |
| p99 | 8192 | 161999.7 | 259.19 | 20.38 |
| max | 8192 | 162309.58 | 261.36 | 22.26 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1_vanilla_mix.json`](../summ_DeepSeek-R1_vanilla_mix.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1" and .method=="vanilla" and .condition=="mix")' ../per_request_raw.jsonl
  ```
