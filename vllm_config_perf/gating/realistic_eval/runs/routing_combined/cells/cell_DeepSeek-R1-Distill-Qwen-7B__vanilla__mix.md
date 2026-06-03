# DeepSeek-R1-Distill-Qwen-7B × vanilla × mix

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 9058.2 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 218.9 |
| total_completion_tokens | 1983139 |
| TTFT p50/p99 ms | 16.8/46.4 |
| TPOT p50/p99 ms | 3.3/3.3 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 91.8 / 632552 |
| CPU util | 2.8 |
| reqtps_avg | 303.8 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 8 | 39.94 | 14.39 | 3.22 |
| p50 | 1895 | 6264.12 | 16.8 | 3.27 |
| p99 | 8192 | 26977.29 | 46.41 | 3.31 |
| max | 8192 | 26982.33 | 49.91 | 3.48 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Qwen-7B_vanilla_mix.json`](../summ_DeepSeek-R1-Distill-Qwen-7B_vanilla_mix.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Qwen-7B" and .method=="vanilla" and .condition=="mix")' ../per_request_raw.jsonl
  ```
