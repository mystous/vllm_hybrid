# DeepSeek-R1-Distill-Llama-70B × vanilla × mix

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 3163.7 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 328.5 |
| total_completion_tokens | 1039276 |
| TTFT p50/p99 ms | 28.4/128.0 |
| TPOT p50/p99 ms | 9.0/9.2 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 98.3 / 1268912 |
| CPU util | 4.9 |
| reqtps_avg | 111.4 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 27.28 | 24.82 | 7.89 |
| p50 | 1291 | 11578.48 | 28.43 | 9.02 |
| p99 | 8192 | 74294.76 | 127.97 | 9.15 |
| max | 8192 | 74336.22 | 129.21 | 9.53 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Llama-70B_vanilla_mix.json`](../summ_DeepSeek-R1-Distill-Llama-70B_vanilla_mix.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Llama-70B" and .method=="vanilla" and .condition=="mix")' ../per_request_raw.jsonl
  ```
