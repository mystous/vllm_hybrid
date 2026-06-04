# DeepSeek-R1 × llm-d × mix

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 1008.2 |
| n_ok/n | 488/500 (err 12) |
| wall_total_s | 1207.2 |
| total_completion_tokens | 1217092 |
| TTFT p50/p99 ms | 93.2/255.5 |
| TPOT p50/p99 ms | 21.3/80.2 |
| accept α (acc/draft) | 0.4642 (321617.0/692894.0) |
| GPU util / mem MiB | 92.2 / 1419623 |
| CPU util | 5.1 |
| reqtps_avg | 40.2 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=488)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 57.69 | 53.03 | 6.73 |
| p50 | 1159 | 30227.13 | 93.23 | 21.29 |
| p99 | 8192 | 277323.2 | 255.53 | 80.16 |
| max | 8192 | 298647.73 | 344.49 | 85.53 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1_llm-d_mix.json`](../summ_DeepSeek-R1_llm-d_mix.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1" and .method=="llm-d" and .condition=="mix")' ../per_request_raw.jsonl
  ```
