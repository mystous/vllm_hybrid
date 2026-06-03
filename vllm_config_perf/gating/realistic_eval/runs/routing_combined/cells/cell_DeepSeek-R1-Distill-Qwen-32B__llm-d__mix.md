# DeepSeek-R1-Distill-Qwen-32B × llm-d × mix

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 5852.3 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 214.5 |
| total_completion_tokens | 1255177 |
| TTFT p50/p99 ms | 29.8/78.9 |
| TPOT p50/p99 ms | 6.5/11.0 |
| accept α (acc/draft) | 0.6698 (596596.0/890676.0) |
| GPU util / mem MiB | 87.3 / 1230912 |
| CPU util | 5.6 |
| reqtps_avg | 263.5 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 7 | 68.76 | 21.6 | 0.94 |
| p50 | 1116 | 7603.98 | 29.83 | 6.55 |
| p99 | 8192 | 56484.74 | 78.87 | 11.04 |
| max | 8192 | 56732.12 | 95.47 | 12.85 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Qwen-32B_llm-d_mix.json`](../summ_DeepSeek-R1-Distill-Qwen-32B_llm-d_mix.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Qwen-32B" and .method=="llm-d" and .condition=="mix")' ../per_request_raw.jsonl
  ```
