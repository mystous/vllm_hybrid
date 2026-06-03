# DeepSeek-R1-Distill-Llama-70B × llm-d × mix

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 2863.5 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 362.3 |
| total_completion_tokens | 1037328 |
| TTFT p50/p99 ms | 40.3/124.6 |
| TPOT p50/p99 ms | 10.2/15.7 |
| accept α (acc/draft) | 0.3929 (326407.0/830767.0) |
| GPU util / mem MiB | 90.5 / 1231062 |
| CPU util | 5.6 |
| reqtps_avg | 123.0 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 31.37 | 28.75 | 1.33 |
| p50 | 1259 | 13241.09 | 40.32 | 10.25 |
| p99 | 8192 | 84550.07 | 124.62 | 15.7 |
| max | 8192 | 88790.38 | 165.89 | 17.35 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Llama-70B_llm-d_mix.json`](../summ_DeepSeek-R1-Distill-Llama-70B_llm-d_mix.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Llama-70B" and .method=="llm-d" and .condition=="mix")' ../per_request_raw.jsonl
  ```
