# DeepSeek-R1-Distill-Llama-70B × llm-d × sharegpt

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 2450.0 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 333.2 |
| total_completion_tokens | 816237 |
| TTFT p50/p99 ms | 39.1/362.0 |
| TPOT p50/p99 ms | 10.3/16.8 |
| accept α (acc/draft) | 0.343 (179710.0/524012.0) |
| GPU util / mem MiB | 88.3 / 1231042 |
| CPU util | 5.5 |
| reqtps_avg | 92.9 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 41.64 | 26.5 | 1.44 |
| p50 | 1154 | 13391.84 | 39.12 | 10.35 |
| p99 | 8192 | 82155.05 | 362.02 | 16.82 |
| max | 8192 | 96952.52 | 363.62 | 18.79 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Llama-70B_llm-d_sharegpt.json`](../summ_DeepSeek-R1-Distill-Llama-70B_llm-d_sharegpt.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Llama-70B" and .method=="llm-d" and (.condition=="sharegpt" or (.condition==null and .corpus=="sharegpt")))' ../per_request_raw.jsonl
  ```
