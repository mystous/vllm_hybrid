# DeepSeek-R1-Distill-Llama-70B × suffix × sharegpt

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 2659.7 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 307.7 |
| total_completion_tokens | 818351 |
| TTFT p50/p99 ms | 38.5/548.8 |
| TPOT p50/p99 ms | 13.8/17.8 |
| accept α (acc/draft) | 0.3459 (345658.0/999349.0) |
| GPU util / mem MiB | 85.8 / 1268752 |
| CPU util | 4.3 |
| reqtps_avg | 92.2 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 39.89 | 28.44 | 1.42 |
| p50 | 1120 | 15422.66 | 38.55 | 13.77 |
| p99 | 8192 | 62403.31 | 548.84 | 17.8 |
| max | 8192 | 93030.2 | 549.5 | 19.72 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Llama-70B_suffix_sharegpt.json`](../summ_DeepSeek-R1-Distill-Llama-70B_suffix_sharegpt.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Llama-70B" and .method=="suffix" and (.condition=="sharegpt" or (.condition==null and .corpus=="sharegpt")))' ../per_request_raw.jsonl
  ```
