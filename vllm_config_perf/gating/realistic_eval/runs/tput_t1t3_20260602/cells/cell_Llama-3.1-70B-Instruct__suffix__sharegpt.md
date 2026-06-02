# Llama-3.1-70B-Instruct × suffix × sharegpt

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 4864.4 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 202.2 |
| total_completion_tokens | 983616 |
| TTFT p50/p99 ms | 45.0/384.1 |
| TPOT p50/p99 ms | 15.4/21.0 |
| accept α (acc/draft) | 0.7353 (732789.0/996534.0) |
| GPU util / mem MiB | 86.4 / 1268754 |
| CPU util | 4.3 |
| reqtps_avg | 131.1 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 44.0 | 29.6 | 1.54 |
| p50 | 562 | 8924.61 | 44.98 | 15.45 |
| p99 | 8192 | 57130.89 | 384.1 | 21.03 |
| max | 8192 | 123282.01 | 385.41 | 23.36 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-70B-Instruct_suffix_sharegpt.json`](../summ_Llama-3.1-70B-Instruct_suffix_sharegpt.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-70B-Instruct" and .method=="suffix" and (.condition=="sharegpt" or (.condition==null and .corpus=="sharegpt")))' ../per_request_raw.jsonl
  ```
