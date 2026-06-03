# Llama-3.1-8B-Instruct × llm-d × sharegpt

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 13907.4 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 274.2 |
| total_completion_tokens | 3813935 |
| TTFT p50/p99 ms | 21.6/280.4 |
| TPOT p50/p99 ms | 1.7/6.0 |
| accept α (acc/draft) | 0.8605 (1984985.0/2306702.0) |
| GPU util / mem MiB | 81.6 / 1228568 |
| CPU util | 5.3 |
| reqtps_avg | 674.0 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 138 | 493.59 | 15.75 | 0.53 |
| p50 | 8192 | 10986.89 | 21.64 | 1.71 |
| p99 | 8192 | 29364.97 | 280.35 | 6.05 |
| max | 8192 | 47678.78 | 283.11 | 7.31 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-8B-Instruct_llm-d_sharegpt.json`](../summ_Llama-3.1-8B-Instruct_llm-d_sharegpt.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-8B-Instruct" and .method=="llm-d" and (.condition=="sharegpt" or (.condition==null and .corpus=="sharegpt")))' ../per_request_raw.jsonl
  ```
