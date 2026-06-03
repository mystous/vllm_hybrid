# Llama-3.1-70B-Instruct × llm-d × sharegpt

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 3319.0 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 290.6 |
| total_completion_tokens | 964352 |
| TTFT p50/p99 ms | 42.7/400.7 |
| TPOT p50/p99 ms | 12.3/20.4 |
| accept α (acc/draft) | 0.7021 (414733.0/590703.0) |
| GPU util / mem MiB | 83.4 / 1231043 |
| CPU util | 5.2 |
| reqtps_avg | 114.0 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 39.76 | 26.11 | 1.42 |
| p50 | 563 | 7671.68 | 42.7 | 12.28 |
| p99 | 8192 | 85838.65 | 400.65 | 20.36 |
| max | 8192 | 85889.11 | 403.14 | 22.66 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-70B-Instruct_llm-d_sharegpt.json`](../summ_Llama-3.1-70B-Instruct_llm-d_sharegpt.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-70B-Instruct" and .method=="llm-d" and (.condition=="sharegpt" or (.condition==null and .corpus=="sharegpt")))' ../per_request_raw.jsonl
  ```
