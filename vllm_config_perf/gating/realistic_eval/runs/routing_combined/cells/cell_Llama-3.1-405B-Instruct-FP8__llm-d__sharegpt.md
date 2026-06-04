# Llama-3.1-405B-Instruct-FP8 × llm-d × sharegpt

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 1267.0 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 751.3 |
| total_completion_tokens | 951960 |
| TTFT p50/p99 ms | 96.5/558.6 |
| TPOT p50/p99 ms | 28.8/49.4 |
| accept α (acc/draft) | 0.7043 (419722.0/595972.0) |
| GPU util / mem MiB | 85.3 / 1234274 |
| CPU util | 4.8 |
| reqtps_avg | 46.6 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 91.95 | 51.14 | 3.3 |
| p50 | 565 | 18716.84 | 96.47 | 28.83 |
| p99 | 8192 | 231492.45 | 558.64 | 49.39 |
| max | 8192 | 285565.22 | 568.92 | 55.59 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-405B-Instruct-FP8_llm-d_sharegpt.json`](../summ_Llama-3.1-405B-Instruct-FP8_llm-d_sharegpt.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-405B-Instruct-FP8" and .method=="llm-d" and (.condition=="sharegpt" or (.condition==null and .corpus=="sharegpt")))' ../per_request_raw.jsonl
  ```
