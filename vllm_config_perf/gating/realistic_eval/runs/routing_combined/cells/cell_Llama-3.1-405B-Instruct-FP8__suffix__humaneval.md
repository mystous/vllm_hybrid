# Llama-3.1-405B-Instruct-FP8 × suffix × humaneval

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 2111.5 |
| n_ok/n | 164/164 (err 0) |
| wall_total_s | 426.8 |
| total_completion_tokens | 901278 |
| TTFT p50/p99 ms | 137.8/311.2 |
| TPOT p50/p99 ms | 14.7/47.8 |
| accept α (acc/draft) | 0.7027 (729436.0/1038058.0) |
| GPU util / mem MiB | 93.9 / 1272464 |
| CPU util | 4.3 |
| reqtps_avg | 87.4 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=164)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 14 | 667.4 | 65.16 | 4.36 |
| p50 | 8192 | 53692.83 | 137.78 | 14.72 |
| p99 | 8192 | 211136.74 | 311.21 | 47.75 |
| max | 8192 | 259260.74 | 313.49 | 51.32 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-405B-Instruct-FP8_suffix_humaneval.json`](../summ_Llama-3.1-405B-Instruct-FP8_suffix_humaneval.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-405B-Instruct-FP8" and .method=="suffix" and (.condition=="humaneval" or (.condition==null and .corpus=="humaneval")))' ../per_request_raw.jsonl
  ```
