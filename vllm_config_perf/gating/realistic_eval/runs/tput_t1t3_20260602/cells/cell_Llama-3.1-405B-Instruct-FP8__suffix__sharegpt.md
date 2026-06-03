# Llama-3.1-405B-Instruct-FP8 × suffix × sharegpt

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 2061.1 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 482.2 |
| total_completion_tokens | 993784 |
| TTFT p50/p99 ms | 97.8/652.2 |
| TPOT p50/p99 ms | 33.4/46.8 |
| accept α (acc/draft) | 0.6874 (713425.0/1037813.0) |
| GPU util / mem MiB | 93.1 / 1272435 |
| CPU util | 4.3 |
| reqtps_avg | 57.5 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 94.82 | 59.25 | 2.99 |
| p50 | 556 | 18766.47 | 97.76 | 33.35 |
| p99 | 8192 | 188692.81 | 652.19 | 46.81 |
| max | 8192 | 286788.08 | 653.45 | 50.4 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-405B-Instruct-FP8_suffix_sharegpt.json`](../summ_Llama-3.1-405B-Instruct-FP8_suffix_sharegpt.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-405B-Instruct-FP8" and .method=="suffix" and (.condition=="sharegpt" or (.condition==null and .corpus=="sharegpt")))' ../per_request_raw.jsonl
  ```
