# Qwen2.5-7B-Instruct × ngram × sharegpt

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 359.1 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 1544.2 |
| total_completion_tokens | 554560 |
| TTFT p50/p99 ms | 301.1/1368.4 |
| TPOT p50/p99 ms | 112.9/150.6 |
| accept α (acc/draft) | 0.5022 (210864.0/419909.0) |
| GPU util / mem MiB | 62.8 / 632556 |
| CPU util | 90.3 |
| reqtps_avg | 12.0 |
| concurrency / max_tokens / stream | 32 / 2048 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 14 | 2317.15 | 44.11 | 16.69 |
| p50 | 800 | 85827.71 | 301.12 | 112.91 |
| p99 | 2048 | 259908.08 | 1368.44 | 150.61 |
| max | 2048 | 280412.26 | 1371.62 | 154.78 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-7B-Instruct_ngram_sharegpt.json`](../summ_Qwen2.5-7B-Instruct_ngram_sharegpt.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-7B-Instruct" and .method=="ngram" and (.condition=="sharegpt" or (.condition==null and .corpus=="sharegpt")))' ../per_request_raw.jsonl
  ```
