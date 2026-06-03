# Qwen2.5-7B-Instruct × suffix × sharegpt

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 6167.1 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 248.7 |
| total_completion_tokens | 1533925 |
| TTFT p50/p99 ms | 39.9/152.6 |
| TPOT p50/p99 ms | 11.3/20.8 |
| accept α (acc/draft) | 1.4134 (2304784.0/1630680.0) |
| GPU util / mem MiB | 38.7 / 632553 |
| CPU util | 2.5 |
| reqtps_avg | 166.8 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 14 | 280.11 | 15.12 | 0.97 |
| p50 | 776 | 10248.22 | 39.95 | 11.32 |
| p99 | 8192 | 73130.98 | 152.58 | 20.82 |
| max | 8192 | 84642.39 | 153.41 | 23.02 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-7B-Instruct_suffix_sharegpt.json`](../summ_Qwen2.5-7B-Instruct_suffix_sharegpt.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-7B-Instruct" and .method=="suffix" and (.condition=="sharegpt" or (.condition==null and .corpus=="sharegpt")))' ../per_request_raw.jsonl
  ```
