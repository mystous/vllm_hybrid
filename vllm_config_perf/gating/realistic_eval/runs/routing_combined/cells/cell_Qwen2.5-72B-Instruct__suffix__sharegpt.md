# Qwen2.5-72B-Instruct × suffix × sharegpt

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 3218.7 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 209.1 |
| total_completion_tokens | 673055 |
| TTFT p50/p99 ms | 43.6/512.1 |
| TPOT p50/p99 ms | 15.3/20.3 |
| accept α (acc/draft) | 0.5305 (383984.0/723869.0) |
| GPU util / mem MiB | 86.4 / 1269115 |
| CPU util | 4.3 |
| reqtps_avg | 90.8 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 27 | 405.27 | 28.78 | 1.65 |
| p50 | 697 | 10624.98 | 43.64 | 15.28 |
| p99 | 8192 | 45106.87 | 512.11 | 20.28 |
| max | 8192 | 119021.77 | 708.24 | 21.57 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-72B-Instruct_suffix_sharegpt.json`](../summ_Qwen2.5-72B-Instruct_suffix_sharegpt.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-72B-Instruct" and .method=="suffix" and (.condition=="sharegpt" or (.condition==null and .corpus=="sharegpt")))' ../per_request_raw.jsonl
  ```
