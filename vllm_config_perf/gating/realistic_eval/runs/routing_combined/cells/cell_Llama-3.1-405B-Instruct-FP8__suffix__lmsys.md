# Llama-3.1-405B-Instruct-FP8 × suffix × lmsys

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 2243.4 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 412.3 |
| total_completion_tokens | 924873 |
| TTFT p50/p99 ms | 102.3/412.2 |
| TPOT p50/p99 ms | 31.5/50.8 |
| accept α (acc/draft) | 0.6733 (693896.0/1030624.0) |
| GPU util / mem MiB | 93.1 / 1272464 |
| CPU util | 4.3 |
| reqtps_avg | 57.0 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 102.87 | 59.9 | 3.37 |
| p50 | 443 | 14795.09 | 102.25 | 31.5 |
| p99 | 8192 | 141229.62 | 412.2 | 50.77 |
| max | 8192 | 267042.4 | 413.97 | 55.32 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-405B-Instruct-FP8_suffix_lmsys.json`](../summ_Llama-3.1-405B-Instruct-FP8_suffix_lmsys.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-405B-Instruct-FP8" and .method=="suffix" and (.condition=="lmsys" or (.condition==null and .corpus=="lmsys")))' ../per_request_raw.jsonl
  ```
