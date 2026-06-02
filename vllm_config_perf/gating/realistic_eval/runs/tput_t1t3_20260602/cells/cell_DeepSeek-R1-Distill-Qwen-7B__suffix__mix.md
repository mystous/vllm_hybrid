# DeepSeek-R1-Distill-Qwen-7B × suffix × mix

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 24458.3 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 76.0 |
| total_completion_tokens | 1858044 |
| TTFT p50/p99 ms | 22.3/56.3 |
| TPOT p50/p99 ms | 0.9/7.2 |
| accept α (acc/draft) | 0.8765 (1652601.0/1885357.0) |
| GPU util / mem MiB | 63.8 / 632552 |
| CPU util | 2.6 |
| reqtps_avg | 950.5 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 8 | 74.03 | 16.25 | 0.69 |
| p50 | 1716 | 2387.65 | 22.25 | 0.93 |
| p99 | 8192 | 22168.34 | 56.33 | 7.23 |
| max | 8192 | 31715.66 | 59.92 | 8.07 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Qwen-7B_suffix_mix.json`](../summ_DeepSeek-R1-Distill-Qwen-7B_suffix_mix.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Qwen-7B" and .method=="suffix" and .condition=="mix")' ../per_request_raw.jsonl
  ```
