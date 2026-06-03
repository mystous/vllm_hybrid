# DeepSeek-R1-Distill-Qwen-32B × suffix × mix

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 9055.6 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 135.0 |
| total_completion_tokens | 1222334 |
| TTFT p50/p99 ms | 37.0/98.8 |
| TPOT p50/p99 ms | 1.6/12.6 |
| accept α (acc/draft) | 0.8011 (1033654.0/1290306.0) |
| GPU util / mem MiB | 79.9 / 1267776 |
| CPU util | 4.4 |
| reqtps_avg | 494.2 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 7 | 83.61 | 26.41 | 1.13 |
| p50 | 1080 | 2641.15 | 37.04 | 1.63 |
| p99 | 8192 | 33597.99 | 98.76 | 12.58 |
| max | 8192 | 58447.91 | 101.72 | 13.4 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Qwen-32B_suffix_mix.json`](../summ_DeepSeek-R1-Distill-Qwen-32B_suffix_mix.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Qwen-32B" and .method=="suffix" and .condition=="mix")' ../per_request_raw.jsonl
  ```
