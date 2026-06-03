# Llama-3.1-405B-Instruct-FP8 × suffix × mix

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 2828.9 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 432.0 |
| total_completion_tokens | 1222079 |
| TTFT p50/p99 ms | 122.2/442.2 |
| TPOT p50/p99 ms | 17.3/49.0 |
| accept α (acc/draft) | 0.7663 (1011592.0/1320036.0) |
| GPU util / mem MiB | 93.0 / 1272464 |
| CPU util | 4.3 |
| reqtps_avg | 94.8 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 116.91 | 71.6 | 3.63 |
| p50 | 530 | 13744.65 | 122.19 | 17.26 |
| p99 | 8192 | 140419.77 | 442.18 | 48.96 |
| max | 8192 | 229542.7 | 446.43 | 58.7 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-405B-Instruct-FP8_suffix_mix.json`](../summ_Llama-3.1-405B-Instruct-FP8_suffix_mix.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-405B-Instruct-FP8" and .method=="suffix" and .condition=="mix")' ../per_request_raw.jsonl
  ```
