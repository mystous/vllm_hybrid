# Llama-3.1-70B-Instruct × suffix × mix

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 10400.5 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 96.3 |
| total_completion_tokens | 1001437 |
| TTFT p50/p99 ms | 56.3/118.6 |
| TPOT p50/p99 ms | 2.5/16.8 |
| accept α (acc/draft) | 0.9146 (903499.0/987822.0) |
| GPU util / mem MiB | 83.4 / 1268784 |
| CPU util | 4.4 |
| reqtps_avg | 341.6 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 45.08 | 37.34 | 1.7 |
| p50 | 503 | 1409.39 | 56.33 | 2.47 |
| p99 | 8192 | 26780.93 | 118.59 | 16.8 |
| max | 8192 | 59170.82 | 134.31 | 21.39 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-70B-Instruct_suffix_mix.json`](../summ_Llama-3.1-70B-Instruct_suffix_mix.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-70B-Instruct" and .method=="suffix" and .condition=="mix")' ../per_request_raw.jsonl
  ```
