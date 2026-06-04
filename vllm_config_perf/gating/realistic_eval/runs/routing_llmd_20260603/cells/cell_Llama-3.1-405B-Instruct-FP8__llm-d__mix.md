# Llama-3.1-405B-Instruct-FP8 × llm-d × mix

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 1429.4 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 833.7 |
| total_completion_tokens | 1191663 |
| TTFT p50/p99 ms | 97.0/458.7 |
| TPOT p50/p99 ms | 27.5/47.2 |
| accept α (acc/draft) | 0.7459 (526008.0/705177.0) |
| GPU util / mem MiB | 86.2 / 1234288 |
| CPU util | 4.9 |
| reqtps_avg | 67.0 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 91.28 | 48.75 | 3.17 |
| p50 | 543 | 13965.84 | 97.03 | 27.46 |
| p99 | 8192 | 249734.6 | 458.72 | 47.19 |
| max | 8192 | 249807.4 | 580.22 | 53.77 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-405B-Instruct-FP8_llm-d_mix.json`](../summ_Llama-3.1-405B-Instruct-FP8_llm-d_mix.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-405B-Instruct-FP8" and .method=="llm-d" and .condition=="mix")' ../per_request_raw.jsonl
  ```
