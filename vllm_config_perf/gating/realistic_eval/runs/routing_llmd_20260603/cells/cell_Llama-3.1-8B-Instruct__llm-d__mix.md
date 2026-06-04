# Llama-3.1-8B-Instruct × llm-d × mix

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 15958.9 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 240.3 |
| total_completion_tokens | 3834327 |
| TTFT p50/p99 ms | 22.4/76.3 |
| TPOT p50/p99 ms | 1.1/4.0 |
| accept α (acc/draft) | 0.8848 (2184035.0/2468358.0) |
| GPU util / mem MiB | 79.5 / 1228587 |
| CPU util | 5.2 |
| reqtps_avg | 857.2 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 21 | 89.62 | 14.18 | 0.49 |
| p50 | 8192 | 8085.12 | 22.39 | 1.1 |
| p99 | 8192 | 29012.2 | 76.31 | 4.04 |
| max | 8192 | 33139.03 | 79.91 | 5.54 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-8B-Instruct_llm-d_mix.json`](../summ_Llama-3.1-8B-Instruct_llm-d_mix.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-8B-Instruct" and .method=="llm-d" and .condition=="mix")' ../per_request_raw.jsonl
  ```
