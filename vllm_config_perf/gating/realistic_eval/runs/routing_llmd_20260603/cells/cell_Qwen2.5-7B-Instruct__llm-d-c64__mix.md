# Qwen2.5-7B-Instruct × llm-d-c64 × mix

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 10346.5 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 133.9 |
| total_completion_tokens | 1385379 |
| TTFT p50/p99 ms | 52.8/120.7 |
| TPOT p50/p99 ms | 4.1/14.6 |
| accept α (acc/draft) | 0.8709 (797097.0/915280.0) |
| GPU util / mem MiB | 55.0 / 614770 |
| CPU util | 3.4 |
| reqtps_avg | 264.0 |
| concurrency / max_tokens / stream | 64 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 2 | 30.09 | 17.74 | 1.13 |
| p50 | 720 | 3230.6 | 52.77 | 4.07 |
| p99 | 8192 | 59330.85 | 120.67 | 14.63 |
| max | 8192 | 119907.96 | 127.67 | 28.07 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-7B-Instruct_llm-d-c64_mix.json`](../summ_Qwen2.5-7B-Instruct_llm-d-c64_mix.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-7B-Instruct" and .method=="llm-d-c64" and .condition=="mix")' ../per_request_raw.jsonl
  ```
