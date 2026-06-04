# Qwen2.5-7B-Instruct × llm-d-c8 × mix

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 3267.2 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 413.0 |
| total_completion_tokens | 1349436 |
| TTFT p50/p99 ms | 19.9/29.1 |
| TPOT p50/p99 ms | 3.5/5.7 |
| accept α (acc/draft) | 0.7565 (726197.0/959979.0) |
| GPU util / mem MiB | 67.9 / 614764 |
| CPU util | 3.1 |
| reqtps_avg | 570.6 |
| concurrency / max_tokens / stream | 8 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 2 | 25.32 | 11.72 | 0.39 |
| p50 | 711 | 2261.77 | 19.87 | 3.48 |
| p99 | 8192 | 37104.23 | 29.1 | 5.67 |
| max | 8192 | 38965.29 | 30.56 | 6.59 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-7B-Instruct_llm-d-c8_mix.json`](../summ_Qwen2.5-7B-Instruct_llm-d-c8_mix.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-7B-Instruct" and .method=="llm-d-c8" and .condition=="mix")' ../per_request_raw.jsonl
  ```
