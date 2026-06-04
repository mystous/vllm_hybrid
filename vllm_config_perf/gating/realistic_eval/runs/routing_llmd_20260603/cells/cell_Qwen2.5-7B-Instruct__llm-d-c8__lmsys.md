# Qwen2.5-7B-Instruct × llm-d-c8 × lmsys

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 3147.0 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 532.4 |
| total_completion_tokens | 1675547 |
| TTFT p50/p99 ms | 19.4/34.2 |
| TPOT p50/p99 ms | 3.9/7.2 |
| accept α (acc/draft) | 0.6971 (859118.0/1232354.0) |
| GPU util / mem MiB | 72.1 / 614764 |
| CPU util | 3.2 |
| reqtps_avg | 398.3 |
| concurrency / max_tokens / stream | 8 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 2 | 23.15 | 11.42 | 0.54 |
| p50 | 665 | 2954.87 | 19.37 | 3.85 |
| p99 | 8192 | 34004.96 | 34.15 | 7.18 |
| max | 8192 | 39692.67 | 37.82 | 8.24 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-7B-Instruct_llm-d-c8_lmsys.json`](../summ_Qwen2.5-7B-Instruct_llm-d-c8_lmsys.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-7B-Instruct" and .method=="llm-d-c8" and (.condition=="lmsys" or (.condition==null and .corpus=="lmsys")))' ../per_request_raw.jsonl
  ```
