# Qwen2.5-7B-Instruct × llm-d-c64 × lmsys

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 9067.5 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 174.3 |
| total_completion_tokens | 1580911 |
| TTFT p50/p99 ms | 45.5/92.5 |
| TPOT p50/p99 ms | 6.3/21.5 |
| accept α (acc/draft) | 0.7089 (805324.0/1136008.0) |
| GPU util / mem MiB | 54.1 / 614770 |
| CPU util | 3.3 |
| reqtps_avg | 199.4 |
| concurrency / max_tokens / stream | 64 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 2 | 36.04 | 16.69 | 1.01 |
| p50 | 648 | 5448.37 | 45.51 | 6.31 |
| p99 | 8192 | 84919.36 | 92.46 | 21.5 |
| max | 8192 | 140993.95 | 97.87 | 28.35 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-7B-Instruct_llm-d-c64_lmsys.json`](../summ_Qwen2.5-7B-Instruct_llm-d-c64_lmsys.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-7B-Instruct" and .method=="llm-d-c64" and (.condition=="lmsys" or (.condition==null and .corpus=="lmsys")))' ../per_request_raw.jsonl
  ```
