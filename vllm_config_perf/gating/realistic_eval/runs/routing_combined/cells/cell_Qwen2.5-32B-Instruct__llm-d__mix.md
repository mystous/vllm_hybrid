# Qwen2.5-32B-Instruct × llm-d × mix

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 5236.1 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 193.3 |
| total_completion_tokens | 1011909 |
| TTFT p50/p99 ms | 32.2/93.4 |
| TPOT p50/p99 ms | 6.1/11.7 |
| accept α (acc/draft) | 0.7686 (515576.0/670797.0) |
| GPU util / mem MiB | 79.7 / 1230928 |
| CPU util | 4.9 |
| reqtps_avg | 308.8 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 25 | 120.76 | 19.92 | 0.93 |
| p50 | 577 | 3015.11 | 32.19 | 6.1 |
| p99 | 8192 | 72730.08 | 93.39 | 11.74 |
| max | 8192 | 73552.85 | 94.18 | 13.24 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-32B-Instruct_llm-d_mix.json`](../summ_Qwen2.5-32B-Instruct_llm-d_mix.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-32B-Instruct" and .method=="llm-d" and .condition=="mix")' ../per_request_raw.jsonl
  ```
