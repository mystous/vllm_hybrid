# Llama-3.1-405B-Instruct-FP8 × llm-d × wildchat

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 1496.7 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 825.4 |
| total_completion_tokens | 1235290 |
| TTFT p50/p99 ms | 98.3/438.9 |
| TPOT p50/p99 ms | 28.8/49.5 |
| accept α (acc/draft) | 0.7205 (550720.0/764405.0) |
| GPU util / mem MiB | 93.4 / 1234288 |
| CPU util | 5.2 |
| reqtps_avg | 53.0 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 85.87 | 45.13 | 3.26 |
| p50 | 683 | 22648.93 | 98.31 | 28.79 |
| p99 | 8192 | 238295.74 | 438.86 | 49.49 |
| max | 8192 | 238353.94 | 593.04 | 51.91 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-405B-Instruct-FP8_llm-d_wildchat.json`](../summ_Llama-3.1-405B-Instruct-FP8_llm-d_wildchat.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-405B-Instruct-FP8" and .method=="llm-d" and (.condition=="wildchat" or (.condition==null and .corpus=="wildchat")))' ../per_request_raw.jsonl
  ```
