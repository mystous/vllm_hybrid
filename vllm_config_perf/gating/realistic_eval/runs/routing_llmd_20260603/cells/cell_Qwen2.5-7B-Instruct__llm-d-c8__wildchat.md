# Qwen2.5-7B-Instruct × llm-d-c8 × wildchat

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 3168.6 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 473.1 |
| total_completion_tokens | 1499224 |
| TTFT p50/p99 ms | 19.0/45.2 |
| TPOT p50/p99 ms | 3.8/6.8 |
| accept α (acc/draft) | 0.7081 (757288.0/1069507.0) |
| GPU util / mem MiB | 72.7 / 614764 |
| CPU util | 3.2 |
| reqtps_avg | 394.9 |
| concurrency / max_tokens / stream | 8 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 6 | 40.39 | 11.91 | 0.55 |
| p50 | 866 | 3364.51 | 18.97 | 3.75 |
| p99 | 8192 | 33646.96 | 45.19 | 6.83 |
| max | 8192 | 34992.56 | 55.95 | 9.06 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-7B-Instruct_llm-d-c8_wildchat.json`](../summ_Qwen2.5-7B-Instruct_llm-d-c8_wildchat.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-7B-Instruct" and .method=="llm-d-c8" and (.condition=="wildchat" or (.condition==null and .corpus=="wildchat")))' ../per_request_raw.jsonl
  ```
