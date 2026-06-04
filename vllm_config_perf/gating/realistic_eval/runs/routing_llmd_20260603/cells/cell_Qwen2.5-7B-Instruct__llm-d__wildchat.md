# Qwen2.5-7B-Instruct × llm-d × wildchat

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 7348.4 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 214.4 |
| total_completion_tokens | 1575486 |
| TTFT p50/p99 ms | 24.1/68.3 |
| TPOT p50/p99 ms | 5.0/12.8 |
| accept α (acc/draft) | 0.6955 (700298.0/1006941.0) |
| GPU util / mem MiB | 62.0 / 614768 |
| CPU util | 3.4 |
| reqtps_avg | 254.1 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 6 | 55.08 | 16.37 | 0.74 |
| p50 | 883 | 5458.4 | 24.12 | 5.01 |
| p99 | 8192 | 46517.42 | 68.33 | 12.84 |
| max | 8192 | 56914.07 | 69.83 | 13.92 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-7B-Instruct_llm-d_wildchat.json`](../summ_Qwen2.5-7B-Instruct_llm-d_wildchat.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-7B-Instruct" and .method=="llm-d" and (.condition=="wildchat" or (.condition==null and .corpus=="wildchat")))' ../per_request_raw.jsonl
  ```
