# Qwen2.5-32B-Instruct × llm-d × lmsys

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 4988.3 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 209.9 |
| total_completion_tokens | 1046974 |
| TTFT p50/p99 ms | 32.2/60.3 |
| TPOT p50/p99 ms | 6.7/12.7 |
| accept α (acc/draft) | 0.7111 (487017.0/684842.0) |
| GPU util / mem MiB | 82.6 / 1230928 |
| CPU util | 5.0 |
| reqtps_avg | 267.6 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=1000)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 2 | 48.94 | 19.23 | 0.94 |
| p50 | 464 | 3541.52 | 31.72 | 7.19 |
| p99 | 8192 | 68382.54 | 107.44 | 14.75 |
| max | 8192 | 84354.22 | 111.83 | 17.72 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-32B-Instruct_llm-d_lmsys.json`](../summ_Qwen2.5-32B-Instruct_llm-d_lmsys.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-32B-Instruct" and .method=="llm-d" and (.condition=="lmsys" or (.condition==null and .corpus=="lmsys")))' ../per_request_raw.jsonl
  ```
