# Qwen2.5-32B-Instruct × llm-d × wildchat

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 5241.7 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 198.0 |
| total_completion_tokens | 1037726 |
| TTFT p50/p99 ms | 33.4/80.9 |
| TPOT p50/p99 ms | 6.7/12.9 |
| accept α (acc/draft) | 0.7578 (554691.0/731941.0) |
| GPU util / mem MiB | 83.2 / 1230928 |
| CPU util | 5.2 |
| reqtps_avg | 252.6 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=1000)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 29 | 102.72 | 19.65 | 0.93 |
| p50 | 719 | 5466.78 | 32.61 | 7.11 |
| p99 | 8192 | 66594.66 | 93.02 | 14.94 |
| max | 8192 | 103492.06 | 107.56 | 17.34 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-32B-Instruct_llm-d_wildchat.json`](../summ_Qwen2.5-32B-Instruct_llm-d_wildchat.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-32B-Instruct" and .method=="llm-d" and (.condition=="wildchat" or (.condition==null and .corpus=="wildchat")))' ../per_request_raw.jsonl
  ```
