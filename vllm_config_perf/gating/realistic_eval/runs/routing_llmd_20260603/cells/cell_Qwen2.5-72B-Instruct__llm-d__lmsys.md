# Qwen2.5-72B-Instruct × llm-d × lmsys

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 3025.7 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 280.6 |
| total_completion_tokens | 849071 |
| TTFT p50/p99 ms | 43.6/135.0 |
| TPOT p50/p99 ms | 12.4/21.2 |
| accept α (acc/draft) | 0.5747 (343286.0/597321.0) |
| GPU util / mem MiB | 84.4 / 1231664 |
| CPU util | 4.9 |
| reqtps_avg | 98.8 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 2 | 63.61 | 27.69 | 1.76 |
| p50 | 509 | 6713.57 | 43.63 | 12.38 |
| p99 | 8192 | 87553.47 | 134.98 | 21.19 |
| max | 8192 | 87909.47 | 139.79 | 25.03 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-72B-Instruct_llm-d_lmsys.json`](../summ_Qwen2.5-72B-Instruct_llm-d_lmsys.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-72B-Instruct" and .method=="llm-d" and (.condition=="lmsys" or (.condition==null and .corpus=="lmsys")))' ../per_request_raw.jsonl
  ```
