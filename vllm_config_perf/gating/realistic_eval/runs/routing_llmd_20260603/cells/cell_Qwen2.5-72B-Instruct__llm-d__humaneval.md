# Qwen2.5-72B-Instruct × llm-d × humaneval

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 2105.4 |
| n_ok/n | 164/164 (err 0) |
| wall_total_s | 26.6 |
| total_completion_tokens | 55963 |
| TTFT p50/p99 ms | 43.3/119.9 |
| TPOT p50/p99 ms | 10.2/13.8 |
| accept α (acc/draft) | 0.4862 (20740.0/42656.0) |
| GPU util / mem MiB | 77.7 / 1231664 |
| CPU util | 4.9 |
| reqtps_avg | 99.1 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=164)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 7 | 175.05 | 28.36 | 2.0 |
| p50 | 214 | 2257.25 | 43.34 | 10.24 |
| p99 | 2512 | 16391.51 | 119.94 | 13.8 |
| max | 8192 | 20935.16 | 120.48 | 16.61 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-72B-Instruct_llm-d_humaneval.json`](../summ_Qwen2.5-72B-Instruct_llm-d_humaneval.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-72B-Instruct" and .method=="llm-d" and (.condition=="humaneval" or (.condition==null and .corpus=="humaneval")))' ../per_request_raw.jsonl
  ```
