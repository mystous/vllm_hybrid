# Llama-3.1-8B-Instruct × llm-d × lmsys

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 15233.2 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 257.1 |
| total_completion_tokens | 3916397 |
| TTFT p50/p99 ms | 22.4/67.9 |
| TPOT p50/p99 ms | 1.4/5.9 |
| accept α (acc/draft) | 0.8484 (2141677.0/2524448.0) |
| GPU util / mem MiB | 81.5 / 1228584 |
| CPU util | 5.3 |
| reqtps_avg | 713.5 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 47 | 174.19 | 14.74 | 0.52 |
| p50 | 8192 | 10552.51 | 22.37 | 1.43 |
| p99 | 8192 | 28866.06 | 67.94 | 5.92 |
| max | 8192 | 35932.97 | 71.32 | 8.14 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-8B-Instruct_llm-d_lmsys.json`](../summ_Llama-3.1-8B-Instruct_llm-d_lmsys.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-8B-Instruct" and .method=="llm-d" and (.condition=="lmsys" or (.condition==null and .corpus=="lmsys")))' ../per_request_raw.jsonl
  ```
