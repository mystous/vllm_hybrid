# Llama-3.1-70B-Instruct × llm-d × humaneval

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 3540.1 |
| n_ok/n | 164/164 (err 0) |
| wall_total_s | 221.7 |
| total_completion_tokens | 785005 |
| TTFT p50/p99 ms | 46.3/129.9 |
| TPOT p50/p99 ms | 10.3/16.7 |
| accept α (acc/draft) | 0.6836 (305660.0/447131.0) |
| GPU util / mem MiB | 88.4 / 1231056 |
| CPU util | 5.5 |
| reqtps_avg | 154.3 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=164)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 24 | 284.19 | 29.34 | 1.66 |
| p50 | 8192 | 20329.73 | 46.32 | 10.29 |
| p99 | 8192 | 88280.12 | 129.92 | 16.73 |
| max | 8192 | 118194.88 | 130.25 | 20.15 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-70B-Instruct_llm-d_humaneval.json`](../summ_Llama-3.1-70B-Instruct_llm-d_humaneval.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-70B-Instruct" and .method=="llm-d" and (.condition=="humaneval" or (.condition==null and .corpus=="humaneval")))' ../per_request_raw.jsonl
  ```
