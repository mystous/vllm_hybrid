# Llama-3.1-8B-Instruct × llm-d × wildchat

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 14789.5 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 261.7 |
| total_completion_tokens | 3870385 |
| TTFT p50/p99 ms | 22.1/64.2 |
| TPOT p50/p99 ms | 1.5/6.0 |
| accept α (acc/draft) | 0.8546 (2101318.0/2458750.0) |
| GPU util / mem MiB | 81.1 / 1228584 |
| CPU util | 5.3 |
| reqtps_avg | 707.8 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 21 | 95.87 | 15.81 | 0.54 |
| p50 | 8192 | 10777.56 | 22.14 | 1.53 |
| p99 | 8192 | 28960.32 | 64.22 | 6.05 |
| max | 8192 | 47776.27 | 67.85 | 7.37 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-8B-Instruct_llm-d_wildchat.json`](../summ_Llama-3.1-8B-Instruct_llm-d_wildchat.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-8B-Instruct" and .method=="llm-d" and (.condition=="wildchat" or (.condition==null and .corpus=="wildchat")))' ../per_request_raw.jsonl
  ```
