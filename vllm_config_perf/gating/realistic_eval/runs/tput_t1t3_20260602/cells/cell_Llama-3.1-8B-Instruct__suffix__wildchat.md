# Llama-3.1-8B-Instruct × suffix × wildchat

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 19856.1 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 195.5 |
| total_completion_tokens | 3882078 |
| TTFT p50/p99 ms | 22.0/70.9 |
| TPOT p50/p99 ms | 1.3/8.1 |
| accept α (acc/draft) | 0.8567 (3341976.0/3901118.0) |
| GPU util / mem MiB | 67.3 / 1265376 |
| CPU util | 4.4 |
| reqtps_avg | 746.8 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 21 | 151.48 | 18.26 | 0.68 |
| p50 | 8192 | 10403.89 | 22.03 | 1.32 |
| p99 | 8192 | 33742.5 | 70.87 | 8.14 |
| max | 8192 | 65165.24 | 73.44 | 9.36 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-8B-Instruct_suffix_wildchat.json`](../summ_Llama-3.1-8B-Instruct_suffix_wildchat.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-8B-Instruct" and .method=="suffix" and (.condition=="wildchat" or (.condition==null and .corpus=="wildchat")))' ../per_request_raw.jsonl
  ```
