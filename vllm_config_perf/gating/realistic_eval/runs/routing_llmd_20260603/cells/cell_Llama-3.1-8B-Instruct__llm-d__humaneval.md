# Llama-3.1-8B-Instruct × llm-d × humaneval

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 12664.6 |
| n_ok/n | 164/164 (err 0) |
| wall_total_s | 104.8 |
| total_completion_tokens | 1327871 |
| TTFT p50/p99 ms | 23.7/53.0 |
| TPOT p50/p99 ms | 3.1/3.9 |
| accept α (acc/draft) | 0.7679 (584437.0/761074.0) |
| GPU util / mem MiB | 80.3 / 1228577 |
| CPU util | 5.3 |
| reqtps_avg | 606.3 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=164)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 81 | 419.89 | 16.44 | 0.62 |
| p50 | 8192 | 21460.07 | 23.74 | 3.1 |
| p99 | 8192 | 28748.5 | 53.0 | 3.85 |
| max | 8192 | 39921.42 | 53.41 | 4.87 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-8B-Instruct_llm-d_humaneval.json`](../summ_Llama-3.1-8B-Instruct_llm-d_humaneval.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-8B-Instruct" and .method=="llm-d" and (.condition=="humaneval" or (.condition==null and .corpus=="humaneval")))' ../per_request_raw.jsonl
  ```
