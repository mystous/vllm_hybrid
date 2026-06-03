# Llama-3.1-8B-Instruct × suffix × humaneval

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 15126.3 |
| n_ok/n | 164/164 (err 0) |
| wall_total_s | 87.3 |
| total_completion_tokens | 1320428 |
| TTFT p50/p99 ms | 22.9/72.5 |
| TPOT p50/p99 ms | 1.3/5.4 |
| accept α (acc/draft) | 0.7654 (1108683.0/1448461.0) |
| GPU util / mem MiB | 68.6 / 1265376 |
| CPU util | 4.5 |
| reqtps_avg | 740.2 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=164)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 81 | 504.93 | 18.7 | 0.74 |
| p50 | 8192 | 10755.94 | 22.89 | 1.32 |
| p99 | 8192 | 41515.6 | 72.49 | 5.42 |
| max | 8192 | 53142.36 | 72.73 | 6.49 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-8B-Instruct_suffix_humaneval.json`](../summ_Llama-3.1-8B-Instruct_suffix_humaneval.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-8B-Instruct" and .method=="suffix" and (.condition=="humaneval" or (.condition==null and .corpus=="humaneval")))' ../per_request_raw.jsonl
  ```
