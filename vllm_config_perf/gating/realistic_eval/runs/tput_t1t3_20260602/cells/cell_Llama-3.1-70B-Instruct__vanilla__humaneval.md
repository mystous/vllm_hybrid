# Llama-3.1-70B-Instruct × vanilla × humaneval

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 3391.1 |
| n_ok/n | 164/164 (err 0) |
| wall_total_s | 232.6 |
| total_completion_tokens | 788781 |
| TTFT p50/p99 ms | 37.9/147.3 |
| TPOT p50/p99 ms | 9.1/9.9 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 98.3 / 1268912 |
| CPU util | 4.8 |
| reqtps_avg | 108.4 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=164)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 24 | 222.95 | 26.27 | 8.3 |
| p50 | 8192 | 74281.75 | 37.88 | 9.12 |
| p99 | 8192 | 75201.2 | 147.29 | 9.95 |
| max | 8192 | 75369.96 | 147.8 | 9.99 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-70B-Instruct_vanilla_humaneval.json`](../summ_Llama-3.1-70B-Instruct_vanilla_humaneval.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-70B-Instruct" and .method=="vanilla" and (.condition=="humaneval" or (.condition==null and .corpus=="humaneval")))' ../per_request_raw.jsonl
  ```
