# Llama-3.1-405B-Instruct-FP8 × llm-d × humaneval

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 1414.6 |
| n_ok/n | 164/164 (err 0) |
| wall_total_s | 636.6 |
| total_completion_tokens | 900552 |
| TTFT p50/p99 ms | 107.9/278.6 |
| TPOT p50/p99 ms | 28.3/42.7 |
| accept α (acc/draft) | 0.6969 (383169.0/549851.0) |
| GPU util / mem MiB | 94.9 / 1234288 |
| CPU util | 5.2 |
| reqtps_avg | 71.4 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=164)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 7 | 488.67 | 74.44 | 3.6 |
| p50 | 8192 | 52133.63 | 107.92 | 28.3 |
| p99 | 8192 | 240666.82 | 278.6 | 42.7 |
| max | 8192 | 290795.05 | 279.63 | 50.95 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-405B-Instruct-FP8_llm-d_humaneval.json`](../summ_Llama-3.1-405B-Instruct-FP8_llm-d_humaneval.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-405B-Instruct-FP8" and .method=="llm-d" and (.condition=="humaneval" or (.condition==null and .corpus=="humaneval")))' ../per_request_raw.jsonl
  ```
