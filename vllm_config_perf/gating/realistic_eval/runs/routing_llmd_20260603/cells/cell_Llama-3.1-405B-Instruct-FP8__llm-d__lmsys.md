# Llama-3.1-405B-Instruct-FP8 × llm-d × lmsys

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 1513.7 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 609.8 |
| total_completion_tokens | 923068 |
| TTFT p50/p99 ms | 97.8/527.8 |
| TPOT p50/p99 ms | 29.1/46.9 |
| accept α (acc/draft) | 0.6659 (426057.0/639830.0) |
| GPU util / mem MiB | 95.3 / 1234288 |
| CPU util | 5.3 |
| reqtps_avg | 46.6 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 96.32 | 44.89 | 3.38 |
| p50 | 450 | 13427.82 | 97.81 | 29.1 |
| p99 | 8192 | 238947.44 | 527.78 | 46.91 |
| max | 8192 | 267252.07 | 533.07 | 59.93 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-405B-Instruct-FP8_llm-d_lmsys.json`](../summ_Llama-3.1-405B-Instruct-FP8_llm-d_lmsys.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-405B-Instruct-FP8" and .method=="llm-d" and (.condition=="lmsys" or (.condition==null and .corpus=="lmsys")))' ../per_request_raw.jsonl
  ```
