# Llama-3.1-405B-Instruct-FP8 × llm-d × mbpp

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 743.0 |
| n_ok/n | 198/198 (err 0) |
| wall_total_s | 250.7 |
| total_completion_tokens | 186225 |
| TTFT p50/p99 ms | 85.5/151.2 |
| TPOT p50/p99 ms | 27.5/28.4 |
| accept α (acc/draft) | 0.6744 (61921.0/91815.0) |
| GPU util / mem MiB | 71.1 / 1234288 |
| CPU util | 4.2 |
| reqtps_avg | 68.9 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=396)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 79 | 727.03 | 42.89 | 2.91 |
| p50 | 433 | 11341.75 | 85.54 | 27.38 |
| p99 | 8192 | 221143.08 | 154.79 | 28.45 |
| max | 8192 | 221569.84 | 155.12 | 28.46 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-405B-Instruct-FP8_llm-d_mbpp.json`](../summ_Llama-3.1-405B-Instruct-FP8_llm-d_mbpp.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-405B-Instruct-FP8" and .method=="llm-d" and (.condition=="mbpp" or (.condition==null and .corpus=="mbpp")))' ../per_request_raw.jsonl
  ```
