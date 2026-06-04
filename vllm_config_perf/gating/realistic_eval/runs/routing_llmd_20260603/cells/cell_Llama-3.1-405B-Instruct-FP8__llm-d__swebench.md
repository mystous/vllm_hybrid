# Llama-3.1-405B-Instruct-FP8 × llm-d × swebench

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 1396.8 |
| n_ok/n | 297/297 (err 0) |
| wall_total_s | 559.5 |
| total_completion_tokens | 781542 |
| TTFT p50/p99 ms | 131.3/1041.0 |
| TPOT p50/p99 ms | 29.7/56.2 |
| accept α (acc/draft) | 0.8066 (334292.0/414436.0) |
| GPU util / mem MiB | 83.8 / 1234284 |
| CPU util | 4.7 |
| reqtps_avg | 56.9 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=297)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 2 | 132.75 | 71.9 | 2.88 |
| p50 | 413 | 12845.77 | 131.31 | 29.7 |
| p99 | 8192 | 244442.43 | 1041.0 | 56.23 |
| max | 8192 | 244444.44 | 1044.96 | 386.65 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-405B-Instruct-FP8_llm-d_swebench.json`](../summ_Llama-3.1-405B-Instruct-FP8_llm-d_swebench.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-405B-Instruct-FP8" and .method=="llm-d" and (.condition=="swebench" or (.condition==null and .corpus=="swebench")))' ../per_request_raw.jsonl
  ```
