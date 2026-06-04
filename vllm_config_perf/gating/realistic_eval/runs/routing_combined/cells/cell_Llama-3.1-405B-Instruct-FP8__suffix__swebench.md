# Llama-3.1-405B-Instruct-FP8 × suffix × swebench

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 2639.3 |
| n_ok/n | 297/297 (err 0) |
| wall_total_s | 344.6 |
| total_completion_tokens | 909478 |
| TTFT p50/p99 ms | 149.2/1031.3 |
| TPOT p50/p99 ms | 29.9/68.8 |
| accept α (acc/draft) | 0.8248 (755808.0/916380.0) |
| GPU util / mem MiB | 93.3 / 1272464 |
| CPU util | 4.3 |
| reqtps_avg | 70.4 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=297)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 2 | 165.31 | 76.66 | 3.54 |
| p50 | 413 | 14088.38 | 149.18 | 29.91 |
| p99 | 8192 | 155855.91 | 1031.29 | 68.82 |
| max | 8192 | 317585.92 | 1034.73 | 73.83 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-405B-Instruct-FP8_suffix_swebench.json`](../summ_Llama-3.1-405B-Instruct-FP8_suffix_swebench.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-405B-Instruct-FP8" and .method=="suffix" and (.condition=="swebench" or (.condition==null and .corpus=="swebench")))' ../per_request_raw.jsonl
  ```
