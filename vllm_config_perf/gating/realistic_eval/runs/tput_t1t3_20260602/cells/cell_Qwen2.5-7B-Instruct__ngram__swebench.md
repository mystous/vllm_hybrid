# Qwen2.5-7B-Instruct × ngram × swebench

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 364.0 |
| n_ok/n | 297/297 (err 0) |
| wall_total_s | 695.1 |
| total_completion_tokens | 252993 |
| TTFT p50/p99 ms | 274.9/327.0 |
| TPOT p50/p99 ms | 106.9/141.1 |
| accept α (acc/draft) | 0.4711 (88477.0/187815.0) |
| GPU util / mem MiB | 57.4 / 632560 |
| CPU util | 90.8 |
| reqtps_avg | 11.4 |
| concurrency / max_tokens / stream | 32 / 2048 / True |

## per-request 분포 (raw 기반, n=297)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 32 | 4651.31 | 52.0 | 20.41 |
| p50 | 643 | 68447.43 | 274.93 | 106.9 |
| p99 | 2048 | 199399.69 | 327.0 | 141.12 |
| max | 2048 | 249209.13 | 354.67 | 146.85 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-7B-Instruct_ngram_swebench.json`](../summ_Qwen2.5-7B-Instruct_ngram_swebench.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-7B-Instruct" and .method=="ngram" and (.condition=="swebench" or (.condition==null and .corpus=="swebench")))' ../per_request_raw.jsonl
  ```
