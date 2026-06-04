# Llama-3.1-405B-Instruct-FP8 × vanilla × swebench

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 1204.5 |
| n_ok/n | 297/297 (err 0) |
| wall_total_s | 635.0 |
| total_completion_tokens | 764834 |
| TTFT p50/p99 ms | 93.7/852.3 |
| TPOT p50/p99 ms | 23.9/27.0 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 98.8 / 1272432 |
| CPU util | 4.8 |
| reqtps_avg | 39.9 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=297)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 2 | 100.44 | 68.63 | 20.53 |
| p50 | 402 | 9866.06 | 93.68 | 23.94 |
| p99 | 8192 | 197127.15 | 852.32 | 26.98 |
| max | 8192 | 197864.08 | 1002.95 | 37.48 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-405B-Instruct-FP8_vanilla_swebench.json`](../summ_Llama-3.1-405B-Instruct-FP8_vanilla_swebench.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-405B-Instruct-FP8" and .method=="vanilla" and (.condition=="swebench" or (.condition==null and .corpus=="swebench")))' ../per_request_raw.jsonl
  ```
