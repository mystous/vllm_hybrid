# Llama-3.1-405B-Instruct-FP8 × vanilla × wildchat

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 1280.3 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 1063.9 |
| total_completion_tokens | 1362089 |
| TTFT p50/p99 ms | 71.5/397.9 |
| TPOT p50/p99 ms | 23.5/24.4 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 98.9 / 1272432 |
| CPU util | 4.8 |
| reqtps_avg | 41.9 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 70.64 | 65.7 | 20.95 |
| p50 | 663 | 15727.27 | 71.46 | 23.48 |
| p99 | 8192 | 193841.14 | 397.87 | 24.4 |
| max | 8192 | 194039.26 | 399.53 | 25.23 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-405B-Instruct-FP8_vanilla_wildchat.json`](../summ_Llama-3.1-405B-Instruct-FP8_vanilla_wildchat.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-405B-Instruct-FP8" and .method=="vanilla" and (.condition=="wildchat" or (.condition==null and .corpus=="wildchat")))' ../per_request_raw.jsonl
  ```
