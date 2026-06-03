# Llama-3.1-405B-Instruct-FP8 × vanilla × mbpp

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 916.3 |
| n_ok/n | 198/198 (err 0) |
| wall_total_s | 223.3 |
| total_completion_tokens | 204594 |
| TTFT p50/p99 ms | 66.9/135.8 |
| TPOT p50/p99 ms | 22.1/22.5 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 98.5 / 1272432 |
| CPU util | 4.8 |
| reqtps_avg | 45.2 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=198)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 100 | 2255.73 | 51.43 | 19.2 |
| p50 | 427 | 9550.81 | 66.95 | 22.09 |
| p99 | 8192 | 169738.27 | 135.83 | 22.54 |
| max | 8192 | 169740.78 | 136.77 | 22.55 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-405B-Instruct-FP8_vanilla_mbpp.json`](../summ_Llama-3.1-405B-Instruct-FP8_vanilla_mbpp.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-405B-Instruct-FP8" and .method=="vanilla" and (.condition=="mbpp" or (.condition==null and .corpus=="mbpp")))' ../per_request_raw.jsonl
  ```
