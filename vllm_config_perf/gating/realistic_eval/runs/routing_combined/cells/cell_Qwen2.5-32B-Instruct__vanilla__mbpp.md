# Qwen2.5-32B-Instruct × vanilla × mbpp

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 2914.6 |
| n_ok/n | 198/198 (err 0) |
| wall_total_s | 176.5 |
| total_completion_tokens | 514365 |
| TTFT p50/p99 ms | 30.1/67.0 |
| TPOT p50/p99 ms | 8.3/10.6 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 93.8 / 1267776 |
| CPU util | 4.4 |
| reqtps_avg | 124.1 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=198)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 75 | 613.17 | 21.79 | 6.08 |
| p50 | 572 | 4437.53 | 30.12 | 8.27 |
| p99 | 8192 | 82932.05 | 66.98 | 10.55 |
| max | 8192 | 82980.36 | 67.93 | 10.57 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-32B-Instruct_vanilla_mbpp.json`](../summ_Qwen2.5-32B-Instruct_vanilla_mbpp.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-32B-Instruct" and .method=="vanilla" and (.condition=="mbpp" or (.condition==null and .corpus=="mbpp")))' ../per_request_raw.jsonl
  ```
