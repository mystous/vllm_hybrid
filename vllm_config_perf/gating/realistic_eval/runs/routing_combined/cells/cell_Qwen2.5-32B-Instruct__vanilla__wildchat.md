# Qwen2.5-32B-Instruct × vanilla × wildchat

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 3127.7 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 325.3 |
| total_completion_tokens | 1017321 |
| TTFT p50/p99 ms | 29.8/97.6 |
| TPOT p50/p99 ms | 9.1/10.8 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 94.3 / 1267776 |
| CPU util | 4.4 |
| reqtps_avg | 115.9 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 27 | 228.16 | 25.3 | 6.0 |
| p50 | 705 | 5949.32 | 29.78 | 9.05 |
| p99 | 8192 | 80293.46 | 97.62 | 10.78 |
| max | 8192 | 80604.08 | 100.84 | 11.24 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-32B-Instruct_vanilla_wildchat.json`](../summ_Qwen2.5-32B-Instruct_vanilla_wildchat.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-32B-Instruct" and .method=="vanilla" and (.condition=="wildchat" or (.condition==null and .corpus=="wildchat")))' ../per_request_raw.jsonl
  ```
