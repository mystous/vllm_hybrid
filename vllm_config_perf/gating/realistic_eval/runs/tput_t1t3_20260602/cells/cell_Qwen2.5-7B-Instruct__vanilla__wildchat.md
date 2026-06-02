# Qwen2.5-7B-Instruct × vanilla × wildchat

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 4184.0 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 365.6 |
| total_completion_tokens | 1529855 |
| TTFT p50/p99 ms | 28.3/62.3 |
| TPOT p50/p99 ms | 7.1/8.8 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 82.1 / 632552 |
| CPU util | 2.7 |
| reqtps_avg | 149.9 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 6 | 57.3 | 16.25 | 3.55 |
| p50 | 858 | 5903.98 | 28.26 | 7.14 |
| p99 | 8192 | 62730.49 | 62.3 | 8.83 |
| max | 8192 | 62824.03 | 63.22 | 9.27 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-7B-Instruct_vanilla_wildchat.json`](../summ_Qwen2.5-7B-Instruct_vanilla_wildchat.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-7B-Instruct" and .method=="vanilla" and (.condition=="wildchat" or (.condition==null and .corpus=="wildchat")))' ../per_request_raw.jsonl
  ```
