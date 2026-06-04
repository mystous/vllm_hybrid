# DeepSeek-R1 × vanilla × wildchat

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 1555.5 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 947.9 |
| total_completion_tokens | 1474503 |
| TTFT p50/p99 ms | 65.3/253.2 |
| TPOT p50/p99 ms | 19.5/20.3 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 98.7 / 1277667 |
| CPU util | 4.8 |
| reqtps_avg | 49.5 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 63.04 | 52.24 | 14.24 |
| p50 | 1528 | 29885.14 | 65.33 | 19.48 |
| p99 | 8192 | 163537.02 | 253.17 | 20.3 |
| max | 8192 | 163647.69 | 254.38 | 20.98 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1_vanilla_wildchat.json`](../summ_DeepSeek-R1_vanilla_wildchat.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1" and .method=="vanilla" and (.condition=="wildchat" or (.condition==null and .corpus=="wildchat")))' ../per_request_raw.jsonl
  ```
