# DeepSeek-R1 × vanilla × humaneval

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 1003.9 |
| n_ok/n | 164/164 (err 0) |
| wall_total_s | 127.6 |
| total_completion_tokens | 128071 |
| TTFT p50/p99 ms | 66.1/180.5 |
| TPOT p50/p99 ms | 18.1/21.4 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 97.0 / 1277664 |
| CPU util | 4.8 |
| reqtps_avg | 51.7 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=164)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 8 | 200.63 | 55.06 | 13.96 |
| p50 | 96 | 1879.01 | 66.11 | 18.11 |
| p99 | 8192 | 119425.18 | 180.46 | 21.42 |
| max | 8192 | 119782.42 | 180.83 | 22.21 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1_vanilla_humaneval.json`](../summ_DeepSeek-R1_vanilla_humaneval.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1" and .method=="vanilla" and (.condition=="humaneval" or (.condition==null and .corpus=="humaneval")))' ../per_request_raw.jsonl
  ```
