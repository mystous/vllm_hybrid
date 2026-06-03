# DeepSeek-R1 × vanilla × sharegpt

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 1474.7 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 848.2 |
| total_completion_tokens | 1250867 |
| TTFT p50/p99 ms | 64.5/12406.9 |
| TPOT p50/p99 ms | 19.6/26.7 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 95.0 / 1274144 |
| CPU util | 4.8 |
| reqtps_avg | 46.7 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 58.37 | 56.83 | 13.06 |
| p50 | 1305 | 25841.28 | 64.46 | 19.59 |
| p99 | 8192 | 176346.51 | 12406.87 | 26.72 |
| max | 8192 | 176348.94 | 12408.31 | 568.01 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1_vanilla_sharegpt.json`](../summ_DeepSeek-R1_vanilla_sharegpt.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1" and .method=="vanilla" and (.condition=="sharegpt" or (.condition==null and .corpus=="sharegpt")))' ../per_request_raw.jsonl
  ```
