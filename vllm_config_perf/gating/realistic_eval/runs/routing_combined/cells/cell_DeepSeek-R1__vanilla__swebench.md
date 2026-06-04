# DeepSeek-R1 × vanilla × swebench

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 1473.7 |
| n_ok/n | 297/297 (err 0) |
| wall_total_s | 594.2 |
| total_completion_tokens | 875716 |
| TTFT p50/p99 ms | 80.6/8862.0 |
| TPOT p50/p99 ms | 19.7/174.6 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 97.1 / 1277660 |
| CPU util | 4.8 |
| reqtps_avg | 41.8 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=297)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 62.51 | 56.93 | 14.38 |
| p50 | 1351 | 26663.46 | 80.62 | 19.72 |
| p99 | 8192 | 163571.71 | 8862.0 | 174.56 |
| max | 8192 | 163713.2 | 8864.19 | 344.2 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1_vanilla_swebench.json`](../summ_DeepSeek-R1_vanilla_swebench.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1" and .method=="vanilla" and (.condition=="swebench" or (.condition==null and .corpus=="swebench")))' ../per_request_raw.jsonl
  ```
