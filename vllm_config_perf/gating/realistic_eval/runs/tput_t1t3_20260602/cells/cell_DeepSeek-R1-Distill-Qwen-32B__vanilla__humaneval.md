# DeepSeek-R1-Distill-Qwen-32B × vanilla × humaneval

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 3461.8 |
| n_ok/n | 164/164 (err 0) |
| wall_total_s | 77.7 |
| total_completion_tokens | 268983 |
| TTFT p50/p99 ms | 24.8/80.5 |
| TPOT p50/p99 ms | 5.7/5.8 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 96.0 / 1267776 |
| CPU util | 4.8 |
| reqtps_avg | 170.9 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=164)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 7 | 108.71 | 20.18 | 5.22 |
| p50 | 564 | 3266.3 | 24.82 | 5.74 |
| p99 | 8192 | 47115.81 | 80.47 | 5.8 |
| max | 8192 | 47116.73 | 81.02 | 5.8 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Qwen-32B_vanilla_humaneval.json`](../summ_DeepSeek-R1-Distill-Qwen-32B_vanilla_humaneval.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Qwen-32B" and .method=="vanilla" and (.condition=="humaneval" or (.condition==null and .corpus=="humaneval")))' ../per_request_raw.jsonl
  ```
