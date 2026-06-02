# DeepSeek-R1-Distill-Qwen-32B × vanilla × lmsys

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 4898.5 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 229.9 |
| total_completion_tokens | 1126271 |
| TTFT p50/p99 ms | 24.2/123.7 |
| TPOT p50/p99 ms | 5.9/6.1 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 97.7 / 1267759 |
| CPU util | 4.8 |
| reqtps_avg | 167.4 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 5 | 49.26 | 21.71 | 5.33 |
| p50 | 953 | 5632.94 | 24.16 | 5.88 |
| p99 | 8192 | 48608.5 | 123.7 | 6.13 |
| max | 8192 | 48743.39 | 124.9 | 6.35 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Qwen-32B_vanilla_lmsys.json`](../summ_DeepSeek-R1-Distill-Qwen-32B_vanilla_lmsys.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Qwen-32B" and .method=="vanilla" and (.condition=="lmsys" or (.condition==null and .corpus=="lmsys")))' ../per_request_raw.jsonl
  ```
