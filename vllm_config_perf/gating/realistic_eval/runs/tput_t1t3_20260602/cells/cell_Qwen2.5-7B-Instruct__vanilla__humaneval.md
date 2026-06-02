# Qwen2.5-7B-Instruct × vanilla × humaneval

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 3754.1 |
| n_ok/n | 164/164 (err 0) |
| wall_total_s | 63.2 |
| total_completion_tokens | 237291 |
| TTFT p50/p99 ms | 20.2/58.5 |
| TPOT p50/p99 ms | 4.4/6.4 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 81.0 / 632552 |
| CPU util | 2.7 |
| reqtps_avg | 217.6 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=164)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 7 | 72.42 | 16.09 | 2.83 |
| p50 | 261 | 1109.36 | 20.22 | 4.36 |
| p99 | 8192 | 52194.33 | 58.48 | 6.37 |
| max | 8192 | 52256.67 | 60.77 | 6.38 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-7B-Instruct_vanilla_humaneval.json`](../summ_Qwen2.5-7B-Instruct_vanilla_humaneval.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-7B-Instruct" and .method=="vanilla" and (.condition=="humaneval" or (.condition==null and .corpus=="humaneval")))' ../per_request_raw.jsonl
  ```
