# Qwen2.5-32B-Instruct × vanilla × lmsys

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 3053.1 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 329.7 |
| total_completion_tokens | 1006667 |
| TTFT p50/p99 ms | 30.0/102.8 |
| TPOT p50/p99 ms | 9.2/11.2 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 94.3 / 1267776 |
| CPU util | 4.4 |
| reqtps_avg | 113.7 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 2 | 42.52 | 21.33 | 5.82 |
| p50 | 467 | 4101.5 | 30.02 | 9.22 |
| p99 | 8192 | 81539.09 | 102.76 | 11.15 |
| max | 8192 | 81809.84 | 103.53 | 11.84 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-32B-Instruct_vanilla_lmsys.json`](../summ_Qwen2.5-32B-Instruct_vanilla_lmsys.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-32B-Instruct" and .method=="vanilla" and (.condition=="lmsys" or (.condition==null and .corpus=="lmsys")))' ../per_request_raw.jsonl
  ```
