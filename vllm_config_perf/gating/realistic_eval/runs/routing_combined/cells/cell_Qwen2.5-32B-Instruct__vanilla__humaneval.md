# Qwen2.5-32B-Instruct × vanilla × humaneval

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 2570.8 |
| n_ok/n | 164/164 (err 0) |
| wall_total_s | 143.2 |
| total_completion_tokens | 368254 |
| TTFT p50/p99 ms | 30.0/83.1 |
| TPOT p50/p99 ms | 7.8/10.1 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 93.6 / 1267776 |
| CPU util | 4.4 |
| reqtps_avg | 126.0 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=164)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 14 | 144.22 | 22.71 | 5.94 |
| p50 | 302 | 2542.15 | 30.0 | 7.84 |
| p99 | 8192 | 79601.6 | 83.06 | 10.06 |
| max | 8192 | 79957.09 | 86.6 | 10.35 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-32B-Instruct_vanilla_humaneval.json`](../summ_Qwen2.5-32B-Instruct_vanilla_humaneval.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-32B-Instruct" and .method=="vanilla" and (.condition=="humaneval" or (.condition==null and .corpus=="humaneval")))' ../per_request_raw.jsonl
  ```
