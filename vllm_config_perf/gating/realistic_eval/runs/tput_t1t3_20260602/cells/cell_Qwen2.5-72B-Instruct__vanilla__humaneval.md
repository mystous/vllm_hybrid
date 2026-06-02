# Qwen2.5-72B-Instruct × vanilla × humaneval

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 806.4 |
| n_ok/n | 164/164 (err 0) |
| wall_total_s | 69.0 |
| total_completion_tokens | 55622 |
| TTFT p50/p99 ms | 34.8/113.6 |
| TPOT p50/p99 ms | 8.9/10.4 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 96.8 / 1269104 |
| CPU util | 4.7 |
| reqtps_avg | 108.1 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=164)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 7 | 161.69 | 25.88 | 7.34 |
| p50 | 207 | 1864.73 | 34.8 | 8.91 |
| p99 | 2510 | 20225.24 | 113.56 | 10.4 |
| max | 8192 | 61793.53 | 114.36 | 12.72 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-72B-Instruct_vanilla_humaneval.json`](../summ_Qwen2.5-72B-Instruct_vanilla_humaneval.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-72B-Instruct" and .method=="vanilla" and (.condition=="humaneval" or (.condition==null and .corpus=="humaneval")))' ../per_request_raw.jsonl
  ```
