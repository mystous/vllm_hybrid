# Qwen2.5-32B-Instruct × suffix × humaneval

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 4858.8 |
| n_ok/n | 164/164 (err 0) |
| wall_total_s | 69.8 |
| total_completion_tokens | 339315 |
| TTFT p50/p99 ms | 57.8/90.3 |
| TPOT p50/p99 ms | 8.4/24.3 |
| accept α (acc/draft) | 0.6828 (271530.0/397665.0) |
| GPU util / mem MiB | 64.0 / 1267776 |
| CPU util | 4.3 |
| reqtps_avg | 137.8 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=164)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 14 | 125.04 | 20.67 | 1.62 |
| p50 | 302 | 4484.58 | 57.79 | 8.44 |
| p99 | 8192 | 43703.33 | 90.3 | 24.32 |
| max | 8192 | 55335.49 | 93.39 | 27.22 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-32B-Instruct_suffix_humaneval.json`](../summ_Qwen2.5-32B-Instruct_suffix_humaneval.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-32B-Instruct" and .method=="suffix" and (.condition=="humaneval" or (.condition==null and .corpus=="humaneval")))' ../per_request_raw.jsonl
  ```
