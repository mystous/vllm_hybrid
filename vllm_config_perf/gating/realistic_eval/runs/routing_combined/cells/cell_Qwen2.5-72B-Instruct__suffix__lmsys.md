# Qwen2.5-72B-Instruct × suffix × lmsys

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 3429.4 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 251.8 |
| total_completion_tokens | 863466 |
| TTFT p50/p99 ms | 47.5/156.4 |
| TPOT p50/p99 ms | 15.2/23.9 |
| accept α (acc/draft) | 0.556 (572571.0/1029794.0) |
| GPU util / mem MiB | 86.2 / 1269136 |
| CPU util | 4.3 |
| reqtps_avg | 95.9 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 2 | 71.27 | 29.03 | 1.81 |
| p50 | 505 | 7829.06 | 47.55 | 15.15 |
| p99 | 8192 | 82537.16 | 156.38 | 23.9 |
| max | 8192 | 117920.45 | 160.02 | 27.37 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-72B-Instruct_suffix_lmsys.json`](../summ_Qwen2.5-72B-Instruct_suffix_lmsys.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-72B-Instruct" and .method=="suffix" and (.condition=="lmsys" or (.condition==null and .corpus=="lmsys")))' ../per_request_raw.jsonl
  ```
