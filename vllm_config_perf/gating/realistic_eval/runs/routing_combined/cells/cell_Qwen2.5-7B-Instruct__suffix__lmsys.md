# Qwen2.5-7B-Instruct × suffix × lmsys

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 5956.5 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 276.9 |
| total_completion_tokens | 1649635 |
| TTFT p50/p99 ms | 44.8/70.9 |
| TPOT p50/p99 ms | 11.2/23.5 |
| accept α (acc/draft) | 0.6841 (1270364.0/1856943.0) |
| GPU util / mem MiB | 33.4 / 632560 |
| CPU util | 2.5 |
| reqtps_avg | 164.2 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 2 | 56.11 | 14.86 | 1.01 |
| p50 | 666 | 9642.55 | 44.81 | 11.2 |
| p99 | 8192 | 91970.97 | 70.88 | 23.51 |
| max | 8192 | 106778.54 | 80.46 | 27.84 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-7B-Instruct_suffix_lmsys.json`](../summ_Qwen2.5-7B-Instruct_suffix_lmsys.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-7B-Instruct" and .method=="suffix" and (.condition=="lmsys" or (.condition==null and .corpus=="lmsys")))' ../per_request_raw.jsonl
  ```
