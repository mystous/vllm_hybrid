# Qwen2.5-32B-Instruct × suffix × lmsys

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 4478.1 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 221.9 |
| total_completion_tokens | 993561 |
| TTFT p50/p99 ms | 42.1/102.1 |
| TPOT p50/p99 ms | 13.7/21.9 |
| accept α (acc/draft) | 0.6246 (714251.0/1143600.0) |
| GPU util / mem MiB | 72.8 / 1267776 |
| CPU util | 4.3 |
| reqtps_avg | 125.1 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 2 | 72.75 | 20.53 | 1.21 |
| p50 | 446 | 6551.64 | 42.14 | 13.65 |
| p99 | 8192 | 79851.36 | 102.08 | 21.88 |
| max | 8192 | 111840.16 | 104.7 | 25.09 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-32B-Instruct_suffix_lmsys.json`](../summ_Qwen2.5-32B-Instruct_suffix_lmsys.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-32B-Instruct" and .method=="suffix" and (.condition=="lmsys" or (.condition==null and .corpus=="lmsys")))' ../per_request_raw.jsonl
  ```
