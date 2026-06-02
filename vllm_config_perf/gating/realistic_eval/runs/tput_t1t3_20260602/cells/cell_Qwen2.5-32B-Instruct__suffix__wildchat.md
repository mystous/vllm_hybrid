# Qwen2.5-32B-Instruct × suffix × wildchat

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 4884.2 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 205.7 |
| total_completion_tokens | 1004616 |
| TTFT p50/p99 ms | 38.8/93.2 |
| TPOT p50/p99 ms | 13.2/18.8 |
| accept α (acc/draft) | 0.619 (695322.0/1123318.0) |
| GPU util / mem MiB | 75.1 / 1267776 |
| CPU util | 4.3 |
| reqtps_avg | 131.7 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 29 | 361.5 | 19.96 | 1.38 |
| p50 | 711 | 8867.85 | 38.81 | 13.18 |
| p99 | 8192 | 46194.56 | 93.16 | 18.79 |
| max | 8192 | 125023.73 | 95.69 | 19.96 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-32B-Instruct_suffix_wildchat.json`](../summ_Qwen2.5-32B-Instruct_suffix_wildchat.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-32B-Instruct" and .method=="suffix" and (.condition=="wildchat" or (.condition==null and .corpus=="wildchat")))' ../per_request_raw.jsonl
  ```
