# Qwen2.5-72B-Instruct × suffix × wildchat

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 2621.0 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 263.3 |
| total_completion_tokens | 690162 |
| TTFT p50/p99 ms | 44.4/125.1 |
| TPOT p50/p99 ms | 15.1/20.3 |
| accept α (acc/draft) | 0.4652 (380173.0/817224.0) |
| GPU util / mem MiB | 86.8 / 1269136 |
| CPU util | 4.3 |
| reqtps_avg | 92.3 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 42.37 | 30.35 | 1.41 |
| p50 | 759 | 11111.4 | 44.44 | 15.1 |
| p99 | 8192 | 55616.29 | 125.05 | 20.29 |
| max | 8192 | 85082.67 | 134.96 | 21.16 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-72B-Instruct_suffix_wildchat.json`](../summ_Qwen2.5-72B-Instruct_suffix_wildchat.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-72B-Instruct" and .method=="suffix" and (.condition=="wildchat" or (.condition==null and .corpus=="wildchat")))' ../per_request_raw.jsonl
  ```
