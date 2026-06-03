# Qwen2.5-72B-Instruct × llm-d × wildchat

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 2524.7 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 275.8 |
| total_completion_tokens | 696291 |
| TTFT p50/p99 ms | 41.4/151.0 |
| TPOT p50/p99 ms | 10.8/19.4 |
| accept α (acc/draft) | 0.4663 (222739.0/477662.0) |
| GPU util / mem MiB | 87.3 / 1231664 |
| CPU util | 5.1 |
| reqtps_avg | 95.3 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 40.91 | 26.97 | 1.42 |
| p50 | 770 | 9163.45 | 41.42 | 10.83 |
| p99 | 8192 | 85686.31 | 151.03 | 19.45 |
| max | 8192 | 127575.68 | 176.2 | 21.77 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-72B-Instruct_llm-d_wildchat.json`](../summ_Qwen2.5-72B-Instruct_llm-d_wildchat.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-72B-Instruct" and .method=="llm-d" and (.condition=="wildchat" or (.condition==null and .corpus=="wildchat")))' ../per_request_raw.jsonl
  ```
