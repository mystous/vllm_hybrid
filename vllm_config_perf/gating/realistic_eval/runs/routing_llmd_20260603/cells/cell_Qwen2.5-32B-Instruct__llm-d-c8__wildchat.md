# Qwen2.5-32B-Instruct × llm-d-c8 × wildchat

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 1813.9 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 552.5 |
| total_completion_tokens | 1002218 |
| TTFT p50/p99 ms | 25.2/79.7 |
| TPOT p50/p99 ms | 6.2/10.9 |
| accept α (acc/draft) | 0.6555 (470087.0/717103.0) |
| GPU util / mem MiB | 88.2 / 1230906 |
| CPU util | 5.1 |
| reqtps_avg | 206.2 |
| concurrency / max_tokens / stream | 8 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 29 | 203.35 | 17.05 | 0.94 |
| p50 | 708 | 4494.06 | 25.25 | 6.21 |
| p99 | 8192 | 51093.33 | 79.67 | 10.9 |
| max | 8192 | 68639.26 | 105.29 | 11.7 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-32B-Instruct_llm-d-c8_wildchat.json`](../summ_Qwen2.5-32B-Instruct_llm-d-c8_wildchat.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-32B-Instruct" and .method=="llm-d-c8" and (.condition=="wildchat" or (.condition==null and .corpus=="wildchat")))' ../per_request_raw.jsonl
  ```
