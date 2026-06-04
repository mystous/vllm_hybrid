# Qwen2.5-32B-Instruct × llm-d-c8 × lmsys

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 1875.7 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 566.8 |
| total_completion_tokens | 1063222 |
| TTFT p50/p99 ms | 25.4/47.7 |
| TPOT p50/p99 ms | 6.2/11.3 |
| accept α (acc/draft) | 0.6725 (529448.0/787297.0) |
| GPU util / mem MiB | 86.4 / 1230906 |
| CPU util | 5.3 |
| reqtps_avg | 214.2 |
| concurrency / max_tokens / stream | 8 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 2 | 34.08 | 15.56 | 0.96 |
| p50 | 492 | 3456.47 | 25.42 | 6.2 |
| p99 | 8192 | 52032.87 | 47.69 | 11.33 |
| max | 8192 | 52954.66 | 59.81 | 12.44 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-32B-Instruct_llm-d-c8_lmsys.json`](../summ_Qwen2.5-32B-Instruct_llm-d-c8_lmsys.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-32B-Instruct" and .method=="llm-d-c8" and (.condition=="lmsys" or (.condition==null and .corpus=="lmsys")))' ../per_request_raw.jsonl
  ```
