# Qwen2.5-72B-Instruct × llm-d × mix

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 3265.2 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 208.3 |
| total_completion_tokens | 680040 |
| TTFT p50/p99 ms | 41.8/140.7 |
| TPOT p50/p99 ms | 9.9/16.6 |
| accept α (acc/draft) | 0.6773 (339414.0/501102.0) |
| GPU util / mem MiB | 88.3 / 1231666 |
| CPU util | 5.1 |
| reqtps_avg | 169.0 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 35.3 | 25.02 | 1.56 |
| p50 | 633 | 5285.14 | 41.81 | 9.93 |
| p99 | 8192 | 87684.48 | 140.65 | 16.58 |
| max | 8192 | 101091.63 | 144.1 | 25.34 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-72B-Instruct_llm-d_mix.json`](../summ_Qwen2.5-72B-Instruct_llm-d_mix.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-72B-Instruct" and .method=="llm-d" and .condition=="mix")' ../per_request_raw.jsonl
  ```
