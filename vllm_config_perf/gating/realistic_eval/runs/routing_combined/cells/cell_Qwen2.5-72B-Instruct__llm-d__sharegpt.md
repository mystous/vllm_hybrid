# Qwen2.5-72B-Instruct × llm-d × sharegpt

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 2845.2 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 261.7 |
| total_completion_tokens | 744542 |
| TTFT p50/p99 ms | 42.2/370.8 |
| TPOT p50/p99 ms | 11.0/19.6 |
| accept α (acc/draft) | 0.6018 (303520.0/504372.0) |
| GPU util / mem MiB | 92.7 / 1231651 |
| CPU util | 5.3 |
| reqtps_avg | 98.9 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 27 | 301.75 | 26.21 | 1.55 |
| p50 | 700 | 8872.14 | 42.18 | 11.01 |
| p99 | 8192 | 85402.87 | 370.76 | 19.6 |
| max | 8192 | 93354.11 | 392.09 | 20.74 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-72B-Instruct_llm-d_sharegpt.json`](../summ_Qwen2.5-72B-Instruct_llm-d_sharegpt.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-72B-Instruct" and .method=="llm-d" and (.condition=="sharegpt" or (.condition==null and .corpus=="sharegpt")))' ../per_request_raw.jsonl
  ```
