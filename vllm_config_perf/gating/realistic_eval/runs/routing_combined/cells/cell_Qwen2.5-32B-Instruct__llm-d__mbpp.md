# Qwen2.5-32B-Instruct × llm-d × mbpp

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 5553.8 |
| n_ok/n | 198/198 (err 0) |
| wall_total_s | 93.2 |
| total_completion_tokens | 517632 |
| TTFT p50/p99 ms | 31.2/54.9 |
| TPOT p50/p99 ms | 5.3/9.2 |
| accept α (acc/draft) | 0.7367 (265623.0/360562.0) |
| GPU util / mem MiB | 85.7 / 1230928 |
| CPU util | 5.4 |
| reqtps_avg | 311.3 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=396)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 71 | 180.34 | 19.17 | 1.09 |
| p50 | 585 | 4403.87 | 30.5 | 6.81 |
| p99 | 8192 | 74500.47 | 54.19 | 11.39 |
| max | 8192 | 74612.23 | 58.62 | 12.09 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-32B-Instruct_llm-d_mbpp.json`](../summ_Qwen2.5-32B-Instruct_llm-d_mbpp.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-32B-Instruct" and .method=="llm-d" and (.condition=="mbpp" or (.condition==null and .corpus=="mbpp")))' ../per_request_raw.jsonl
  ```
