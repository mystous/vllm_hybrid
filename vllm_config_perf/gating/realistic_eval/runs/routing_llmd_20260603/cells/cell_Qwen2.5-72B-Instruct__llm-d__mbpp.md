# Qwen2.5-72B-Instruct × llm-d × mbpp

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 3609.0 |
| n_ok/n | 198/198 (err 0) |
| wall_total_s | 33.1 |
| total_completion_tokens | 119433 |
| TTFT p50/p99 ms | 34.3/67.9 |
| TPOT p50/p99 ms | 9.8/10.1 |
| accept α (acc/draft) | 0.4916 (40447.0/82283.0) |
| GPU util / mem MiB | 84.4 / 1231670 |
| CPU util | 5.3 |
| reqtps_avg | 192.2 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=396)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 32 | 133.36 | 23.39 | 1.18 |
| p50 | 551 | 4986.59 | 33.89 | 9.86 |
| p99 | 1567 | 10868.09 | 74.53 | 10.8 |
| max | 8192 | 81613.33 | 78.69 | 11.16 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-72B-Instruct_llm-d_mbpp.json`](../summ_Qwen2.5-72B-Instruct_llm-d_mbpp.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-72B-Instruct" and .method=="llm-d" and (.condition=="mbpp" or (.condition==null and .corpus=="mbpp")))' ../per_request_raw.jsonl
  ```
