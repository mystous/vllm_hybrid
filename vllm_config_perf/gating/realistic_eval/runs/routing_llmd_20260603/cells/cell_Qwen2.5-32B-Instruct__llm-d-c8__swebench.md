# Qwen2.5-32B-Instruct × llm-d-c8 × swebench

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 1706.8 |
| n_ok/n | 297/297 (err 0) |
| wall_total_s | 287.7 |
| total_completion_tokens | 491059 |
| TTFT p50/p99 ms | 29.7/68.3 |
| TPOT p50/p99 ms | 6.1/9.8 |
| accept α (acc/draft) | 0.6209 (230092.0/370604.0) |
| GPU util / mem MiB | 85.2 / 1230906 |
| CPU util | 5.2 |
| reqtps_avg | 203.2 |
| concurrency / max_tokens / stream | 8 / 8192 / True |

## per-request 분포 (raw 기반, n=297)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 41 | 265.19 | 18.21 | 0.94 |
| p50 | 643 | 4154.64 | 29.75 | 6.1 |
| p99 | 8192 | 49759.11 | 68.29 | 9.84 |
| max | 8192 | 50361.94 | 86.48 | 10.09 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-32B-Instruct_llm-d-c8_swebench.json`](../summ_Qwen2.5-32B-Instruct_llm-d-c8_swebench.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-32B-Instruct" and .method=="llm-d-c8" and (.condition=="swebench" or (.condition==null and .corpus=="swebench")))' ../per_request_raw.jsonl
  ```
