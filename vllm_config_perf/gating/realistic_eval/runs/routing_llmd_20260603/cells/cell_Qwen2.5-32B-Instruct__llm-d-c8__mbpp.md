# Qwen2.5-32B-Instruct × llm-d-c8 × mbpp

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 1919.6 |
| n_ok/n | 198/198 (err 0) |
| wall_total_s | 287.0 |
| total_completion_tokens | 550863 |
| TTFT p50/p99 ms | 24.3/37.1 |
| TPOT p50/p99 ms | 6.0/9.7 |
| accept α (acc/draft) | 0.6571 (294600.0/448325.0) |
| GPU util / mem MiB | 85.0 / 1230906 |
| CPU util | 5.2 |
| reqtps_avg | 237.7 |
| concurrency / max_tokens / stream | 8 / 8192 / True |

## per-request 분포 (raw 기반, n=198)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 71 | 443.49 | 17.82 | 1.38 |
| p50 | 605 | 3933.7 | 24.31 | 5.97 |
| p99 | 8192 | 51025.17 | 37.07 | 9.67 |
| max | 8192 | 51142.03 | 37.49 | 10.21 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-32B-Instruct_llm-d-c8_mbpp.json`](../summ_Qwen2.5-32B-Instruct_llm-d-c8_mbpp.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-32B-Instruct" and .method=="llm-d-c8" and (.condition=="mbpp" or (.condition==null and .corpus=="mbpp")))' ../per_request_raw.jsonl
  ```
