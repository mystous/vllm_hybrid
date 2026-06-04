# Qwen2.5-7B-Instruct × llm-d-c8 × mbpp

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 2528.6 |
| n_ok/n | 198/198 (err 0) |
| wall_total_s | 100.2 |
| total_completion_tokens | 253437 |
| TTFT p50/p99 ms | 17.1/25.7 |
| TPOT p50/p99 ms | 3.4/5.2 |
| accept α (acc/draft) | 0.5125 (104507.0/203915.0) |
| GPU util / mem MiB | 67.6 / 614764 |
| CPU util | 3.1 |
| reqtps_avg | 327.5 |
| concurrency / max_tokens / stream | 8 / 8192 / True |

## per-request 분포 (raw 기반, n=198)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 42 | 188.52 | 12.98 | 0.75 |
| p50 | 575 | 1997.94 | 17.12 | 3.42 |
| p99 | 8192 | 31120.55 | 25.73 | 5.15 |
| max | 8192 | 31406.28 | 31.59 | 5.68 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-7B-Instruct_llm-d-c8_mbpp.json`](../summ_Qwen2.5-7B-Instruct_llm-d-c8_mbpp.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-7B-Instruct" and .method=="llm-d-c8" and (.condition=="mbpp" or (.condition==null and .corpus=="mbpp")))' ../per_request_raw.jsonl
  ```
