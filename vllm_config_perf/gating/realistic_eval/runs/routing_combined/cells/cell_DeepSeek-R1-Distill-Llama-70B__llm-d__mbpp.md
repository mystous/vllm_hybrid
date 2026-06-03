# DeepSeek-R1-Distill-Llama-70B × llm-d × mbpp

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 2305.3 |
| n_ok/n | 198/198 (err 0) |
| wall_total_s | 223.2 |
| total_completion_tokens | 514610 |
| TTFT p50/p99 ms | 38.9/69.9 |
| TPOT p50/p99 ms | 10.2/13.2 |
| accept α (acc/draft) | 0.2646 (123680.0/467444.0) |
| GPU util / mem MiB | 89.2 / 1231056 |
| CPU util | 5.5 |
| reqtps_avg | 98.6 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=198)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 604 | 6739.74 | 27.98 | 2.1 |
| p50 | 1860 | 19414.0 | 38.87 | 10.18 |
| p99 | 8192 | 82612.78 | 69.92 | 13.21 |
| max | 8192 | 84338.23 | 70.75 | 14.27 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Llama-70B_llm-d_mbpp.json`](../summ_DeepSeek-R1-Distill-Llama-70B_llm-d_mbpp.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Llama-70B" and .method=="llm-d" and (.condition=="mbpp" or (.condition==null and .corpus=="mbpp")))' ../per_request_raw.jsonl
  ```
