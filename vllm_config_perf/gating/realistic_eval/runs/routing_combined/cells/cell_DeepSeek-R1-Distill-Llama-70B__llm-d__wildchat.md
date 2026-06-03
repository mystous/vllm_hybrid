# DeepSeek-R1-Distill-Llama-70B × llm-d × wildchat

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 2731.0 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 379.2 |
| total_completion_tokens | 1035699 |
| TTFT p50/p99 ms | 41.1/156.4 |
| TPOT p50/p99 ms | 10.4/17.5 |
| accept α (acc/draft) | 0.3595 (259659.0/722272.0) |
| GPU util / mem MiB | 89.3 / 1231056 |
| CPU util | 5.5 |
| reqtps_avg | 95.1 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 32.58 | 28.16 | 1.58 |
| p50 | 1295 | 15722.52 | 41.13 | 10.42 |
| p99 | 8192 | 84119.07 | 156.39 | 17.46 |
| max | 8192 | 89513.69 | 176.84 | 21.73 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Llama-70B_llm-d_wildchat.json`](../summ_DeepSeek-R1-Distill-Llama-70B_llm-d_wildchat.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Llama-70B" and .method=="llm-d" and (.condition=="wildchat" or (.condition==null and .corpus=="wildchat")))' ../per_request_raw.jsonl
  ```
