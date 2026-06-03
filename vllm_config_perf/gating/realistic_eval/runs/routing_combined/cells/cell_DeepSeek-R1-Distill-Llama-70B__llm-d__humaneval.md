# DeepSeek-R1-Distill-Llama-70B × llm-d × humaneval

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 2379.2 |
| n_ok/n | 164/164 (err 0) |
| wall_total_s | 189.2 |
| total_completion_tokens | 450228 |
| TTFT p50/p99 ms | 44.1/121.3 |
| TPOT p50/p99 ms | 10.1/13.7 |
| accept α (acc/draft) | 0.3845 (129593.0/337080.0) |
| GPU util / mem MiB | 82.2 / 1231056 |
| CPU util | 5.2 |
| reqtps_avg | 106.2 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=164)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 7 | 169.27 | 28.42 | 2.07 |
| p50 | 1804 | 19893.91 | 44.13 | 10.14 |
| p99 | 8192 | 83057.13 | 121.26 | 13.69 |
| max | 8192 | 89601.58 | 122.23 | 14.88 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Llama-70B_llm-d_humaneval.json`](../summ_DeepSeek-R1-Distill-Llama-70B_llm-d_humaneval.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Llama-70B" and .method=="llm-d" and (.condition=="humaneval" or (.condition==null and .corpus=="humaneval")))' ../per_request_raw.jsonl
  ```
