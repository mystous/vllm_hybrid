# DeepSeek-R1-Distill-Llama-70B × llm-d × swebench

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 2858.4 |
| n_ok/n | 297/297 (err 0) |
| wall_total_s | 354.4 |
| total_completion_tokens | 1012945 |
| TTFT p50/p99 ms | 52.6/273.6 |
| TPOT p50/p99 ms | 10.5/16.2 |
| accept α (acc/draft) | 0.3548 (293756.0/827997.0) |
| GPU util / mem MiB | 93.4 / 1231056 |
| CPU util | 5.8 |
| reqtps_avg | 100.4 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=297)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 733 | 7629.87 | 32.79 | 2.04 |
| p50 | 2264 | 26125.05 | 52.64 | 10.49 |
| p99 | 8192 | 84902.05 | 273.65 | 16.25 |
| max | 8192 | 101595.44 | 276.37 | 16.96 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Llama-70B_llm-d_swebench.json`](../summ_DeepSeek-R1-Distill-Llama-70B_llm-d_swebench.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Llama-70B" and .method=="llm-d" and (.condition=="swebench" or (.condition==null and .corpus=="swebench")))' ../per_request_raw.jsonl
  ```
