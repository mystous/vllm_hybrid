# Qwen2.5-7B-Instruct × llm-d-c64 × swebench

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 7532.9 |
| n_ok/n | 297/297 (err 0) |
| wall_total_s | 75.8 |
| total_completion_tokens | 571006 |
| TTFT p50/p99 ms | 41.8/141.5 |
| TPOT p50/p99 ms | 5.0/15.3 |
| accept α (acc/draft) | 0.696 (243931.0/350456.0) |
| GPU util / mem MiB | 51.8 / 614770 |
| CPU util | 3.1 |
| reqtps_avg | 224.2 |
| concurrency / max_tokens / stream | 64 / 8192 / True |

## per-request 분포 (raw 기반, n=297)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 32 | 199.87 | 19.36 | 0.8 |
| p50 | 634 | 3322.49 | 41.79 | 5.01 |
| p99 | 8192 | 53506.36 | 141.48 | 15.31 |
| max | 8192 | 53697.03 | 145.53 | 16.09 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-7B-Instruct_llm-d-c64_swebench.json`](../summ_Qwen2.5-7B-Instruct_llm-d-c64_swebench.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-7B-Instruct" and .method=="llm-d-c64" and (.condition=="swebench" or (.condition==null and .corpus=="swebench")))' ../per_request_raw.jsonl
  ```
