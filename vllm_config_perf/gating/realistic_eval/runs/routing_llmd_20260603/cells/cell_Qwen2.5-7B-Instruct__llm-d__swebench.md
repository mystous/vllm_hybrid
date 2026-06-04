# Qwen2.5-7B-Instruct × llm-d × swebench

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 6282.9 |
| n_ok/n | 297/297 (err 0) |
| wall_total_s | 80.6 |
| total_completion_tokens | 506554 |
| TTFT p50/p99 ms | 28.3/109.5 |
| TPOT p50/p99 ms | 4.6/9.8 |
| accept α (acc/draft) | 0.6229 (231050.0/370945.0) |
| GPU util / mem MiB | 61.6 / 614764 |
| CPU util | 3.1 |
| reqtps_avg | 238.6 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=297)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 32 | 139.99 | 17.81 | 0.7 |
| p50 | 630 | 3078.55 | 28.26 | 4.56 |
| p99 | 8192 | 38800.85 | 109.53 | 9.8 |
| max | 8192 | 39041.39 | 110.38 | 10.32 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-7B-Instruct_llm-d_swebench.json`](../summ_Qwen2.5-7B-Instruct_llm-d_swebench.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-7B-Instruct" and .method=="llm-d" and (.condition=="swebench" or (.condition==null and .corpus=="swebench")))' ../per_request_raw.jsonl
  ```
