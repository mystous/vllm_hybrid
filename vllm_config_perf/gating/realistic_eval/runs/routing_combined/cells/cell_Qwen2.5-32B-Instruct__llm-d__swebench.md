# Qwen2.5-32B-Instruct × llm-d × swebench

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 3734.5 |
| n_ok/n | 297/297 (err 0) |
| wall_total_s | 120.5 |
| total_completion_tokens | 449978 |
| TTFT p50/p99 ms | 40.1/147.5 |
| TPOT p50/p99 ms | 6.9/11.0 |
| accept α (acc/draft) | 0.6715 (201064.0/299404.0) |
| GPU util / mem MiB | 67.2 / 1230928 |
| CPU util | 4.2 |
| reqtps_avg | 243.9 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=594)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 45 | 229.57 | 23.78 | 0.95 |
| p50 | 636 | 4958.41 | 38.44 | 7.11 |
| p99 | 8192 | 63936.13 | 174.12 | 12.98 |
| max | 8192 | 66034.09 | 176.46 | 13.91 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-32B-Instruct_llm-d_swebench.json`](../summ_Qwen2.5-32B-Instruct_llm-d_swebench.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-32B-Instruct" and .method=="llm-d" and (.condition=="swebench" or (.condition==null and .corpus=="swebench")))' ../per_request_raw.jsonl
  ```
