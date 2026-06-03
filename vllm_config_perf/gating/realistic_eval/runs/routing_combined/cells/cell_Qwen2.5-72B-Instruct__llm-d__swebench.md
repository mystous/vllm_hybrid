# Qwen2.5-72B-Instruct × llm-d × swebench

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 1971.4 |
| n_ok/n | 297/297 (err 0) |
| wall_total_s | 149.0 |
| total_completion_tokens | 293797 |
| TTFT p50/p99 ms | 48.2/276.2 |
| TPOT p50/p99 ms | 10.6/15.0 |
| accept α (acc/draft) | 0.4286 (80142.0/186983.0) |
| GPU util / mem MiB | 74.9 / 1231664 |
| CPU util | 4.4 |
| reqtps_avg | 98.6 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=297)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 19 | 266.03 | 30.99 | 1.96 |
| p50 | 651 | 7047.57 | 48.18 | 10.57 |
| p99 | 8192 | 83818.24 | 276.17 | 14.97 |
| max | 8192 | 84880.65 | 279.31 | 16.16 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-72B-Instruct_llm-d_swebench.json`](../summ_Qwen2.5-72B-Instruct_llm-d_swebench.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-72B-Instruct" and .method=="llm-d" and (.condition=="swebench" or (.condition==null and .corpus=="swebench")))' ../per_request_raw.jsonl
  ```
