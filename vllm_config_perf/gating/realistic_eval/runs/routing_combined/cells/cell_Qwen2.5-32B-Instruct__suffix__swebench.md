# Qwen2.5-32B-Instruct × suffix × swebench

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 5002.5 |
| n_ok/n | 297/297 (err 0) |
| wall_total_s | 96.6 |
| total_completion_tokens | 483243 |
| TTFT p50/p99 ms | 46.0/193.9 |
| TPOT p50/p99 ms | 11.1/15.4 |
| accept α (acc/draft) | 0.5994 (336259.0/560970.0) |
| GPU util / mem MiB | 75.6 / 1267776 |
| CPU util | 4.3 |
| reqtps_avg | 131.2 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=297)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 45 | 436.85 | 25.15 | 1.09 |
| p50 | 626 | 7137.85 | 45.96 | 11.14 |
| p99 | 8192 | 33319.43 | 193.9 | 15.36 |
| max | 8192 | 43062.9 | 206.09 | 16.34 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-32B-Instruct_suffix_swebench.json`](../summ_Qwen2.5-32B-Instruct_suffix_swebench.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-32B-Instruct" and .method=="suffix" and (.condition=="swebench" or (.condition==null and .corpus=="swebench")))' ../per_request_raw.jsonl
  ```
