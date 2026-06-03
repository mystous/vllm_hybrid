# Qwen2.5-72B-Instruct × suffix × swebench

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 2646.7 |
| n_ok/n | 297/297 (err 0) |
| wall_total_s | 105.1 |
| total_completion_tokens | 278261 |
| TTFT p50/p99 ms | 48.8/323.6 |
| TPOT p50/p99 ms | 12.2/16.0 |
| accept α (acc/draft) | 0.3817 (143242.0/375282.0) |
| GPU util / mem MiB | 85.0 / 1269124 |
| CPU util | 4.3 |
| reqtps_avg | 91.0 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=297)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 19 | 316.79 | 32.87 | 1.61 |
| p50 | 646 | 7947.58 | 48.84 | 12.25 |
| p99 | 8192 | 34027.12 | 323.56 | 15.97 |
| max | 8192 | 57792.47 | 340.16 | 18.72 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-72B-Instruct_suffix_swebench.json`](../summ_Qwen2.5-72B-Instruct_suffix_swebench.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-72B-Instruct" and .method=="suffix" and (.condition=="swebench" or (.condition==null and .corpus=="swebench")))' ../per_request_raw.jsonl
  ```
