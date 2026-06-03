# Qwen2.5-7B-Instruct × vanilla × swebench

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 4120.1 |
| n_ok/n | 297/297 (err 0) |
| wall_total_s | 138.5 |
| total_completion_tokens | 570688 |
| TTFT p50/p99 ms | 27.8/96.0 |
| TPOT p50/p99 ms | 6.4/8.6 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 83.0 / 632549 |
| CPU util | 2.7 |
| reqtps_avg | 170.2 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=297)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 32 | 227.48 | 16.54 | 3.64 |
| p50 | 631 | 3686.37 | 27.81 | 6.38 |
| p99 | 8192 | 58357.26 | 96.01 | 8.57 |
| max | 8192 | 58635.13 | 100.28 | 9.07 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-7B-Instruct_vanilla_swebench.json`](../summ_Qwen2.5-7B-Instruct_vanilla_swebench.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-7B-Instruct" and .method=="vanilla" and (.condition=="swebench" or (.condition==null and .corpus=="swebench")))' ../per_request_raw.jsonl
  ```
