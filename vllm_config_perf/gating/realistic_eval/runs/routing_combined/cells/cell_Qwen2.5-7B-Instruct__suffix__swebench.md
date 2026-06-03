# Qwen2.5-7B-Instruct × suffix × swebench

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 5416.5 |
| n_ok/n | 297/297 (err 0) |
| wall_total_s | 101.4 |
| total_completion_tokens | 549438 |
| TTFT p50/p99 ms | 40.5/124.6 |
| TPOT p50/p99 ms | 9.5/14.6 |
| accept α (acc/draft) | 1.1062 (779308.0/704513.0) |
| GPU util / mem MiB | 41.4 / 632560 |
| CPU util | 2.6 |
| reqtps_avg | 155.4 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=297)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 32 | 292.15 | 18.43 | 0.85 |
| p50 | 650 | 6389.88 | 40.52 | 9.47 |
| p99 | 8192 | 35936.55 | 124.6 | 14.57 |
| max | 8192 | 62959.9 | 125.27 | 14.95 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-7B-Instruct_suffix_swebench.json`](../summ_Qwen2.5-7B-Instruct_suffix_swebench.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-7B-Instruct" and .method=="suffix" and (.condition=="swebench" or (.condition==null and .corpus=="swebench")))' ../per_request_raw.jsonl
  ```
