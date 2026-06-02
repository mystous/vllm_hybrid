# Qwen2.5-72B-Instruct × vanilla × swebench

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 2361.3 |
| n_ok/n | 297/297 (err 0) |
| wall_total_s | 143.4 |
| total_completion_tokens | 338540 |
| TTFT p50/p99 ms | 40.6/320.5 |
| TPOT p50/p99 ms | 9.3/10.1 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 97.4 / 1269104 |
| CPU util | 4.6 |
| reqtps_avg | 106.6 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=297)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 19 | 214.66 | 26.62 | 8.36 |
| p50 | 654 | 6203.46 | 40.63 | 9.33 |
| p99 | 8192 | 77163.15 | 320.52 | 10.05 |
| max | 8192 | 77340.99 | 320.88 | 10.11 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-72B-Instruct_vanilla_swebench.json`](../summ_Qwen2.5-72B-Instruct_vanilla_swebench.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-72B-Instruct" and .method=="vanilla" and (.condition=="swebench" or (.condition==null and .corpus=="swebench")))' ../per_request_raw.jsonl
  ```
