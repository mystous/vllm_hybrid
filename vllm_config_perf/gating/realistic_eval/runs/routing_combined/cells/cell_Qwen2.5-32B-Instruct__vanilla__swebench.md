# Qwen2.5-32B-Instruct × vanilla × swebench

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 2891.6 |
| n_ok/n | 297/297 (err 0) |
| wall_total_s | 174.4 |
| total_completion_tokens | 504304 |
| TTFT p50/p99 ms | 33.7/209.1 |
| TPOT p50/p99 ms | 8.4/11.0 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 94.1 / 1267776 |
| CPU util | 4.4 |
| reqtps_avg | 122.0 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=297)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 45 | 414.28 | 24.04 | 6.05 |
| p50 | 618 | 4926.07 | 33.72 | 8.37 |
| p99 | 8192 | 77673.97 | 209.1 | 11.02 |
| max | 8192 | 77740.22 | 210.38 | 11.15 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-32B-Instruct_vanilla_swebench.json`](../summ_Qwen2.5-32B-Instruct_vanilla_swebench.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-32B-Instruct" and .method=="vanilla" and (.condition=="swebench" or (.condition==null and .corpus=="swebench")))' ../per_request_raw.jsonl
  ```
