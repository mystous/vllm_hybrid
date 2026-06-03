# Qwen2.5-7B-Instruct × vanilla × sharegpt

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 4188.7 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 373.1 |
| total_completion_tokens | 1563020 |
| TTFT p50/p99 ms | 28.1/172.9 |
| TPOT p50/p99 ms | 7.1/8.3 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 81.7 / 632544 |
| CPU util | 2.7 |
| reqtps_avg | 151.4 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 14 | 105.63 | 15.69 | 3.63 |
| p50 | 769 | 5093.78 | 28.09 | 7.12 |
| p99 | 8192 | 61896.1 | 172.94 | 8.26 |
| max | 8192 | 62075.98 | 173.5 | 8.51 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-7B-Instruct_vanilla_sharegpt.json`](../summ_Qwen2.5-7B-Instruct_vanilla_sharegpt.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-7B-Instruct" and .method=="vanilla" and (.condition=="sharegpt" or (.condition==null and .corpus=="sharegpt")))' ../per_request_raw.jsonl
  ```
