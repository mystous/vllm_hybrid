# Llama-3.1-8B-Instruct × vanilla × wildchat

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 9001.8 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 435.4 |
| total_completion_tokens | 3919030 |
| TTFT p50/p99 ms | 41.1/83.4 |
| TPOT p50/p99 ms | 3.5/3.6 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 94.8 / 1265360 |
| CPU util | 4.6 |
| reqtps_avg | 283.2 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 21 | 139.58 | 15.97 | 3.48 |
| p50 | 8192 | 28736.27 | 41.1 | 3.51 |
| p99 | 8192 | 29267.71 | 83.44 | 3.57 |
| max | 8192 | 29275.77 | 84.66 | 4.79 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-8B-Instruct_vanilla_wildchat.json`](../summ_Llama-3.1-8B-Instruct_vanilla_wildchat.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-8B-Instruct" and .method=="vanilla" and (.condition=="wildchat" or (.condition==null and .corpus=="wildchat")))' ../per_request_raw.jsonl
  ```
