# Llama-3.1-8B-Instruct × vanilla × humaneval

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 9048.1 |
| n_ok/n | 164/164 (err 0) |
| wall_total_s | 145.1 |
| total_completion_tokens | 1312490 |
| TTFT p50/p99 ms | 48.4/61.9 |
| TPOT p50/p99 ms | 3.5/3.5 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 94.1 / 1265360 |
| CPU util | 4.7 |
| reqtps_avg | 286.6 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=164)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 81 | 334.21 | 16.42 | 3.43 |
| p50 | 8192 | 28520.04 | 48.35 | 3.48 |
| p99 | 8192 | 28767.93 | 61.87 | 3.51 |
| max | 8192 | 28912.88 | 63.79 | 3.53 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-8B-Instruct_vanilla_humaneval.json`](../summ_Llama-3.1-8B-Instruct_vanilla_humaneval.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-8B-Instruct" and .method=="vanilla" and (.condition=="humaneval" or (.condition==null and .corpus=="humaneval")))' ../per_request_raw.jsonl
  ```
