# Llama-3.1-70B-Instruct × vanilla × lmsys

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 3039.5 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 260.1 |
| total_completion_tokens | 790684 |
| TTFT p50/p99 ms | 28.8/144.6 |
| TPOT p50/p99 ms | 9.1/9.6 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 98.3 / 1268912 |
| CPU util | 4.8 |
| reqtps_avg | 107.7 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 27.44 | 25.27 | 8.1 |
| p50 | 438 | 3947.89 | 28.79 | 9.12 |
| p99 | 8192 | 75724.95 | 144.56 | 9.57 |
| max | 8192 | 75799.22 | 148.15 | 9.87 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-70B-Instruct_vanilla_lmsys.json`](../summ_Llama-3.1-70B-Instruct_vanilla_lmsys.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-70B-Instruct" and .method=="vanilla" and (.condition=="lmsys" or (.condition==null and .corpus=="lmsys")))' ../per_request_raw.jsonl
  ```
