# Qwen2.5-7B-Instruct × vanilla × lmsys

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 4090.1 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 401.4 |
| total_completion_tokens | 1641947 |
| TTFT p50/p99 ms | 27.7/66.7 |
| TPOT p50/p99 ms | 7.2/9.4 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 82.0 / 632552 |
| CPU util | 2.7 |
| reqtps_avg | 148.7 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 2 | 34.62 | 16.64 | 3.36 |
| p50 | 629 | 4277.64 | 27.7 | 7.25 |
| p99 | 8192 | 63580.1 | 66.66 | 9.39 |
| max | 8192 | 63893.56 | 70.09 | 10.0 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-7B-Instruct_vanilla_lmsys.json`](../summ_Qwen2.5-7B-Instruct_vanilla_lmsys.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-7B-Instruct" and .method=="vanilla" and (.condition=="lmsys" or (.condition==null and .corpus=="lmsys")))' ../per_request_raw.jsonl
  ```
