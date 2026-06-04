# Llama-3.1-405B-Instruct-FP8 × vanilla × lmsys

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 1219.5 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 754.5 |
| total_completion_tokens | 920145 |
| TTFT p50/p99 ms | 71.0/407.4 |
| TPOT p50/p99 ms | 23.4/24.2 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 98.9 / 1272432 |
| CPU util | 4.8 |
| reqtps_avg | 41.6 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 71.32 | 65.89 | 19.88 |
| p50 | 442 | 10318.23 | 71.02 | 23.39 |
| p99 | 8192 | 192994.43 | 407.42 | 24.18 |
| max | 8192 | 193104.84 | 412.27 | 26.42 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-405B-Instruct-FP8_vanilla_lmsys.json`](../summ_Llama-3.1-405B-Instruct-FP8_vanilla_lmsys.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-405B-Instruct-FP8" and .method=="vanilla" and (.condition=="lmsys" or (.condition==null and .corpus=="lmsys")))' ../per_request_raw.jsonl
  ```
