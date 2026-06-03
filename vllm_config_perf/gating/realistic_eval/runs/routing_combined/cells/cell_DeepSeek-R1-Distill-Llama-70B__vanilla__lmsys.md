# DeepSeek-R1-Distill-Llama-70B × vanilla × lmsys

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 2992.0 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 271.8 |
| total_completion_tokens | 813099 |
| TTFT p50/p99 ms | 28.6/147.3 |
| TPOT p50/p99 ms | 9.0/9.2 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 98.3 / 1268912 |
| CPU util | 4.9 |
| reqtps_avg | 111.6 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 28.73 | 25.35 | 7.46 |
| p50 | 916 | 8247.81 | 28.63 | 8.99 |
| p99 | 8192 | 74319.88 | 147.27 | 9.2 |
| max | 8192 | 74629.74 | 151.56 | 9.32 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Llama-70B_vanilla_lmsys.json`](../summ_DeepSeek-R1-Distill-Llama-70B_vanilla_lmsys.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Llama-70B" and .method=="vanilla" and (.condition=="lmsys" or (.condition==null and .corpus=="lmsys")))' ../per_request_raw.jsonl
  ```
