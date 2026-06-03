# DeepSeek-R1-Distill-Llama-70B × vanilla × wildchat

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 3126.6 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 319.3 |
| total_completion_tokens | 998349 |
| TTFT p50/p99 ms | 28.6/129.4 |
| TPOT p50/p99 ms | 9.0/9.4 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 98.3 / 1268912 |
| CPU util | 4.8 |
| reqtps_avg | 110.6 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 29.53 | 25.27 | 7.9 |
| p50 | 1313 | 11712.89 | 28.56 | 9.0 |
| p99 | 8192 | 75130.76 | 129.36 | 9.36 |
| max | 8192 | 75254.98 | 133.24 | 9.39 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Llama-70B_vanilla_wildchat.json`](../summ_DeepSeek-R1-Distill-Llama-70B_vanilla_wildchat.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Llama-70B" and .method=="vanilla" and (.condition=="wildchat" or (.condition==null and .corpus=="wildchat")))' ../per_request_raw.jsonl
  ```
