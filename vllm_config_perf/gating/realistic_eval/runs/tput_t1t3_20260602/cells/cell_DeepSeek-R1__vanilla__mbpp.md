# DeepSeek-R1 × vanilla × mbpp

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 1436.9 |
| n_ok/n | 198/198 (err 0) |
| wall_total_s | 353.3 |
| total_completion_tokens | 507680 |
| TTFT p50/p99 ms | 62.3/161.8 |
| TPOT p50/p99 ms | 18.8/19.6 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 98.3 / 1277664 |
| CPU util | 4.8 |
| reqtps_avg | 53.7 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=198)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 65.11 | 48.88 | 13.68 |
| p50 | 1025 | 19963.91 | 62.26 | 18.75 |
| p99 | 8192 | 156550.84 | 161.84 | 19.63 |
| max | 8192 | 158327.52 | 164.05 | 19.86 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1_vanilla_mbpp.json`](../summ_DeepSeek-R1_vanilla_mbpp.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1" and .method=="vanilla" and (.condition=="mbpp" or (.condition==null and .corpus=="mbpp")))' ../per_request_raw.jsonl
  ```
