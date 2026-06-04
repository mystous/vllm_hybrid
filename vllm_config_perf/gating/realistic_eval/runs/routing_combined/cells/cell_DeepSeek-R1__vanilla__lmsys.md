# DeepSeek-R1 × vanilla × lmsys

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 1533.0 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 742.6 |
| total_completion_tokens | 1138338 |
| TTFT p50/p99 ms | 64.5/193.6 |
| TPOT p50/p99 ms | 19.4/20.1 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 98.6 / 1277680 |
| CPU util | 4.8 |
| reqtps_avg | 50.1 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 62.73 | 55.47 | 13.9 |
| p50 | 1000 | 19333.35 | 64.45 | 19.43 |
| p99 | 8192 | 161084.61 | 193.61 | 20.09 |
| max | 8192 | 161110.03 | 196.96 | 21.03 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1_vanilla_lmsys.json`](../summ_DeepSeek-R1_vanilla_lmsys.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1" and .method=="vanilla" and (.condition=="lmsys" or (.condition==null and .corpus=="lmsys")))' ../per_request_raw.jsonl
  ```
