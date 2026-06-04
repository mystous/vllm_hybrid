# DeepSeek-R1 × llm-d × wildchat

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 1131.1 |
| n_ok/n | 494/500 (err 6) |
| wall_total_s | 1224.9 |
| total_completion_tokens | 1385486 |
| TTFT p50/p99 ms | 95.7/8337.9 |
| TPOT p50/p99 ms | 22.1/83.3 |
| accept α (acc/draft) | 0.5417 (361137.0/666708.0) |
| GPU util / mem MiB | 94.6 / 1419621 |
| CPU util | 5.3 |
| reqtps_avg | 40.8 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=494)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 50.81 | 51.02 | 4.44 |
| p50 | 1510 | 47713.71 | 95.71 | 22.11 |
| p99 | 8192 | 252932.88 | 8337.93 | 83.32 |
| max | 8192 | 299654.96 | 8339.13 | 102.52 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1_llm-d_wildchat.json`](../summ_DeepSeek-R1_llm-d_wildchat.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1" and .method=="llm-d" and (.condition=="wildchat" or (.condition==null and .corpus=="wildchat")))' ../per_request_raw.jsonl
  ```
