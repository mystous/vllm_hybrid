# DeepSeek-R1 × llm-d × lmsys

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 1167.9 |
| n_ok/n | 496/500 (err 4) |
| wall_total_s | 959.5 |
| total_completion_tokens | 1120623 |
| TTFT p50/p99 ms | 91.4/269.7 |
| TPOT p50/p99 ms | 22.8/74.3 |
| accept α (acc/draft) | 0.5362 (295206.0/550555.0) |
| GPU util / mem MiB | 95.8 / 1419622 |
| CPU util | 5.3 |
| reqtps_avg | 39.9 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=496)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 66.37 | 53.8 | 5.36 |
| p50 | 955 | 28024.52 | 91.36 | 22.78 |
| p99 | 8192 | 237293.91 | 269.69 | 74.28 |
| max | 8192 | 278916.12 | 271.55 | 77.18 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1_llm-d_lmsys.json`](../summ_DeepSeek-R1_llm-d_lmsys.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1" and .method=="llm-d" and (.condition=="lmsys" or (.condition==null and .corpus=="lmsys")))' ../per_request_raw.jsonl
  ```
