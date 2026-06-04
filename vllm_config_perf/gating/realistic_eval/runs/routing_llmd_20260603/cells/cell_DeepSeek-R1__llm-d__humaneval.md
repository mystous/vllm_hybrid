# DeepSeek-R1 × llm-d × humaneval

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 861.7 |
| n_ok/n | 164/164 (err 0) |
| wall_total_s | 144.3 |
| total_completion_tokens | 124348 |
| TTFT p50/p99 ms | 86.5/192.7 |
| TPOT p50/p99 ms | 22.8/36.5 |
| accept α (acc/draft) | 0.6589 (27458.0/41674.0) |
| GPU util / mem MiB | 79.8 / 1419616 |
| CPU util | 4.7 |
| reqtps_avg | 43.9 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=164)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 8 | 225.2 | 59.74 | 3.64 |
| p50 | 105 | 2698.12 | 86.54 | 22.81 |
| p99 | 8192 | 133968.82 | 192.69 | 36.52 |
| max | 8192 | 134568.97 | 196.26 | 38.73 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1_llm-d_humaneval.json`](../summ_DeepSeek-R1_llm-d_humaneval.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1" and .method=="llm-d" and (.condition=="humaneval" or (.condition==null and .corpus=="humaneval")))' ../per_request_raw.jsonl
  ```
