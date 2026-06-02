# Qwen2.5-72B-Instruct × suffix × humaneval

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 2488.6 |
| n_ok/n | 164/164 (err 0) |
| wall_total_s | 16.2 |
| total_completion_tokens | 40409 |
| TTFT p50/p99 ms | 34.0/123.8 |
| TPOT p50/p99 ms | 9.8/14.8 |
| accept α (acc/draft) | 0.2771 (16933.0/61115.0) |
| GPU util / mem MiB | 78.7 / 1269134 |
| CPU util | 4.3 |
| reqtps_avg | 102.0 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=164)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 7 | 170.75 | 29.78 | 4.26 |
| p50 | 216 | 2084.48 | 33.97 | 9.75 |
| p99 | 1072 | 9014.16 | 123.78 | 14.83 |
| max | 2503 | 15846.81 | 126.28 | 17.94 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-72B-Instruct_suffix_humaneval.json`](../summ_Qwen2.5-72B-Instruct_suffix_humaneval.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-72B-Instruct" and .method=="suffix" and (.condition=="humaneval" or (.condition==null and .corpus=="humaneval")))' ../per_request_raw.jsonl
  ```
