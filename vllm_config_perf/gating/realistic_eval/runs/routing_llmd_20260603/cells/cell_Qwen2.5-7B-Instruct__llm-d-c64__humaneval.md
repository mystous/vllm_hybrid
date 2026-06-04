# Qwen2.5-7B-Instruct × llm-d-c64 × humaneval

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 7047.1 |
| n_ok/n | 164/164 (err 0) |
| wall_total_s | 43.8 |
| total_completion_tokens | 308461 |
| TTFT p50/p99 ms | 36.1/88.6 |
| TPOT p50/p99 ms | 2.9/6.3 |
| accept α (acc/draft) | 0.7572 (147935.0/195371.0) |
| GPU util / mem MiB | 49.8 / 614770 |
| CPU util | 3.0 |
| reqtps_avg | 383.1 |
| concurrency / max_tokens / stream | 64 / 8192 / True |

## per-request 분포 (raw 기반, n=164)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 14 | 96.12 | 19.04 | 0.72 |
| p50 | 287 | 876.94 | 36.12 | 2.9 |
| p99 | 8192 | 43298.16 | 88.63 | 6.28 |
| max | 8192 | 43299.75 | 89.47 | 8.71 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-7B-Instruct_llm-d-c64_humaneval.json`](../summ_Qwen2.5-7B-Instruct_llm-d-c64_humaneval.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-7B-Instruct" and .method=="llm-d-c64" and (.condition=="humaneval" or (.condition==null and .corpus=="humaneval")))' ../per_request_raw.jsonl
  ```
