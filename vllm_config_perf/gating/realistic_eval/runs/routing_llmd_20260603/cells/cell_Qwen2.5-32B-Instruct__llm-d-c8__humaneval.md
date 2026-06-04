# Qwen2.5-32B-Instruct × llm-d-c8 × humaneval

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 1853.0 |
| n_ok/n | 164/164 (err 0) |
| wall_total_s | 190.3 |
| total_completion_tokens | 352536 |
| TTFT p50/p99 ms | 25.2/48.8 |
| TPOT p50/p99 ms | 5.9/8.9 |
| accept α (acc/draft) | 0.6664 (152050.0/228154.0) |
| GPU util / mem MiB | 86.1 / 1230906 |
| CPU util | 5.3 |
| reqtps_avg | 230.2 |
| concurrency / max_tokens / stream | 8 / 8192 / True |

## per-request 분포 (raw 기반, n=164)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 14 | 146.25 | 16.83 | 0.88 |
| p50 | 320 | 1942.0 | 25.19 | 5.94 |
| p99 | 8192 | 53420.91 | 48.83 | 8.9 |
| max | 8192 | 53745.28 | 51.91 | 9.52 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-32B-Instruct_llm-d-c8_humaneval.json`](../summ_Qwen2.5-32B-Instruct_llm-d-c8_humaneval.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-32B-Instruct" and .method=="llm-d-c8" and (.condition=="humaneval" or (.condition==null and .corpus=="humaneval")))' ../per_request_raw.jsonl
  ```
