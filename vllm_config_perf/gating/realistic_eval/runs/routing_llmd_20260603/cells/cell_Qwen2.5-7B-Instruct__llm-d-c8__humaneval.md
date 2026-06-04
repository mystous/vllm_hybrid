# Qwen2.5-7B-Instruct × llm-d-c8 × humaneval

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 2737.1 |
| n_ok/n | 164/164 (err 0) |
| wall_total_s | 91.5 |
| total_completion_tokens | 250582 |
| TTFT p50/p99 ms | 17.9/30.3 |
| TPOT p50/p99 ms | 3.4/5.7 |
| accept α (acc/draft) | 0.5512 (110900.0/201184.0) |
| GPU util / mem MiB | 67.8 / 614764 |
| CPU util | 3.1 |
| reqtps_avg | 325.9 |
| concurrency / max_tokens / stream | 8 / 8192 / True |

## per-request 분포 (raw 기반, n=164)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 7 | 42.4 | 14.57 | 0.93 |
| p50 | 277 | 959.64 | 17.91 | 3.44 |
| p99 | 8192 | 32937.15 | 30.27 | 5.67 |
| max | 8192 | 33258.51 | 31.8 | 6.49 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-7B-Instruct_llm-d-c8_humaneval.json`](../summ_Qwen2.5-7B-Instruct_llm-d-c8_humaneval.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-7B-Instruct" and .method=="llm-d-c8" and (.condition=="humaneval" or (.condition==null and .corpus=="humaneval")))' ../per_request_raw.jsonl
  ```
