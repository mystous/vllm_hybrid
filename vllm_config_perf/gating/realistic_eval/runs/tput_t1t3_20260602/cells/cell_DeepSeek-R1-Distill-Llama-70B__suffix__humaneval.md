# DeepSeek-R1-Distill-Llama-70B × suffix × humaneval

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 2787.7 |
| n_ok/n | 164/164 (err 0) |
| wall_total_s | 165.1 |
| total_completion_tokens | 460143 |
| TTFT p50/p99 ms | 46.7/131.8 |
| TPOT p50/p99 ms | 11.8/14.4 |
| accept α (acc/draft) | 0.381 (267285.0/701546.0) |
| GPU util / mem MiB | 86.4 / 1268784 |
| CPU util | 4.4 |
| reqtps_avg | 113.2 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=164)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 7 | 197.96 | 28.61 | 1.75 |
| p50 | 1789 | 21518.12 | 46.7 | 11.84 |
| p99 | 8192 | 92151.11 | 131.83 | 14.38 |
| max | 8192 | 101439.05 | 135.06 | 14.91 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Llama-70B_suffix_humaneval.json`](../summ_DeepSeek-R1-Distill-Llama-70B_suffix_humaneval.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Llama-70B" and .method=="suffix" and (.condition=="humaneval" or (.condition==null and .corpus=="humaneval")))' ../per_request_raw.jsonl
  ```
