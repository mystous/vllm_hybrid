# Qwen2.5-7B-Instruct × ngram × humaneval

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 370.0 |
| n_ok/n | 164/164 (err 0) |
| wall_total_s | 266.2 |
| total_completion_tokens | 98500 |
| TTFT p50/p99 ms | 237.4/316.6 |
| TPOT p50/p99 ms | 101.5/137.3 |
| accept α (acc/draft) | 0.5143 (42571.0/82780.0) |
| GPU util / mem MiB | 58.3 / 632560 |
| CPU util | 88.9 |
| reqtps_avg | 16.7 |
| concurrency / max_tokens / stream | 32 / 2048 / True |

## per-request 분포 (raw 기반, n=164)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 7 | 80.74 | 16.0 | 3.81 |
| p50 | 278 | 31281.37 | 237.43 | 101.46 |
| p99 | 2048 | 169218.16 | 316.61 | 137.3 |
| max | 2048 | 210443.99 | 364.31 | 141.65 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-7B-Instruct_ngram_humaneval.json`](../summ_Qwen2.5-7B-Instruct_ngram_humaneval.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-7B-Instruct" and .method=="ngram" and (.condition=="humaneval" or (.condition==null and .corpus=="humaneval")))' ../per_request_raw.jsonl
  ```
