# Qwen2.5-7B-Instruct × suffix × humaneval

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 5213.4 |
| n_ok/n | 164/164 (err 0) |
| wall_total_s | 48.6 |
| total_completion_tokens | 253263 |
| TTFT p50/p99 ms | 37.3/61.5 |
| TPOT p50/p99 ms | 7.0/16.5 |
| accept α (acc/draft) | 0.6207 (195310.0/314657.0) |
| GPU util / mem MiB | 35.6 / 632560 |
| CPU util | 2.5 |
| reqtps_avg | 175.0 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=164)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 14 | 93.04 | 15.16 | 1.12 |
| p50 | 276 | 2835.02 | 37.35 | 7.01 |
| p99 | 8192 | 38004.94 | 61.48 | 16.53 |
| max | 8192 | 44907.92 | 63.17 | 21.95 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-7B-Instruct_suffix_humaneval.json`](../summ_Qwen2.5-7B-Instruct_suffix_humaneval.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-7B-Instruct" and .method=="suffix" and (.condition=="humaneval" or (.condition==null and .corpus=="humaneval")))' ../per_request_raw.jsonl
  ```
