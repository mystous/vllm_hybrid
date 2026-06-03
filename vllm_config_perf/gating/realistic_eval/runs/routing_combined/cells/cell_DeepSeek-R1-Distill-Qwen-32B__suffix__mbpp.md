# DeepSeek-R1-Distill-Qwen-32B × suffix × mbpp

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 5690.0 |
| n_ok/n | 198/198 (err 0) |
| wall_total_s | 156.5 |
| total_completion_tokens | 890522 |
| TTFT p50/p99 ms | 34.7/77.4 |
| TPOT p50/p99 ms | 8.0/10.6 |
| accept α (acc/draft) | 0.4874 (619195.0/1270450.0) |
| GPU util / mem MiB | 81.0 / 1267776 |
| CPU util | 4.4 |
| reqtps_avg | 202.6 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=198)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 478 | 3775.64 | 25.41 | 1.17 |
| p50 | 2964 | 19766.22 | 34.69 | 7.97 |
| p99 | 8192 | 60613.19 | 77.4 | 10.64 |
| max | 8192 | 65900.63 | 77.7 | 11.58 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Qwen-32B_suffix_mbpp.json`](../summ_DeepSeek-R1-Distill-Qwen-32B_suffix_mbpp.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Qwen-32B" and .method=="suffix" and (.condition=="mbpp" or (.condition==null and .corpus=="mbpp")))' ../per_request_raw.jsonl
  ```
