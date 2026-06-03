# DeepSeek-R1 × suffix × mbpp

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 676.9 |
| n_ok/n | 198/198 (err 0) |
| wall_total_s | 690.6 |
| total_completion_tokens | 467530 |
| TTFT p50/p99 ms | 186.8/239.1 |
| TPOT p50/p99 ms | 43.3/82.0 |
| accept α (acc/draft) | 0.3658 (258057.0/705446.0) |
| GPU util / mem MiB | 94.1 / 1273376 |
| CPU util | 4.3 |
| reqtps_avg | 30.4 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=198)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 186.37 | 64.36 | 4.77 |
| p50 | 881 | 38292.59 | 186.84 | 43.26 |
| p99 | 8192 | 440723.01 | 239.06 | 82.01 |
| max | 8192 | 543380.04 | 242.99 | 87.27 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1_suffix_mbpp.json`](../summ_DeepSeek-R1_suffix_mbpp.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1" and .method=="suffix" and (.condition=="mbpp" or (.condition==null and .corpus=="mbpp")))' ../per_request_raw.jsonl
  ```
