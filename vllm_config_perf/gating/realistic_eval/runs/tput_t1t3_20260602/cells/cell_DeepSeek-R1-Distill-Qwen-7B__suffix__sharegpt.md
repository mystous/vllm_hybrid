# DeepSeek-R1-Distill-Qwen-7B × suffix × sharegpt

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 11960.6 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 136.4 |
| total_completion_tokens | 1631359 |
| TTFT p50/p99 ms | 18.5/534.2 |
| TPOT p50/p99 ms | 5.6/7.5 |
| accept α (acc/draft) | 0.6213 (1151089.0/1852744.0) |
| GPU util / mem MiB | 67.8 / 632551 |
| CPU util | 2.6 |
| reqtps_avg | 394.2 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 44.93 | 14.7 | 0.59 |
| p50 | 1426 | 7199.65 | 18.51 | 5.56 |
| p99 | 8192 | 25088.1 | 534.23 | 7.48 |
| max | 8192 | 58000.17 | 534.85 | 7.81 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Qwen-7B_suffix_sharegpt.json`](../summ_DeepSeek-R1-Distill-Qwen-7B_suffix_sharegpt.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Qwen-7B" and .method=="suffix" and (.condition=="sharegpt" or (.condition==null and .corpus=="sharegpt")))' ../per_request_raw.jsonl
  ```
