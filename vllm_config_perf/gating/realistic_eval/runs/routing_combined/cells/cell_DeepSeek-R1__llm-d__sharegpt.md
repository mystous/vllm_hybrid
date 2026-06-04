# DeepSeek-R1 × llm-d × sharegpt

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 996.8 |
| n_ok/n | 491/500 (err 9) |
| wall_total_s | 1183.9 |
| total_completion_tokens | 1180172 |
| TTFT p50/p99 ms | 83.5/26564.5 |
| TPOT p50/p99 ms | 22.9/91.1 |
| accept α (acc/draft) | 0.5278 (305287.0/578372.0) |
| GPU util / mem MiB | 89.9 / 1419607 |
| CPU util | 5.1 |
| reqtps_avg | 38.0 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=491)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 64.3 | 50.11 | 5.12 |
| p50 | 1314 | 39560.74 | 83.49 | 22.85 |
| p99 | 8192 | 258400.33 | 26564.52 | 91.14 |
| max | 8192 | 292039.92 | 26565.91 | 611.4 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1_llm-d_sharegpt.json`](../summ_DeepSeek-R1_llm-d_sharegpt.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1" and .method=="llm-d" and (.condition=="sharegpt" or (.condition==null and .corpus=="sharegpt")))' ../per_request_raw.jsonl
  ```
