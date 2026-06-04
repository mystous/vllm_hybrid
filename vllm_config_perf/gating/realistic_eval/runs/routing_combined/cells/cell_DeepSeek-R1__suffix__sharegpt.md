# DeepSeek-R1 × suffix × sharegpt

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 797.0 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 1566.1 |
| total_completion_tokens | 1248234 |
| TTFT p50/p99 ms | 173.2/13558.4 |
| TPOT p50/p99 ms | 60.7/87.4 |
| accept α (acc/draft) | 0.5041 (698371.0/1385509.0) |
| GPU util / mem MiB | 92.8 / 1273360 |
| CPU util | 4.3 |
| reqtps_avg | 29.0 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 131.42 | 75.53 | 5.26 |
| p50 | 1303 | 68709.76 | 173.22 | 60.74 |
| p99 | 8192 | 437974.48 | 13558.43 | 87.41 |
| max | 8192 | 595244.01 | 13560.04 | 1283.4 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1_suffix_sharegpt.json`](../summ_DeepSeek-R1_suffix_sharegpt.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1" and .method=="suffix" and (.condition=="sharegpt" or (.condition==null and .corpus=="sharegpt")))' ../per_request_raw.jsonl
  ```
