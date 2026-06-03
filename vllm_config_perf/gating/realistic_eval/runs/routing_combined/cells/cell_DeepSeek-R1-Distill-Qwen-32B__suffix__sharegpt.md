# DeepSeek-R1-Distill-Qwen-32B × suffix × sharegpt

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 4996.5 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 230.3 |
| total_completion_tokens | 1150611 |
| TTFT p50/p99 ms | 31.9/496.7 |
| TPOT p50/p99 ms | 10.6/14.3 |
| accept α (acc/draft) | 0.5511 (723420.0/1312650.0) |
| GPU util / mem MiB | 82.4 / 1267769 |
| CPU util | 4.4 |
| reqtps_avg | 169.3 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 9 | 131.46 | 21.56 | 1.05 |
| p50 | 1201 | 11589.98 | 31.91 | 10.61 |
| p99 | 8192 | 47652.94 | 496.68 | 14.29 |
| max | 8192 | 72977.73 | 500.85 | 15.85 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Qwen-32B_suffix_sharegpt.json`](../summ_DeepSeek-R1-Distill-Qwen-32B_suffix_sharegpt.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Qwen-32B" and .method=="suffix" and (.condition=="sharegpt" or (.condition==null and .corpus=="sharegpt")))' ../per_request_raw.jsonl
  ```
