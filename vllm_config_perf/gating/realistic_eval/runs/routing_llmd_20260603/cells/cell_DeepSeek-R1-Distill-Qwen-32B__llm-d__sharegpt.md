# DeepSeek-R1-Distill-Qwen-32B × llm-d × sharegpt

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 4824.6 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 232.9 |
| total_completion_tokens | 1123698 |
| TTFT p50/p99 ms | 28.9/324.9 |
| TPOT p50/p99 ms | 6.7/12.9 |
| accept α (acc/draft) | 0.5687 (456536.0/802711.0) |
| GPU util / mem MiB | 88.8 / 1230900 |
| CPU util | 5.7 |
| reqtps_avg | 175.0 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 9 | 131.66 | 19.88 | 1.03 |
| p50 | 1200 | 9578.67 | 28.88 | 6.7 |
| p99 | 8192 | 52702.87 | 324.93 | 12.92 |
| max | 8192 | 53081.0 | 328.21 | 13.7 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Qwen-32B_llm-d_sharegpt.json`](../summ_DeepSeek-R1-Distill-Qwen-32B_llm-d_sharegpt.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Qwen-32B" and .method=="llm-d" and (.condition=="sharegpt" or (.condition==null and .corpus=="sharegpt")))' ../per_request_raw.jsonl
  ```
