# DeepSeek-R1-Distill-Qwen-32B × suffix × swebench

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 5240.8 |
| n_ok/n | 297/297 (err 0) |
| wall_total_s | 125.1 |
| total_completion_tokens | 655558 |
| TTFT p50/p99 ms | 41.8/189.2 |
| TPOT p50/p99 ms | 10.1/13.6 |
| accept α (acc/draft) | 0.5878 (464792.0/790677.0) |
| GPU util / mem MiB | 81.7 / 1267776 |
| CPU util | 4.4 |
| reqtps_avg | 165.7 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=297)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 52 | 789.78 | 23.81 | 1.15 |
| p50 | 588 | 6166.85 | 41.78 | 10.12 |
| p99 | 8192 | 44580.67 | 189.18 | 13.59 |
| max | 8192 | 69592.48 | 214.78 | 14.58 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Qwen-32B_suffix_swebench.json`](../summ_DeepSeek-R1-Distill-Qwen-32B_suffix_swebench.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Qwen-32B" and .method=="suffix" and (.condition=="swebench" or (.condition==null and .corpus=="swebench")))' ../per_request_raw.jsonl
  ```
