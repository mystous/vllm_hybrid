# DeepSeek-R1-Distill-Qwen-7B × suffix × swebench

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 15421.6 |
| n_ok/n | 297/297 (err 0) |
| wall_total_s | 96.8 |
| total_completion_tokens | 1492267 |
| TTFT p50/p99 ms | 24.1/122.3 |
| TPOT p50/p99 ms | 2.2/7.5 |
| accept α (acc/draft) | 0.6983 (1191701.0/1706664.0) |
| GPU util / mem MiB | 66.3 / 632552 |
| CPU util | 2.6 |
| reqtps_avg | 489.6 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=297)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 118 | 786.01 | 16.9 | 0.69 |
| p50 | 8192 | 9188.76 | 24.09 | 2.17 |
| p99 | 8192 | 27454.32 | 122.28 | 7.48 |
| max | 8192 | 34588.5 | 123.16 | 8.06 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Qwen-7B_suffix_swebench.json`](../summ_DeepSeek-R1-Distill-Qwen-7B_suffix_swebench.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Qwen-7B" and .method=="suffix" and (.condition=="swebench" or (.condition==null and .corpus=="swebench")))' ../per_request_raw.jsonl
  ```
