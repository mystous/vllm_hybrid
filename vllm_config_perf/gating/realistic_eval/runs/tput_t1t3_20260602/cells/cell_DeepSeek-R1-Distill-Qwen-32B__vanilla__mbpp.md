# DeepSeek-R1-Distill-Qwen-32B × vanilla × mbpp

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 4689.6 |
| n_ok/n | 198/198 (err 0) |
| wall_total_s | 181.2 |
| total_completion_tokens | 849679 |
| TTFT p50/p99 ms | 23.4/66.1 |
| TPOT p50/p99 ms | 5.8/5.9 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 97.5 / 1267776 |
| CPU util | 4.8 |
| reqtps_avg | 171.9 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=198)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 480 | 2755.02 | 21.89 | 5.64 |
| p50 | 2592 | 14916.56 | 23.37 | 5.84 |
| p99 | 8192 | 48119.84 | 66.1 | 5.94 |
| max | 8192 | 48155.22 | 68.9 | 5.99 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Qwen-32B_vanilla_mbpp.json`](../summ_DeepSeek-R1-Distill-Qwen-32B_vanilla_mbpp.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Qwen-32B" and .method=="vanilla" and (.condition=="mbpp" or (.condition==null and .corpus=="mbpp")))' ../per_request_raw.jsonl
  ```
