# DeepSeek-R1-Distill-Qwen-7B × vanilla × mbpp

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 8440.1 |
| n_ok/n | 198/198 (err 0) |
| wall_total_s | 112.9 |
| total_completion_tokens | 952734 |
| TTFT p50/p99 ms | 16.8/43.5 |
| TPOT p50/p99 ms | 3.3/3.3 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 91.0 / 632552 |
| CPU util | 2.8 |
| reqtps_avg | 304.3 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=198)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 199 | 664.93 | 15.08 | 3.23 |
| p50 | 4623 | 15165.13 | 16.76 | 3.28 |
| p99 | 8192 | 26931.98 | 43.52 | 3.3 |
| max | 8192 | 26932.04 | 43.76 | 3.31 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Qwen-7B_vanilla_mbpp.json`](../summ_DeepSeek-R1-Distill-Qwen-7B_vanilla_mbpp.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Qwen-7B" and .method=="vanilla" and (.condition=="mbpp" or (.condition==null and .corpus=="mbpp")))' ../per_request_raw.jsonl
  ```
