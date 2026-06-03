# DeepSeek-R1-Distill-Llama-70B × vanilla × swebench

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 3235.9 |
| n_ok/n | 297/297 (err 0) |
| wall_total_s | 314.6 |
| total_completion_tokens | 1018086 |
| TTFT p50/p99 ms | 35.3/311.9 |
| TPOT p50/p99 ms | 9.1/9.3 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 98.4 / 1268910 |
| CPU util | 4.9 |
| reqtps_avg | 110.0 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=297)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 571 | 5174.5 | 27.17 | 7.93 |
| p50 | 2421 | 22154.73 | 35.3 | 9.14 |
| p99 | 8192 | 75424.01 | 311.94 | 9.3 |
| max | 8192 | 75540.37 | 312.91 | 9.35 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Llama-70B_vanilla_swebench.json`](../summ_DeepSeek-R1-Distill-Llama-70B_vanilla_swebench.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Llama-70B" and .method=="vanilla" and (.condition=="swebench" or (.condition==null and .corpus=="swebench")))' ../per_request_raw.jsonl
  ```
