# DeepSeek-R1-Distill-Qwen-7B × vanilla × swebench

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 8835.3 |
| n_ok/n | 297/297 (err 0) |
| wall_total_s | 164.9 |
| total_completion_tokens | 1456816 |
| TTFT p50/p99 ms | 21.6/99.7 |
| TPOT p50/p99 ms | 3.3/3.4 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 91.7 / 632549 |
| CPU util | 2.8 |
| reqtps_avg | 299.9 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=297)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 122 | 427.82 | 15.79 | 3.24 |
| p50 | 8192 | 26915.35 | 21.6 | 3.3 |
| p99 | 8192 | 27252.04 | 99.73 | 3.4 |
| max | 8192 | 27252.51 | 105.73 | 3.42 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Qwen-7B_vanilla_swebench.json`](../summ_DeepSeek-R1-Distill-Qwen-7B_vanilla_swebench.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Qwen-7B" and .method=="vanilla" and (.condition=="swebench" or (.condition==null and .corpus=="swebench")))' ../per_request_raw.jsonl
  ```
