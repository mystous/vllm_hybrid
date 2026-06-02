# DeepSeek-R1-Distill-Qwen-32B × vanilla × swebench

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 4408.9 |
| n_ok/n | 297/297 (err 0) |
| wall_total_s | 150.0 |
| total_completion_tokens | 661394 |
| TTFT p50/p99 ms | 27.1/196.3 |
| TPOT p50/p99 ms | 6.0/6.2 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 97.3 / 1267771 |
| CPU util | 4.7 |
| reqtps_avg | 165.9 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=297)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 72 | 483.04 | 21.16 | 5.7 |
| p50 | 592 | 3543.0 | 27.06 | 5.97 |
| p99 | 8192 | 49703.35 | 196.25 | 6.2 |
| max | 8192 | 49767.86 | 203.39 | 6.32 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Qwen-32B_vanilla_swebench.json`](../summ_DeepSeek-R1-Distill-Qwen-32B_vanilla_swebench.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Qwen-32B" and .method=="vanilla" and (.condition=="swebench" or (.condition==null and .corpus=="swebench")))' ../per_request_raw.jsonl
  ```
