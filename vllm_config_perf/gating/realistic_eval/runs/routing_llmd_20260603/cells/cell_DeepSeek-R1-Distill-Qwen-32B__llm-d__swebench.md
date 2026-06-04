# DeepSeek-R1-Distill-Qwen-32B × llm-d × swebench

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 4954.6 |
| n_ok/n | 297/297 (err 0) |
| wall_total_s | 148.5 |
| total_completion_tokens | 735821 |
| TTFT p50/p99 ms | 35.8/163.4 |
| TPOT p50/p99 ms | 6.8/12.0 |
| accept α (acc/draft) | 0.6411 (308980.0/481950.0) |
| GPU util / mem MiB | 81.8 / 1230904 |
| CPU util | 5.3 |
| reqtps_avg | 179.3 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=297)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 72 | 578.32 | 21.56 | 1.23 |
| p50 | 613 | 5019.86 | 35.85 | 6.84 |
| p99 | 8192 | 56247.49 | 163.4 | 12.03 |
| max | 8192 | 59545.09 | 165.55 | 12.65 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Qwen-32B_llm-d_swebench.json`](../summ_DeepSeek-R1-Distill-Qwen-32B_llm-d_swebench.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Qwen-32B" and .method=="llm-d" and (.condition=="swebench" or (.condition==null and .corpus=="swebench")))' ../per_request_raw.jsonl
  ```
