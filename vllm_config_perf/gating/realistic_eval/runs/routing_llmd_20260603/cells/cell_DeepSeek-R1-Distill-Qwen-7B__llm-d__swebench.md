# DeepSeek-R1-Distill-Qwen-7B × llm-d × swebench

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 12072.0 |
| n_ok/n | 297/297 (err 0) |
| wall_total_s | 121.9 |
| total_completion_tokens | 1471845 |
| TTFT p50/p99 ms | 25.7/109.4 |
| TPOT p50/p99 ms | 3.2/6.3 |
| accept α (acc/draft) | 0.7082 (732123.0/1033731.0) |
| GPU util / mem MiB | 78.3 / 614764 |
| CPU util | 3.7 |
| reqtps_avg | 471.5 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=297)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 116 | 475.39 | 15.4 | 0.61 |
| p50 | 8192 | 8515.24 | 25.69 | 3.21 |
| p99 | 8192 | 26496.54 | 109.44 | 6.34 |
| max | 8192 | 26512.9 | 112.59 | 6.67 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Qwen-7B_llm-d_swebench.json`](../summ_DeepSeek-R1-Distill-Qwen-7B_llm-d_swebench.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Qwen-7B" and .method=="llm-d" and (.condition=="swebench" or (.condition==null and .corpus=="swebench")))' ../per_request_raw.jsonl
  ```
