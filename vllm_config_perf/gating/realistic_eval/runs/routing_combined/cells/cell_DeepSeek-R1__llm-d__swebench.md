# DeepSeek-R1 × llm-d × swebench

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 877.4 |
| n_ok/n | 278/297 (err 19) |
| wall_total_s | 826.4 |
| total_completion_tokens | 725131 |
| TTFT p50/p99 ms | 127.3/6948.4 |
| TPOT p50/p99 ms | 21.6/86.4 |
| accept α (acc/draft) | 0.372 (147911.0/397591.0) |
| GPU util / mem MiB | 94.4 / 1419614 |
| CPU util | 5.2 |
| reqtps_avg | 34.7 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=278)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 69.12 | 65.42 | 4.19 |
| p50 | 1157 | 29218.76 | 127.29 | 21.58 |
| p99 | 8192 | 269199.36 | 6948.37 | 86.4 |
| max | 8192 | 293631.39 | 6949.98 | 282.98 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1_llm-d_swebench.json`](../summ_DeepSeek-R1_llm-d_swebench.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1" and .method=="llm-d" and (.condition=="swebench" or (.condition==null and .corpus=="swebench")))' ../per_request_raw.jsonl
  ```
