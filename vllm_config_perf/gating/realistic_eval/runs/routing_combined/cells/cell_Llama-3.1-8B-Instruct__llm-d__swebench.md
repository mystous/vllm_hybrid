# Llama-3.1-8B-Instruct × llm-d × swebench

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 14526.5 |
| n_ok/n | 297/297 (err 0) |
| wall_total_s | 133.4 |
| total_completion_tokens | 1937384 |
| TTFT p50/p99 ms | 25.5/89.5 |
| TPOT p50/p99 ms | 3.5/6.0 |
| accept α (acc/draft) | 0.8881 (1016337.0/1144426.0) |
| GPU util / mem MiB | 75.3 / 1228576 |
| CPU util | 5.0 |
| reqtps_avg | 697.8 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=297)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 42 | 198.45 | 16.23 | 0.52 |
| p50 | 8192 | 7612.01 | 25.53 | 3.48 |
| p99 | 8192 | 29094.27 | 89.47 | 5.99 |
| max | 8192 | 29107.24 | 90.02 | 7.0 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-8B-Instruct_llm-d_swebench.json`](../summ_Llama-3.1-8B-Instruct_llm-d_swebench.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-8B-Instruct" and .method=="llm-d" and (.condition=="swebench" or (.condition==null and .corpus=="swebench")))' ../per_request_raw.jsonl
  ```
