# Llama-3.1-70B-Instruct × vanilla × swebench

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 2878.2 |
| n_ok/n | 297/297 (err 0) |
| wall_total_s | 225.7 |
| total_completion_tokens | 649540 |
| TTFT p50/p99 ms | 39.6/308.6 |
| TPOT p50/p99 ms | 9.3/10.3 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 98.2 / 1268907 |
| CPU util | 4.8 |
| reqtps_avg | 105.5 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=297)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 2 | 47.02 | 26.11 | 8.04 |
| p50 | 323 | 3004.53 | 39.59 | 9.29 |
| p99 | 8192 | 76970.96 | 308.62 | 10.26 |
| max | 8192 | 77192.43 | 311.99 | 11.23 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-70B-Instruct_vanilla_swebench.json`](../summ_Llama-3.1-70B-Instruct_vanilla_swebench.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-70B-Instruct" and .method=="vanilla" and (.condition=="swebench" or (.condition==null and .corpus=="swebench")))' ../per_request_raw.jsonl
  ```
