# Llama-3.1-8B-Instruct × vanilla × swebench

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 8347.9 |
| n_ok/n | 297/297 (err 0) |
| wall_total_s | 229.3 |
| total_completion_tokens | 1914354 |
| TTFT p50/p99 ms | 49.7/102.2 |
| TPOT p50/p99 ms | 3.5/3.8 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 94.6 / 1265352 |
| CPU util | 4.6 |
| reqtps_avg | 282.3 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=297)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 42 | 215.51 | 16.58 | 3.39 |
| p50 | 8192 | 28536.16 | 49.69 | 3.49 |
| p99 | 8192 | 28721.84 | 102.23 | 3.82 |
| max | 8192 | 28739.95 | 105.58 | 3.98 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-8B-Instruct_vanilla_swebench.json`](../summ_Llama-3.1-8B-Instruct_vanilla_swebench.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-8B-Instruct" and .method=="vanilla" and (.condition=="swebench" or (.condition==null and .corpus=="swebench")))' ../per_request_raw.jsonl
  ```
