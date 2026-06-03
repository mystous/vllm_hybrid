# Llama-3.1-8B-Instruct × vanilla × mbpp

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 8730.3 |
| n_ok/n | 198/198 (err 0) |
| wall_total_s | 172.7 |
| total_completion_tokens | 1508031 |
| TTFT p50/p99 ms | 32.1/47.5 |
| TPOT p50/p99 ms | 3.5/3.6 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 94.2 / 1265360 |
| CPU util | 4.6 |
| reqtps_avg | 286.3 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=198)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 337 | 1190.9 | 16.22 | 3.43 |
| p50 | 8192 | 28322.65 | 32.1 | 3.45 |
| p99 | 8192 | 29122.88 | 47.49 | 3.57 |
| max | 8192 | 29123.49 | 49.77 | 3.6 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-8B-Instruct_vanilla_mbpp.json`](../summ_Llama-3.1-8B-Instruct_vanilla_mbpp.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-8B-Instruct" and .method=="vanilla" and (.condition=="mbpp" or (.condition==null and .corpus=="mbpp")))' ../per_request_raw.jsonl
  ```
