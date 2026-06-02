# Qwen2.5-7B-Instruct × vanilla × mbpp

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 3813.9 |
| n_ok/n | 198/198 (err 0) |
| wall_total_s | 62.2 |
| total_completion_tokens | 237109 |
| TTFT p50/p99 ms | 19.7/46.9 |
| TPOT p50/p99 ms | 4.2/5.5 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 84.2 / 632552 |
| CPU util | 2.6 |
| reqtps_avg | 230.4 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=198)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 42 | 211.14 | 14.72 | 3.5 |
| p50 | 571 | 2442.81 | 19.69 | 4.25 |
| p99 | 8192 | 45070.95 | 46.91 | 5.5 |
| max | 8192 | 45158.65 | 48.84 | 5.51 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-7B-Instruct_vanilla_mbpp.json`](../summ_Qwen2.5-7B-Instruct_vanilla_mbpp.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-7B-Instruct" and .method=="vanilla" and (.condition=="mbpp" or (.condition==null and .corpus=="mbpp")))' ../per_request_raw.jsonl
  ```
