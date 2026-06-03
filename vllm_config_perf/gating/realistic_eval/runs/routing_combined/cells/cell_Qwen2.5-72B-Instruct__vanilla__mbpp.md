# Qwen2.5-72B-Instruct × vanilla × mbpp

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 3395.0 |
| n_ok/n | 198/198 (err 0) |
| wall_total_s | 33.1 |
| total_completion_tokens | 112402 |
| TTFT p50/p99 ms | 26.8/79.0 |
| TPOT p50/p99 ms | 8.6/8.6 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 94.9 / 1269104 |
| CPU util | 4.8 |
| reqtps_avg | 116.4 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=198)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 32 | 303.15 | 25.09 | 7.83 |
| p50 | 543 | 4655.44 | 26.8 | 8.59 |
| p99 | 1107 | 9002.0 | 79.03 | 8.62 |
| max | 1551 | 13343.51 | 81.62 | 8.65 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-72B-Instruct_vanilla_mbpp.json`](../summ_Qwen2.5-72B-Instruct_vanilla_mbpp.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-72B-Instruct" and .method=="vanilla" and (.condition=="mbpp" or (.condition==null and .corpus=="mbpp")))' ../per_request_raw.jsonl
  ```
