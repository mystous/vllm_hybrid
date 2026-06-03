# Qwen2.5-72B-Instruct × vanilla × wildchat

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 2802.6 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 243.5 |
| total_completion_tokens | 682532 |
| TTFT p50/p99 ms | 30.8/125.6 |
| TPOT p50/p99 ms | 9.3/10.0 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 97.9 / 1269104 |
| CPU util | 4.6 |
| reqtps_avg | 106.8 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 29.66 | 25.89 | 8.47 |
| p50 | 760 | 7165.64 | 30.79 | 9.33 |
| p99 | 8192 | 78577.79 | 125.64 | 10.04 |
| max | 8192 | 78866.64 | 131.03 | 10.58 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-72B-Instruct_vanilla_wildchat.json`](../summ_Qwen2.5-72B-Instruct_vanilla_wildchat.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-72B-Instruct" and .method=="vanilla" and (.condition=="wildchat" or (.condition==null and .corpus=="wildchat")))' ../per_request_raw.jsonl
  ```
