# Llama-3.1-70B-Instruct × vanilla × wildchat

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 3172.2 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 384.7 |
| total_completion_tokens | 1220462 |
| TTFT p50/p99 ms | 29.0/127.6 |
| TPOT p50/p99 ms | 9.2/9.5 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 98.5 / 1268912 |
| CPU util | 4.8 |
| reqtps_avg | 108.1 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 29.28 | 25.28 | 8.16 |
| p50 | 665 | 6144.2 | 29.04 | 9.22 |
| p99 | 8192 | 76435.13 | 127.56 | 9.53 |
| max | 8192 | 76500.87 | 129.93 | 10.44 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-70B-Instruct_vanilla_wildchat.json`](../summ_Llama-3.1-70B-Instruct_vanilla_wildchat.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-70B-Instruct" and .method=="vanilla" and (.condition=="wildchat" or (.condition==null and .corpus=="wildchat")))' ../per_request_raw.jsonl
  ```
