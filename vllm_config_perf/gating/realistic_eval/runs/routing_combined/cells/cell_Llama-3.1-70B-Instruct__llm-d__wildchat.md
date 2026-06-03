# Llama-3.1-70B-Instruct × llm-d × wildchat

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 3898.2 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 309.7 |
| total_completion_tokens | 1207329 |
| TTFT p50/p99 ms | 44.1/145.2 |
| TPOT p50/p99 ms | 10.5/20.6 |
| accept α (acc/draft) | 0.7582 (584546.0/770917.0) |
| GPU util / mem MiB | 86.2 / 1231058 |
| CPU util | 5.4 |
| reqtps_avg | 135.2 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 45.77 | 28.1 | 1.43 |
| p50 | 685 | 8857.99 | 44.13 | 10.48 |
| p99 | 8192 | 86583.73 | 145.22 | 20.62 |
| max | 8192 | 140113.05 | 170.84 | 24.22 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-70B-Instruct_llm-d_wildchat.json`](../summ_Llama-3.1-70B-Instruct_llm-d_wildchat.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-70B-Instruct" and .method=="llm-d" and (.condition=="wildchat" or (.condition==null and .corpus=="wildchat")))' ../per_request_raw.jsonl
  ```
