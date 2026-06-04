# Qwen2.5-7B-Instruct × llm-d-c64 × wildchat

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 9475.6 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 161.4 |
| total_completion_tokens | 1529151 |
| TTFT p50/p99 ms | 49.7/89.7 |
| TPOT p50/p99 ms | 6.4/23.3 |
| accept α (acc/draft) | 0.7131 (783493.0/1098665.0) |
| GPU util / mem MiB | 56.1 / 614770 |
| CPU util | 3.6 |
| reqtps_avg | 191.0 |
| concurrency / max_tokens / stream | 64 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 6 | 136.0 | 17.51 | 0.9 |
| p50 | 866 | 8212.27 | 49.73 | 6.43 |
| p99 | 8192 | 64797.13 | 89.67 | 23.27 |
| max | 8192 | 82227.3 | 91.54 | 24.85 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-7B-Instruct_llm-d-c64_wildchat.json`](../summ_Qwen2.5-7B-Instruct_llm-d-c64_wildchat.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-7B-Instruct" and .method=="llm-d-c64" and (.condition=="wildchat" or (.condition==null and .corpus=="wildchat")))' ../per_request_raw.jsonl
  ```
