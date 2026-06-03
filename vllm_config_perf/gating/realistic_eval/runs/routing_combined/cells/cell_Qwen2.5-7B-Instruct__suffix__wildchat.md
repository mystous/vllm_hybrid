# Qwen2.5-7B-Instruct × suffix × wildchat

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 6284.9 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 236.4 |
| total_completion_tokens | 1485684 |
| TTFT p50/p99 ms | 39.2/62.9 |
| TPOT p50/p99 ms | 10.8/19.9 |
| accept α (acc/draft) | 0.6864 (1121472.0/1633750.0) |
| GPU util / mem MiB | 36.0 / 632560 |
| CPU util | 2.5 |
| reqtps_avg | 174.1 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 6 | 108.1 | 17.92 | 0.83 |
| p50 | 860 | 10182.83 | 39.17 | 10.85 |
| p99 | 8192 | 65394.16 | 62.88 | 19.87 |
| max | 8192 | 80445.96 | 73.25 | 22.3 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-7B-Instruct_suffix_wildchat.json`](../summ_Qwen2.5-7B-Instruct_suffix_wildchat.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-7B-Instruct" and .method=="suffix" and (.condition=="wildchat" or (.condition==null and .corpus=="wildchat")))' ../per_request_raw.jsonl
  ```
