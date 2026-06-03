# Llama-3.1-70B-Instruct × suffix × wildchat

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 5261.4 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 233.7 |
| total_completion_tokens | 1229562 |
| TTFT p50/p99 ms | 46.8/134.5 |
| TPOT p50/p99 ms | 14.9/21.2 |
| accept α (acc/draft) | 0.7529 (948810.0/1260267.0) |
| GPU util / mem MiB | 86.1 / 1268784 |
| CPU util | 4.4 |
| reqtps_avg | 154.2 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 59.82 | 26.25 | 1.47 |
| p50 | 676 | 10846.68 | 46.8 | 14.92 |
| p99 | 8192 | 58395.26 | 134.46 | 21.22 |
| max | 8192 | 128677.34 | 135.47 | 25.83 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-70B-Instruct_suffix_wildchat.json`](../summ_Llama-3.1-70B-Instruct_suffix_wildchat.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-70B-Instruct" and .method=="suffix" and (.condition=="wildchat" or (.condition==null and .corpus=="wildchat")))' ../per_request_raw.jsonl
  ```
