# Llama-3.1-8B-Instruct × suffix × sharegpt

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 19053.6 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 199.8 |
| total_completion_tokens | 3806276 |
| TTFT p50/p99 ms | 21.2/166.3 |
| TPOT p50/p99 ms | 1.4/8.5 |
| accept α (acc/draft) | 0.8509 (3234788.0/3801505.0) |
| GPU util / mem MiB | 68.5 / 1265358 |
| CPU util | 4.4 |
| reqtps_avg | 719.5 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 138 | 1168.82 | 17.84 | 0.7 |
| p50 | 8192 | 10727.42 | 21.17 | 1.37 |
| p99 | 8192 | 43070.96 | 166.29 | 8.49 |
| max | 8192 | 68395.18 | 167.33 | 8.92 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-8B-Instruct_suffix_sharegpt.json`](../summ_Llama-3.1-8B-Instruct_suffix_sharegpt.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-8B-Instruct" and .method=="suffix" and (.condition=="sharegpt" or (.condition==null and .corpus=="sharegpt")))' ../per_request_raw.jsonl
  ```
