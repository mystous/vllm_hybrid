# Qwen2.5-32B-Instruct × suffix × sharegpt

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 4662.4 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 227.1 |
| total_completion_tokens | 1058657 |
| TTFT p50/p99 ms | 41.1/465.7 |
| TPOT p50/p99 ms | 13.9/23.3 |
| accept α (acc/draft) | 0.654 (743768.0/1137337.0) |
| GPU util / mem MiB | 79.8 / 1267775 |
| CPU util | 4.3 |
| reqtps_avg | 121.4 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 13 | 175.73 | 19.43 | 1.59 |
| p50 | 597 | 8776.73 | 41.08 | 13.9 |
| p99 | 8192 | 82495.39 | 465.69 | 23.33 |
| max | 8192 | 114798.57 | 656.24 | 25.63 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-32B-Instruct_suffix_sharegpt.json`](../summ_Qwen2.5-32B-Instruct_suffix_sharegpt.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-32B-Instruct" and .method=="suffix" and (.condition=="sharegpt" or (.condition==null and .corpus=="sharegpt")))' ../per_request_raw.jsonl
  ```
