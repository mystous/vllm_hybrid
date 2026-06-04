# Qwen2.5-7B-Instruct × llm-d-c8 × sharegpt

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 3040.0 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 538.4 |
| total_completion_tokens | 1636603 |
| TTFT p50/p99 ms | 18.4/237.9 |
| TPOT p50/p99 ms | 3.7/6.8 |
| accept α (acc/draft) | 0.7193 (744643.0/1035161.0) |
| GPU util / mem MiB | 73.8 / 614763 |
| CPU util | 3.3 |
| reqtps_avg | 389.9 |
| concurrency / max_tokens / stream | 8 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 14 | 109.27 | 11.05 | 0.54 |
| p50 | 794 | 3360.79 | 18.37 | 3.71 |
| p99 | 8192 | 34986.84 | 237.87 | 6.78 |
| max | 8192 | 41658.12 | 309.1 | 8.92 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-7B-Instruct_llm-d-c8_sharegpt.json`](../summ_Qwen2.5-7B-Instruct_llm-d-c8_sharegpt.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-7B-Instruct" and .method=="llm-d-c8" and (.condition=="sharegpt" or (.condition==null and .corpus=="sharegpt")))' ../per_request_raw.jsonl
  ```
