# Qwen2.5-7B-Instruct × llm-d-c64 × sharegpt

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 9728.9 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 158.1 |
| total_completion_tokens | 1537890 |
| TTFT p50/p99 ms | 42.9/117.0 |
| TPOT p50/p99 ms | 6.4/21.2 |
| accept α (acc/draft) | 0.7116 (767945.0/1079194.0) |
| GPU util / mem MiB | 56.2 / 614769 |
| CPU util | 3.6 |
| reqtps_avg | 192.5 |
| concurrency / max_tokens / stream | 64 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 14 | 142.05 | 18.57 | 0.85 |
| p50 | 777 | 6568.56 | 42.92 | 6.36 |
| p99 | 8192 | 88008.31 | 116.97 | 21.24 |
| max | 8192 | 110424.18 | 144.21 | 26.36 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-7B-Instruct_llm-d-c64_sharegpt.json`](../summ_Qwen2.5-7B-Instruct_llm-d-c64_sharegpt.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-7B-Instruct" and .method=="llm-d-c64" and (.condition=="sharegpt" or (.condition==null and .corpus=="sharegpt")))' ../per_request_raw.jsonl
  ```
