# Qwen2.5-32B-Instruct × llm-d-c8 × sharegpt

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 1772.9 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 640.9 |
| total_completion_tokens | 1136358 |
| TTFT p50/p99 ms | 24.8/237.6 |
| TPOT p50/p99 ms | 6.3/11.3 |
| accept α (acc/draft) | 0.6693 (476448.0/711898.0) |
| GPU util / mem MiB | 89.0 / 1230905 |
| CPU util | 5.3 |
| reqtps_avg | 199.9 |
| concurrency / max_tokens / stream | 8 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 13 | 152.7 | 15.44 | 0.81 |
| p50 | 622 | 4431.16 | 24.8 | 6.28 |
| p99 | 8192 | 53065.43 | 237.6 | 11.34 |
| max | 8192 | 57742.92 | 326.81 | 12.9 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-32B-Instruct_llm-d-c8_sharegpt.json`](../summ_Qwen2.5-32B-Instruct_llm-d-c8_sharegpt.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-32B-Instruct" and .method=="llm-d-c8" and (.condition=="sharegpt" or (.condition==null and .corpus=="sharegpt")))' ../per_request_raw.jsonl
  ```
