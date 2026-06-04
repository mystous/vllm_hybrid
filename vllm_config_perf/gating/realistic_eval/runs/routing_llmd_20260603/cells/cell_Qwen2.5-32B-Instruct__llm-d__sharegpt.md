# Qwen2.5-32B-Instruct × llm-d × sharegpt

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 5150.2 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 202.1 |
| total_completion_tokens | 1040773 |
| TTFT p50/p99 ms | 33.8/79.1 |
| TPOT p50/p99 ms | 6.8/13.5 |
| accept α (acc/draft) | 0.7336 (509223.0/694171.0) |
| GPU util / mem MiB | 81.0 / 1230928 |
| CPU util | 4.8 |
| reqtps_avg | 237.8 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=1000)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 13 | 74.71 | 19.82 | 1.16 |
| p50 | 607 | 4971.41 | 32.59 | 7.19 |
| p99 | 8192 | 67400.12 | 125.58 | 14.98 |
| max | 8192 | 69759.35 | 140.45 | 17.15 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-32B-Instruct_llm-d_sharegpt.json`](../summ_Qwen2.5-32B-Instruct_llm-d_sharegpt.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-32B-Instruct" and .method=="llm-d" and (.condition=="sharegpt" or (.condition==null and .corpus=="sharegpt")))' ../per_request_raw.jsonl
  ```
