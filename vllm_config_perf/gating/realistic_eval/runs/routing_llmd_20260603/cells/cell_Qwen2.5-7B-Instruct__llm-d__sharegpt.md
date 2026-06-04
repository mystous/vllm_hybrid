# Qwen2.5-7B-Instruct × llm-d × sharegpt

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 7505.2 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 216.7 |
| total_completion_tokens | 1626673 |
| TTFT p50/p99 ms | 24.7/351.7 |
| TPOT p50/p99 ms | 5.2/13.4 |
| accept α (acc/draft) | 0.7214 (729238.0/1010933.0) |
| GPU util / mem MiB | 63.7 / 614761 |
| CPU util | 3.4 |
| reqtps_avg | 251.3 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 14 | 80.65 | 17.38 | 0.87 |
| p50 | 810 | 5434.51 | 24.65 | 5.17 |
| p99 | 8192 | 49799.15 | 351.69 | 13.45 |
| max | 8192 | 75510.22 | 353.9 | 15.67 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-7B-Instruct_llm-d_sharegpt.json`](../summ_Qwen2.5-7B-Instruct_llm-d_sharegpt.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-7B-Instruct" and .method=="llm-d" and (.condition=="sharegpt" or (.condition==null and .corpus=="sharegpt")))' ../per_request_raw.jsonl
  ```
