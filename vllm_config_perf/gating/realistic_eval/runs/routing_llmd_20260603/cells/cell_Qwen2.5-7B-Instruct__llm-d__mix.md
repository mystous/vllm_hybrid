# Qwen2.5-7B-Instruct × llm-d × mix

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 7739.2 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 177.7 |
| total_completion_tokens | 1375085 |
| TTFT p50/p99 ms | 25.5/52.4 |
| TPOT p50/p99 ms | 4.6/11.6 |
| accept α (acc/draft) | 0.7499 (670797.0/894458.0) |
| GPU util / mem MiB | 61.0 / 614772 |
| CPU util | 3.2 |
| reqtps_avg | 327.1 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 2 | 55.71 | 15.19 | 0.66 |
| p50 | 735 | 3627.78 | 25.54 | 4.58 |
| p99 | 8192 | 47218.22 | 52.41 | 11.56 |
| max | 8192 | 54510.91 | 53.94 | 18.94 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-7B-Instruct_llm-d_mix.json`](../summ_Qwen2.5-7B-Instruct_llm-d_mix.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-7B-Instruct" and .method=="llm-d" and .condition=="mix")' ../per_request_raw.jsonl
  ```
