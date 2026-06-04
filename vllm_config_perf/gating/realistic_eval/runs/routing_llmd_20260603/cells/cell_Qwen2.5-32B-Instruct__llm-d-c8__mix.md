# Qwen2.5-32B-Instruct × llm-d-c8 × mix

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 1471.6 |
| n_ok/n | 226/500 (err 274) |
| wall_total_s | 329.5 |
| total_completion_tokens | 484859 |
| TTFT p50/p99 ms | 26.2/49.0 |
| TPOT p50/p99 ms | 6.0/10.6 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 84.8 / 1225695 |
| CPU util | 4.4 |
| reqtps_avg | 330.6 |
| concurrency / max_tokens / stream | 8 / 8192 / True |

## per-request 분포 (raw 기반, n=226)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 28 | 66.81 | 15.23 | 0.69 |
| p50 | 568 | 3261.26 | 26.23 | 6.05 |
| p99 | 8192 | 53954.03 | 49.0 | 10.55 |
| max | 8192 | 54189.87 | 55.16 | 11.02 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-32B-Instruct_llm-d-c8_mix.json`](../summ_Qwen2.5-32B-Instruct_llm-d-c8_mix.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-32B-Instruct" and .method=="llm-d-c8" and .condition=="mix")' ../per_request_raw.jsonl
  ```
