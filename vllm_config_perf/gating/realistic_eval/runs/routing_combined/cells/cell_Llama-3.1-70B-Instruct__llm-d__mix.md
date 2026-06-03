# Llama-3.1-70B-Instruct × llm-d × mix

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 4003.6 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 253.6 |
| total_completion_tokens | 1015158 |
| TTFT p50/p99 ms | 43.8/148.4 |
| TPOT p50/p99 ms | 9.5/16.7 |
| accept α (acc/draft) | 0.8247 (526141.0/638014.0) |
| GPU util / mem MiB | 83.7 / 1231064 |
| CPU util | 5.3 |
| reqtps_avg | 201.5 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 39.68 | 28.72 | 1.45 |
| p50 | 495 | 4051.16 | 43.84 | 9.51 |
| p99 | 8192 | 90487.85 | 148.37 | 16.68 |
| max | 8192 | 97033.84 | 167.53 | 18.37 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-70B-Instruct_llm-d_mix.json`](../summ_Llama-3.1-70B-Instruct_llm-d_mix.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-70B-Instruct" and .method=="llm-d" and .condition=="mix")' ../per_request_raw.jsonl
  ```
