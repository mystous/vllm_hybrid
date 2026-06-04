# Llama-3.1-70B-Instruct × llm-d × lmsys

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 3897.4 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 214.7 |
| total_completion_tokens | 836719 |
| TTFT p50/p99 ms | 44.2/167.1 |
| TPOT p50/p99 ms | 11.1/20.6 |
| accept α (acc/draft) | 0.6924 (405910.0/586209.0) |
| GPU util / mem MiB | 92.4 / 1231063 |
| CPU util | 5.7 |
| reqtps_avg | 114.2 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 37.26 | 27.9 | 1.51 |
| p50 | 425 | 5650.0 | 44.25 | 11.08 |
| p99 | 8192 | 84563.29 | 167.06 | 20.63 |
| max | 8192 | 84716.96 | 169.0 | 22.75 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-70B-Instruct_llm-d_lmsys.json`](../summ_Llama-3.1-70B-Instruct_llm-d_lmsys.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-70B-Instruct" and .method=="llm-d" and (.condition=="lmsys" or (.condition==null and .corpus=="lmsys")))' ../per_request_raw.jsonl
  ```
