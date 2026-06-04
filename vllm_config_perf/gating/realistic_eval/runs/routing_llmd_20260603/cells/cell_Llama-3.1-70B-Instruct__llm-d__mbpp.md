# Llama-3.1-70B-Instruct × llm-d × mbpp

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 1405.0 |
| n_ok/n | 198/198 (err 0) |
| wall_total_s | 96.9 |
| total_completion_tokens | 136164 |
| TTFT p50/p99 ms | 37.7/67.1 |
| TPOT p50/p99 ms | 9.7/10.1 |
| accept α (acc/draft) | 0.6197 (43316.0/69896.0) |
| GPU util / mem MiB | 66.8 / 1231064 |
| CPU util | 4.4 |
| reqtps_avg | 191.2 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=396)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 183 | 430.24 | 22.52 | 1.34 |
| p50 | 427 | 3917.07 | 36.37 | 9.69 |
| p99 | 8192 | 78195.73 | 74.65 | 11.33 |
| max | 8192 | 78534.0 | 76.07 | 11.9 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-70B-Instruct_llm-d_mbpp.json`](../summ_Llama-3.1-70B-Instruct_llm-d_mbpp.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-70B-Instruct" and .method=="llm-d" and (.condition=="mbpp" or (.condition==null and .corpus=="mbpp")))' ../per_request_raw.jsonl
  ```
