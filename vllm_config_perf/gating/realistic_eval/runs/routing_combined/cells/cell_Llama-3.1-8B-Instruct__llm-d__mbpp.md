# Llama-3.1-8B-Instruct × llm-d × mbpp

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 13474.1 |
| n_ok/n | 198/198 (err 0) |
| wall_total_s | 111.9 |
| total_completion_tokens | 1507933 |
| TTFT p50/p99 ms | 21.5/44.9 |
| TPOT p50/p99 ms | 1.8/4.8 |
| accept α (acc/draft) | 0.7853 (751384.0/956821.0) |
| GPU util / mem MiB | 80.4 / 1228584 |
| CPU util | 5.3 |
| reqtps_avg | 626.8 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=198)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 332 | 1354.91 | 15.99 | 0.54 |
| p50 | 8192 | 11265.52 | 21.53 | 1.77 |
| p99 | 8192 | 28803.83 | 44.93 | 4.85 |
| max | 8192 | 29593.63 | 48.19 | 5.46 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-8B-Instruct_llm-d_mbpp.json`](../summ_Llama-3.1-8B-Instruct_llm-d_mbpp.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-8B-Instruct" and .method=="llm-d" and (.condition=="mbpp" or (.condition==null and .corpus=="mbpp")))' ../per_request_raw.jsonl
  ```
