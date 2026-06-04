# Qwen2.5-7B-Instruct × llm-d-c64 × mbpp

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 5858.6 |
| n_ok/n | 198/198 (err 0) |
| wall_total_s | 38.1 |
| total_completion_tokens | 223148 |
| TTFT p50/p99 ms | 28.8/94.7 |
| TPOT p50/p99 ms | 3.9/7.0 |
| accept α (acc/draft) | 0.5755 (110767.0/192465.0) |
| GPU util / mem MiB | 59.5 / 614770 |
| CPU util | 3.0 |
| reqtps_avg | 306.6 |
| concurrency / max_tokens / stream | 64 / 8192 / True |

## per-request 분포 (raw 기반, n=198)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 42 | 92.76 | 16.57 | 0.93 |
| p50 | 562 | 2386.93 | 28.78 | 3.93 |
| p99 | 8192 | 32207.65 | 94.67 | 7.04 |
| max | 8192 | 32346.41 | 98.67 | 7.76 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-7B-Instruct_llm-d-c64_mbpp.json`](../summ_Qwen2.5-7B-Instruct_llm-d-c64_mbpp.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-7B-Instruct" and .method=="llm-d-c64" and (.condition=="mbpp" or (.condition==null and .corpus=="mbpp")))' ../per_request_raw.jsonl
  ```
