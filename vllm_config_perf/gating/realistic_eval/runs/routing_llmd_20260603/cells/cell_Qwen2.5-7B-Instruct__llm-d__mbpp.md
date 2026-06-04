# Qwen2.5-7B-Instruct × llm-d × mbpp

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 4440.1 |
| n_ok/n | 198/198 (err 0) |
| wall_total_s | 48.8 |
| total_completion_tokens | 216464 |
| TTFT p50/p99 ms | 20.7/45.6 |
| TPOT p50/p99 ms | 4.0/7.0 |
| accept α (acc/draft) | 0.4518 (69011.0/152746.0) |
| GPU util / mem MiB | 57.3 / 614764 |
| CPU util | 2.8 |
| reqtps_avg | 265.9 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=198)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 42 | 189.58 | 13.94 | 0.97 |
| p50 | 575 | 2328.15 | 20.67 | 4.03 |
| p99 | 8192 | 34545.75 | 45.58 | 6.97 |
| max | 8192 | 34591.54 | 48.71 | 7.09 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-7B-Instruct_llm-d_mbpp.json`](../summ_Qwen2.5-7B-Instruct_llm-d_mbpp.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-7B-Instruct" and .method=="llm-d" and (.condition=="mbpp" or (.condition==null and .corpus=="mbpp")))' ../per_request_raw.jsonl
  ```
