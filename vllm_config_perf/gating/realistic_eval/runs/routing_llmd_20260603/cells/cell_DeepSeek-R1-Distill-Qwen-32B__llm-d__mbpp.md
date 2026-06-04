# DeepSeek-R1-Distill-Qwen-32B × llm-d × mbpp

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 4947.1 |
| n_ok/n | 198/198 (err 0) |
| wall_total_s | 175.1 |
| total_completion_tokens | 866181 |
| TTFT p50/p99 ms | 28.5/51.9 |
| TPOT p50/p99 ms | 6.5/9.8 |
| accept α (acc/draft) | 0.4923 (361956.0/735309.0) |
| GPU util / mem MiB | 88.4 / 1230904 |
| CPU util | 5.8 |
| reqtps_avg | 200.4 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=198)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 480 | 3030.16 | 22.18 | 1.12 |
| p50 | 3178 | 17559.45 | 28.49 | 6.48 |
| p99 | 8192 | 54045.5 | 51.9 | 9.79 |
| max | 8192 | 58931.47 | 52.45 | 10.36 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Qwen-32B_llm-d_mbpp.json`](../summ_DeepSeek-R1-Distill-Qwen-32B_llm-d_mbpp.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Qwen-32B" and .method=="llm-d" and (.condition=="mbpp" or (.condition==null and .corpus=="mbpp")))' ../per_request_raw.jsonl
  ```
