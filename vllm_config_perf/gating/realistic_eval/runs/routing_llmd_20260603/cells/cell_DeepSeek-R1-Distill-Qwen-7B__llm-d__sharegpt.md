# DeepSeek-R1-Distill-Qwen-7B × llm-d × sharegpt

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 11191.9 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 149.8 |
| total_completion_tokens | 1676058 |
| TTFT p50/p99 ms | 21.5/287.9 |
| TPOT p50/p99 ms | 3.2/6.7 |
| accept α (acc/draft) | 0.6352 (760534.0/1197315.0) |
| GPU util / mem MiB | 80.8 / 614754 |
| CPU util | 3.8 |
| reqtps_avg | 404.6 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 38.4 | 13.82 | 0.57 |
| p50 | 1505 | 5998.32 | 21.52 | 3.23 |
| p99 | 8192 | 26518.49 | 287.93 | 6.65 |
| max | 8192 | 31355.51 | 289.6 | 7.2 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Qwen-7B_llm-d_sharegpt.json`](../summ_DeepSeek-R1-Distill-Qwen-7B_llm-d_sharegpt.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Qwen-7B" and .method=="llm-d" and (.condition=="sharegpt" or (.condition==null and .corpus=="sharegpt")))' ../per_request_raw.jsonl
  ```
