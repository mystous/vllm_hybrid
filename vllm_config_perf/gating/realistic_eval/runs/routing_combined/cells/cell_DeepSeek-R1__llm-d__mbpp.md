# DeepSeek-R1 × llm-d × mbpp

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 917.8 |
| n_ok/n | 195/198 (err 3) |
| wall_total_s | 480.6 |
| total_completion_tokens | 441071 |
| TTFT p50/p99 ms | 76.0/195.4 |
| TPOT p50/p99 ms | 22.0/63.9 |
| accept α (acc/draft) | 0.3617 (100527.0/277961.0) |
| GPU util / mem MiB | 87.8 / 1419616 |
| CPU util | 4.9 |
| reqtps_avg | 41.5 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=195)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 137.99 | 51.17 | 4.88 |
| p50 | 841 | 23647.04 | 75.96 | 22.04 |
| p99 | 8192 | 273253.04 | 195.39 | 63.94 |
| max | 8192 | 297944.62 | 206.2 | 67.1 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1_llm-d_mbpp.json`](../summ_DeepSeek-R1_llm-d_mbpp.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1" and .method=="llm-d" and (.condition=="mbpp" or (.condition==null and .corpus=="mbpp")))' ../per_request_raw.jsonl
  ```
