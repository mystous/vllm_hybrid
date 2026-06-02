# DeepSeek-R1-Distill-Llama-70B × vanilla × mbpp

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 2777.1 |
| n_ok/n | 198/198 (err 0) |
| wall_total_s | 190.0 |
| total_completion_tokens | 527539 |
| TTFT p50/p99 ms | 27.9/77.4 |
| TPOT p50/p99 ms | 8.8/9.0 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 98.1 / 1268912 |
| CPU util | 4.9 |
| reqtps_avg | 114.6 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=198)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 632 | 5386.08 | 25.73 | 7.48 |
| p50 | 1849 | 16248.68 | 27.86 | 8.8 |
| p99 | 8192 | 72790.59 | 77.42 | 9.02 |
| max | 8192 | 73194.84 | 77.6 | 9.03 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Llama-70B_vanilla_mbpp.json`](../summ_DeepSeek-R1-Distill-Llama-70B_vanilla_mbpp.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Llama-70B" and .method=="vanilla" and (.condition=="mbpp" or (.condition==null and .corpus=="mbpp")))' ../per_request_raw.jsonl
  ```
