# DeepSeek-R1-Distill-Qwen-7B × suffix × mbpp

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 12397.8 |
| n_ok/n | 198/198 (err 0) |
| wall_total_s | 73.4 |
| total_completion_tokens | 910449 |
| TTFT p50/p99 ms | 18.4/49.9 |
| TPOT p50/p99 ms | 4.4/6.3 |
| accept α (acc/draft) | 0.5799 (681822.0/1175770.0) |
| GPU util / mem MiB | 66.3 / 632552 |
| CPU util | 2.6 |
| reqtps_avg | 561.0 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=198)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 199 | 1045.33 | 15.56 | 0.63 |
| p50 | 3311 | 7488.09 | 18.41 | 4.44 |
| p99 | 8192 | 35216.58 | 49.92 | 6.26 |
| max | 8192 | 41910.61 | 52.97 | 6.38 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Qwen-7B_suffix_mbpp.json`](../summ_DeepSeek-R1-Distill-Qwen-7B_suffix_mbpp.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Qwen-7B" and .method=="suffix" and (.condition=="mbpp" or (.condition==null and .corpus=="mbpp")))' ../per_request_raw.jsonl
  ```
