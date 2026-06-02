# DeepSeek-R1-Distill-Llama-70B × suffix × mbpp

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 2426.0 |
| n_ok/n | 198/198 (err 0) |
| wall_total_s | 210.4 |
| total_completion_tokens | 510551 |
| TTFT p50/p99 ms | 43.7/85.7 |
| TPOT p50/p99 ms | 11.7/13.8 |
| accept α (acc/draft) | 0.2652 (243466.0/918113.0) |
| GPU util / mem MiB | 86.1 / 1268784 |
| CPU util | 4.3 |
| reqtps_avg | 95.1 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=198)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 538 | 6717.59 | 32.86 | 1.52 |
| p50 | 1817 | 21108.99 | 43.71 | 11.69 |
| p99 | 8192 | 86539.62 | 85.65 | 13.85 |
| max | 8192 | 97542.07 | 86.19 | 14.37 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Llama-70B_suffix_mbpp.json`](../summ_DeepSeek-R1-Distill-Llama-70B_suffix_mbpp.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Llama-70B" and .method=="suffix" and (.condition=="mbpp" or (.condition==null and .corpus=="mbpp")))' ../per_request_raw.jsonl
  ```
