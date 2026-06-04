# Llama-3.1-405B-Instruct-FP8 × suffix × mbpp

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 1725.4 |
| n_ok/n | 198/198 (err 0) |
| wall_total_s | 98.5 |
| total_completion_tokens | 169998 |
| TTFT p50/p99 ms | 87.8/134.0 |
| TPOT p50/p99 ms | 23.1/31.9 |
| accept α (acc/draft) | 0.4888 (111351.0/227817.0) |
| GPU util / mem MiB | 91.8 / 1272464 |
| CPU util | 4.3 |
| reqtps_avg | 52.0 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=198)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 64 | 2094.51 | 53.0 | 3.04 |
| p50 | 429 | 9987.33 | 87.8 | 23.11 |
| p99 | 8192 | 53939.7 | 134.01 | 31.95 |
| max | 8192 | 67238.57 | 137.0 | 32.27 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-405B-Instruct-FP8_suffix_mbpp.json`](../summ_Llama-3.1-405B-Instruct-FP8_suffix_mbpp.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-405B-Instruct-FP8" and .method=="suffix" and (.condition=="mbpp" or (.condition==null and .corpus=="mbpp")))' ../per_request_raw.jsonl
  ```
