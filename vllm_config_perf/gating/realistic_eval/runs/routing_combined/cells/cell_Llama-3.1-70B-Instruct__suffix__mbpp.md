# Llama-3.1-70B-Instruct × suffix × mbpp

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 3265.6 |
| n_ok/n | 198/198 (err 0) |
| wall_total_s | 46.7 |
| total_completion_tokens | 152593 |
| TTFT p50/p99 ms | 38.2/87.4 |
| TPOT p50/p99 ms | 9.7/13.4 |
| accept α (acc/draft) | 0.4429 (94156.0/212605.0) |
| GPU util / mem MiB | 84.1 / 1268784 |
| CPU util | 4.3 |
| reqtps_avg | 113.9 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=198)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 197 | 1543.9 | 27.75 | 1.92 |
| p50 | 432 | 4265.28 | 38.16 | 9.74 |
| p99 | 8192 | 28774.41 | 87.37 | 13.39 |
| max | 8192 | 46722.73 | 88.69 | 14.05 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-70B-Instruct_suffix_mbpp.json`](../summ_Llama-3.1-70B-Instruct_suffix_mbpp.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-70B-Instruct" and .method=="suffix" and (.condition=="mbpp" or (.condition==null and .corpus=="mbpp")))' ../per_request_raw.jsonl
  ```
