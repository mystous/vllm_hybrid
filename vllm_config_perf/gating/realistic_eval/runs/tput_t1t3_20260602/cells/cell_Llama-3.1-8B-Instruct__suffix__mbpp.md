# Llama-3.1-8B-Instruct × suffix × mbpp

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 17824.9 |
| n_ok/n | 198/198 (err 0) |
| wall_total_s | 84.1 |
| total_completion_tokens | 1499618 |
| TTFT p50/p99 ms | 22.1/58.4 |
| TPOT p50/p99 ms | 1.4/6.6 |
| accept α (acc/draft) | 0.7898 (1281016.0/1621875.0) |
| GPU util / mem MiB | 66.7 / 1265376 |
| CPU util | 4.4 |
| reqtps_avg | 672.3 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=198)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 281 | 1959.28 | 17.35 | 0.79 |
| p50 | 8192 | 11253.05 | 22.07 | 1.43 |
| p99 | 8192 | 27462.57 | 58.42 | 6.61 |
| max | 8192 | 38754.56 | 60.39 | 7.11 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-8B-Instruct_suffix_mbpp.json`](../summ_Llama-3.1-8B-Instruct_suffix_mbpp.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-8B-Instruct" and .method=="suffix" and (.condition=="mbpp" or (.condition==null and .corpus=="mbpp")))' ../per_request_raw.jsonl
  ```
