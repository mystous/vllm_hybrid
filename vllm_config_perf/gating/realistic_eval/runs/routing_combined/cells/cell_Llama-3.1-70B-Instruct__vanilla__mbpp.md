# Llama-3.1-70B-Instruct × vanilla × mbpp

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 1772.7 |
| n_ok/n | 198/198 (err 0) |
| wall_total_s | 81.2 |
| total_completion_tokens | 143934 |
| TTFT p50/p99 ms | 26.6/73.8 |
| TPOT p50/p99 ms | 8.4/8.6 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 97.1 / 1268912 |
| CPU util | 4.8 |
| reqtps_avg | 118.6 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=198)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 129 | 1113.64 | 25.02 | 7.39 |
| p50 | 430 | 3646.34 | 26.59 | 8.39 |
| p99 | 8192 | 63134.48 | 73.75 | 8.58 |
| max | 8192 | 63453.1 | 75.34 | 8.58 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-70B-Instruct_vanilla_mbpp.json`](../summ_Llama-3.1-70B-Instruct_vanilla_mbpp.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-70B-Instruct" and .method=="vanilla" and (.condition=="mbpp" or (.condition==null and .corpus=="mbpp")))' ../per_request_raw.jsonl
  ```
