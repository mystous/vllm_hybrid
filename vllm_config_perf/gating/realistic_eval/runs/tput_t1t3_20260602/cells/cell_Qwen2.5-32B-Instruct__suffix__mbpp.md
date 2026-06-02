# Qwen2.5-32B-Instruct × suffix × mbpp

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 5138.2 |
| n_ok/n | 198/198 (err 0) |
| wall_total_s | 89.7 |
| total_completion_tokens | 460816 |
| TTFT p50/p99 ms | 47.4/69.3 |
| TPOT p50/p99 ms | 11.4/18.2 |
| accept α (acc/draft) | 0.6436 (353525.0/549304.0) |
| GPU util / mem MiB | 66.1 / 1267776 |
| CPU util | 4.3 |
| reqtps_avg | 137.5 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=198)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 70 | 1202.97 | 20.95 | 2.0 |
| p50 | 549 | 7063.84 | 47.44 | 11.43 |
| p99 | 8192 | 34911.16 | 69.29 | 18.23 |
| max | 8192 | 43237.54 | 71.35 | 19.2 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-32B-Instruct_suffix_mbpp.json`](../summ_Qwen2.5-32B-Instruct_suffix_mbpp.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-32B-Instruct" and .method=="suffix" and (.condition=="mbpp" or (.condition==null and .corpus=="mbpp")))' ../per_request_raw.jsonl
  ```
