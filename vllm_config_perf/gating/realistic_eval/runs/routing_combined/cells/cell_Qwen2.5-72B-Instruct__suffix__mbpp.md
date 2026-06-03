# Qwen2.5-72B-Instruct × suffix × mbpp

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 3233.7 |
| n_ok/n | 198/198 (err 0) |
| wall_total_s | 34.6 |
| total_completion_tokens | 111816 |
| TTFT p50/p99 ms | 31.5/75.3 |
| TPOT p50/p99 ms | 9.2/11.0 |
| accept α (acc/draft) | 0.2657 (47425.0/178509.0) |
| GPU util / mem MiB | 80.2 / 1269134 |
| CPU util | 4.4 |
| reqtps_avg | 110.9 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=198)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 32 | 253.53 | 28.94 | 5.02 |
| p50 | 556 | 5066.95 | 31.45 | 9.21 |
| p99 | 1113 | 9694.56 | 75.26 | 11.02 |
| max | 1411 | 10445.44 | 78.37 | 11.16 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-72B-Instruct_suffix_mbpp.json`](../summ_Qwen2.5-72B-Instruct_suffix_mbpp.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-72B-Instruct" and .method=="suffix" and (.condition=="mbpp" or (.condition==null and .corpus=="mbpp")))' ../per_request_raw.jsonl
  ```
