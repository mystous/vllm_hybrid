# DeepSeek-R1-Distill-Qwen-7B × suffix × lmsys

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 11359.9 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 124.8 |
| total_completion_tokens | 1418123 |
| TTFT p50/p99 ms | 18.6/71.5 |
| TPOT p50/p99 ms | 5.4/7.4 |
| accept α (acc/draft) | 0.5756 (1004545.0/1745065.0) |
| GPU util / mem MiB | 67.2 / 632552 |
| CPU util | 2.6 |
| reqtps_avg | 375.1 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 4 | 37.66 | 15.06 | 0.63 |
| p50 | 1251 | 6007.27 | 18.55 | 5.44 |
| p99 | 8192 | 27509.52 | 71.47 | 7.39 |
| max | 8192 | 40011.64 | 74.69 | 7.92 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Qwen-7B_suffix_lmsys.json`](../summ_DeepSeek-R1-Distill-Qwen-7B_suffix_lmsys.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Qwen-7B" and .method=="suffix" and (.condition=="lmsys" or (.condition==null and .corpus=="lmsys")))' ../per_request_raw.jsonl
  ```
