# DeepSeek-R1 × suffix × swebench

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 537.6 |
| n_ok/n | 297/297 (err 0) |
| wall_total_s | 1713.7 |
| total_completion_tokens | 921313 |
| TTFT p50/p99 ms | 269.6/5413.2 |
| TPOT p50/p99 ms | 66.4/109.1 |
| accept α (acc/draft) | 0.3614 (486196.0/1345229.0) |
| GPU util / mem MiB | 95.2 / 1273376 |
| CPU util | 4.3 |
| reqtps_avg | 20.5 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=297)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 117.77 | 84.12 | 4.03 |
| p50 | 1409 | 83288.16 | 269.65 | 66.4 |
| p99 | 8192 | 670668.63 | 5413.19 | 109.11 |
| max | 8192 | 723177.29 | 5415.65 | 114.68 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1_suffix_swebench.json`](../summ_DeepSeek-R1_suffix_swebench.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1" and .method=="suffix" and (.condition=="swebench" or (.condition==null and .corpus=="swebench")))' ../per_request_raw.jsonl
  ```
