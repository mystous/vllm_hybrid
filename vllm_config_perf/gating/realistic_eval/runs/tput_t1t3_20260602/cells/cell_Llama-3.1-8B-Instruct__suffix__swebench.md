# Llama-3.1-8B-Instruct × suffix × swebench

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 21352.7 |
| n_ok/n | 297/297 (err 0) |
| wall_total_s | 90.8 |
| total_completion_tokens | 1938384 |
| TTFT p50/p99 ms | 29.8/118.9 |
| TPOT p50/p99 ms | 1.2/8.2 |
| accept α (acc/draft) | 0.8891 (1706556.0/1919489.0) |
| GPU util / mem MiB | 66.7 / 1265361 |
| CPU util | 4.5 |
| reqtps_avg | 706.1 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=297)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 42 | 307.07 | 19.03 | 0.76 |
| p50 | 8192 | 8838.21 | 29.84 | 1.24 |
| p99 | 8192 | 23743.27 | 118.88 | 8.17 |
| max | 8192 | 64242.75 | 122.97 | 10.76 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-8B-Instruct_suffix_swebench.json`](../summ_Llama-3.1-8B-Instruct_suffix_swebench.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-8B-Instruct" and .method=="suffix" and (.condition=="swebench" or (.condition==null and .corpus=="swebench")))' ../per_request_raw.jsonl
  ```
