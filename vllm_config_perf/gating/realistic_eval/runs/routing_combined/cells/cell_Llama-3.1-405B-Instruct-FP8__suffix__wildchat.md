# Llama-3.1-405B-Instruct-FP8 × suffix × wildchat

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 2290.3 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 538.4 |
| total_completion_tokens | 1233162 |
| TTFT p50/p99 ms | 104.3/344.5 |
| TPOT p50/p99 ms | 32.5/47.6 |
| accept α (acc/draft) | 0.6872 (924838.0/1345904.0) |
| GPU util / mem MiB | 93.2 / 1272464 |
| CPU util | 4.3 |
| reqtps_avg | 64.2 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 91.03 | 59.23 | 3.19 |
| p50 | 670 | 23767.38 | 104.28 | 32.52 |
| p99 | 8192 | 182035.7 | 344.49 | 47.63 |
| max | 8192 | 276803.25 | 412.45 | 52.4 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-405B-Instruct-FP8_suffix_wildchat.json`](../summ_Llama-3.1-405B-Instruct-FP8_suffix_wildchat.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-405B-Instruct-FP8" and .method=="suffix" and (.condition=="wildchat" or (.condition==null and .corpus=="wildchat")))' ../per_request_raw.jsonl
  ```
