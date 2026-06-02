# Llama-3.1-70B-Instruct × suffix × lmsys

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 3958.5 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 203.3 |
| total_completion_tokens | 804834 |
| TTFT p50/p99 ms | 46.1/162.8 |
| TPOT p50/p99 ms | 14.5/21.6 |
| accept α (acc/draft) | 0.6533 (593722.0/908814.0) |
| GPU util / mem MiB | 86.2 / 1268784 |
| CPU util | 4.3 |
| reqtps_avg | 117.0 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 38.68 | 28.51 | 1.43 |
| p50 | 427 | 6538.39 | 46.07 | 14.51 |
| p99 | 8192 | 58608.58 | 162.83 | 21.55 |
| max | 8192 | 81614.16 | 167.1 | 24.28 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-70B-Instruct_suffix_lmsys.json`](../summ_Llama-3.1-70B-Instruct_suffix_lmsys.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-70B-Instruct" and .method=="suffix" and (.condition=="lmsys" or (.condition==null and .corpus=="lmsys")))' ../per_request_raw.jsonl
  ```
