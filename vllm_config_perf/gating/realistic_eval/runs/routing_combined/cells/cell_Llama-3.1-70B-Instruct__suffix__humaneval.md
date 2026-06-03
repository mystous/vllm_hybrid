# Llama-3.1-70B-Instruct × suffix × humaneval

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 4728.0 |
| n_ok/n | 164/164 (err 0) |
| wall_total_s | 168.6 |
| total_completion_tokens | 797275 |
| TTFT p50/p99 ms | 54.4/126.3 |
| TPOT p50/p99 ms | 7.0/19.8 |
| accept α (acc/draft) | 0.6938 (637656.0/919125.0) |
| GPU util / mem MiB | 86.6 / 1268784 |
| CPU util | 4.3 |
| reqtps_avg | 198.5 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=164)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 24 | 305.38 | 28.14 | 1.79 |
| p50 | 8192 | 18357.69 | 54.35 | 7.01 |
| p99 | 8192 | 115613.61 | 126.31 | 19.78 |
| max | 8192 | 127775.88 | 126.93 | 20.74 |

## raw / 원시 데이터
- 집계 원본: [`summ_Llama-3.1-70B-Instruct_suffix_humaneval.json`](../summ_Llama-3.1-70B-Instruct_suffix_humaneval.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Llama-3.1-70B-Instruct" and .method=="suffix" and (.condition=="humaneval" or (.condition==null and .corpus=="humaneval")))' ../per_request_raw.jsonl
  ```
