# DeepSeek-R1-Distill-Qwen-32B × suffix × lmsys

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 5355.6 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 206.1 |
| total_completion_tokens | 1103670 |
| TTFT p50/p99 ms | 33.2/111.8 |
| TPOT p50/p99 ms | 10.1/14.3 |
| accept α (acc/draft) | 0.5366 (744106.0/1386638.0) |
| GPU util / mem MiB | 82.1 / 1267776 |
| CPU util | 4.4 |
| reqtps_avg | 182.4 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 5 | 78.22 | 20.19 | 1.09 |
| p50 | 984 | 10065.64 | 33.18 | 10.11 |
| p99 | 8192 | 54446.24 | 111.8 | 14.3 |
| max | 8192 | 82825.88 | 114.18 | 16.02 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Qwen-32B_suffix_lmsys.json`](../summ_DeepSeek-R1-Distill-Qwen-32B_suffix_lmsys.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Qwen-32B" and .method=="suffix" and (.condition=="lmsys" or (.condition==null and .corpus=="lmsys")))' ../per_request_raw.jsonl
  ```
