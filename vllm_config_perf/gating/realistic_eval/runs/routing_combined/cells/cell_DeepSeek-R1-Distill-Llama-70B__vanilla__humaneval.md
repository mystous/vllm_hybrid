# DeepSeek-R1-Distill-Llama-70B × vanilla × humaneval

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 2851.5 |
| n_ok/n | 164/164 (err 0) |
| wall_total_s | 168.4 |
| total_completion_tokens | 480067 |
| TTFT p50/p99 ms | 35.2/124.1 |
| TPOT p50/p99 ms | 8.8/9.2 |
| accept α (acc/draft) | None (vanilla) |
| GPU util / mem MiB | 98.0 / 1268912 |
| CPU util | 4.8 |
| reqtps_avg | 112.7 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=164)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 7 | 172.37 | 26.96 | 7.94 |
| p50 | 1893 | 16930.45 | 35.15 | 8.77 |
| p99 | 8192 | 73628.04 | 124.13 | 9.24 |
| max | 8192 | 73934.88 | 124.53 | 9.29 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Llama-70B_vanilla_humaneval.json`](../summ_DeepSeek-R1-Distill-Llama-70B_vanilla_humaneval.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Llama-70B" and .method=="vanilla" and (.condition=="humaneval" or (.condition==null and .corpus=="humaneval")))' ../per_request_raw.jsonl
  ```
