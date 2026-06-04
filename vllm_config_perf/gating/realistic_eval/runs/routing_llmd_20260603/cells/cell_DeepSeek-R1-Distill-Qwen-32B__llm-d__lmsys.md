# DeepSeek-R1-Distill-Qwen-32B × llm-d × lmsys

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 5042.1 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 217.7 |
| total_completion_tokens | 1097583 |
| TTFT p50/p99 ms | 30.4/87.8 |
| TPOT p50/p99 ms | 6.5/13.0 |
| accept α (acc/draft) | 0.5443 (470185.0/863761.0) |
| GPU util / mem MiB | 88.2 / 1230912 |
| CPU util | 5.7 |
| reqtps_avg | 181.7 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 5 | 69.73 | 19.16 | 1.03 |
| p50 | 934 | 7267.55 | 30.42 | 6.55 |
| p99 | 8192 | 52820.25 | 87.76 | 13.02 |
| max | 8192 | 77922.92 | 92.67 | 13.94 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Qwen-32B_llm-d_lmsys.json`](../summ_DeepSeek-R1-Distill-Qwen-32B_llm-d_lmsys.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Qwen-32B" and .method=="llm-d" and (.condition=="lmsys" or (.condition==null and .corpus=="lmsys")))' ../per_request_raw.jsonl
  ```
