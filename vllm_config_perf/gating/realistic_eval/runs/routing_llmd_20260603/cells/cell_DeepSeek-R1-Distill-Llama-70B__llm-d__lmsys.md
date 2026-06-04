# DeepSeek-R1-Distill-Llama-70B × llm-d × lmsys

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 2651.0 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 307.3 |
| total_completion_tokens | 814699 |
| TTFT p50/p99 ms | 40.2/142.0 |
| TPOT p50/p99 ms | 10.5/17.5 |
| accept α (acc/draft) | 0.4086 (244445.0/598273.0) |
| GPU util / mem MiB | 83.9 / 1231062 |
| CPU util | 5.3 |
| reqtps_avg | 101.3 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 36.11 | 26.67 | 1.55 |
| p50 | 904 | 11172.72 | 40.16 | 10.51 |
| p99 | 8192 | 82639.79 | 142.03 | 17.49 |
| max | 8192 | 100860.5 | 143.65 | 31.4 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Llama-70B_llm-d_lmsys.json`](../summ_DeepSeek-R1-Distill-Llama-70B_llm-d_lmsys.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Llama-70B" and .method=="llm-d" and (.condition=="lmsys" or (.condition==null and .corpus=="lmsys")))' ../per_request_raw.jsonl
  ```
