# DeepSeek-R1-Distill-Qwen-7B × llm-d × lmsys

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 14768.3 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 98.3 |
| total_completion_tokens | 1451768 |
| TTFT p50/p99 ms | 22.0/53.4 |
| TPOT p50/p99 ms | 2.9/5.6 |
| accept α (acc/draft) | 0.7763 (838343.0/1079972.0) |
| GPU util / mem MiB | 80.8 / 614764 |
| CPU util | 3.9 |
| reqtps_avg | 681.0 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=1000)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 15 | 60.77 | 14.34 | 0.53 |
| p50 | 1212 | 4520.68 | 22.27 | 3.21 |
| p99 | 8192 | 26689.97 | 54.56 | 6.42 |
| max | 8192 | 30303.46 | 58.59 | 7.02 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Qwen-7B_llm-d_lmsys.json`](../summ_DeepSeek-R1-Distill-Qwen-7B_llm-d_lmsys.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Qwen-7B" and .method=="llm-d" and (.condition=="lmsys" or (.condition==null and .corpus=="lmsys")))' ../per_request_raw.jsonl
  ```
