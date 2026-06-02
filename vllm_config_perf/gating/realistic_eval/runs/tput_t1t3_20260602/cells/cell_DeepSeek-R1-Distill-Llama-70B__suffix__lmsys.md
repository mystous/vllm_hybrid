# DeepSeek-R1-Distill-Llama-70B × suffix × lmsys

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 2848.5 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 275.5 |
| total_completion_tokens | 784745 |
| TTFT p50/p99 ms | 42.3/169.8 |
| TPOT p50/p99 ms | 13.8/17.8 |
| accept α (acc/draft) | 0.3738 (395860.0/1059101.0) |
| GPU util / mem MiB | 85.9 / 1268784 |
| CPU util | 4.4 |
| reqtps_avg | 96.2 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 38.86 | 28.18 | 1.5 |
| p50 | 939 | 12856.88 | 42.28 | 13.83 |
| p99 | 8192 | 63361.38 | 169.78 | 17.84 |
| max | 8192 | 103506.95 | 180.2 | 23.27 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Llama-70B_suffix_lmsys.json`](../summ_DeepSeek-R1-Distill-Llama-70B_suffix_lmsys.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Llama-70B" and .method=="suffix" and (.condition=="lmsys" or (.condition==null and .corpus=="lmsys")))' ../per_request_raw.jsonl
  ```
