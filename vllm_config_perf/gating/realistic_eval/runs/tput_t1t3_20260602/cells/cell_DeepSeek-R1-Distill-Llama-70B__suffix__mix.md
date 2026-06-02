# DeepSeek-R1-Distill-Llama-70B × suffix × mix

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 6127.0 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 172.0 |
| total_completion_tokens | 1053570 |
| TTFT p50/p99 ms | 50.1/130.6 |
| TPOT p50/p99 ms | 2.1/16.0 |
| accept α (acc/draft) | 0.7864 (883697.0/1123761.0) |
| GPU util / mem MiB | 85.0 / 1268784 |
| CPU util | 4.4 |
| reqtps_avg | 390.9 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 1 | 47.34 | 39.87 | 1.71 |
| p50 | 1305 | 3288.86 | 50.06 | 2.09 |
| p99 | 8192 | 76908.34 | 130.65 | 15.96 |
| max | 8192 | 95055.93 | 134.51 | 16.78 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Llama-70B_suffix_mix.json`](../summ_DeepSeek-R1-Distill-Llama-70B_suffix_mix.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Llama-70B" and .method=="suffix" and .condition=="mix")' ../per_request_raw.jsonl
  ```
