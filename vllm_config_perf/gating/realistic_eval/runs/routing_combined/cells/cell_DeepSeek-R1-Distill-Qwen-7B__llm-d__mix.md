# DeepSeek-R1-Distill-Qwen-7B × llm-d × mix

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 13000.3 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 143.2 |
| total_completion_tokens | 1861070 |
| TTFT p50/p99 ms | 21.6/56.2 |
| TPOT p50/p99 ms | 3.2/5.9 |
| accept α (acc/draft) | 0.711 (946932.0/1331864.0) |
| GPU util / mem MiB | 81.0 / 614764 |
| CPU util | 3.9 |
| reqtps_avg | 617.5 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 8 | 42.63 | 14.46 | 0.51 |
| p50 | 1704 | 5279.7 | 21.59 | 3.2 |
| p99 | 8192 | 26614.44 | 56.18 | 5.9 |
| max | 8192 | 26641.05 | 60.18 | 6.43 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Qwen-7B_llm-d_mix.json`](../summ_DeepSeek-R1-Distill-Qwen-7B_llm-d_mix.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Qwen-7B" and .method=="llm-d" and .condition=="mix")' ../per_request_raw.jsonl
  ```
