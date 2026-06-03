# DeepSeek-R1-Distill-Qwen-7B × llm-d × wildchat

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 11028.7 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 158.0 |
| total_completion_tokens | 1742001 |
| TTFT p50/p99 ms | 22.2/60.4 |
| TPOT p50/p99 ms | 3.3/6.8 |
| accept α (acc/draft) | 0.6267 (801251.0/1278583.0) |
| GPU util / mem MiB | 80.1 / 614764 |
| CPU util | 3.8 |
| reqtps_avg | 410.6 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 8 | 91.01 | 13.72 | 0.53 |
| p50 | 1643 | 6335.3 | 22.16 | 3.26 |
| p99 | 8192 | 26579.33 | 60.38 | 6.82 |
| max | 8192 | 28434.6 | 64.51 | 7.88 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Qwen-7B_llm-d_wildchat.json`](../summ_DeepSeek-R1-Distill-Qwen-7B_llm-d_wildchat.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Qwen-7B" and .method=="llm-d" and (.condition=="wildchat" or (.condition==null and .corpus=="wildchat")))' ../per_request_raw.jsonl
  ```
