# DeepSeek-R1-Distill-Qwen-32B × llm-d × wildchat

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 5469.2 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 256.7 |
| total_completion_tokens | 1404112 |
| TTFT p50/p99 ms | 30.1/95.7 |
| TPOT p50/p99 ms | 6.7/13.6 |
| accept α (acc/draft) | 0.5828 (604190.0/1036738.0) |
| GPU util / mem MiB | 89.3 / 1230909 |
| CPU util | 5.8 |
| reqtps_avg | 191.5 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 8 | 75.91 | 21.02 | 1.0 |
| p50 | 1394 | 11046.33 | 30.09 | 6.66 |
| p99 | 8192 | 53716.53 | 95.74 | 13.64 |
| max | 8192 | 53796.05 | 124.78 | 15.73 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Qwen-32B_llm-d_wildchat.json`](../summ_DeepSeek-R1-Distill-Qwen-32B_llm-d_wildchat.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Qwen-32B" and .method=="llm-d" and (.condition=="wildchat" or (.condition==null and .corpus=="wildchat")))' ../per_request_raw.jsonl
  ```
