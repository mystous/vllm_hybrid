# Qwen2.5-7B-Instruct × ngram × wildchat

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 352.7 |
| n_ok/n | 500/500 (err 0) |
| wall_total_s | 1553.0 |
| total_completion_tokens | 547753 |
| TTFT p50/p99 ms | 291.7/352.1 |
| TPOT p50/p99 ms | 117.1/146.1 |
| accept α (acc/draft) | 0.4775 (203026.0/425220.0) |
| GPU util / mem MiB | 61.1 / 632560 |
| CPU util | 90.6 |
| reqtps_avg | 12.3 |
| concurrency / max_tokens / stream | 32 / 2048 / True |

## per-request 분포 (raw 기반, n=500)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 6 | 889.52 | 35.07 | 10.09 |
| p50 | 854 | 84335.07 | 291.65 | 117.1 |
| p99 | 2048 | 255184.77 | 352.1 | 146.06 |
| max | 2048 | 280807.77 | 494.8 | 155.21 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-7B-Instruct_ngram_wildchat.json`](../summ_Qwen2.5-7B-Instruct_ngram_wildchat.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-7B-Instruct" and .method=="ngram" and (.condition=="wildchat" or (.condition==null and .corpus=="wildchat")))' ../per_request_raw.jsonl
  ```
