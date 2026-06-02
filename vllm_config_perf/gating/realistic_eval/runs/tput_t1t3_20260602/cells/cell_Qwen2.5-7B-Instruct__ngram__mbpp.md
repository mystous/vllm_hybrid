# Qwen2.5-7B-Instruct × ngram × mbpp

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 378.4 |
| n_ok/n | 198/198 (err 0) |
| wall_total_s | 374.2 |
| total_completion_tokens | 141574 |
| TTFT p50/p99 ms | 198.0/315.0 |
| TPOT p50/p99 ms | 87.2/130.0 |
| accept α (acc/draft) | 0.4281 (34738.0/81145.0) |
| GPU util / mem MiB | 58.9 / 632560 |
| CPU util | 88.1 |
| reqtps_avg | 13.7 |
| concurrency / max_tokens / stream | 32 / 2048 / True |

## per-request 분포 (raw 기반, n=198)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 42 | 5239.8 | 31.56 | 29.52 |
| p50 | 560 | 50042.19 | 198.04 | 87.21 |
| p99 | 2048 | 157426.05 | 315.03 | 129.96 |
| max | 2048 | 165705.61 | 316.71 | 131.96 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-7B-Instruct_ngram_mbpp.json`](../summ_Qwen2.5-7B-Instruct_ngram_mbpp.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-7B-Instruct" and .method=="ngram" and (.condition=="mbpp" or (.condition==null and .corpus=="mbpp")))' ../per_request_raw.jsonl
  ```
