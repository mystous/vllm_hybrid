# Qwen2.5-7B-Instruct × suffix × mbpp

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 5506.2 |
| n_ok/n | 198/198 (err 0) |
| wall_total_s | 44.5 |
| total_completion_tokens | 245284 |
| TTFT p50/p99 ms | 25.5/47.6 |
| TPOT p50/p99 ms | 6.8/10.4 |
| accept α (acc/draft) | 0.485 (159567.0/329037.0) |
| GPU util / mem MiB | 46.6 / 632560 |
| CPU util | 2.6 |
| reqtps_avg | 187.3 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=198)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 42 | 409.47 | 14.92 | 1.27 |
| p50 | 551 | 3769.97 | 25.47 | 6.76 |
| p99 | 8192 | 20178.96 | 47.62 | 10.45 |
| max | 8192 | 33846.12 | 48.09 | 10.98 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-7B-Instruct_suffix_mbpp.json`](../summ_Qwen2.5-7B-Instruct_suffix_mbpp.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-7B-Instruct" and .method=="suffix" and (.condition=="mbpp" or (.condition==null and .corpus=="mbpp")))' ../per_request_raw.jsonl
  ```
