# Qwen2.5-7B-Instruct × llm-d × humaneval

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 4704.6 |
| n_ok/n | 164/164 (err 0) |
| wall_total_s | 51.4 |
| total_completion_tokens | 241866 |
| TTFT p50/p99 ms | 21.7/59.8 |
| TPOT p50/p99 ms | 4.1/7.5 |
| accept α (acc/draft) | 0.4627 (60088.0/129858.0) |
| GPU util / mem MiB | 63.7 / 614764 |
| CPU util | 3.2 |
| reqtps_avg | 244.4 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=164)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 7 | 79.14 | 16.2 | 1.17 |
| p50 | 276 | 1228.09 | 21.67 | 4.08 |
| p99 | 8192 | 42690.6 | 59.76 | 7.45 |
| max | 8192 | 42732.17 | 60.18 | 9.02 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-7B-Instruct_llm-d_humaneval.json`](../summ_Qwen2.5-7B-Instruct_llm-d_humaneval.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-7B-Instruct" and .method=="llm-d" and (.condition=="humaneval" or (.condition==null and .corpus=="humaneval")))' ../per_request_raw.jsonl
  ```
