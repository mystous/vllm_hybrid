# Qwen2.5-7B-Instruct × llm-d × lmsys

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 7513.5 |
| n_ok/n | 499/500 (err 1) |
| wall_total_s | 215.5 |
| total_completion_tokens | 1619143 |
| TTFT p50/p99 ms | 23.3/63.6 |
| TPOT p50/p99 ms | 5.0/13.9 |
| accept α (acc/draft) | 0.6793 (723024.0/1064340.0) |
| GPU util / mem MiB | 63.0 / 614772 |
| CPU util | 3.4 |
| reqtps_avg | 249.1 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=499)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 2 | 38.59 | 14.14 | 0.81 |
| p50 | 629 | 3980.04 | 23.29 | 5.04 |
| p99 | 8192 | 48249.74 | 63.62 | 13.88 |
| max | 8192 | 62536.47 | 66.3 | 15.8 |

## raw / 원시 데이터
- 집계 원본: [`summ_Qwen2.5-7B-Instruct_llm-d_lmsys.json`](../summ_Qwen2.5-7B-Instruct_llm-d_lmsys.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="Qwen2.5-7B-Instruct" and .method=="llm-d" and (.condition=="lmsys" or (.condition==null and .corpus=="lmsys")))' ../per_request_raw.jsonl
  ```
