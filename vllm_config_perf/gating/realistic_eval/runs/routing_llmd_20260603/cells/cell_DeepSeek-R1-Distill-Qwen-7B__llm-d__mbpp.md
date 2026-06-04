# DeepSeek-R1-Distill-Qwen-7B × llm-d × mbpp

## 메트릭 (집계)
| 메트릭 | 값 |
|---|---|
| output_tps | 11984.8 |
| n_ok/n | 198/198 (err 0) |
| wall_total_s | 77.8 |
| total_completion_tokens | 932579 |
| TTFT p50/p99 ms | 22.5/43.4 |
| TPOT p50/p99 ms | 3.3/5.5 |
| accept α (acc/draft) | 0.6029 (471319.0/781800.0) |
| GPU util / mem MiB | 74.1 / 614764 |
| CPU util | 3.7 |
| reqtps_avg | 512.9 |
| concurrency / max_tokens / stream | 32 / 8192 / True |

## per-request 분포 (raw 기반, n=198)
| 통계 | completion_tokens | wall_ms | ttft_ms | tpot_ms |
|---|---|---|---|---|
| min | 199 | 831.5 | 14.32 | 0.55 |
| p50 | 3973 | 7260.64 | 22.46 | 3.26 |
| p99 | 8192 | 26766.59 | 43.39 | 5.5 |
| max | 8192 | 31791.98 | 47.11 | 6.15 |

## raw / 원시 데이터
- 집계 원본: [`summ_DeepSeek-R1-Distill-Qwen-7B_llm-d_mbpp.json`](../summ_DeepSeek-R1-Distill-Qwen-7B_llm-d_mbpp.json)
- per-request raw: 공용 `../per_request_raw.jsonl` 에서 아래 필터
  ```bash
  jq -c 'select(.model=="DeepSeek-R1-Distill-Qwen-7B" and .method=="llm-d" and (.condition=="mbpp" or (.condition==null and .corpus=="mbpp")))' ../per_request_raw.jsonl
  ```
