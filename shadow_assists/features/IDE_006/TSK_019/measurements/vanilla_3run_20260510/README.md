# vanilla 3-run sequential — reference (2026-05-10)

> NEO OFF (vanilla vllm path) 의 500p × 8192 baseline. 3 회 sequential.
> 본 dir = TSK_019 의 best 추적 reference. NEO best (v1.6 2,157 / Phase 3.1+KMP=200 Run 2 2,256) 의 vs vanilla 분모.

## 결과

| Run | output_tps | wall (s) | init (s) |
|---|---:|---:|---:|
| vanilla run 1 | 4,690.7 | 873 | 88 |
| vanilla run 2 | 4,690.4 | 873 | 91 |
| vanilla run 3 | 4,691.0 | 873 | 87 |
| **avg** | **4,690.7** | 873 | 89 |
| **min** | 4,690.4 | 873 | 87 |
| **max** | 4,691.0 | 873 | 91 |
| std | 0.3 | 0 | 2 |
| **CV** | **0.006%** | 0% | 2.2% |

→ vanilla = **perfectly deterministic** (3 회 spread 0.6 tps).

## fact

| 항목 | 값 |
|---|---|
| source dir | `eval/results/20260510_081620_perf_compare_v13_vs_vanilla/` |
| launch script | `eval/run_v13_vs_vanilla_3x3.sh` |
| 측정 시각 (KST) | 2026-05-10 17:16:20 → 18:05:23 |
| NEO env | 모두 unset (vanilla path) |
| `--enable-neo-asymmetric` | OFF |
| model | meta-llama/Llama-3.3-70B-Instruct |
| TP | 8 (H100 80GB × 8) |
| gpu_memory_utilization | 0.85 |
| num_prompts | 500 |
| target_input_len / max_tokens | 8,192 / 8,192 |
| kv_cache_dtype | fp8 |
| async_scheduling | True |
| enforce_eager | False |

## 의미

- 동일 hardware + workload 에서 **vanilla 는 run-간 noise 0**.
- NEO 측정의 3-run variance (CV 5-9%) 는 vanilla 의 0% 와 대조 — variance source 는 NEO path 내부 (CPU OMP thread placement, scheduler bistability) 로 추정.
- best NEO vs vanilla 비교 분모 = **4,691 tps**.

## 파일

| file | 내용 |
|---|---|
| `vanilla_run1/result.json` | run 1 metrics |
| `vanilla_run1/engine.log.stdout` | run 1 full stdout (130 KB) |
| `vanilla_run2/` ... | run 2 동일 |
| `vanilla_run3/` ... | run 3 동일 |
| `suite.log` | 3-run wrapper timeline |
