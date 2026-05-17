# Best — S1-S9 (NEO 원본 100% 정합, 3-run avg 2,238.6 tps, 2026-05-17 KST)

> NEO §4.4 algorithm-correct path 정통 implement 도달. S1-S9 9 단계 rewrite 적용 (NEO `swiftllm/worker/layers/transformer_layer.py` 10/10 함수 정합).
> 측정 archive = [`measurements/neo_s1_s9_500p_3run_20260517/`](measurements/neo_s1_s9_500p_3run_20260517/).
> Timeline 분석 = [`measurements/timeline_v16_s1_s9_20260517/`](measurements/timeline_v16_s1_s9_20260517/).

## 3-run 결과

| Run | tps | wall (s) | shape_mismatch | crash | 측정 dir |
|---|---:|---:|:-:|:-:|---|
| **run 1** | **2,303.4** | **1,763** | 0 ✓ | 0 ✓ | `eval/results/20260517_142533_neo_standard` |
| run 2 | 2,153.6 | 1,889 | 0 ✓ | 0 ✓ | `20260517_145648_neo_standard` |
| run 3 | 2,258.9 | 1,806 | 0 ✓ | 0 ✓ | `20260517_153134_neo_standard` |
| **avg** | **2,238.6** | **1,819** | 0 | 0 | — |
| min / max | 2,153.6 / 2,303.4 | 1,763 / 1,889 | — | — | — |
| std / **CV** | 76.9 / **3.44%** | — | — | — | — |

vs v1.6 best 3-run avg (2,197.4): **+1.9%**.
vs vanilla 3-run avg (4,690.7): **47.7%**.

## S1-S9 적용 영역

| step | 영역 | 위치 |
|---|---|---|
| S1 | `_neo_comm_wait_compute` / `_neo_compute_wait_comm` helper 추가 (NEO `_comm_wait_compute` / `_compute_wait_comm` 정합) | attention.py |
| S2 | `_get_neo_communication_stream` 이미 있음 | attention.py:1372 |
| S3 | Option B (async deque) path 제거 — Option A sync path 만 유지 | attention.py:1133 |
| S4 | `forward_double` 의 `with cuda.stream(s0/s1):` 제거 | sub_batch_executor.py:forward_double |
| S5 | `ThreadPoolExecutor.submit` → `_neo_cdec_compute_cpu` 직접 호출 + `_NeoDirectFuture` wrap | attention.py |
| S6 | `qkvtr_e` event 영역 — 이미 `_xfer_stream.record_event()` 사용 | attention.py |
| S7 | `_get_batch_streams()` (s0/s1 stream pair) dead code 제거 | sub_batch_executor.py |
| S8 | `forward_double` NEO `_forward_pipeline_stage(cur_stage)` ordering 정합 | sub_batch_executor.py:forward_double |
| S9 | result D2D copy 를 `cpu_communication_stream` 위 async + `_compute_wait_comm()` 호출 | attention.py |

→ NEO 원본 `swiftllm/worker/layers/transformer_layer.py + model.py` 의 **10/10 함수 정합** ✓.

## fact

| 항목 | 값 |
|---|---|
| base commit | `64f9e0c48` (v1.6) + S1-S9 변경 |
| branch | `feat/neo-option-b` |
| 측정 시각 (KST) | 2026-05-17 14:25:33 → 16:04:48 |
| launch script | `/tmp/S1S9_500p_3run.sh` |
| 3-run summary | `eval/results/20260517_142533_S1S9_500p_3run/SUMMARY.txt` |
| pacpu .so | v1.6 시점 빌드 (S1-S9 모두 Python only) |

### workload

| | |
|---|---|
| model | meta-llama/Llama-3.3-70B-Instruct |
| tensor_parallel_size | 8 (H100 80GB × 8) |
| max_model_len | 16,384 |
| max_num_seqs | 256 |
| max_num_batched_tokens | 8,192 |
| num_prompts | 500 |
| target_input_len / max_tokens | 8,192 / 8,192 |
| kv_cache_dtype | fp8 |
| async_scheduling | True |
| gpu_memory_utilization | 0.92 |

## NEO 의도 동작 검증

### Static 분석

NEO 원본 함수 10/10 정합 (위 표). 자세한 trace = [`analysis/G_neo_rewrite_plan.md`](analysis/G_neo_rewrite_plan.md).

### Dynamic 분석

| NEO 의도 | 측정 fact (Run 1) | 동작 |
|---|---|:-:|
| KV exclusive ownership | swap_out 1,567 async + 1,821 sync | ✓ |
| b1 sub-batch cdec dispatch | cdec_count ~38k / step, chain fire 74% | ✓ |
| batch interleave (NEO §4.4) | forward_double Stage 0/1 ordering | ✓ |
| paged_attention_cpu 직접 호출 | cdec_wait_avg 0.00 ms (future 즉시 return) | ✓ |
| result copy on cpu_communication_stream | S9 적용 | ✓ |
| 22 strict (shape_mismatch + crash) | 0 / 0 | ✓ |

## paper claim H100 +14% (vanilla 대비) 도달 X — fact

| 원인 | 영역 |
|---|---|
| vllm baseline 차이 | 우리 vanilla = 4,690 tps (paper 시점 vllm 보다 빠름) |
| workload 차이 | paper sweet spot = 짧은 context / 작은 batch. 우리 500p × 8192 long context |
| HBM 영역 | H100 80GB × 8 = 640 GB. vllm max_num_seqs=256 batch 가 이미 HBM 안. NEO KV offload + batch 확장 효과 작음 |

→ NEO §4.4 algorithm-correct 정통 implement 완성 ✓. 단 paper claim 영역 fundamental 도달 어려움.

## 재현 절차

```bash
# v1.6 base + S1-S9 변경 (feat/neo-option-b branch)
git checkout feat/neo-option-b

# pacpu rebuild (v1.6 시점, 단 S1-S9 모두 Python only 라 미필요 가능)
CXX=/tmp/gcc12/usr/bin/g++-12 bash csrc/cpu/pacpu/build.sh llama3_3_70b 8

# 3-run 측정
bash /tmp/S1S9_500p_3run.sh
```

## 한계

- CV 3.44% (v1.6 best 1.62% 보다 큼) — variance 영역 작은 redesign 영향
- min 2,153.6 (Run 2) 가 v1.6 best 1-run (2,156.9) 와 비슷 — variance bottom 도 안정
- paper claim H100 +14% 도달 X (위 fact)
