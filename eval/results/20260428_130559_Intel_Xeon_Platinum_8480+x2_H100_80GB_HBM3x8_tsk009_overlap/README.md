# 20260428_130559_Intel_Xeon_Platinum_8480+x2_H100_80GB_HBM3x8_tsk009_overlap

- timestamp: 20260428_130559
- hw_tag:    Intel_Xeon_Platinum_8480+x2_H100_80GB_HBM3x8
- branch:    feat/ide006-cold-kv-cpu-partial-attention
- commit:    dea755ffbd28632a1c8fc07bc34b3de8ae09f3bc
- python:    /workspace/vllm_dev_prj/bin/python
- vllm:      0.1.dev15917+g0a6396b45

## scope
- baseline: envs/vllm_original_long_ctx.env (split off)
- split_on: envs/ide006_cold_kv_split_on_long_ctx.env (split on, overlap profile)
- max-prompts: 20
- max-tokens:  16
- logprobs:    1

## measurement
- per layer: T_hot_GPU (CUDA event) + T_cold_CPU (perf_counter).
- verdict per layer: cold ≤ hot → 'hidden' (overlap success), cold > hot → 'wait'.
- aggregate: hidden / wait counters + wall-time sums.
