# 20260428_134903_Intel_Xeon_Platinum_8480+x2_H100_80GB_HBM3x8_tsk009_reload_profile

- timestamp: 20260428_134903
- hw_tag:    Intel_Xeon_Platinum_8480+x2_H100_80GB_HBM3x8
- branch:    feat/ide006-cold-kv-cpu-partial-attention
- commit:    dea755ffbd28632a1c8fc07bc34b3de8ae09f3bc

## scope
- baseline: envs/vllm_original_long_ctx.env
- split_on: envs/ide006_cold_kv_split_on_long_ctx.env (overlap+reload profile)
- max-prompts: 500   (PCIe transfer 시간이 의미 있는 영역)
- max-tokens:  16
- logprobs:    0      (0 — wall-time 만 측정)

## measurement
- per-layer:  T_hot_GPU (CUDA event) + T_cold_CPU (perf_counter).
- per-reload: T_PCIe (cuda.Event) + size.
- 핵심 비교: cold_cpu_ms vs PCIe transfer_ms — TSK_009 (a) 영역의 결정값.
