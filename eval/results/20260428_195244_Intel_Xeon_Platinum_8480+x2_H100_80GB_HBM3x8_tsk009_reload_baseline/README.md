# 20260428_195244_Intel_Xeon_Platinum_8480+x2_H100_80GB_HBM3x8_tsk009_reload_baseline

- timestamp: 20260428_195244
- hw_tag:    Intel_Xeon_Platinum_8480+x2_H100_80GB_HBM3x8
- branch:    feat/ide006-cold-kv-cpu-partial-attention
- commit:    dea755ffbd28632a1c8fc07bc34b3de8ae09f3bc

## scope
- baseline: vllm_original_long_ctx.env (cold-tier OFF — A 비교)
- split_on: ide006_cold_kv_long_ctx.env (cold-tier ON, IDE_006 OFF — B)
- max-prompts: 500
- max-tokens:  16

## measurement
- per-reload: T_PCIe (cuda.Event) + size.
- B 회차의 PCIe reload 시간이 진짜 baseline 의 *cold-tier cost*.
- A vs B 의 wall-time 차이 = cold-tier 자체 cost (vLLM 자연 reload+paged FA).

## exit code
- e2e RC: 0

## reload events
- log lines: 0
0

## generate timing
```
[baseline] env=vllm_original_long_ctx.env model=meta-llama/Llama-3.3-70B-Instruct TP=8 max_model_len=16384
[baseline]   kv_transfer_config=None
  [baseline] batched generate: 100 prompts in flight
  [baseline] batched generate complete: 100 prompts in 43.4s (avg 0.43s/prompt)
[baseline] done in 139.9s
[split_on] env=ide006_cold_kv_long_ctx.env model=meta-llama/Llama-3.3-70B-Instruct TP=8 max_model_len=16384
[split_on]   kv_transfer_config={"kv_connector": "OffloadingConnector", "kv_role": "kv_both", "kv_connector_extra_config": {"cpu_bytes_to_use": 549755813888}}
  [split_on] batched generate: 100 prompts in flight
  [split_on] batched generate complete: 100 prompts in 43.6s (avg 0.44s/prompt)
[split_on] done in 171.4s
```
