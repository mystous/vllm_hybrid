# 20260428_222855_Intel_Xeon_Platinum_8480+x2_H100_80GB_HBM3x8_tsk009_three_scen_baseline_fix

- timestamp: 20260428_222855
- mode: baseline_fix
- num_prompts: 100
- branch: feat/ide006-cold-kv-cpu-partial-attention
- commit: dea755ffbd28632a1c8fc07bc34b3de8ae09f3bc

## input_heavy (input=15360, output=1024)
- exit: 0
```
[baseline] env=baseline.env model=meta-llama/Llama-3.3-70B-Instruct TP=8 max_model_len=16384
[baseline]   kv_transfer_config=None
  [baseline] batched generate: 100 prompts in flight
  [baseline] batched generate complete: 100 prompts in 68.1s (avg 0.68s/prompt)
[baseline] done in 164.1s
[split_on] env=split_on.env model=meta-llama/Llama-3.3-70B-Instruct TP=8 max_model_len=16384
[split_on]   kv_transfer_config={"kv_connector": "OffloadingConnector", "kv_role": "kv_both", "enable_cpu_partial_attention": true, "kv_connector_extra_config": {"cpu_bytes_to_use": 549755813888}}
  [split_on] batched generate: 100 prompts in flight
  [split_on] batched generate complete: 100 prompts in 82.2s (avg 0.82s/prompt)
[split_on] done in 216.1s
```

## output_heavy (input=1024, output=15360)
- exit: 0
```
[baseline] env=baseline.env model=meta-llama/Llama-3.3-70B-Instruct TP=8 max_model_len=16384
[baseline]   kv_transfer_config=None
  [baseline] batched generate: 100 prompts in flight
  [baseline] batched generate complete: 100 prompts in 352.0s (avg 3.52s/prompt)
[baseline] done in 502.9s
[split_on] env=split_on.env model=meta-llama/Llama-3.3-70B-Instruct TP=8 max_model_len=16384
[split_on]   kv_transfer_config={"kv_connector": "OffloadingConnector", "kv_role": "kv_both", "enable_cpu_partial_attention": true, "kv_connector_extra_config": {"cpu_bytes_to_use": 549755813888}}
  [split_on] batched generate: 100 prompts in flight
  [split_on] batched generate complete: 100 prompts in 436.1s (avg 4.36s/prompt)
[split_on] done in 571.1s
```

## equal (input=8192, output=8192)
- exit: 0
```
[baseline] env=baseline.env model=meta-llama/Llama-3.3-70B-Instruct TP=8 max_model_len=16384
[baseline]   kv_transfer_config=None
  [baseline] batched generate: 100 prompts in flight
  [baseline] batched generate complete: 100 prompts in 170.9s (avg 1.71s/prompt)
[baseline] done in 321.1s
[split_on] env=split_on.env model=meta-llama/Llama-3.3-70B-Instruct TP=8 max_model_len=16384
[split_on]   kv_transfer_config={"kv_connector": "OffloadingConnector", "kv_role": "kv_both", "enable_cpu_partial_attention": true, "kv_connector_extra_config": {"cpu_bytes_to_use": 549755813888}}
  [split_on] batched generate: 100 prompts in flight
  [split_on] batched generate complete: 100 prompts in 198.3s (avg 1.98s/prompt)
[split_on] done in 335.1s
```
