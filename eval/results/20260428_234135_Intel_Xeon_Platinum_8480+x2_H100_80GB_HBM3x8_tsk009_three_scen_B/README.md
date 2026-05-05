# 20260428_234135_Intel_Xeon_Platinum_8480+x2_H100_80GB_HBM3x8_tsk009_three_scen_B
- mode: B (cold-tier ON, IDE_006 OFF)
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
[baseline] done in 165.0s
[split_on] env=split_on.env model=meta-llama/Llama-3.3-70B-Instruct TP=8 max_model_len=16384
[split_on]   kv_transfer_config={"kv_connector": "OffloadingConnector", "kv_role": "kv_both", "kv_connector_extra_config": {"cpu_bytes_to_use": 549755813888}}
  [split_on] batched generate: 100 prompts in flight
  [split_on] batched generate complete: 100 prompts in 70.4s (avg 0.70s/prompt)
[split_on] done in 200.0s
```

## output_heavy (input=1024, output=15360)
- exit: 0
```
[baseline] env=baseline.env model=meta-llama/Llama-3.3-70B-Instruct TP=8 max_model_len=16384
[baseline]   kv_transfer_config=None
  [baseline] batched generate: 100 prompts in flight
  [baseline] batched generate complete: 100 prompts in 352.0s (avg 3.52s/prompt)
[baseline] done in 488.8s
[split_on] env=split_on.env model=meta-llama/Llama-3.3-70B-Instruct TP=8 max_model_len=16384
[split_on]   kv_transfer_config={"kv_connector": "OffloadingConnector", "kv_role": "kv_both", "kv_connector_extra_config": {"cpu_bytes_to_use": 549755813888}}
  [split_on] batched generate: 100 prompts in flight
  [split_on] batched generate complete: 100 prompts in 392.7s (avg 3.93s/prompt)
[split_on] done in 522.7s
```

## equal (input=8192, output=8192)
- exit: 0
```
[baseline] env=baseline.env model=meta-llama/Llama-3.3-70B-Instruct TP=8 max_model_len=16384
[baseline]   kv_transfer_config=None
  [baseline] batched generate: 100 prompts in flight
  [baseline] batched generate complete: 100 prompts in 171.0s (avg 1.71s/prompt)
[baseline] done in 301.1s
[split_on] env=split_on.env model=meta-llama/Llama-3.3-70B-Instruct TP=8 max_model_len=16384
[split_on]   kv_transfer_config={"kv_connector": "OffloadingConnector", "kv_role": "kv_both", "kv_connector_extra_config": {"cpu_bytes_to_use": 549755813888}}
  [split_on] batched generate: 100 prompts in flight
  [split_on] batched generate complete: 100 prompts in 184.7s (avg 1.85s/prompt)
[split_on] done in 314.5s
```
