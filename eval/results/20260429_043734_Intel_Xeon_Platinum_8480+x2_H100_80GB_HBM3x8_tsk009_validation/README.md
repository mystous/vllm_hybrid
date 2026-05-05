# 20260429_043734_Intel_Xeon_Platinum_8480+x2_H100_80GB_HBM3x8_tsk009_validation
- num_prompts: 100
- branch: feat/ide006-cold-kv-cpu-partial-attention
- commit: dea755ffbd28632a1c8fc07bc34b3de8ae09f3bc

## invariants
- 1: B (cold-tier ON, IDE_006 OFF) wall-time vs C (IDE_006 ON, fix) — C ≤ B
- 2: C 회차의 cold-outcome merged > 0 (CPU 작업 활용된 layer 수)

## input_heavy_B (input=15360, output=1024)
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
  [split_on] batched generate complete: 100 prompts in 70.3s (avg 0.70s/prompt)
[split_on] done in 200.0s
```

## input_heavy_C (input=15360, output=1024)
- exit: 0
```
[baseline] env=baseline.env model=meta-llama/Llama-3.3-70B-Instruct TP=8 max_model_len=16384
[baseline]   kv_transfer_config=None
  [baseline] batched generate: 100 prompts in flight
  [baseline] batched generate complete: 100 prompts in 68.1s (avg 0.68s/prompt)
[baseline] done in 203.4s
[split_on] env=split_on.env model=meta-llama/Llama-3.3-70B-Instruct TP=8 max_model_len=16384
[split_on]   kv_transfer_config={"kv_connector": "OffloadingConnector", "kv_role": "kv_both", "enable_cpu_partial_attention": true, "kv_connector_extra_config": {"cpu_bytes_to_use": 549755813888}}
  [split_on] batched generate: 100 prompts in flight
  [split_on] batched generate complete: 100 prompts in 84.2s (avg 0.84s/prompt)
[split_on] done in 220.6s

### cold-outcome (last 3)
(Worker_TP2 pid=3369453) [IDE_006/TSK_009 cold-outcome pid=3369453] #26/50 merged=0 dropped=5200 merged_pct=0.00%
(Worker_TP3 pid=3369454) [IDE_006/TSK_009 cold-outcome pid=3369454] #26/50 merged=0 dropped=5200 merged_pct=0.00%
(Worker_TP5 pid=3369456) [IDE_006/TSK_009 cold-outcome pid=3369456] #26/50 merged=0 dropped=5200 merged_pct=0.00%
```

## output_heavy_B (input=1024, output=15360)
- exit: 0
```
[baseline] env=baseline.env model=meta-llama/Llama-3.3-70B-Instruct TP=8 max_model_len=16384
[baseline]   kv_transfer_config=None
  [baseline] batched generate: 100 prompts in flight
  [baseline] batched generate complete: 100 prompts in 352.3s (avg 3.52s/prompt)
[baseline] done in 505.6s
[split_on] env=split_on.env model=meta-llama/Llama-3.3-70B-Instruct TP=8 max_model_len=16384
[split_on]   kv_transfer_config={"kv_connector": "OffloadingConnector", "kv_role": "kv_both", "kv_connector_extra_config": {"cpu_bytes_to_use": 549755813888}}
  [split_on] batched generate: 100 prompts in flight
  [split_on] batched generate complete: 100 prompts in 392.2s (avg 3.92s/prompt)
[split_on] done in 528.1s
```

## output_heavy_C (input=1024, output=15360)
- exit: 0
```
[baseline] env=baseline.env model=meta-llama/Llama-3.3-70B-Instruct TP=8 max_model_len=16384
[baseline]   kv_transfer_config=None
  [baseline] batched generate: 100 prompts in flight
  [baseline] batched generate complete: 100 prompts in 352.2s (avg 3.52s/prompt)
[baseline] done in 486.8s
[split_on] env=split_on.env model=meta-llama/Llama-3.3-70B-Instruct TP=8 max_model_len=16384
[split_on]   kv_transfer_config={"kv_connector": "OffloadingConnector", "kv_role": "kv_both", "enable_cpu_partial_attention": true, "kv_connector_extra_config": {"cpu_bytes_to_use": 549755813888}}
  [split_on] batched generate: 100 prompts in flight
  [split_on] batched generate complete: 100 prompts in 439.6s (avg 4.40s/prompt)
[split_on] done in 573.9s

### cold-outcome (last 3)
(Worker_TP6 pid=3424056) [IDE_006/TSK_009 cold-outcome pid=3424056] #1/50 merged=0 dropped=200 merged_pct=0.00%
(Worker_TP7 pid=3424057) [IDE_006/TSK_009 cold-outcome pid=3424057] #1/50 merged=0 dropped=200 merged_pct=0.00%
(Worker_TP1 pid=3424051) [IDE_006/TSK_009 cold-outcome pid=3424051] #1/50 merged=0 dropped=200 merged_pct=0.00%
```

## equal_B (input=8192, output=8192)
- exit: 0
```
[baseline] env=baseline.env model=meta-llama/Llama-3.3-70B-Instruct TP=8 max_model_len=16384
[baseline]   kv_transfer_config=None
  [baseline] batched generate: 100 prompts in flight
  [baseline] batched generate complete: 100 prompts in 170.8s (avg 1.71s/prompt)
[baseline] done in 318.8s
[split_on] env=split_on.env model=meta-llama/Llama-3.3-70B-Instruct TP=8 max_model_len=16384
[split_on]   kv_transfer_config={"kv_connector": "OffloadingConnector", "kv_role": "kv_both", "kv_connector_extra_config": {"cpu_bytes_to_use": 549755813888}}
  [split_on] batched generate: 100 prompts in flight
  [split_on] batched generate complete: 100 prompts in 184.7s (avg 1.85s/prompt)
[split_on] done in 314.8s
```

## equal_C (input=8192, output=8192)
- exit: 0
```
[baseline] env=baseline.env model=meta-llama/Llama-3.3-70B-Instruct TP=8 max_model_len=16384
[baseline]   kv_transfer_config=None
  [baseline] batched generate: 100 prompts in flight
  [baseline] batched generate complete: 100 prompts in 171.0s (avg 1.71s/prompt)
[baseline] done in 306.3s
[split_on] env=split_on.env model=meta-llama/Llama-3.3-70B-Instruct TP=8 max_model_len=16384
[split_on]   kv_transfer_config={"kv_connector": "OffloadingConnector", "kv_role": "kv_both", "enable_cpu_partial_attention": true, "kv_connector_extra_config": {"cpu_bytes_to_use": 549755813888}}
  [split_on] batched generate: 100 prompts in flight
  [split_on] batched generate complete: 100 prompts in 202.8s (avg 2.03s/prompt)
[split_on] done in 339.2s

### cold-outcome (last 3)
(Worker_TP6 pid=3472542) [IDE_006/TSK_009 cold-outcome pid=3472542] #13/50 merged=0 dropped=2600 merged_pct=0.00%
(Worker_TP7 pid=3472543) [IDE_006/TSK_009 cold-outcome pid=3472543] #13/50 merged=0 dropped=2600 merged_pct=0.00%
(Worker_TP1 pid=3472537) [IDE_006/TSK_009 cold-outcome pid=3472537] #13/50 merged=0 dropped=2600 merged_pct=0.00%
```
